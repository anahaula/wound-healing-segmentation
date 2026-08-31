import argparse
import csv
import hashlib
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
import queue
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from xml.sax.saxutils import escape as xml_escape

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

from area_eval_interface import (
    default_folder_from_base_path,
    enhance_roi_editor_image,
    mask_to_polygon,
    polygon_to_mask,
    resize_mask,
)
from matlab_style_area_eval import (
    AreaEvalArtifacts,
    AreaEvalConfig,
    AreaEvalNoRefResults,
    contour_mask_with_thickness,
    overlay_perimeter,
    run_area_eval_no_reference,
)

import matplotlib.pyplot as plt
from matplotlib.widgets import EllipseSelector, PolygonSelector, RadioButtons, RectangleSelector


# Estruturas simples usadas para transportar dados entre processamento e UI.
@dataclass
class ProcessedTimepoint:
    path: Path
    results: AreaEvalNoRefResults
    artifacts: AreaEvalArtifacts


@dataclass(frozen=True)
class ParsedImageInfo:
    group_key: str
    slot_tag: str
    stem: str
    prefix: str | None = None
    sample_id: str | None = None
    time_tag: str | None = None


TIME_FINDER = re.compile(r"(?i)(\d+)\s*h")
TRAILING_NUMERIC_SUFFIX = re.compile(r"(?i)[\s._-]*\d+(?:[.,]\d+)*$")
IMAGE_NAME_PATTERN = re.compile(
    r"(?ix)^"
    r"(?P<prefix>.+?)"
    r"(?:[\s._-]+(?P<sample>\d+(?:[.,]\d+)?))?"
    r"(?:[\s._-]+(?P<time>\d+\s*h))?"
    r"$"
)
SUPPORTED_IMAGE_SUFFIXES = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp",
    ".webp", ".pgm", ".ppm",
}

# Limiares para classificar brilho medio da imagem
BRIGHT_IMAGE_THRESHOLD = 165.0
BRIGHT_0H_IMAGE_THRESHOLD = 155.0
MID_BRIGHT_IMAGE_THRESHOLD = 140.0
DARK_IMAGE_THRESHOLD = 105.0

# Limiares para imagens 24h
BRIGHT_24H_IMAGE_THRESHOLD = 150.0
MID_24H_IMAGE_THRESHOLD = 120.0
DARK_24H_IMAGE_THRESHOLD = 105.0


# ---------------------------------------------------------------------------
# Utilitarios de descoberta e agrupamento de arquivos
# Esta parte nao processa imagem; ela apenas interpreta nomes e monta grupos.
# ---------------------------------------------------------------------------
def normalize_group_name(text: str) -> str:
    cleaned = text.replace("_", " ").strip()
    cleaned = TIME_FINDER.sub(" ", cleaned)
    cleaned = TRAILING_NUMERIC_SUFFIX.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_.")
    return cleaned or text.strip()


def build_group_key_from_stem(stem: str) -> str:
    cleaned = stem.replace("_", " ").strip()
    cleaned = TIME_FINDER.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_.")
    return normalize_group_name(cleaned) if cleaned else stem.strip()


def sort_slot_key(tag: str) -> tuple[int, int | str]:
    match = TIME_FINDER.fullmatch(tag.strip())
    if match is not None:
        return (0, int(match.group(1)))
    if tag == "img":
        return (2, tag)
    img_match = re.fullmatch(r"img_(\d+)", tag.strip(), re.IGNORECASE)
    if img_match is not None:
        return (3, int(img_match.group(1)))
    time_variant_match = re.fullmatch(r"(\d+)h_(\d+)", tag.strip(), re.IGNORECASE)
    if time_variant_match is not None:
        return (1, int(time_variant_match.group(1)) * 1000 + int(time_variant_match.group(2)))
    return (1, tag)


def natural_group_sort_key(text: str) -> tuple[object, ...]:
    tokens = re.split(r"(\d+(?:[.,]\d+)?)", text.casefold())
    key: list[object] = []
    for token in tokens:
        if not token:
            continue
        normalized = token.replace(",", ".")
        if re.fullmatch(r"\d+(?:\.\d+)?", normalized):
            if "." in normalized:
                key.append((0, float(normalized)))
            else:
                key.append((0, int(normalized)))
        else:
            key.append((1, token.strip()))
    return tuple(key)


def normalize_sample_id(text: str) -> str:
    normalized = text.strip().replace(",", ".")
    match = re.fullmatch(r"(\d+)\.(\d{2})\d+", normalized)
    if match is not None:
        # Alguns arquivos trazem um sufixo numerico extra apos o id real.
        # Ex.: 1.111418 -> 1.11, 1.21419 -> 1.21.
        return f"{match.group(1)}.{match.group(2)}"
    return normalized


def parse_flexible_group_image(path: Path) -> ParsedImageInfo | None:
    stem = path.stem.strip()
    if not stem:
        return None

    stem_normalized = re.sub(r"\s+", " ", stem.replace("_", " ")).strip(" -_.")
    candidates = [stem_normalized, *[part for part in path.parts[-4:-1] if part]]
    time_tag = None
    for candidate in candidates:
        match = TIME_FINDER.search(candidate)
        if match is not None:
            time_tag = f"{int(match.group(1))}h"
            break

    if time_tag == "48h":
        return None

    without_time = TIME_FINDER.sub(" ", stem_normalized)
    without_time = re.sub(r"\s+", " ", without_time).strip(" -_.")
    if not without_time:
        return None

    match = IMAGE_NAME_PATTERN.match(without_time)
    prefix = None
    sample_id = None
    if match is not None:
        prefix_candidate = (match.group("prefix") or "").strip(" -_.")
        sample_candidate = (match.group("sample") or "").strip()
        if prefix_candidate:
            prefix = re.sub(r"\s+", " ", prefix_candidate.replace("_", " ")).strip(" -_.")
        if sample_candidate:
            sample_id = normalize_sample_id(sample_candidate)

    if prefix and sample_id:
        # Agrupa pelo identificador completo da amostra, ex.: "i 1.1".
        group_key = f"{prefix} {sample_id}".strip()
    elif prefix:
        group_key = prefix
    else:
        group_key = build_group_key_from_stem(stem_normalized)

    slot_tag = "img" if time_tag is None else time_tag.lower()
    return ParsedImageInfo(
        group_key=group_key,
        slot_tag=slot_tag,
        stem=stem,
        prefix=prefix,
        sample_id=sample_id,
        time_tag=time_tag.lower() if time_tag is not None else None,
    )


def discover_image_groups(folder: Path, progress_callback=None) -> dict[str, dict[str, Path]]:
    groups: dict[str, dict[str, Path]] = {}
    all_paths = sorted(folder.rglob("*"))
    total = len(all_paths)
    if progress_callback is not None:
        progress_callback(0, total)

    for idx, path in enumerate(all_paths, start=1):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            if progress_callback is not None and (idx == total or (idx % 40) == 0):
                progress_callback(idx, total)
            continue

        parsed = parse_flexible_group_image(path)
        if parsed is None:
            if progress_callback is not None and (idx == total or (idx % 40) == 0):
                progress_callback(idx, total)
            continue

        group_key, slot_tag = parsed.group_key, parsed.slot_tag
        slot = groups.setdefault(group_key, {})
        if slot_tag not in slot:
            slot[slot_tag] = path
        else:
            # Evita perder arquivos quando nao existe timepoint no nome.
            counter = 1
            alt_tag = f"{slot_tag}_{counter}"
            while alt_tag in slot:
                counter += 1
                alt_tag = f"{slot_tag}_{counter}"
            slot[alt_tag] = path

        if progress_callback is not None and (idx == total or (idx % 40) == 0):
            progress_callback(idx, total)

    if progress_callback is not None and total == 0:
        progress_callback(0, 0)
    return groups


class ProcessamentoSemGabaritoApp:
    """App Tkinter que coordena interface, agrupamento, ROI e pipeline.

    Regra pratica de leitura:
    - metodos de UI/app: constroem a janela e apresentam resultados;
    - metodos de processamento: leem imagem, realcam contraste, limpam
      mascaras e atualizam os artefatos calculados.
    """

    DEFAULT_RED_CONTOUR_MIN_LENGTH = 30.0

    def __init__(self, folder_path: str = "", output_dir: str = "", manual_mask_dir: str = "") -> None:
        self.root = tk.Tk()
        self.root.title("Processamento Sem Gabarito")
        self.root.geometry("1440x860")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.folder_var = tk.StringVar(value=folder_path)
        self.output_dir_var = tk.StringVar(value=output_dir)
        self.manual_mask_dir_var = tk.StringVar(value=manual_mask_dir)
        self.group_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Pronto")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="0.0% | 0/0 imagens")
        self.center_progress_title_var = tk.StringVar(value="Processando imagens")
        self.center_progress_detail_var = tk.StringVar(value="")

        self.config = AreaEvalConfig()
        self.group_files: dict[str, dict[str, Path]] = {}
        self.group_labels: dict[str, str] = {}
        self.label_to_group: dict[str, str] = {}
        self.group_order: list[str] = []
        self.current_group_key: str | None = None

        self.roi_by_item: dict[tuple[str, str], np.ndarray] = {}
        self.processed_by_group: dict[str, dict[str, ProcessedTimepoint]] = {}
        self.processing_errors: dict[tuple[str, str], str] = {}

        # Guarda informacoes do modo adaptativo usado em cada imagem
        self.processing_mode_by_item: dict[tuple[str, str], dict[str, float | str]] = {}

        self.ui_busy = False
        self.worker_thread: threading.Thread | None = None
        self.worker_queue: queue.Queue[tuple] = queue.Queue()

        # Pasta temporaria para salvar a imagem adaptada antes do pipeline principal
        self.temp_workdir = tempfile.TemporaryDirectory(prefix="wh_adapt_")

        self._build_layout()
        self._set_ui_busy(False)
        self._refresh_metrics()
        self._refresh_figure()
        self.root.after(120, self._prompt_output_dir_on_start)

    def _on_close(self) -> None:
        try:
            if hasattr(self, "temp_workdir") and self.temp_workdir is not None:
                self.temp_workdir.cleanup()
        except Exception:
            pass
        self.root.destroy()

    # ------------------------------------------------------------------
    # Camada de app/UI
    # Layout, estado da janela, selecao de pasta e navegacao dos grupos.
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Pasta de imagens:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(top, textvariable=self.folder_var, width=95).grid(row=0, column=1, sticky=tk.EW, padx=6)
        self.browse_btn = ttk.Button(top, text="Procurar pasta", command=self._browse_folder)
        self.browse_btn.grid(row=0, column=2, padx=4)
        self.scan_btn = ttk.Button(top, text="Carregar grupos", command=self._scan_folder)
        self.scan_btn.grid(row=0, column=3, padx=4)

        ttk.Label(top, text="Pasta de saida:").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(top, textvariable=self.output_dir_var, width=95).grid(row=1, column=1, sticky=tk.EW, padx=6, pady=(8, 0))
        self.browse_output_btn = ttk.Button(top, text="Escolher saida", command=self._browse_output_dir)
        self.browse_output_btn.grid(row=1, column=2, padx=4, pady=(8, 0))

        ttk.Label(top, text="Pasta de mascaras manuais:").grid(row=2, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(top, textvariable=self.manual_mask_dir_var, width=95).grid(row=2, column=1, sticky=tk.EW, padx=6, pady=(8, 0))
        self.browse_manual_mask_btn = ttk.Button(top, text="Escolher mascaras", command=self._browse_manual_mask_dir)
        self.browse_manual_mask_btn.grid(row=2, column=2, padx=4, pady=(8, 0))

        self.process_all_btn = ttk.Button(top, text="Processar todos os grupos", command=self._start_full_processing)
        self.process_all_btn.grid(row=3, column=1, columnspan=3, sticky=tk.EW, padx=4, pady=(8, 0))

        ttk.Label(top, text="Grupo em exibicao:").grid(row=4, column=0, sticky=tk.W, pady=(8, 0))
        self.group_combo = ttk.Combobox(top, textvariable=self.group_var, state="disabled", width=87)
        self.group_combo.grid(row=4, column=1, sticky=tk.EW, padx=6, pady=(8, 0))
        self.group_combo.bind("<<ComboboxSelected>>", self._on_group_selected)

        nav = ttk.Frame(top)
        nav.grid(row=4, column=2, columnspan=2, sticky=tk.E, padx=4, pady=(8, 0))
        self.prev_btn = ttk.Button(nav, text="Grupo anterior", command=self._show_prev_group)
        self.prev_btn.pack(side=tk.LEFT, padx=3)
        self.next_btn = ttk.Button(nav, text="Proximo grupo", command=self._show_next_group)
        self.next_btn.pack(side=tk.LEFT, padx=3)

        actions = ttk.Frame(top)
        actions.grid(row=5, column=0, columnspan=4, sticky=tk.EW, pady=(8, 0))
        self.redefine_roi_btn = ttk.Button(actions, text="Redefinir ROI do grupo", command=self._redefine_roi_current_group)
        self.redefine_roi_btn.grid(row=0, column=0, padx=(0, 4), sticky=tk.W)
        self.clear_roi_btn = ttk.Button(actions, text="Limpar ROI personalizada", command=self._clear_roi_current_group)
        self.clear_roi_btn.grid(row=0, column=1, padx=4, sticky=tk.W)
        self.reprocess_btn = ttk.Button(actions, text="Processar grupo atual", command=self._reprocess_current_group)
        self.reprocess_btn.grid(row=0, column=2, padx=4, sticky=tk.W)
        self.save_btn = ttk.Button(actions, text="Salvar resultados atuais", command=self._save_results_clicked)
        self.save_btn.grid(row=0, column=3, padx=4, sticky=tk.E)

        top.columnconfigure(1, weight=1)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        left = ttk.Frame(body, padding=8)
        right = ttk.Frame(body, padding=8)
        body.add(left, weight=1)
        body.add(right, weight=3)

        ttk.Label(left, text="Resultados do grupo").pack(anchor=tk.W)
        self.metrics_text = tk.Text(left, width=46, height=30, state=tk.DISABLED)
        self.metrics_text.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        ttk.Label(right, text="Imagens do grupo (contorno automatico)").pack(anchor=tk.W)
        self.figure = Figure(figsize=(11.4, 6.9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        progress_row = ttk.Frame(bottom)
        progress_row.pack(fill=tk.X)
        ttk.Progressbar(progress_row, mode="determinate", variable=self.progress_var, maximum=100.0).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(progress_row, textvariable=self.progress_text_var, width=24, anchor=tk.E).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(bottom, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, pady=(4, 0))

        self.center_progress_panel = tk.Frame(
            self.root,
            bg="#ffffff",
            bd=2,
            relief=tk.RIDGE,
            highlightthickness=1,
            highlightbackground="#777777",
        )
        self.center_progress_panel.columnconfigure(0, weight=1)
        tk.Label(
            self.center_progress_panel,
            textvariable=self.center_progress_title_var,
            bg="#ffffff",
            fg="#111111",
            font=("Segoe UI", 16, "bold"),
        ).grid(row=0, column=0, sticky=tk.EW, padx=22, pady=(18, 6))
        ttk.Progressbar(
            self.center_progress_panel,
            mode="determinate",
            variable=self.progress_var,
            maximum=100.0,
        ).grid(row=1, column=0, sticky=tk.EW, padx=24, pady=(4, 6))
        tk.Label(
            self.center_progress_panel,
            textvariable=self.progress_text_var,
            bg="#ffffff",
            fg="#111111",
            font=("Segoe UI", 14, "bold"),
        ).grid(row=2, column=0, sticky=tk.EW, padx=22, pady=(2, 4))
        tk.Label(
            self.center_progress_panel,
            textvariable=self.center_progress_detail_var,
            bg="#ffffff",
            fg="#333333",
            font=("Segoe UI", 10),
            wraplength=560,
            justify=tk.CENTER,
        ).grid(row=3, column=0, sticky=tk.EW, padx=22, pady=(0, 16))
        self.center_progress_panel.place_forget()

    def _set_ui_busy(self, busy: bool) -> None:
        self.ui_busy = busy
        btn_state = tk.DISABLED if busy else tk.NORMAL
        for btn in (
            self.browse_btn,
            self.browse_output_btn,
            self.browse_manual_mask_btn,
            self.scan_btn,
            self.process_all_btn,
            self.prev_btn,
            self.next_btn,
            self.redefine_roi_btn,
            self.clear_roi_btn,
            self.reprocess_btn,
            self.save_btn,
        ):
            btn.configure(state=btn_state)
        self.group_combo.configure(state="readonly" if (not busy and self.group_order) else "disabled")
        self.root.configure(cursor="watch" if busy else "")
        if busy:
            self._show_center_progress()
        else:
            self._hide_center_progress()

    def _show_center_progress(self) -> None:
        if not hasattr(self, "center_progress_panel"):
            return
        self.center_progress_panel.place(relx=0.5, rely=0.52, anchor=tk.CENTER, width=640, height=175)
        self.center_progress_panel.lift()

    def _hide_center_progress(self) -> None:
        if hasattr(self, "center_progress_panel"):
            self.center_progress_panel.place_forget()

    def _update_center_progress_detail(self, detail: str) -> None:
        self.center_progress_detail_var.set(detail)
        if self.ui_busy:
            self._show_center_progress()

    def _browse_folder(self) -> None:
        folder = filedialog.askdirectory(title="Selecione a pasta com as imagens")
        if folder:
            self.folder_var.set(folder)

    def _browse_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="Selecione a pasta de saida dos resultados")
        if folder:
            self.output_dir_var.set(folder)

    def _browse_manual_mask_dir(self) -> None:
        folder = filedialog.askdirectory(title="Selecione a pasta com as mascaras manuais")
        if folder:
            self.manual_mask_dir_var.set(folder)

    def _prompt_output_dir_on_start(self) -> None:
        if not self.output_dir_var.get().strip():
            folder = filedialog.askdirectory(title="Escolha a pasta de saida dos resultados")
            if folder:
                self.output_dir_var.set(folder)

    def _ensure_output_dir_selected(self) -> bool:
        if self.output_dir_var.get().strip():
            return True
        self._prompt_output_dir_on_start()
        if self.output_dir_var.get().strip():
            return True
        messagebox.showerror("Erro", "Selecione uma pasta de saida para salvar os resultados.")
        return False

    def _scan_folder(self) -> None:
        folder_text = self.folder_var.get().strip()
        if not folder_text:
            messagebox.showerror("Erro", "Informe a pasta de imagens.")
            return
        folder = Path(folder_text)
        if not folder.exists() or not folder.is_dir():
            messagebox.showerror("Erro", f"Pasta invalida: {folder}")
            return

        def on_scan_progress(done: int, total: int) -> None:
            pct = 100.0 if total <= 0 else (100.0 * float(done) / float(total))
            self.progress_var.set(pct)
            self._set_progress_text(pct, done, total, suffix="arquivos")
            self.status_var.set(f"Separando imagens em grupos: {done}/{total}")
            self.root.update_idletasks()

        groups = discover_image_groups(folder, progress_callback=on_scan_progress)
        self.group_files = groups
        self.group_labels = {}
        self.label_to_group = {}
        self.group_order = sorted(groups.keys(), key=natural_group_sort_key)
        self.current_group_key = None
        self.roi_by_item = {}
        self.processed_by_group = {}
        self.processing_errors = {}
        self.processing_mode_by_item = {}

        if not groups:
            self.group_combo["values"] = ()
            self.group_var.set("")
            self._set_progress_text(0.0, 0, 0, suffix="imagens")
            self._refresh_metrics()
            self._refresh_figure()
            messagebox.showwarning("Aviso", "Nenhuma imagem com padrao i|ni + 0h/24h/48h + id foi encontrada na pasta.")
            return

        labels = []
        for group_key in self.group_order:
            times = self._group_times(group_key)
            label = f"{group_key} | {'/'.join(times)}"
            labels.append(label)
            self.group_labels[group_key] = label
            self.label_to_group[label] = group_key

        self.group_combo["values"] = labels
        self._show_group(self.group_order[0], status_text=f"{len(labels)} grupo(s) encontrado(s). Iniciando processamento completo...")
        self.root.after(120, self._auto_process_all_after_scan)

    def _group_times(self, group_key: str) -> list[str]:
        times = sorted(self.group_files.get(group_key, {}).keys(), key=sort_slot_key)
        if not times:
            return []

        # "img" costuma representar a imagem inicial quando o arquivo nao traz
        # o timepoint explicito. Mantem esse slot antes de 24h para que a ROI
        # herdada esteja disponivel durante o processamento.
        preferred = [t for t in ("0h", "img", "24h") if t in times]
        remaining = [t for t in times if t not in ("0h", "img", "24h")]
        return preferred + remaining

    def _all_items(self) -> list[tuple[str, str, Path]]:
        return [
            (group_key, time_tag, self.group_files[group_key][time_tag])
            for group_key in self.group_order
            for time_tag in self._group_times(group_key)
        ]

    def _current_group_items(self) -> list[tuple[str, str, Path]]:
        if self.current_group_key is None:
            return []
        return [
            (self.current_group_key, time_tag, self.group_files[self.current_group_key][time_tag])
            for time_tag in self._group_times(self.current_group_key)
        ]

    def _roi_for_item(self, group_key: str, time_tag: str) -> np.ndarray | None:
        key = (group_key, time_tag)
        roi = self.roi_by_item.get(key)
        if roi is None:
            existing = self.processed_by_group.get(group_key, {}).get(time_tag)
            if existing is not None:
                roi = existing.artifacts.roi_mask
        return None if roi is None else np.asarray(roi).astype(bool)

    # ------------------------------------------------------------------
    # Regras de suporte ao pipeline
    # Estas funcoes ajudam o processamento a decidir tempo, ROI e leitura.
    # ------------------------------------------------------------------
    @staticmethod
    def _is_24h_time_tag(time_tag: str) -> bool:
        return time_tag.strip().lower() == "24h"

    @staticmethod
    def _is_0h_time_tag(time_tag: str | None) -> bool:
        return (time_tag or "").strip().lower() == "0h"

    @staticmethod
    def _uses_0h_brightness_buckets(time_tag: str | None) -> bool:
        # "img" aparece quando o arquivo nao traz tempo explicito; nestes lotes,
        # ele normalmente representa a imagem inicial/0h.
        return (time_tag or "").strip().lower() in {"0h", "img"}

    @staticmethod
    def _resize_roi_mask(roi_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        roi_arr = np.asarray(roi_mask).astype(bool)
        if roi_arr.shape == shape:
            return roi_arr
        return cv2.resize(
            roi_arr.astype(np.uint8),
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    @staticmethod
    def _expand_mask_for_roi(mask: np.ndarray, padding_px: int) -> np.ndarray:
        roi = np.asarray(mask).astype(bool)
        if not np.any(roi):
            return roi
        kernel_size = max(3, (2 * int(max(1, padding_px))) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        expanded = cv2.dilate(roi.astype(np.uint8), kernel, iterations=1) > 0
        return expanded

    def _derive_24h_spatial_context_from_0h(
        self,
        group_key: str,
        image_shape: tuple[int, int],
        processed_cache: dict[tuple[str, str], ProcessedTimepoint] | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
        """Prepara o contexto espacial de 24h a partir da mascara FINAL de 0h.

        IMPORTANTE: esta funcao apenas LE o resultado de 0h. Ela nao altera,
        recalcula ou limpa a mascara de 0h.

        Retorna:
        - search_roi: limite externo amplo para descartar regioes muito distantes;
        - spatial_prior: mapa continuo 0..1 usado como preferencia, nao como corte;
        - ref_mask: mascara de 0h redimensionada para a geometria da imagem 24h;
        - descricao da origem do contexto.

        O prior vale aproximadamente 0.90-1.00 dentro da ferida de 0h e cai
        gradualmente fora dela. Assim, a borda de 24h pode se ajustar aos dados
        da propria imagem sem ficar presa a um corredor central, enquanto regioes
        muito distantes continuam excluidas pela search_roi.
        """
        ref_candidates = (("0h", "0h"), ("img", "img_inicial"))
        cache = processed_cache or {}

        for ref_time_tag, ref_label in ref_candidates:
            ref_key = (group_key, ref_time_tag)
            ref_proc = cache.get(ref_key)
            if ref_proc is None:
                ref_proc = self.processed_by_group.get(group_key, {}).get(ref_time_tag)

            if ref_proc is not None and np.any(ref_proc.artifacts.mask_auto):
                ref_mask = self._resize_roi_mask(ref_proc.artifacts.mask_auto, image_shape)
                h, w = image_shape
                min_dim = min(h, w)

                # O limite externo e deliberadamente mais amplo que o padding antigo
                # de 1.2%. A mascara de 0h agora funciona principalmente como PRIOR,
                # enquanto este limite serve apenas para excluir regioes muito longe.
                max_outside_px = max(18, int(round(min_dim * 0.055)))
                search_roi = self._expand_mask_for_roi(ref_mask, max_outside_px)

                explicit_ref_roi = self.roi_by_item.get(ref_key)
                if explicit_ref_roi is not None and np.any(explicit_ref_roi):
                    explicit_ref_roi = self._resize_roi_mask(explicit_ref_roi, image_shape)
                    # A ROI explicita de 0h continua sendo apenas um limite de seguranca.
                    explicit_expanded = self._expand_mask_for_roi(
                        explicit_ref_roi,
                        max(6, int(round(min_dim * 0.018))),
                    )
                    intersection = search_roi & explicit_expanded
                    if np.any(intersection):
                        search_roi = intersection

                # Distancia para FORA da mascara de 0h: dentro dela a distancia e zero.
                outside_dist = cv2.distanceTransform(
                    (~ref_mask).astype(np.uint8),
                    cv2.DIST_L2,
                    5,
                ).astype(np.float32)

                # Pequeno bonus de profundidade interna (no maximo 10%). Isso evita
                # transformar o prior em um novo "corredor central": toda a mascara
                # de 0h permanece fortemente favorecida, inclusive perto da borda.
                inside_dist = cv2.distanceTransform(
                    ref_mask.astype(np.uint8),
                    cv2.DIST_L2,
                    5,
                ).astype(np.float32)
                inside_max = float(np.max(inside_dist))
                inside_depth = inside_dist / max(inside_max, 1.0)

                decay_scale = max(8.0, float(max_outside_px) * 0.42)
                spatial_prior = np.exp(-outside_dist / decay_scale).astype(np.float32)
                spatial_prior[ref_mask] = (0.90 + (0.10 * inside_depth[ref_mask])).astype(np.float32)
                spatial_prior[~search_roi] = 0.0
                spatial_prior = cv2.GaussianBlur(spatial_prior, (0, 0), sigmaX=3.0, sigmaY=3.0)
                spatial_prior = np.clip(spatial_prior, 0.0, 1.0).astype(np.float32)
                spatial_prior[~search_roi] = 0.0

                return (
                    search_roi,
                    spatial_prior,
                    ref_mask,
                    f"prior_24h_gradual_pela_mascara_{ref_label}",
                )

            # Fallback sem mascara automatica de 0h: uma ROI desenhada pelo usuario
            # ainda pode limitar o 24h, mas nao gera prior artificial.
            explicit_ref_roi = self.roi_by_item.get(ref_key)
            if explicit_ref_roi is not None and np.any(explicit_ref_roi):
                search_roi = self._resize_roi_mask(explicit_ref_roi, image_shape)
                return (
                    search_roi,
                    None,
                    None,
                    f"roi_24h_pre_segmentacao_pela_roi_{ref_label}",
                )

        return None, None, None, "roi_padrao_sem_referencia_0h"

    def _derive_24h_roi_from_0h(
        self,
        group_key: str,
        image_shape: tuple[int, int],
        processed_cache: dict[tuple[str, str], ProcessedTimepoint] | None = None,
    ) -> tuple[np.ndarray | None, str]:
        """Compatibilidade: retorna apenas a ROI externa e sua origem."""
        roi, _prior, _ref_mask, origin = self._derive_24h_spatial_context_from_0h(
            group_key,
            image_shape,
            processed_cache=processed_cache,
        )
        return roi, origin

    @staticmethod
    def _read_image_unicode(path: Path) -> np.ndarray:
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
        except OSError as exc:
            raise ValueError(f"Nao consegui acessar a imagem: {path}") from exc

        if data.size == 0:
            raise ValueError(f"Arquivo de imagem vazio ou corrompido: {path}")

        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raw_image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if raw_image is None:
                raise ValueError(f"Nao consegui decodificar a imagem: {path}")
            if raw_image.ndim == 2:
                image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2BGR)
            elif raw_image.ndim == 3 and raw_image.shape[2] == 4:
                image = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)
            elif raw_image.ndim == 3 and raw_image.shape[2] == 3:
                image = raw_image
            else:
                raise ValueError(f"Formato de imagem nao suportado: {path}")

        return np.ascontiguousarray(image)

    # ------------------------------------------------------------------
    # Mascaras manuais e Dice
    # A avaliacao e apenas impressa no terminal; nada e salvo no Excel.
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_mask_match_key(text: str) -> str:
        value = str(text).strip().casefold()
        value = re.sub(r"(?i)\.(?:tif|tiff|png|jpg|jpeg|bmp|webp|pgm|ppm)$", "", value)
        value = re.sub(r"[_.-]+", " ", value)
        value = re.sub(r"(?i)\bground\s*truth\b", " ", value)
        value = re.sub(r"(?i)\b(?:manual|mask|mascara|máscara|gt|gabarito)\b", " ", value)
        value = re.sub(r"(?i)(\d+)\s*h", r"\1h", value)
        return re.sub(r"[^a-z0-9]+", "", value)

    def _manual_mask_index(self) -> dict[str, list[Path]]:
        folder_text = self.manual_mask_dir_var.get().strip()
        if not folder_text:
            return {}

        folder = Path(folder_text)
        if not folder.exists() or not folder.is_dir():
            print(f"[DICE] Pasta de mascaras manuais invalida: {folder}")
            return {}

        index: dict[str, list[Path]] = {}
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            key = self._normalize_mask_match_key(path.stem)
            if key:
                index.setdefault(key, []).append(path)
        return index

    def _find_manual_mask_path(
        self,
        group_key: str,
        time_tag: str,
        source_path: Path,
        mask_index: dict[str, list[Path]],
    ) -> Path | None:
        if not mask_index:
            return None

        base_time = time_tag.strip().lower().split("_", 1)[0]
        source_parsed = parse_flexible_group_image(source_path)

        candidates = [
            source_path.stem,
            f"{group_key} {base_time}",
            f"{group_key}_{base_time}",
        ]
        if source_parsed is not None:
            candidates.extend([
                source_parsed.stem,
                f"{source_parsed.group_key} {base_time}",
            ])
            if source_parsed.prefix and source_parsed.sample_id:
                candidates.extend([
                    f"{source_parsed.prefix} {base_time} {source_parsed.sample_id}",
                    f"{source_parsed.prefix} {source_parsed.sample_id} {base_time}",
                ])

        normalized_candidates: list[str] = []
        for candidate in candidates:
            key = self._normalize_mask_match_key(candidate)
            if key and key not in normalized_candidates:
                normalized_candidates.append(key)

        # Primeiro tenta igualdade exata apos retirar apenas sufixos tipicos
        # de mascara manual. Isso evita pareamentos ambiguos.
        for key in normalized_candidates:
            paths = mask_index.get(key, [])
            if len(paths) == 1:
                return paths[0]
            if len(paths) > 1:
                # Em duplicatas, prefere o nome mais parecido com o original.
                return min(paths, key=lambda p: abs(len(p.stem) - len(source_path.stem)))

        # Fallback conservador: aceita inclusao de chave apenas quando o par
        # permanece unico. E util para nomes como *_manual_mask.png.
        fallback: list[Path] = []
        for candidate_key in normalized_candidates:
            if len(candidate_key) < 4:
                continue
            for mask_key, paths in mask_index.items():
                if candidate_key in mask_key or mask_key in candidate_key:
                    fallback.extend(paths)
        unique_fallback = list(dict.fromkeys(fallback))
        return unique_fallback[0] if len(unique_fallback) == 1 else None

    @staticmethod
    def _read_binary_manual_mask(path: Path, target_shape: tuple[int, int]) -> np.ndarray:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            raise ValueError(f"Mascara manual vazia: {path}")

        raw = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Nao consegui ler mascara manual: {path}")
        if raw.ndim == 3 and raw.shape[2] == 4:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY)
        elif raw.ndim == 3:
            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        else:
            gray = raw

        mask = gray > 0
        if mask.shape != target_shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        return mask.astype(bool)

    @staticmethod
    def _dice_coefficient(mask_auto: np.ndarray, mask_manual: np.ndarray) -> float:
        auto = np.asarray(mask_auto).astype(bool)
        manual = np.asarray(mask_manual).astype(bool)
        if auto.shape != manual.shape:
            raise ValueError("Mascaras automatica e manual possuem dimensoes diferentes.")

        auto_n = int(np.count_nonzero(auto))
        manual_n = int(np.count_nonzero(manual))
        denom = auto_n + manual_n
        if denom == 0:
            return 1.0
        intersection = int(np.count_nonzero(auto & manual))
        return (2.0 * float(intersection)) / float(denom)

    def _print_dice_summary(self, success: list[tuple[str, str]]) -> None:
        folder_text = self.manual_mask_dir_var.get().strip()
        if not folder_text:
            print("\n[DICE] Pasta de mascaras manuais nao selecionada; Dice nao calculado.\n")
            return

        mask_index = self._manual_mask_index()
        if not mask_index:
            print("\n[DICE] Nenhuma mascara manual valida encontrada; Dice nao calculado.\n")
            return

        values_by_time: dict[str, list[float]] = {}
        all_values: list[float] = []
        matched = 0
        missing: list[str] = []

        print("\n" + "=" * 72)
        print("DICE - COMPARACAO COM MASCARAS MANUAIS")
        print("=" * 72)

        for group_key, time_tag in success:
            proc = self.processed_by_group.get(group_key, {}).get(time_tag)
            source_path = self.group_files.get(group_key, {}).get(time_tag)
            if proc is None or source_path is None:
                continue

            manual_path = self._find_manual_mask_path(
                group_key,
                time_tag,
                source_path,
                mask_index,
            )
            if manual_path is None:
                missing.append(f"{group_key} {time_tag} -> {source_path.name}")
                continue

            try:
                manual_mask = self._read_binary_manual_mask(
                    manual_path,
                    proc.artifacts.mask_auto.shape,
                )
                dice = self._dice_coefficient(proc.artifacts.mask_auto, manual_mask)
            except Exception as exc:
                print(f"[DICE] ERRO {source_path.name}: {exc}")
                continue

            base_time = time_tag.strip().lower().split("_", 1)[0]
            values_by_time.setdefault(base_time, []).append(dice)
            all_values.append(dice)
            matched += 1
            print(
                f"[DICE] {source_path.name} | mascara={manual_path.name} | "
                f"Dice={dice:.4f}"
            )

        print("-" * 72)
        for base_time in sorted(values_by_time.keys(), key=sort_slot_key):
            values = np.asarray(values_by_time[base_time], dtype=np.float64)
            print(
                f"[DICE] Media {base_time.upper()}: {float(np.mean(values)):.4f} "
                f"(n={values.size})"
            )
        if all_values:
            values = np.asarray(all_values, dtype=np.float64)
            print(f"[DICE] MEDIA GERAL: {float(np.mean(values)):.4f} (n={values.size})")
        else:
            print("[DICE] Nenhum par automatico/manual foi encontrado.")
        print(f"[DICE] Mascaras pareadas: {matched}/{len(success)}")
        if missing:
            print("[DICE] Sem mascara manual correspondente:")
            for item in missing:
                print(f"       - {item}")
        print("=" * 72 + "\n")

    # ------------------------------------------------------------------
    # Pipeline de processamento de imagem
    # Este bloco contem o nucleo tecnico: realce, segmentacao e limpeza.
    # ------------------------------------------------------------------
    @staticmethod
    def _mean_gray_intensity(image_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    @staticmethod
    def _adjust_gamma(image_bgr: np.ndarray, gamma: float) -> np.ndarray:
        gamma = max(float(gamma), 1e-6)
        table = np.array(
            [np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)],
            dtype=np.uint8
        )
        return cv2.LUT(image_bgr, table)

    @staticmethod
    def _clahe_lab(
        image_bgr: np.ndarray,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8)
    ) -> np.ndarray:
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l2 = clahe.apply(l)
        merged = cv2.merge((l2, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _homomorphic_gray_u8(gray_u8: np.ndarray, sigma: float = 18.0, ksize: int = 81) -> np.ndarray:
        src = gray_u8.astype(np.float32) / 255.0
        src = np.clip(src, 1e-6, 1.0)
        img_log = np.log1p(src)

        k = max(3, int(ksize))
        if (k % 2) == 0:
            k += 1

        illum = cv2.GaussianBlur(img_log, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))
        reflect = img_log - illum
        out = np.expm1(reflect)
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _local_clahe_gray(gray_u8: np.ndarray, clip_limit: float = 2.4, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)
        return clahe.apply(gray_u8)

    @staticmethod
    def _edge_texture_confidence_gray(gray_u8: np.ndarray, local_win: int = 13) -> tuple[np.ndarray, np.ndarray]:
        src = gray_u8.astype(np.float32) / 255.0

        gx = cv2.Scharr(src, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(src, cv2.CV_32F, 0, 1)
        grad = (1.20 * np.abs(gx)) + (0.35 * np.abs(gy))
        grad = cv2.normalize(grad, None, 0.0, 1.0, cv2.NORM_MINMAX)

        win = max(3, int(local_win))
        if (win % 2) == 0:
            win += 1
        mean = cv2.blur(src, (win, win))
        mean2 = cv2.blur(src * src, (win, win))
        texture = np.sqrt(np.maximum(mean2 - (mean * mean), 0.0))
        texture = cv2.normalize(texture, None, 0.0, 1.0, cv2.NORM_MINMAX)

        low_texture_gate = np.clip(1.0 - (0.72 * texture), 0.0, 1.0)
        edge_support = grad * ((0.42) + (0.58 * low_texture_gate))
        edge_support = cv2.normalize(edge_support, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        texture_noise = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return edge_support, texture_noise

    @staticmethod
    def _bright_image_border_enhance(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        homo = ProcessamentoSemGabaritoApp._homomorphic_gray_u8(gray, sigma=16.0, ksize=71)
        eq = ProcessamentoSemGabaritoApp._local_clahe_gray(homo, clip_limit=2.3, tile_grid_size=(8, 8))
        eq = cv2.bilateralFilter(eq, d=9, sigmaColor=22, sigmaSpace=22)
        eq = cv2.medianBlur(eq, 5)

        # Remove textura celular miuda para que a segmentacao favoreca bordas maiores.
        se_denoise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        base = cv2.morphologyEx(eq, cv2.MORPH_OPEN, se_denoise)
        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, se_denoise)

        se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        se_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

        top_small = cv2.morphologyEx(base, cv2.MORPH_TOPHAT, se_small)
        bot_small = cv2.morphologyEx(base, cv2.MORPH_BLACKHAT, se_small)
        top_large = cv2.morphologyEx(base, cv2.MORPH_TOPHAT, se_large)
        bot_large = cv2.morphologyEx(base, cv2.MORPH_BLACKHAT, se_large)

        grad = cv2.morphologyEx(base, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # Reforca frentes/bordas e reduz restos celulares pequenos no centro.
        border_support = cv2.addWeighted(top_large, 0.82, grad, 0.58, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, bot_large, 0.30, 0.0)
        central_noise = cv2.addWeighted(top_small, 1.05, bot_small, 1.15, 0.0)

        refined = cv2.addWeighted(base, 1.0, border_support, 0.52, 0.0)
        refined = cv2.addWeighted(refined, 1.0, central_noise, -0.72, 0.0)
        refined = cv2.GaussianBlur(refined, (5, 5), 0)

        return cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _bright_image_homogeneous_border_enhance(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Homomorfico mais amplo: corrige fundo/iluminacao sem tentar criar borda artificial.
        homo = ProcessamentoSemGabaritoApp._homomorphic_gray_u8(gray, sigma=24.0, ksize=101)
        smooth = cv2.bilateralFilter(homo, d=9, sigmaColor=26, sigmaSpace=30)
        smooth = cv2.medianBlur(smooth, 5)

        # CLAHE fraco: devolve contraste de borda sem trazer demais a textura miuda do fundo.
        eq = ProcessamentoSemGabaritoApp._local_clahe_gray(smooth, clip_limit=1.18, tile_grid_size=(8, 8))

        base = cv2.GaussianBlur(eq, (5, 5), 0)
        se_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        base = cv2.morphologyEx(base, cv2.MORPH_OPEN, se_bg)
        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, se_bg)
        base = cv2.bilateralFilter(base, d=5, sigmaColor=14, sigmaSpace=16)
        shape_base = cv2.morphologyEx(
            base,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
        )

        se_long = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        se_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        top_long = cv2.morphologyEx(shape_base, cv2.MORPH_TOPHAT, se_long)
        bot_long = cv2.morphologyEx(shape_base, cv2.MORPH_BLACKHAT, se_long)
        grad = cv2.morphologyEx(shape_base, cv2.MORPH_GRADIENT, se_edge)
        top_small = cv2.morphologyEx(shape_base, cv2.MORPH_TOPHAT, se_small)
        bot_small = cv2.morphologyEx(shape_base, cv2.MORPH_BLACKHAT, se_small)
        edge_support, texture_noise = ProcessamentoSemGabaritoApp._edge_texture_confidence_gray(base, local_win=17)

        border_support = cv2.addWeighted(top_long, 0.52, bot_long, 0.42, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, grad, 0.54, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, edge_support, 0.58, 0.0)
        fine_texture = cv2.addWeighted(top_small, 0.42, bot_small, 0.54, 0.0)
        fine_texture = cv2.addWeighted(fine_texture, 1.0, texture_noise, 0.44, 0.0)

        refined = cv2.addWeighted(base, 0.98, border_support, 0.34, 0.0)
        refined = cv2.addWeighted(refined, 1.0, fine_texture, -0.40, 0.0)
        refined = cv2.bilateralFilter(refined, d=3, sigmaColor=10, sigmaSpace=12)
        return cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _fill_mask_holes(mask: np.ndarray) -> np.ndarray:
        source = np.asarray(mask).astype(bool)
        if not np.any(source):
            return source

        flood = (source.astype(np.uint8) * 255)
        padded = cv2.copyMakeBorder(flood, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        fill_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(padded, fill_mask, (0, 0), 255)
        background = padded[1:-1, 1:-1] > 0
        holes = ~background & ~source
        return source | holes

    @staticmethod
    def _keep_largest_mask_component(mask: np.ndarray) -> np.ndarray:
        source = np.asarray(mask).astype(bool)
        if not np.any(source):
            return source

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            source.astype(np.uint8),
            connectivity=8,
            ltype=cv2.CV_32S,
        )
        if n_labels <= 1:
            return source
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest

    @staticmethod
    def _smooth_mask_boundary(
        mask: np.ndarray,
        roi: np.ndarray | None = None,
        *,
        close_scale: float = 1.0,
        open_scale: float = 1.0,
        blur_scale: float = 1.0,
    ) -> np.ndarray:
        source = np.asarray(mask).astype(bool)
        if not np.any(source):
            return source

        roi_bool = None if roi is None else np.asarray(roi).astype(bool)
        if roi_bool is not None and roi_bool.shape == source.shape:
            source = source & roi_bool

        h, w = source.shape
        min_dim = min(h, w)
        close_k = max(5, int(round(min_dim * 0.012 * float(close_scale))))
        open_k = max(3, int(round(min_dim * 0.006 * float(open_scale))))
        blur_k = max(5, int(round(min_dim * 0.010 * float(blur_scale))))

        if (close_k % 2) == 0:
            close_k += 1
        if (open_k % 2) == 0:
            open_k += 1
        if (blur_k % 2) == 0:
            blur_k += 1

        smooth = source.astype(np.uint8) * 255
        smooth = cv2.morphologyEx(
            smooth,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
        )
        smooth = cv2.morphologyEx(
            smooth,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)),
        )
        smooth = cv2.GaussianBlur(smooth, (blur_k, blur_k), 0)
        smooth = smooth >= 127

        if roi_bool is not None and roi_bool.shape == source.shape:
            smooth = smooth & roi_bool

        smooth = ProcessamentoSemGabaritoApp._fill_mask_holes(smooth)
        smooth = ProcessamentoSemGabaritoApp._keep_largest_mask_component(smooth)
        return smooth

    def _smooth_24h_mask_boundary(self, mask: np.ndarray, roi: np.ndarray) -> np.ndarray:
        smooth = self._smooth_mask_boundary(
            mask,
            roi,
            close_scale=1.55,
            open_scale=1.10,
            blur_scale=0.85,
        )
        if not np.any(smooth):
            return smooth

        smooth = cv2.morphologyEx(
            smooth.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        ) > 0
        smooth = self._fill_mask_holes(smooth & np.asarray(roi).astype(bool))
        smooth = self._keep_largest_mask_component(smooth & np.asarray(roi).astype(bool))
        return smooth

    @staticmethod
    def _red_contour_components(overlay_rgb_u8: np.ndarray) -> np.ndarray:
        c0 = overlay_rgb_u8[:, :, 0].astype(np.int16)
        c1 = overlay_rgb_u8[:, :, 1].astype(np.int16)
        c2 = overlay_rgb_u8[:, :, 2].astype(np.int16)

        red_rgb = (
            (c0 >= 150)
            & (c1 <= 50)
            & (c2 <= 50)
            & (c0 - c1 >= 50)
            & (c0 - c2 >= 50)
        )
        red_bgr = (
            (c2 >= 150)
            & (c1 <= 50)
            & (c0 <= 50)
            & (c2 - c1 >= 50)
            & (c2 - c0 >= 50)
        )
        return red_rgb | red_bgr

    def _filter_small_red_contours_overlay(
        self,
        overlay_rgb_u8: np.ndarray,
        base_rgb_u8: np.ndarray,
    ) -> tuple[np.ndarray, list[float]]:
        red_bin = self._red_contour_components(overlay_rgb_u8).astype(np.uint8)
        if not np.any(red_bin):
            return overlay_rgb_u8.copy(), []

        num_labels, labels = cv2.connectedComponents(red_bin, connectivity=8)
        filtered_overlay = overlay_rgb_u8.copy()
        contour_lengths: list[float] = []
        min_length = float(getattr(self, "red_contour_min_length", self.DEFAULT_RED_CONTOUR_MIN_LENGTH))

        for label in range(1, num_labels):
            component_mask = (labels == label).astype(np.uint8)
            contours, _ = cv2.findContours(
                component_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE,
            )
            if not contours:
                continue

            length = float(cv2.arcLength(contours[0], True))
            contour_lengths.append(length)
            if length < min_length:
                filtered_overlay[labels == label] = base_rgb_u8[labels == label]

        return filtered_overlay, contour_lengths

    def _create_mask_from_red_contours(
        self,
        overlay_rgb_u8: np.ndarray,
    ) -> np.ndarray:
        red_bin = self._red_contour_components(overlay_rgb_u8).astype(np.uint8)
        if not np.any(red_bin):
            return np.zeros(overlay_rgb_u8.shape[:2], dtype=bool)

        h, w = red_bin.shape
        inverted_mask = np.where(red_bin > 0, 0, 255).astype(np.uint8)
        filled_from_outside = inverted_mask.copy()

        for point in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
            cv2.floodFill(filled_from_outside, None, point, 127)

        return filled_from_outside == 255

    def _apply_filtered_red_contour_artifacts(
        self,
        results: AreaEvalNoRefResults,
        artifacts: AreaEvalArtifacts,
    ) -> None:
        overlay_rgb = getattr(artifacts, "contour_overlay_rgb_u8", None)
        base_rgb = getattr(artifacts, "base_rgb_u8", None)
        if overlay_rgb is None or base_rgb is None:
            return

        filtered_overlay, red_lengths = self._filter_small_red_contours_overlay(
            overlay_rgb,
            base_rgb,
        )
        artifacts.filtered_contour_overlay_rgb_u8 = filtered_overlay
        artifacts.filtered_red_contour_lengths = red_lengths

        red_mask = self._create_mask_from_red_contours(filtered_overlay)
        if np.any(red_mask):
            roi_mask = getattr(artifacts, "roi_mask", None)
            # O overlay vermelho serve para visualizacao; aqui ele volta a
            # virar mascara binaria para podermos limpar artefatos pequenos.
            red_mask = self._smooth_mask_boundary(red_mask, roi_mask)
        if np.any(red_mask):
            artifacts.filtered_contour_mask = contour_mask_with_thickness(
                red_mask,
                radius=self.config.r_auto,
            )
            artifacts.mask_auto = red_mask
            artifacts.contour_mask = artifacts.filtered_contour_mask
            artifacts.contour_overlay_rgb_u8 = filtered_overlay
            results.area_auto = int(np.count_nonzero(red_mask))
        else:
            artifacts.filtered_contour_mask = getattr(
                artifacts,
                "contour_mask",
                np.zeros(overlay_rgb.shape[:2], dtype=bool),
            )

    def _cleanup_bright_mode_artifacts(
        self,
        results: AreaEvalNoRefResults,
        artifacts: AreaEvalArtifacts,
        time_tag: str | None = None,
    ) -> None:
        mask = np.asarray(artifacts.mask_auto).astype(bool)
        roi = np.asarray(artifacts.roi_mask).astype(bool)
        if mask.shape != roi.shape or not np.any(mask):
            return

        cleaned = self._fill_mask_holes(mask)
        cleaned = cv2.morphologyEx(
            cleaned.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
        ) > 0
        cleaned = self._fill_mask_holes(cleaned)
        cleaned = self._keep_largest_mask_component(cleaned & roi)
        if self._is_0h_time_tag(time_tag):
            # 0h clara costuma vazar lateralmente; por isso recebe duas
            # correcoes extras antes da suavizacao final do contorno.
            cleaned = self._stabilize_0h_bright_mask(cleaned, roi)
            cleaned = self._suppress_0h_bright_lateral_leakage(cleaned, roi)
        cleaned = self._smooth_mask_boundary(cleaned, roi)
        if not np.any(cleaned):
            return

        artifacts.mask_auto = cleaned
        artifacts.contour_mask = contour_mask_with_thickness(cleaned, radius=self.config.r_auto)
        artifacts.contour_overlay_rgb_u8 = overlay_perimeter(
            artifacts.base_rgb_u8,
            artifacts.contour_mask,
            self.config.col_auto,
        )
        results.area_auto = int(np.count_nonzero(cleaned))

    def _stabilize_0h_bright_mask(self, mask: np.ndarray, roi: np.ndarray) -> np.ndarray:
        source = np.asarray(mask).astype(bool)
        roi_bool = np.asarray(roi).astype(bool)
        if source.shape != roi_bool.shape or not np.any(source):
            return source

        h, w = source.shape
        if min(h, w) < 80:
            return source

        # Remove recortes finos e granulado interno que em 0h clara costumam
        # surgir no fundo da cicatriz e invadir a mascara principal.
        opened = cv2.morphologyEx(
            source.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        ) > 0
        closed = cv2.morphologyEx(
            opened.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        ) > 0
        closed = self._fill_mask_holes(closed & roi_bool)
        closed = self._keep_largest_mask_component(closed & roi_bool)
        if not np.any(closed):
            return source

        preserved_fraction = float(np.count_nonzero(closed)) / max(float(np.count_nonzero(source)), 1.0)
        if preserved_fraction < 0.58:
            return source
        return closed

    def _suppress_0h_bright_lateral_leakage(self, mask: np.ndarray, roi: np.ndarray) -> np.ndarray:
        source = np.asarray(mask).astype(bool)
        roi_bool = np.asarray(roi).astype(bool)
        if source.shape != roi_bool.shape or not np.any(source):
            return source

        h, w = source.shape
        if w < 80:
            return source

        guard = max(6, int(round(0.12 * w)))
        border = max(2, int(round(0.015 * w)))
        side_pixels = int(np.count_nonzero(source[:, :guard]) + np.count_nonzero(source[:, w - guard:]))
        border_touch = bool(np.any(source[:, :border]) or np.any(source[:, w - border:]))
        total_pixels = int(np.count_nonzero(source))
        side_fraction = float(side_pixels) / max(float(total_pixels), 1.0)
        area_fraction = float(total_pixels) / max(float(source.size), 1.0)

        # Em 0h clara, vazamento lateral costuma aparecer como contato com as bordas
        # ou como excesso de mascara nas faixas laterais. Mantem o resultado antigo
        # quando o sinal de vazamento e fraco.
        if not border_touch and side_fraction < 0.035:
            return source
        if area_fraction < 0.12 and side_fraction < 0.08:
            return source

        corridor = np.zeros_like(source, dtype=bool)
        corridor[:, guard:w - guard] = True
        candidate = source & roi_bool & corridor

        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        ) > 0
        candidate = self._fill_mask_holes(candidate)
        candidate = self._keep_largest_mask_component(candidate & roi_bool & corridor)
        candidate_pixels = int(np.count_nonzero(candidate))
        if candidate_pixels == 0:
            return source

        preserved_fraction = float(candidate_pixels) / max(float(total_pixels), 1.0)
        if preserved_fraction < 0.45:
            return source
        return candidate

    @staticmethod
    def _mid_bright_image_enhance(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        homo = ProcessamentoSemGabaritoApp._homomorphic_gray_u8(gray, sigma=12.0, ksize=55)
        eq = ProcessamentoSemGabaritoApp._local_clahe_gray(homo, clip_limit=2.2, tile_grid_size=(8, 8))
        eq = cv2.bilateralFilter(eq, d=7, sigmaColor=22, sigmaSpace=22)

        # Base suavizada para reduzir granulado, mantendo estrutura maior das frentes.
        base = cv2.GaussianBlur(eq, (5, 5), 0)

        se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        se_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        se_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

        top_small = cv2.morphologyEx(base, cv2.MORPH_TOPHAT, se_small)
        bot_small = cv2.morphologyEx(base, cv2.MORPH_BLACKHAT, se_small)
        top_mid = cv2.morphologyEx(base, cv2.MORPH_TOPHAT, se_mid)
        bot_mid = cv2.morphologyEx(base, cv2.MORPH_BLACKHAT, se_mid)
        top_large = cv2.morphologyEx(base, cv2.MORPH_TOPHAT, se_large)
        grad = cv2.morphologyEx(base, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        border_support = cv2.addWeighted(top_large, 0.72, top_mid, 0.55, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, grad, 0.62, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, bot_mid, 0.22, 0.0)
        central_noise = cv2.addWeighted(top_small, 0.72, bot_small, 0.78, 0.0)

        refined = cv2.addWeighted(base, 1.0, border_support, 0.68, 0.0)
        refined = cv2.addWeighted(refined, 1.0, central_noise, -0.40, 0.0)
        refined = cv2.GaussianBlur(refined, (3, 3), 0)

        return cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _medium_0h_image_enhance(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 0h medio: mais correcao de fundo que o modo intermediario antigo,
        # mas com suavizacao para nao transformar textura celular em borda.
        homo = ProcessamentoSemGabaritoApp._homomorphic_gray_u8(gray, sigma=18.0, ksize=83)
        eq = ProcessamentoSemGabaritoApp._local_clahe_gray(homo, clip_limit=2.9, tile_grid_size=(8, 8))
        eq = cv2.bilateralFilter(eq, d=11, sigmaColor=34, sigmaSpace=34)
        eq = cv2.medianBlur(eq, 5)

        base = cv2.GaussianBlur(eq, (5, 5), 0)
        se_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        base = cv2.morphologyEx(base, cv2.MORPH_OPEN, se_bg)
        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, se_bg)
        shape_base = cv2.bilateralFilter(base, d=7, sigmaColor=22, sigmaSpace=26)
        shape_base = cv2.morphologyEx(shape_base, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))

        se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        se_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        se_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        grad = cv2.morphologyEx(shape_base, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        top_small = cv2.morphologyEx(shape_base, cv2.MORPH_TOPHAT, se_small)
        bot_small = cv2.morphologyEx(shape_base, cv2.MORPH_BLACKHAT, se_small)
        top_mid = cv2.morphologyEx(shape_base, cv2.MORPH_TOPHAT, se_mid)
        bot_mid = cv2.morphologyEx(shape_base, cv2.MORPH_BLACKHAT, se_mid)
        top_large = cv2.morphologyEx(shape_base, cv2.MORPH_TOPHAT, se_large)
        edge_support, texture_noise = ProcessamentoSemGabaritoApp._edge_texture_confidence_gray(base, local_win=13)

        border_support = cv2.addWeighted(top_large, 0.42, top_mid, 0.38, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, bot_mid, 0.20, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, grad, 0.58, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, edge_support, 0.72, 0.0)
        small_texture = cv2.addWeighted(top_small, 0.32, bot_small, 0.42, 0.0)
        small_texture = cv2.addWeighted(small_texture, 1.0, texture_noise, 0.34, 0.0)

        refined = cv2.addWeighted(base, 1.0, border_support, 0.66, 0.0)
        refined = cv2.addWeighted(refined, 1.0, small_texture, -0.42, 0.0)
        refined = cv2.bilateralFilter(refined, d=5, sigmaColor=16, sigmaSpace=18)
        return cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _dark_image_edge_enhance_core(
        gray_u8: np.ndarray,
        *,
        homo_sigma: float,
        homo_ksize: int,
        clahe_clip: float,
        bg_kernel: int,
        local_win: int,
        border_gain: float,
        texture_penalty_gain: float,
        shape_open_kernel: int = 11,
        extra_texture_suppression: bool = False,
        base_blur_kernel: int = 5,
        final_blur_kernel: int = 3,
    ) -> np.ndarray:
        # Nucleo comum para imagens escuras: corrige iluminacao, reduz textura
        # miuda e reforca bordas maiores em varias escalas.
        homo = ProcessamentoSemGabaritoApp._homomorphic_gray_u8(
            gray_u8,
            sigma=homo_sigma,
            ksize=homo_ksize,
        )
        eq = ProcessamentoSemGabaritoApp._local_clahe_gray(
            homo,
            clip_limit=clahe_clip,
            tile_grid_size=(8, 8),
        )
        eq = cv2.bilateralFilter(eq, d=11, sigmaColor=30, sigmaSpace=32)
        eq = cv2.medianBlur(eq, 5)

        base_k = max(3, int(base_blur_kernel))
        if (base_k % 2) == 0:
            base_k += 1
        base = cv2.GaussianBlur(eq, (base_k, base_k), 0)
        se_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_kernel, bg_kernel))
        base = cv2.morphologyEx(base, cv2.MORPH_OPEN, se_bg)
        base = cv2.morphologyEx(base, cv2.MORPH_CLOSE, se_bg)

        shape_base = cv2.bilateralFilter(base, d=7, sigmaColor=20, sigmaSpace=22)
        shape_base = cv2.morphologyEx(
            shape_base,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shape_open_kernel, shape_open_kernel)),
        )
        if extra_texture_suppression:
            shape_base = cv2.morphologyEx(
                shape_base,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            )
            shape_base = cv2.bilateralFilter(shape_base, d=9, sigmaColor=18, sigmaSpace=24)

        edge_support, texture_noise = ProcessamentoSemGabaritoApp._edge_texture_confidence_gray(
            shape_base,
            local_win=local_win,
        )
        grad = cv2.morphologyEx(
            shape_base,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )

        se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        se_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        se_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (33, 33))
        top_small = cv2.morphologyEx(shape_base, cv2.MORPH_TOPHAT, se_small)
        bot_small = cv2.morphologyEx(shape_base, cv2.MORPH_BLACKHAT, se_small)
        top_mid = cv2.morphologyEx(shape_base, cv2.MORPH_TOPHAT, se_mid)
        bot_mid = cv2.morphologyEx(shape_base, cv2.MORPH_BLACKHAT, se_mid)
        top_large = cv2.morphologyEx(shape_base, cv2.MORPH_TOPHAT, se_large)
        bot_large = cv2.morphologyEx(shape_base, cv2.MORPH_BLACKHAT, se_large)

        border_support = cv2.addWeighted(top_large, 0.56, top_mid, 0.42, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, bot_mid, 0.20, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, bot_large, 0.12, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, grad, 0.74, 0.0)
        border_support = cv2.addWeighted(border_support, 1.0, edge_support, 0.88, 0.0)

        texture_penalty = cv2.addWeighted(top_small, 0.34, bot_small, 0.46, 0.0)
        texture_penalty = cv2.addWeighted(texture_penalty, 1.0, texture_noise, 0.44, 0.0)
        if extra_texture_suppression:
            texture_penalty = cv2.addWeighted(texture_penalty, 1.0, top_mid, 0.22, 0.0)
            texture_penalty = cv2.addWeighted(texture_penalty, 1.0, bot_mid, 0.26, 0.0)

        refined = cv2.addWeighted(base, 1.0, border_support, border_gain, 0.0)
        refined = cv2.addWeighted(refined, 1.0, texture_penalty, -texture_penalty_gain, 0.0)
        refined = cv2.bilateralFilter(refined, d=5, sigmaColor=16, sigmaSpace=18)
        final_k = max(1, int(final_blur_kernel))
        if final_k > 1:
            if (final_k % 2) == 0:
                final_k += 1
            refined = cv2.GaussianBlur(refined, (final_k, final_k), 0)
        return cv2.cvtColor(refined, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _dark_0h_image_enhance(image_bgr: np.ndarray) -> np.ndarray:
        processed = ProcessamentoSemGabaritoApp._adjust_gamma(image_bgr, gamma=0.88)
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        return ProcessamentoSemGabaritoApp._dark_image_edge_enhance_core(
            gray,
            homo_sigma=18.0,
            homo_ksize=83,
            clahe_clip=3.0,
            bg_kernel=11,
            local_win=17,
            border_gain=0.78,
            texture_penalty_gain=0.46,
            shape_open_kernel=11,
            extra_texture_suppression=False,
            base_blur_kernel=5,
            final_blur_kernel=3,
        )

    @staticmethod
    def _dark_24h_image_enhance(image_bgr: np.ndarray) -> np.ndarray:
        """Realce de 24h escura com prioridade maior para bordas largas.

        Esta funcao pertence exclusivamente ao ramo 24h. O objetivo e evitar
        que a forte supressao de textura apague a transicao ferida/monocamada.
        """
        processed = ProcessamentoSemGabaritoApp._adjust_gamma(image_bgr, gamma=0.78)
        processed = ProcessamentoSemGabaritoApp._clahe_lab(
            processed, clip_limit=2.8, tile_grid_size=(8, 8)
        )
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        # Nas imagens escuras, preserva mais estrutura de borda e reduz um pouco
        # a penalizacao de textura. A abertura menor evita apagar frentes finas.
        edge_enhanced = ProcessamentoSemGabaritoApp._dark_image_edge_enhance_core(
            gray,
            homo_sigma=22.0,
            homo_ksize=91,
            clahe_clip=2.8,
            bg_kernel=13,
            local_win=19,
            border_gain=0.98,
            texture_penalty_gain=0.42,
            shape_open_kernel=9,
            extra_texture_suppression=True,
            base_blur_kernel=3,
            final_blur_kernel=1,
        )

        soft_detail = ProcessamentoSemGabaritoApp._soft_24h_image_enhance(processed)
        adaptive_separation = ProcessamentoSemGabaritoApp._dark_24h_adaptive_separation(processed)

        # Antes o componente de separacao adaptativa tinha peso suficiente para
        # puxar o resultado para uma faixa central. Agora ele atua como suporte,
        # enquanto a informacao de borda permanece dominante.
        combined = cv2.addWeighted(edge_enhanced, 0.80, soft_detail, 0.20, 0.0)
        return cv2.addWeighted(combined, 0.84, adaptive_separation, 0.16, 0.0)

    @staticmethod
    def _gray_contrast_percentile(image_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        p95 = float(np.percentile(gray, 95))
        p5 = float(np.percentile(gray, 5))
        return max(0.0, p95 - p5)

    @staticmethod
    def _suppress_24h_cell_texture(image_bgr: np.ndarray, texture_weight: float = 0.40, local_win: int = 21) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        win = max(3, int(local_win))
        if (win % 2) == 0:
            win += 1
        mean = cv2.blur(gray, (win, win))
        mean2 = cv2.blur(gray * gray, (win, win))
        std = np.sqrt(np.maximum(mean2 - (mean * mean), 0.0))
        std = cv2.normalize(std, None, 0.0, 1.0, cv2.NORM_MINMAX)
        suppression = np.clip(gray - (texture_weight * std * gray), 0.0, 255.0).astype(np.uint8)
        suppression = cv2.GaussianBlur(suppression, (5, 5), 0)
        return cv2.cvtColor(suppression, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _bright_24h_image_enhance(image_bgr: np.ndarray) -> np.ndarray:
        enhanced = ProcessamentoSemGabaritoApp._flattened_24h_image(
            image_bgr,
            gamma=0.96,
            clahe_clip=1.2,
            homomorphic_sigma=15.0,
            homomorphic_ksize=71,
            flatten_scale=150.0,
            bilateral_sigma_color=12,
            bilateral_sigma_space=16,
            top_hat_weight=0.28,
            black_hat_weight=0.22,
            gradient_weight=0.08,
            unsharp_amount=0.12,
            raw_texture_weight=0.04,
        )
        return ProcessamentoSemGabaritoApp._suppress_24h_cell_texture(enhanced, texture_weight=0.32, local_win=19)

    @staticmethod
    def _medium_24h_image_enhance(image_bgr: np.ndarray) -> np.ndarray:
        enhanced = ProcessamentoSemGabaritoApp._flattened_24h_image(
            image_bgr,
            gamma=0.92,
            clahe_clip=1.55,
            homomorphic_sigma=17.0,
            homomorphic_ksize=81,
            flatten_scale=160.0,
            bilateral_sigma_color=14,
            bilateral_sigma_space=18,
            top_hat_weight=0.34,
            black_hat_weight=0.26,
            gradient_weight=0.12,
            unsharp_amount=0.18,
            raw_texture_weight=0.06,
        )
        return ProcessamentoSemGabaritoApp._suppress_24h_cell_texture(enhanced, texture_weight=0.36, local_win=21)

    @staticmethod
    def _dark_24h_adaptive_separation(image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        h, w = blurred.shape
        block_size = max(31, min(h, w) // 15 * 2 + 1)
        if (block_size % 2) == 0:
            block_size += 1

        center = blurred[h // 3:(2 * h) // 3, w // 3:(2 * w) // 3]
        if center.size == 0:
            center_mean = float(np.mean(blurred))
        else:
            center_mean = float(np.mean(center))
        global_mean = float(np.mean(blurred))

        if center_mean >= global_mean:
            thresh_type = cv2.THRESH_BINARY
        else:
            thresh_type = cv2.THRESH_BINARY_INV

        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresh_type,
            block_size,
            8,
        )

        se_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        se_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se_close)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, se_open)
        thresh = cv2.GaussianBlur(thresh, (9, 9), 0)
        thresh = cv2.normalize(thresh, None, 0, 255, cv2.NORM_MINMAX)

        enhanced = cv2.addWeighted(gray, 0.60, thresh, 0.40, 0.0)
        enhanced = cv2.equalizeHist(enhanced)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _soft_24h_image_enhance(image_bgr: np.ndarray) -> np.ndarray:
        return ProcessamentoSemGabaritoApp._flattened_24h_image(
            image_bgr,
            gamma=0.96,
            clahe_clip=1.47,
            homomorphic_sigma=17.0,
            homomorphic_ksize=81,
            flatten_scale=170.0,
            bilateral_sigma_color=16,
            bilateral_sigma_space=20,
            top_hat_weight=0.44,
            black_hat_weight=0.34,
            gradient_weight=0.15,
            unsharp_amount=0.28,
            raw_texture_weight=0.08,
        )

    @staticmethod
    def _flattened_24h_image(
        image_bgr: np.ndarray,
        *,
        gamma: float,
        clahe_clip: float,
        homomorphic_sigma: float,
        homomorphic_ksize: int,
        flatten_scale: float,
        bilateral_sigma_color: int,
        bilateral_sigma_space: int,
        top_hat_weight: float,
        black_hat_weight: float,
        gradient_weight: float,
        unsharp_amount: float,
        raw_texture_weight: float,
    ) -> np.ndarray:
        processed = ProcessamentoSemGabaritoApp._adjust_gamma(image_bgr, gamma=gamma)
        source_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        gray = ProcessamentoSemGabaritoApp._homomorphic_gray_u8(
            source_gray,
            sigma=homomorphic_sigma,
            ksize=homomorphic_ksize,
        )

        gray_f = gray.astype(np.float32)
        bg = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=31.0, sigmaY=31.0)
        flattened = cv2.divide(gray_f, bg + 1.0, scale=flatten_scale)
        flattened = np.clip(flattened, 0, 255).astype(np.uint8)

        flattened = cv2.bilateralFilter(
            flattened,
            d=7,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space,
        )
        flattened = cv2.GaussianBlur(flattened, (3, 3), 0)
        flattened = ProcessamentoSemGabaritoApp._local_clahe_gray(
            flattened,
            clip_limit=clahe_clip,
            tile_grid_size=(8, 8),
        )
        hat_base = cv2.GaussianBlur(flattened, (3, 3), 0)
        se_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        se_mid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        se_tiny = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        top_small = cv2.morphologyEx(hat_base, cv2.MORPH_TOPHAT, se_small)
        bot_small = cv2.morphologyEx(hat_base, cv2.MORPH_BLACKHAT, se_small)
        top_mid = cv2.morphologyEx(hat_base, cv2.MORPH_TOPHAT, se_mid)
        bot_mid = cv2.morphologyEx(hat_base, cv2.MORPH_BLACKHAT, se_mid)
        top_tiny = cv2.morphologyEx(hat_base, cv2.MORPH_TOPHAT, se_tiny)
        bot_tiny = cv2.morphologyEx(hat_base, cv2.MORPH_BLACKHAT, se_tiny)
        grad = cv2.morphologyEx(
            hat_base,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        raw_hat_base = cv2.GaussianBlur(source_gray, (3, 3), 0)
        raw_top_tiny = cv2.morphologyEx(raw_hat_base, cv2.MORPH_TOPHAT, se_tiny)
        raw_bot_tiny = cv2.morphologyEx(raw_hat_base, cv2.MORPH_BLACKHAT, se_tiny)
        raw_top_small = cv2.morphologyEx(raw_hat_base, cv2.MORPH_TOPHAT, se_small)
        raw_bot_small = cv2.morphologyEx(raw_hat_base, cv2.MORPH_BLACKHAT, se_small)
        raw_texture = cv2.addWeighted(raw_top_tiny, 0.64, raw_bot_tiny, 0.58, 0.0)
        raw_texture = cv2.addWeighted(raw_texture, 1.0, raw_top_small, 0.32, 0.0)
        raw_texture = cv2.addWeighted(raw_texture, 1.0, raw_bot_small, 0.28, 0.0)
        raw_texture = cv2.GaussianBlur(raw_texture, (3, 3), 0)
        cell_support = cv2.addWeighted(top_tiny, 0.50, top_small, 0.76, 0.0)
        cell_support = cv2.addWeighted(cell_support, 1.0, top_mid, 0.54, 0.0)
        cell_shadow = cv2.addWeighted(bot_tiny, 0.46, bot_small, 0.70, 0.0)
        cell_shadow = cv2.addWeighted(cell_shadow, 1.0, bot_mid, 0.50, 0.0)
        flattened = cv2.addWeighted(flattened, 1.0, cell_support, top_hat_weight, 0.0)
        flattened = cv2.addWeighted(flattened, 1.0, cell_shadow, black_hat_weight, 0.0)
        flattened = cv2.addWeighted(flattened, 1.0, raw_texture, raw_texture_weight, 0.0)
        flattened = cv2.addWeighted(flattened, 1.0, grad, gradient_weight, 0.0)
        blur_for_unsharp = cv2.GaussianBlur(flattened, (0, 0), 1.2)
        flattened = cv2.addWeighted(flattened, 1.0 + unsharp_amount, blur_for_unsharp, -unsharp_amount, 0.0)
        flattened = cv2.GaussianBlur(flattened, (3, 3), 0)
        flattened = cv2.medianBlur(flattened, 5)
        return cv2.cvtColor(flattened, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _area_eval_overrides_for_24h(mode: str) -> dict[str, int | float]:
        return {}

    def _cleanup_24h_artifacts(
        self,
        group_key: str,
        results: AreaEvalNoRefResults,
        artifacts: AreaEvalArtifacts,
        mode: str = "",
        spatial_prior: np.ndarray | None = None,
        reference_0h_mask: np.ndarray | None = None,
    ) -> None:
        """Limpa exclusivamente a mascara de 24h.

        Para 24h escura, usa morfologia menos agressiva e um refinamento guiado
        por bordas. O prior de 0h entra como preferencia gradual, nunca como
        substituto da evidencia presente na imagem 24h.
        """
        mask = np.asarray(artifacts.mask_auto).astype(bool)
        roi = np.asarray(artifacts.roi_mask).astype(bool)
        if mask.shape != roi.shape or not np.any(mask):
            return

        original = self._fill_mask_holes(mask & roi)
        cleaned = original.copy()

        if mode == "24h_escura":
            # Kernels menores preservam irregularidades verdadeiras da fronteira.
            close_kernel = 9
            open_kernel = 3
        else:
            close_kernel = 13
            open_kernel = 7

        cleaned = cv2.morphologyEx(
            cleaned.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)),
        ) > 0
        cleaned = cv2.morphologyEx(
            cleaned.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)),
        ) > 0

        cleaned = self._refine_24h_background_separation(
            cleaned,
            roi,
            artifacts.base_rgb_u8,
            mode=mode,
            spatial_prior=spatial_prior,
            reference_0h_mask=reference_0h_mask,
        )

        final_close = 11 if mode == "24h_escura" else 15
        cleaned = cv2.morphologyEx(
            cleaned.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (final_close, final_close)),
        ) > 0

        cleaned = self._fill_mask_holes(cleaned & roi)
        cleaned = self._keep_largest_mask_component(cleaned & roi)

        if mode == "24h_escura":
            # Suavizacao menor que a versao comum de 24h para nao arredondar nem
            # puxar novamente as bordas escuras para o centro.
            cleaned = self._smooth_mask_boundary(
                cleaned,
                roi,
                close_scale=1.10,
                open_scale=0.72,
                blur_scale=0.58,
            )
            cleaned = cv2.morphologyEx(
                cleaned.astype(np.uint8) * 255,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            ) > 0
            cleaned = self._fill_mask_holes(cleaned & roi)
            cleaned = self._keep_largest_mask_component(cleaned & roi)
        else:
            cleaned = self._smooth_24h_mask_boundary(cleaned, roi)

        if not np.any(cleaned):
            return

        artifacts.mask_auto = cleaned
        artifacts.contour_mask = contour_mask_with_thickness(cleaned, radius=self.config.r_auto)
        artifacts.contour_overlay_rgb_u8 = overlay_perimeter(
            artifacts.base_rgb_u8,
            artifacts.contour_mask,
            self.config.col_auto,
        )
        results.area_auto = int(np.count_nonzero(cleaned))

    def _refine_24h_background_separation(
        self,
        mask: np.ndarray,
        roi: np.ndarray,
        base_rgb_u8: np.ndarray,
        mode: str = "",
        spatial_prior: np.ndarray | None = None,
        reference_0h_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Refina a ferida em 24h sem forcar a mascara para o centro.

        O algoritmo combina quatro sinais:
        1) homogeneidade local do fundo da ferida;
        2) diferenca para um modelo de fundo;
        3) continuidade com a mascara inicial calculada em 24h;
        4) prior espacial gradual derivado da mascara de 0h.

        Em 24h escura, acrescenta evidencia de borda por Scharr, com enfase
        horizontal (gradiente em x), porque as frentes do scratch tendem a ser
        aproximadamente verticais. Nao ha mais corridor_half nem penalizacao
        pela distancia ao centro da imagem.
        """
        source = np.asarray(mask).astype(bool)
        roi_bool = np.asarray(roi).astype(bool)
        if source.shape != roi_bool.shape or not np.any(source):
            return source

        gray = cv2.cvtColor(base_rgb_u8, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        gray_f = gray.astype(np.float32)

        # Textura local: fundo de ferida tende a ser mais homogeneo que a
        # monocamada celular, especialmente depois do realce adaptativo.
        local_mean = cv2.blur(gray_f, (21, 21))
        local_mean2 = cv2.blur(gray_f * gray_f, (21, 21))
        local_std = np.sqrt(np.maximum(local_mean2 - (local_mean * local_mean), 0.0))
        local_std = cv2.normalize(local_std, None, 0.0, 1.0, cv2.NORM_MINMAX)

        bg_model = cv2.morphologyEx(
            gray,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        )
        bg_diff = cv2.absdiff(gray, bg_model).astype(np.float32)
        bg_diff = cv2.normalize(bg_diff, None, 0.0, 1.0, cv2.NORM_MINMAX)

        roi_ys, roi_xs = np.nonzero(roi_bool)
        if roi_xs.size < 64 or roi_ys.size < 64:
            return source

        h, w = source.shape

        # Continuidade espacial em torno da mascara que o pipeline encontrou em
        # 24h. Diferente do corredor central, esta medida acompanha qualquer forma.
        dist_from_source = cv2.distanceTransform(
            (~source).astype(np.uint8),
            cv2.DIST_L2,
            5,
        ).astype(np.float32)
        continuity_scale = max(8.0, float(min(h, w)) * (0.030 if mode == "24h_escura" else 0.022))
        continuity = np.exp(-dist_from_source / continuity_scale).astype(np.float32)
        continuity[source] = 1.0
        continuity[~roi_bool] = 0.0

        if spatial_prior is not None and np.asarray(spatial_prior).shape == source.shape:
            prior = np.clip(np.asarray(spatial_prior, dtype=np.float32), 0.0, 1.0)
        else:
            # Sem 0h valida, o prior fica neutro dentro da ROI.
            prior = roi_bool.astype(np.float32)

        # Evidencia de borda. Scharr em x recebe peso maior para capturar as
        # duas frentes laterais, mas o termo em y continua permitindo curvaturas.
        src01 = gray_f / 255.0
        gx = cv2.Scharr(src01, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(src01, cv2.CV_32F, 0, 1)
        edge_support = (1.35 * np.abs(gx)) + (0.30 * np.abs(gy))
        edge_support = cv2.normalize(edge_support, None, 0.0, 1.0, cv2.NORM_MINMAX)
        edge_support = cv2.GaussianBlur(edge_support, (3, 3), 0)
        edge_support[~roi_bool] = 0.0

        # Pesos sem tendencia ao centro. Nas escuras, o prior e a continuidade
        # passam a substituir dist_x/dist_y, e a borda recebe bonus explicito.
        if mode == "24h_clara":
            texture_weight = 0.50
            bg_weight = 0.16
            prior_weight = 0.22
            continuity_weight = 0.12
            edge_bonus = 0.04
            quantile_cut = 0.72
            open_kernel = 5
            close_kernel = 17
        elif mode == "24h_media":
            texture_weight = 0.42
            bg_weight = 0.16
            prior_weight = 0.26
            continuity_weight = 0.16
            edge_bonus = 0.06
            quantile_cut = 0.65
            open_kernel = 5
            close_kernel = 15
        else:
            texture_weight = 0.28
            bg_weight = 0.14
            prior_weight = 0.34
            continuity_weight = 0.24
            edge_bonus = 0.12
            quantile_cut = 0.56
            open_kernel = 3
            close_kernel = 13

        interior_score = (
            (texture_weight * (1.0 - local_std))
            + (bg_weight * (1.0 - bg_diff))
            + (prior_weight * prior)
            + (continuity_weight * continuity)
        )
        combined_score = np.clip(interior_score + (edge_bonus * edge_support), 0.0, 1.0)

        roi_scores = combined_score[roi_bool]
        if roi_scores.size == 0:
            return source
        thr = float(np.quantile(roi_scores, quantile_cut))

        # A mascara inicial SEMPRE faz parte do candidato. Isso impede que um
        # threshold muito seletivo reduza o resultado para uma ilha central.
        candidate = roi_bool & ((combined_score >= thr) | source)

        if mode == "24h_escura":
            # Bordas fortes proximas da ferida entram como suporte adicional.
            # O prior impede que bordas celulares distantes dominem o resultado.
            edge_values = edge_support[roi_bool]
            edge_thr = float(np.quantile(edge_values, 0.70)) if edge_values.size else 1.0
            border_supported = (
                roi_bool
                & (edge_support >= edge_thr)
                & (continuity >= 0.16)
                & (prior >= 0.12)
            )
            border_supported = cv2.dilate(
                border_supported.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            ) > 0
            candidate |= border_supported & roi_bool

            # Quando existe referencia 0h, preserva uma faixa fina ao redor da
            # forma esperada apenas como zona de busca, nunca como mascara final.
            if reference_0h_mask is not None and np.asarray(reference_0h_mask).shape == source.shape:
                ref = np.asarray(reference_0h_mask).astype(bool)
                ref_band = cv2.morphologyEx(
                    ref.astype(np.uint8) * 255,
                    cv2.MORPH_GRADIENT,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)),
                ) > 0
                supported_ref_band = ref_band & (edge_support >= edge_thr) & roi_bool
                candidate |= supported_ref_band

        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8) * 255,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)),
        ) > 0
        candidate = cv2.morphologyEx(
            candidate.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)),
        ) > 0
        candidate = self._fill_mask_holes(candidate & roi_bool)
        if not np.any(candidate):
            return source

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8),
            connectivity=8,
            ltype=cv2.CV_32S,
        )
        if n_labels <= 1:
            return source

        seed_mask = source & roi_bool
        seed_area = float(np.count_nonzero(seed_mask))
        best_mask = None
        best_score = float("-inf")

        for label_idx in range(1, n_labels):
            comp = labels == label_idx
            comp_area = float(stats[label_idx, cv2.CC_STAT_AREA])
            if comp_area <= 0:
                continue

            overlap = float(np.count_nonzero(comp & seed_mask)) / max(seed_area, 1.0)
            prior_mean = float(np.mean(prior[comp])) if np.any(comp) else 0.0
            continuity_mean = float(np.mean(continuity[comp])) if np.any(comp) else 0.0
            edge_mean = float(np.mean(edge_support[comp])) if np.any(comp) else 0.0

            # Sem center_penalty: componentes sao avaliados pela relacao com a
            # segmentacao inicial, com a forma 0h e com a evidencia da imagem.
            score = (
                (0.52 * overlap)
                + (0.22 * prior_mean)
                + (0.16 * continuity_mean)
                + (0.10 * edge_mean)
            )
            if score > best_score:
                best_score = score
                best_mask = comp

        if best_mask is None or not np.any(best_mask):
            return source

        final_kernel = 13 if mode == "24h_escura" else 19
        best_mask = cv2.morphologyEx(
            best_mask.astype(np.uint8) * 255,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (final_kernel, final_kernel)),
        ) > 0
        best_mask = self._fill_mask_holes(best_mask & roi_bool)
        best_mask = self._keep_largest_mask_component(best_mask & roi_bool)
        if not np.any(best_mask):
            return source

        preserved_fraction = float(np.count_nonzero(best_mask)) / max(
            float(np.count_nonzero(source)), 1.0
        )
        min_preserved = 0.55 if mode == "24h_escura" else 0.30
        if preserved_fraction < min_preserved:
            return source
        return best_mask

    def _adaptive_preprocess_by_mean_intensity(
        self,
        image_bgr: np.ndarray,
        time_tag: str | None = None,
    ) -> tuple[np.ndarray, dict[str, float | str]]:
        # Seletor principal do pipeline: escolhe o realce mais adequado para
        # a imagem com base no brilho medio e no timepoint.
        mean_gray = self._mean_gray_intensity(image_bgr)

        if time_tag is not None and self._is_24h_time_tag(time_tag):
            contrast_gray = self._gray_contrast_percentile(image_bgr)
            if mean_gray >= BRIGHT_24H_IMAGE_THRESHOLD:
                mode = "24h_clara"
                processed = self._bright_24h_image_enhance(image_bgr)
            elif mean_gray >= MID_24H_IMAGE_THRESHOLD:
                mode = "24h_media"
                processed = self._medium_24h_image_enhance(image_bgr)
            else:
                mode = "24h_escura"
                processed = self._dark_24h_image_enhance(image_bgr)

            info = {
                "mode": mode,
                "mean_gray": mean_gray,
                "contrast_gray": contrast_gray,
            }
            return processed, info

        if self._uses_0h_brightness_buckets(time_tag):
            if mean_gray >= BRIGHT_0H_IMAGE_THRESHOLD:
                mode = "clara"
                processed = self._bright_image_homogeneous_border_enhance(image_bgr)
            elif mean_gray <= DARK_IMAGE_THRESHOLD:
                mode = "escura"
                processed = self._dark_0h_image_enhance(image_bgr)
            else:
                mode = "media"
                processed = self._medium_0h_image_enhance(image_bgr)

            info = {
                "mode": mode,
                "mean_gray": mean_gray,
            }
            return processed, info

        # Imagem clara: escurece um pouco e reforca contraste local
        if mean_gray >= BRIGHT_IMAGE_THRESHOLD:
            mode = "clara"
            processed = self._bright_image_homogeneous_border_enhance(image_bgr)

        # Imagem escura: mantem mais proximo do pipeline original
        elif mean_gray <= DARK_IMAGE_THRESHOLD:
            mode = "escura"
            processed = image_bgr.copy()

        # Imagem quase clara: reforco moderado de borda sem exagerar no contraste
        elif mean_gray >= MID_BRIGHT_IMAGE_THRESHOLD:
            mode = "intermediaria_clara"
            processed = self._adjust_gamma(image_bgr, gamma=1.25)
            processed = self._clahe_lab(processed, clip_limit=1.9, tile_grid_size=(8, 8))
            processed = self._mid_bright_image_enhance(processed)

        # Imagem intermediaria: ajuste leve
        else:
            mode = "normal"
            processed = self._clahe_lab(image_bgr, clip_limit=1.6, tile_grid_size=(8, 8))

        info = {
            "mode": mode,
            "mean_gray": mean_gray,
        }
        return processed, info

    def _prepare_adaptive_input(
        self,
        path: Path,
        time_tag: str | None = None,
        image_bgr: np.ndarray | None = None,
    ) -> tuple[Path, dict[str, float | str], np.ndarray]:
        if image_bgr is None:
            image_bgr = self._read_image_unicode(path)
        processed_bgr, info = self._adaptive_preprocess_by_mean_intensity(image_bgr, time_tag=time_tag)

        temp_name = f"{self._safe_filename(path.stem)}_adapt.png"
        temp_path = Path(self.temp_workdir.name) / temp_name

        ok = self._write_image_file(temp_path, processed_bgr)
        if not ok:
            raise RuntimeError(f"Nao foi possivel salvar imagem temporaria adaptada: {temp_path}")

        return temp_path, info, processed_bgr

    # ------------------------------------------------------------------
    # Orquestracao do processamento em lote
    # O app prepara jobs, executa o pipeline em thread separada e recebe
    # os resultados de volta para atualizar a interface.
    # ------------------------------------------------------------------
    def _start_processing_job(self, items: list[tuple[str, str, Path]], title: str) -> None:
        if self.ui_busy or not items:
            return
        self.worker_queue = queue.Queue()
        self.progress_var.set(0.0)
        self._set_progress_text(0.0, 0, len(items), suffix="imagens")
        self.center_progress_title_var.set(title)
        self.center_progress_detail_var.set(f"Iniciando processamento de {len(items)} imagem(ns)...")
        self.status_var.set(f"{title} (0/{len(items)})...")
        self._set_ui_busy(True)

        def worker() -> None:
            success: list[tuple[str, str]] = []
            failures: list[tuple[str, str, str]] = []
            total = len(items)

            # Cache local do lote. Garante que a mascara 0h calculada nesta
            # mesma thread esteja disponivel imediatamente quando chegar 24h.
            worker_processed_cache: dict[tuple[str, str], ProcessedTimepoint] = {}

            for idx, (group_key, time_tag, path) in enumerate(items, start=1):
                self.worker_queue.put(("stage", idx, total, group_key, time_tag, "Preparando imagem", 0.0))

                def on_stage(stage: str, stage_progress: float) -> None:
                    self.worker_queue.put(("stage", idx, total, group_key, time_tag, stage, stage_progress))

                try:
                    self.worker_queue.put(("stage", idx, total, group_key, time_tag, "Analisando brilho medio", 0.05))
                    image_bgr = self._read_image_unicode(path)
                    image_shape = image_bgr.shape[:2]
                    roi_origin = "roi_padrao"
                    spatial_prior_24h: np.ndarray | None = None
                    reference_0h_mask: np.ndarray | None = None

                    if self._is_24h_time_tag(time_tag):
                        # O contexto de 0h e sempre calculado para 24h: a mascara
                        # final de 0h gera um PRIOR continuo e uma ROI externa ampla.
                        derived_roi, spatial_prior_24h, reference_0h_mask, derived_origin = (
                            self._derive_24h_spatial_context_from_0h(
                                group_key,
                                image_shape,
                                processed_cache=worker_processed_cache,
                            )
                        )

                        # Uma ROI explicitamente desenhada para a propria imagem
                        # 24h continua tendo prioridade como area de busca, mas o
                        # prior de 0h ainda orienta a escolha dentro dela.
                        roi_mask = self.roi_by_item.get((group_key, time_tag))
                        if roi_mask is not None and np.any(roi_mask):
                            roi_mask = self._resize_roi_mask(roi_mask, image_shape)
                            roi_origin = f"roi_personalizada_24h + {derived_origin}"
                        else:
                            roi_mask = derived_roi
                            roi_origin = derived_origin
                    else:
                        # Caminho original de 0h e dos demais timepoints: intacto.
                        roi_mask = self._roi_for_item(group_key, time_tag)
                        if roi_mask is not None:
                            roi_origin = "roi_personalizada_ou_atual"

                    if self._is_24h_time_tag(time_tag):
                        self.worker_queue.put((
                            "stage",
                            idx,
                            total,
                            group_key,
                            time_tag,
                            f"Preparando ROI especial de 24h ({roi_origin})",
                            0.09
                        ))

                    adaptive_path, adaptive_info, processed_bgr = self._prepare_adaptive_input(
                        path,
                        time_tag=time_tag,
                        image_bgr=image_bgr,
                    )
                    adaptive_info["roi_source"] = roi_origin

                    self.worker_queue.put((
                        "stage",
                        idx,
                        total,
                        group_key,
                        time_tag,
                        f"Modo adaptativo: {adaptive_info['mode']} (media={adaptive_info['mean_gray']:.1f})",
                        0.12
                    ))

                    area_eval_overrides: dict[str, int | float] = {}
                    if self._is_24h_time_tag(time_tag):
                        area_eval_overrides = self._area_eval_overrides_for_24h(
                            str(adaptive_info.get("mode", "")),
                        )
                        adaptive_info["eval_profile"] = "24h_fundo_suave"

                    results, artifacts = run_area_eval_no_reference(
                        base_path=str(adaptive_path),
                        config=self.config,
                        roi_mask=roi_mask,
                        show=False,
                        verbose=False,
                        return_artifacts=True,
                        progress_callback=on_stage,
                        **area_eval_overrides,
                    )
                    if self._is_24h_time_tag(time_tag):
                        self._cleanup_24h_artifacts(
                            group_key,
                            results,
                            artifacts,
                            mode=str(adaptive_info.get("mode", "")),
                            spatial_prior=spatial_prior_24h,
                            reference_0h_mask=reference_0h_mask,
                        )
                    elif adaptive_info.get("mode") == "clara":
                        self._cleanup_bright_mode_artifacts(results, artifacts, time_tag=time_tag)

                    # A mascara e calculada pelo pipeline; o desenho final do
                    # contorno acontece sobre a imagem realcada para facilitar
                    # a inspecao visual no app e na exportacao.
                    artifacts.base_rgb_u8 = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
                    artifacts.contour_overlay_rgb_u8 = overlay_perimeter(
                        artifacts.base_rgb_u8,
                        artifacts.contour_mask,
                        self.config.col_auto,
                    )
                    if self._is_24h_time_tag(time_tag):
                        # Em 24h a limpeza acima ja produz uma unica mascara coerente.
                        # Filtra apenas o overlay para visualizacao; nao reconstrui a
                        # mascara a partir do vermelho, pois essa etapa generica faria
                        # uma nova suavizacao e poderia apagar a borda escura recuperada.
                        filtered_overlay, red_lengths = self._filter_small_red_contours_overlay(
                            artifacts.contour_overlay_rgb_u8,
                            artifacts.base_rgb_u8,
                        )
                        artifacts.filtered_contour_overlay_rgb_u8 = filtered_overlay
                        artifacts.filtered_red_contour_lengths = red_lengths
                        artifacts.filtered_contour_mask = artifacts.contour_mask.copy()
                        artifacts.contour_overlay_rgb_u8 = filtered_overlay
                    else:
                        # Caminho original de 0h: inalterado.
                        self._apply_filtered_red_contour_artifacts(results, artifacts)

                    processed_item = ProcessedTimepoint(
                        path=path,
                        results=results,
                        artifacts=artifacts,
                    )
                    worker_processed_cache[(group_key, time_tag)] = processed_item

                    success.append((group_key, time_tag))
                    self.worker_queue.put(("ok", group_key, time_tag, path, results, artifacts, adaptive_info))

                except Exception as exc:
                    failures.append((group_key, time_tag, str(exc)))
                    err_text = str(exc)
                    if "Nao consegui ler:" in err_text or "Arquivo de imagem nao encontrado:" in err_text:
                        err_text = f"Falha ao ler imagem: {path.name}\n{err_text}"
                    else:
                        err_text = f"Falha no processamento: {path.name}\n{err_text}"
                    self.worker_queue.put(("err", group_key, time_tag, err_text))

                self.worker_queue.put(("progress", idx, total, group_key, time_tag))

            self.worker_queue.put(("done", success, failures))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
        self.root.after(90, self._poll_worker_queue)

    def _poll_worker_queue(self) -> None:
        while True:
            try:
                msg = self.worker_queue.get_nowait()
            except queue.Empty:
                break

            tag = msg[0]

            if tag == "ok":
                _, group_key, time_tag, path, results, artifacts, adaptive_info = msg
                self.processed_by_group.setdefault(group_key, {})[time_tag] = ProcessedTimepoint(
                    path=path,
                    results=results,
                    artifacts=artifacts
                )
                self.processing_mode_by_item[(group_key, time_tag)] = adaptive_info
                self.processing_errors.pop((group_key, time_tag), None)

            elif tag == "err":
                _, group_key, time_tag, err = msg
                self.processing_errors[(group_key, time_tag)] = err

            elif tag == "stage":
                _, idx, total, group_key, time_tag, stage, stage_progress = msg
                pct = 100.0 * ((float(idx - 1) + min(max(float(stage_progress), 0.0), 1.0)) / float(max(total, 1)))
                self.progress_var.set(pct)
                self._set_progress_text(pct, idx - 1, total, suffix="imagens")
                detail = f"Processando: {group_key} {time_tag.upper()} | {stage} ({idx}/{total})"
                self.status_var.set(detail)
                self._update_center_progress_detail(detail)

            elif tag == "progress":
                _, idx, total, group_key, time_tag = msg
                pct = 100.0 * (float(idx) / float(max(total, 1)))
                self.progress_var.set(pct)
                self._set_progress_text(pct, idx, total, suffix="imagens")
                detail = f"Processando: {group_key} {time_tag.upper()} concluida ({idx}/{total})"
                self.status_var.set(detail)
                self._update_center_progress_detail(detail)

            elif tag == "done":
                _, success, failures = msg
                self.worker_thread = None
                self._print_dice_summary(success)
                total_done = len(success) + len(failures)
                self._set_progress_text(100.0, total_done, total_done, suffix="imagens")
                self.progress_var.set(100.0)
                self._update_center_progress_detail("Processamento concluido.")
                self._set_ui_busy(False)
                if failures:
                    preview = "\n".join(f"{g} {t}: {e}" for g, t, e in failures[:12])
                    self.status_var.set("Processamento concluido com avisos.")
                    messagebox.showwarning("Processamento concluido com avisos", preview)
                else:
                    self.status_var.set(f"Processamento concluido. {len(success)} imagem(ns) processada(s).")
                if self.current_group_key is not None:
                    self._show_group(self.current_group_key, status_text=self.status_var.get())

        if self.ui_busy:
            self.root.after(90, self._poll_worker_queue)

    def _set_progress_text(self, pct: float, done: int, total: int, suffix: str = "imagens") -> None:
        pct_clamped = min(max(float(pct), 0.0), 100.0)
        total_int = max(int(total), 0)
        done_int = min(max(int(done), 0), total_int) if total_int > 0 else max(int(done), 0)
        self.progress_text_var.set(f"{pct_clamped:.1f}% | {done_int}/{total_int} {suffix}")

    def _start_full_processing(self) -> None:
        if not self.group_order:
            messagebox.showinfo("Aviso", "Carregue os grupos antes de processar.")
            return
        self.processed_by_group = {}
        self.processing_errors = {}
        self.processing_mode_by_item = {}
        self._start_processing_job(self._all_items(), "Processando grupos")

    def _reprocess_current_group(self) -> None:
        items = self._current_group_items()
        if not items:
            messagebox.showinfo("Aviso", "Selecione um grupo para processar.")
            return
        self._start_processing_job(items, "Processando grupo atual")

    def _auto_process_all_after_scan(self) -> None:
        if self.ui_busy or not self.group_order:
            return
        # Ao carregar a pasta, processa o conjunto completo de uma vez.
        # A ordem interna de cada grupo continua 0h -> img -> 24h, necessaria
        # para que 24h receba a delimitacao espacial produzida em 0h.
        self.processed_by_group = {}
        self.processing_errors = {}
        self.processing_mode_by_item = {}
        self._start_processing_job(self._all_items(), "Processando todos os grupos")

    # ------------------------------------------------------------------
    # Exibicao e interacao com resultados
    # Daqui em diante o codigo apenas le o estado processado e mostra na UI.
    # ------------------------------------------------------------------
    def _show_group(self, group_key: str, status_text: str | None = None) -> None:
        self.current_group_key = group_key
        self.group_var.set(self.group_labels.get(group_key, group_key))
        self._refresh_metrics(group_key)
        self._refresh_figure(group_key)
        if status_text is not None:
            self.status_var.set(status_text)
        idx = self.group_order.index(group_key)
        self.prev_btn.configure(state=tk.NORMAL if idx > 0 and not self.ui_busy else tk.DISABLED)
        self.next_btn.configure(state=tk.NORMAL if idx < len(self.group_order) - 1 and not self.ui_busy else tk.DISABLED)

    def _on_group_selected(self, _event=None) -> None:
        group_key = self.label_to_group.get(self.group_var.get().strip())
        if group_key:
            self._show_group(group_key)

    def _show_prev_group(self) -> None:
        if self.current_group_key in self.group_order:
            idx = self.group_order.index(self.current_group_key)
            if idx > 0:
                self._show_group(self.group_order[idx - 1])

    def _show_next_group(self) -> None:
        if self.current_group_key in self.group_order:
            idx = self.group_order.index(self.current_group_key)
            if idx < len(self.group_order) - 1:
                self._show_group(self.group_order[idx + 1])

    def _refresh_metrics(self, group_key: str | None = None) -> None:
        self.metrics_text.configure(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)
        group_key = group_key or self.current_group_key

        if group_key is None:
            self.metrics_text.insert(tk.END, "Sem grupo selecionado.")
            self.metrics_text.configure(state=tk.DISABLED)
            return

        lines = [
            "=========== RESULTADOS (SEM GABARITO) ===========",
            f"Grupo: {group_key}",
            ""
        ]

        for time_tag in self._group_times(group_key):
            key = (group_key, time_tag)
            proc = self.processed_by_group.get(group_key, {}).get(time_tag)
            adaptive_info = self.processing_mode_by_item.get(key)

            lines.append(f"[{time_tag.upper()}] {self.group_files[group_key][time_tag].name}")
            lines.append(f"ROI personalizada: {'SIM' if key in self.roi_by_item else 'NAO'}")

            if adaptive_info is not None:
                lines.append(f"Modo adaptativo: {adaptive_info['mode']}")
                lines.append(f"Media intensidade (cinza): {adaptive_info['mean_gray']:.1f}")
                roi_source = str(adaptive_info.get("roi_source", "")).strip()
                if roi_source:
                    lines.append(f"Origem ROI: {roi_source}")

            if proc is None:
                lines.append(f"Status: SEM RESULTADO ({self.processing_errors.get(key, 'Nao processada.')})")
            else:
                lines.append(f"Area automatica: {proc.results.area_auto:.0f} px")
                lines.append(f"Tempo proc.: {proc.results.processing_time_s:.3f} s")

            lines.append("")

        self.metrics_text.insert(tk.END, "\n".join(lines).rstrip())
        self.metrics_text.configure(state=tk.DISABLED)

    def _refresh_figure(self, group_key: str | None = None) -> None:
        self.figure.clear()
        group_key = group_key or self.current_group_key

        if group_key is None or group_key not in self.group_files:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Sem grupo selecionado", ha="center", va="center")
            ax.axis("off")
            self.canvas.draw_idle()
            return

        times = self._group_times(group_key)
        ncols = max(len(times), 1)

        for idx, time_tag in enumerate(times, start=1):
            ax = self.figure.add_subplot(1, ncols, idx)
            proc = self.processed_by_group.get(group_key, {}).get(time_tag)
            adaptive_info = self.processing_mode_by_item.get((group_key, time_tag))

            if proc is None:
                ax.text(
                    0.5,
                    0.5,
                    self.processing_errors.get((group_key, time_tag), "Sem resultado"),
                    ha="center",
                    va="center",
                    wrap=True
                )
            else:
                ax.imshow(proc.artifacts.contour_overlay_rgb_u8)
                if adaptive_info is not None:
                    ax.set_title(
                        f"{time_tag.upper()} | {adaptive_info['mode']}\narea={proc.results.area_auto:.0f} px",
                        fontsize=10
                    )
                else:
                    ax.set_title(f"{time_tag.upper()} | area={proc.results.area_auto:.0f} px", fontsize=10)

            ax.axis("off")

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _open_roi_editor(self, image: np.ndarray, initial_mask: np.ndarray, title: str) -> np.ndarray | None:
        preview_image = enhance_roi_editor_image(image)
        fig, ax = plt.subplots(figsize=(12, 9))
        fig.subplots_adjust(right=0.82)
        ax.imshow(preview_image)
        ax.set_title(title)
        ax.axis("off")
        ax.contour(initial_mask.astype(np.uint8), levels=[0.5], colors="black", linewidths=3.2, alpha=0.95)
        ax.contour(initial_mask.astype(np.uint8), levels=[0.5], colors="#ffe600", linewidths=1.6, alpha=0.98)

        selected: dict[str, object] = {"verts": None, "accepted": False}
        selector_ref: dict[str, object] = {"selector": None}
        mode_ref: dict[str, str] = {"mode": "MATLAB (poligono)"}
        initial_poly = mask_to_polygon(initial_mask)
        if initial_poly.shape[0] >= 3:
            selected["verts"] = initial_poly.tolist()

        def _set_selected_verts(verts: object) -> None:
            arr = np.asarray(verts, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] == 2:
                selected["verts"] = arr.tolist()

        def _disconnect_selector() -> None:
            selector = selector_ref.get("selector")
            if selector is not None:
                try:
                    selector.disconnect_events()
                except Exception:
                    pass
                selector_ref["selector"] = None

        def _build_polygon_selector() -> PolygonSelector:
            try:
                selector = PolygonSelector(
                    ax,
                    _set_selected_verts,
                    useblit=False,
                    props={"color": "#00ffd0", "linewidth": 2.4, "alpha": 0.95},
                    handle_props={"marker": "o", "markersize": 5.5, "mec": "black", "mfc": "#fff176", "alpha": 0.96}
                )
            except TypeError:
                selector = PolygonSelector(
                    ax,
                    _set_selected_verts,
                    useblit=False,
                    lineprops={"color": "#00ffd0", "linewidth": 2.4, "alpha": 0.95},
                    markerprops={"marker": "o", "markersize": 5.5, "mec": "black", "mfc": "#fff176", "alpha": 0.96}
                )
            if selected["verts"] is not None:
                try:
                    selector.verts = selected["verts"]
                except Exception:
                    pass
            return selector

        def _verts_from_bounds(bounds: tuple[float, float, float, float], mode: str) -> np.ndarray | None:
            x_min, y_min, x_max, y_max = bounds
            if mode == "Retangulo":
                return np.asarray([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.float32)
            if mode == "Elipse":
                cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
                rx, ry = 0.5 * (x_max - x_min), 0.5 * (y_max - y_min)
                theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False, dtype=np.float32)
                return np.column_stack((cx + rx * np.cos(theta), cy + ry * np.sin(theta))).astype(np.float32)
            return None

        def _selector_bounds(selector: object) -> tuple[float, float, float, float] | None:
            if selector is None or not hasattr(selector, "extents"):
                return None
            try:
                x0, x1, y0, y1 = [float(v) for v in selector.extents[:4]]
            except Exception:
                return None
            x_min, x_max = sorted((x0, x1))
            y_min, y_max = sorted((y0, y1))
            return None if (x_max - x_min) < 1e-3 or (y_max - y_min) < 1e-3 else (x_min, y_min, x_max, y_max)

        def _activate_mode(mode: str) -> None:
            _disconnect_selector()
            mode_ref["mode"] = mode
            if mode == "Retangulo":
                selector_ref["selector"] = RectangleSelector(ax, lambda *_: None, useblit=False)
            elif mode == "Elipse":
                selector_ref["selector"] = EllipseSelector(ax, lambda *_: None, useblit=False)
            else:
                selector_ref["selector"] = _build_polygon_selector()
            fig.canvas.draw_idle()

        mode_ax = fig.add_axes([0.835, 0.68, 0.15, 0.23])
        RadioButtons(mode_ax, ("MATLAB (poligono)", "Retangulo", "Elipse"), active=0).on_clicked(_activate_mode)
        _activate_mode("MATLAB (poligono)")

        def on_key(event):
            if event.key in ("enter", "return"):
                verts = selected["verts"]
                if mode_ref["mode"] in ("Retangulo", "Elipse"):
                    bounds = _selector_bounds(selector_ref.get("selector"))
                    if bounds is not None:
                        verts = _verts_from_bounds(bounds, mode_ref["mode"])
                elif verts is None and hasattr(selector_ref.get("selector"), "verts"):
                    verts = selector_ref["selector"].verts
                if verts is not None and len(verts) >= 3:
                    selected["verts"] = np.asarray(verts, dtype=np.float32).tolist()
                    selected["accepted"] = True
                    plt.close(fig)
            elif event.key == "escape":
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.show(block=True)
        _disconnect_selector()

        if not bool(selected["accepted"]) or selected["verts"] is None:
            return None

        return polygon_to_mask(image.shape[:2], np.asarray(selected["verts"], dtype=np.float32))

    def _redefine_roi_current_group(self) -> None:
        items = self._current_group_items()
        if not items:
            messagebox.showinfo("Aviso", "Selecione um grupo para redefinir a ROI.")
            return

        updated = 0
        for idx, (group_key, time_tag, _path) in enumerate(items, start=1):
            proc = self.processed_by_group.get(group_key, {}).get(time_tag)
            if proc is None:
                continue
            key = (group_key, time_tag)
            seed_roi = self.roi_by_item.get(key)
            seed_roi = proc.artifacts.mask_auto if seed_roi is None else resize_mask(seed_roi, proc.artifacts.base_rgb_u8.shape[:2])
            new_roi = self._open_roi_editor(
                proc.artifacts.base_rgb_u8,
                seed_roi,
                f"Grupo {group_key} | {time_tag.upper()} ({idx}/{len(items)})"
            )
            if new_roi is not None and np.any(new_roi):
                self.roi_by_item[key] = new_roi
                updated += 1

        self.status_var.set(f"ROI atualizada para {updated} imagem(ns) do grupo." if updated else "Nenhuma ROI foi alterada.")
        if self.current_group_key is not None:
            self._refresh_metrics(self.current_group_key)
            self._refresh_figure(self.current_group_key)

    def _clear_roi_current_group(self) -> None:
        removed = 0
        for group_key, time_tag, _path in self._current_group_items():
            key = (group_key, time_tag)
            if key in self.roi_by_item:
                del self.roi_by_item[key]
                removed += 1
        self.status_var.set(
            f"ROI personalizada removida de {removed} imagem(ns)."
            if removed else
            "O grupo ja estava sem ROI personalizada."
        )

    # ------------------------------------------------------------------
    # Exportacao
    # Esta parte salva artefatos e tabelas; nao recalcula a segmentacao.
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_filename(name: str, max_len: int = 120) -> str:
        clean = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in name).strip("._")
        if len(clean) <= max_len:
            return clean or "resultado"
        return f"{clean[:max_len-11]}_{hashlib.sha1(clean.encode('utf-8')).hexdigest()[:10]}"

    @staticmethod
    def _write_image_file(path: Path, image: np.ndarray) -> bool:
        suffix = path.suffix.lower() or ".png"
        ok, encoded = cv2.imencode(suffix, image)
        if not ok:
            return False
        try:
            encoded.tofile(str(path))
        except Exception:
            return False
        return True

    @staticmethod
    def _write_simple_xlsx(path: Path, sheet_name: str, headers: list[str], rows: list[list[object]]) -> None:
        ProcessamentoSemGabaritoApp._write_multi_sheet_xlsx(
            path,
            sheets=[(sheet_name, headers, rows)],
        )

    @staticmethod
    def _write_multi_sheet_xlsx(
        path: Path,
        sheets: list[tuple[str, list[str], list[list[object]]]],
    ) -> None:
        def make_cell(value: object) -> str:
            if value is None or value == "":
                return "<c/>"
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return f'<c><v>{value}</v></c>'
            text = xml_escape(str(value))
            return f'<c t="inlineStr"><is><t>{text}</t></is></c>'

        worksheet_parts: list[tuple[str, str]] = []
        workbook_sheet_entries: list[str] = []
        workbook_relationships: list[str] = []

        for idx, (sheet_name, headers, rows) in enumerate(sheets, start=1):
            safe_sheet_name = (sheet_name or f"planilha_{idx}")[:31]
            header_cells = "".join(make_cell(title) for title in headers)
            row_xml_parts = [f'<row r="1">{header_cells}</row>']

            for row_idx, row in enumerate(rows, start=2):
                cells = "".join(make_cell(value) for value in row)
                row_xml_parts.append(f'<row r="{row_idx}">{cells}</row>')

            worksheet_xml = (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                f'<sheetData>{"".join(row_xml_parts)}</sheetData>'
                '</worksheet>'
            )
            worksheet_parts.append((f"xl/worksheets/sheet{idx}.xml", worksheet_xml))
            workbook_sheet_entries.append(
                f'<sheet name="{xml_escape(safe_sheet_name)}" sheetId="{idx}" r:id="rId{idx}"/>'
            )
            workbook_relationships.append(
                '<Relationship '
                f'Id="rId{idx}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{idx}.xml"/>'
            )

        styles_rel_id = len(sheets) + 1
        workbook_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets>{"".join(workbook_sheet_entries)}</sheets>'
            '</workbook>'
        )
        workbook_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'{"".join(workbook_relationships)}'
            '<Relationship '
            f'Id="rId{styles_rel_id}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
            'Target="styles.xml"/>'
            '</Relationships>'
        )
        root_rels_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            '</Relationships>'
        )
        styles_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
            '<borders count="1"><border/></borders>'
            '<cellStyleXfs count="1"><xf/></cellStyleXfs>'
            '<cellXfs count="1"><xf xfId="0"/></cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            '</styleSheet>'
        )
        content_types_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for idx in range(1, len(sheets) + 1)
            )
            +
            '<Override PartName="/xl/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '</Types>'
        )

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as workbook_zip:
            workbook_zip.writestr("[Content_Types].xml", content_types_xml)
            workbook_zip.writestr("_rels/.rels", root_rels_xml)
            workbook_zip.writestr("xl/workbook.xml", workbook_xml)
            workbook_zip.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
            for worksheet_path, worksheet_xml in worksheet_parts:
                workbook_zip.writestr(worksheet_path, worksheet_xml)
            workbook_zip.writestr("xl/styles.xml", styles_xml)

    def _build_area_summary_rows(self) -> list[list[object]]:
        summary_map: dict[str, dict[str, object]] = {}
        fallback_counter = 0

        for group_key in self.group_order:
            for time_tag in self._group_times(group_key):
                src_path = self.group_files.get(group_key, {}).get(time_tag)
                if src_path is None:
                    continue

                parsed = parse_flexible_group_image(src_path)
                if parsed is not None and parsed.prefix and parsed.sample_id:
                    sample_key = f"{parsed.prefix} {parsed.sample_id}".strip()
                    sample_id = parsed.sample_id
                else:
                    fallback_counter += 1
                    sample_key = f"{group_key} #{fallback_counter}"
                    sample_id = group_key

                row = summary_map.setdefault(sample_key, {
                    "amostra": sample_key,
                    "sample_id": sample_id,
                    "arquivo_0h": "",
                    "area_0h": "",
                    "erro_0h": "",
                    "arquivo_24h": "",
                    "area_24h": "",
                    "erro_24h": "",
                })

                normalized_time = time_tag.strip().lower()
                base_time = normalized_time.split("_", 1)[0]
                is_0h = base_time == "0h" or base_time == "img"
                is_24h = base_time == "24h"
                proc = self.processed_by_group.get(group_key, {}).get(time_tag)
                err = self.processing_errors.get((group_key, time_tag), "")

                if is_0h:
                    row["arquivo_0h"] = str(src_path)
                    row["area_0h"] = "" if proc is None else int(round(proc.results.area_auto))
                    row["erro_0h"] = err or ("Tempo original tratado como img (assumido como 0h)." if base_time == "img" else "")
                elif is_24h:
                    row["arquivo_24h"] = str(src_path)
                    row["area_24h"] = "" if proc is None else int(round(proc.results.area_auto))
                    row["erro_24h"] = err

        summary_rows: list[list[object]] = []
        for sample_key in sorted(summary_map.keys(), key=natural_group_sort_key):
            row = summary_map[sample_key]
            summary_rows.append([
                row["amostra"],
                row["sample_id"],
                row["arquivo_0h"],
                row["area_0h"],
                row["erro_0h"],
                row["arquivo_24h"],
                row["area_24h"],
                row["erro_24h"],
            ])

        return summary_rows

    @staticmethod
    def _normalize_ground_truth_key(name: str) -> str:
        return re.sub(r"[^a-z0-9.]+", "", str(name).strip().casefold())

    @classmethod
    def _ground_truth_area_map(cls) -> dict[str, int]:
        raw_values = {
            "3 c- i 0h 1.11418.tif": 1254414,
            "3 c- i 0h 1.21419.tif": 1484506,
            "3 c- i 0h 1.31420.tif": 1357893,
            "3 c- i 0h 2.11421.tif": 1523944,
            "3 c- i 0h 2.21422.tif": 1484541,
            "3 c- i 0h 2.31423.tif": 1555283,
            "3 c- i 0h 3.11424.tif": 1376203,
            "3 c- i 0h 3.21425.tif": 1361001,
            "3 c- i 0h 3.31426.tif": 1114978,
            "3 c- i 0h 4.11427.tif": 1349449,
            "3 c- i 0h 4.21428.tif": 1467663,
            "3 c- i 0h 4.31429.tif": 1135380,
            "3 c- ni 0h 1.11323.tif": 1558989,
            "3 c- ni 0h 1.21324.tif": 1841087,
            "3 c- ni 0h 1.31325.tif": 1806609,
            "3 c- ni 0h 2.11326.tif": 2020738,
            "3 c- ni 0h 2.21327.tif": 1416819,
            "3 c- ni 0h 2.31328.tif": 1514952,
            "3 c- ni 0h 3.11329.tif": 1442248,
            "3 c- i 24h 1.11418.tif": 403281,
            "3 c- i 24h 1.21419.tif": 573881,
            "3 c- i 24h 1.31420.tif": 381912,
            "3 c- i 24h 2.11421.tif": 353344,
            "3 c- i 24h 2.21422.tif": 602564,
            "3 c- i 24h 2.31423.tif": 602564,
            "3 c- i 24h 3.11424.tif": 428313,
            "3 c- i 24h 3.21425.tif": 498901,
            "3 c- i 24h 3.31426.tif": 199355,
            "3 c- i 24h 4.11427.tif": 174364,
            "3 c- i 24h 4.21428.tif": 141256,
            "3 c- i 24h 4.31429.tif": 141256,
            "3 c- ni 24h 1.11323.tif": 623858,
            "3 c- ni 24h 1.21324.tif": 560846,
            "3 c- ni 24h 1.31325.tif": 429197,
            "3 c- ni 24h 2.11326.tif": 1223601,
            "3 c- ni 24h 2.21327.tif": 637207,
            "3 c- ni 24h 2.31328.tif": 435479,
            "3 c- ni 24h 3.11329.tif": 446177,
        }
        return {
            cls._normalize_ground_truth_key(filename): int(area)
            for filename, area in raw_values.items()
        }

    @classmethod
    def _ground_truth_sample_time_map(cls) -> dict[str, int]:
        raw_values: dict[tuple[str, str], int] = {
            ("ct1", "0h"): 1437412,
            ("ct1", "24h"): 584000,
            ("ct1.1", "24h"): 277556,
            ("ct2", "0h"): 1912996,
            ("ct2", "24h"): 713556,
            ("ct2.1", "24h"): 872008,
            ("ct2.2", "0h"): 1887280,
            ("ct3", "0h"): 1818404,
            ("ct3", "24h"): 796668,
            ("ct4", "0h"): 1890280,
            ("ct4", "24h"): 898352,
            ("ct_tg1", "0h"): 2159804,
            ("ct_tg1", "24h"): 975032,
            ("ct_tg2", "0h"): 2177360,
            ("ct_tg2", "24h"): 1411976,
            ("ct_tg2.1", "24h"): 1146304,
            ("ct_tg2.2", "0h"): 2130548,
            ("ct_tg3", "0h"): 2019448,
            ("ct_tg3", "24h"): 961572,
            ("ct_tg3.3", "0h"): 2028984,
            ("ct_tg4", "0h"): 2100532,
            ("ct_tg4", "24h"): 1490336,
            ("ct_tg4.1", "24h"): 1156868,
            ("gal3_1", "0h"): 1497376,
            ("gal3_1", "24h"): 717272,
            ("gal3_1.1", "0h"): 1245020,
            ("gal3_1.1", "24h"): 129512,
            ("gal3_2", "0h"): 1488876,
            ("gal3_2", "24h"): 623688,
            ("gal3_2.1", "24h"): 755800,
            ("gal3_2.2", "0h"): 1453616,
            ("gal3_2.2", "24h"): 252276,
            ("gal3_3", "0h"): 1504528,
            ("gal3_3", "24h"): 519288,
            ("gal3_3.1", "24h"): 349008,
            ("gal3_4", "24h"): 623548,
            ("gal3_4.1", "24h"): 420372,
            ("gal3_tg1", "0h"): 2073536,
            ("gal3_tg1", "24h"): 595020,
            ("gal3_tg1.1", "0h"): 2072240,
            ("gal3_tg1.1", "24h"): 675724,
            ("gal3_tg1.2", "24h"): 675724,
            ("gal3_tg2", "0h"): 1838872,
            ("gal3_tg2", "24h"): 870488,
            ("gal3_tg2.1", "24h"): 769876,
            ("gal3_tg3", "0h"): 1721556,
            ("gal3_tg3", "24h"): 685900,
            ("gal3_tg3.1", "24h"): 781200,
            ("gal3_tg4", "0h"): 1724240,
            ("gal3_tg4", "24h"): 887660,
            ("wt1", "0h"): 2127852,
            ("wt1", "24h"): 552252,
            ("wt1.1", "0h"): 1611512,
            ("wt2", "0h"): 1680488,
            ("wt2", "24h"): 712544,
            ("wt2.2", "24h"): 615036,
            ("wt3", "0h"): 1946184,
            ("wt3", "24h"): 728344,
            ("wt4", "0h"): 1786900,
            ("wt4", "24h"): 444928,
            ("wt4.1", "0h"): 1992764,
            ("wt_tg1", "0h"): 1991300,
            ("wt_tg1", "24h"): 807172,
            ("wt_tg1.1", "0h"): 2146264,
            ("wt_tg1.1", "24h"): 1469416,
            ("wt_tg2", "0h"): 2060940,
            ("wt_tg2", "24h"): 1040460,
            ("wt_tg2.1", "0h"): 1806472,
            ("wt_tg2.1", "24h"): 881352,
            ("wt_tg3", "0h"): 2392204,
            ("wt_tg3", "24h"): 1350312,
            ("wt_tg3.1", "24h"): 1369328,
            ("wt_tg3.2", "24h"): 1403432,
            ("wt_tg4", "0h"): 1641344,
            ("wt_tg4", "24h"): 793464,
        }
        return {
            cls._normalize_ground_truth_key(f"{sample} {time_tag}"): int(area)
            for (sample, time_tag), area in raw_values.items()
        }

    def _build_ground_truth_comparison_rows(self) -> list[list[object]]:
        gt_map = self._ground_truth_area_map()
        gt_sample_time_map = self._ground_truth_sample_time_map()
        rows: list[list[object]] = []

        for group_key in self.group_order:
            for time_tag in self._group_times(group_key):
                src_path = self.group_files.get(group_key, {}).get(time_tag)
                if src_path is None:
                    continue

                proc = self.processed_by_group.get(group_key, {}).get(time_tag)
                err = self.processing_errors.get((group_key, time_tag), "")
                adaptive_info = self.processing_mode_by_item.get((group_key, time_tag), {})
                parsed = parse_flexible_group_image(src_path)
                normalized_time = time_tag.strip().lower()
                base_time = normalized_time.split("_", 1)[0]
                sample_id = parsed.sample_id if parsed is not None and parsed.sample_id else group_key
                sample_name = (
                    f"{parsed.prefix} {parsed.sample_id}".strip()
                    if parsed is not None and parsed.prefix and parsed.sample_id
                    else group_key
                )
                gt_area = self._lookup_ground_truth_area(
                    src_path=src_path,
                    time_tag=base_time,
                    parsed=parsed,
                    gt_map=gt_map,
                    gt_sample_time_map=gt_sample_time_map,
                )
                area_auto = "" if proc is None else int(round(proc.results.area_auto))
                signed_error = ""
                abs_error = ""
                abs_percent_error = ""

                if isinstance(gt_area, int) and isinstance(area_auto, int):
                    signed_error = int(area_auto - gt_area)
                    abs_error = abs(signed_error)
                    abs_percent_error = (abs_error / float(gt_area)) * 100.0 if gt_area > 0 else ""

                rows.append([
                    src_path.name,
                    str(src_path),
                    sample_name,
                    sample_id,
                    base_time,
                    adaptive_info.get("mode", ""),
                    adaptive_info.get("mean_gray", ""),
                    gt_area,
                    area_auto,
                    signed_error,
                    abs_error,
                    abs_percent_error,
                    err,
                ])

        rows.sort(key=lambda item: natural_group_sort_key(f"{item[2]} {item[4]}"))
        return rows

    def _lookup_ground_truth_area(
        self,
        *,
        src_path: Path,
        time_tag: str,
        parsed: ParsedImageInfo | None,
        gt_map: dict[str, int],
        gt_sample_time_map: dict[str, int],
    ) -> int | str:
        candidates = [src_path.name]

        if parsed is not None and parsed.prefix and parsed.sample_id:
            candidates.append(f"{parsed.prefix} {parsed.sample_id}{src_path.suffix}")
            if time_tag in {"0h", "24h"}:
                candidates.append(f"{parsed.prefix} {time_tag} {parsed.sample_id}{src_path.suffix}")

        for candidate in candidates:
            key = self._normalize_ground_truth_key(candidate)
            if key in gt_map:
                return gt_map[key]

        if time_tag in {"0h", "24h"}:
            sample_candidates: list[str] = []
            if parsed is not None:
                if parsed.group_key:
                    sample_candidates.append(parsed.group_key)
                if parsed.prefix and parsed.sample_id:
                    sample_candidates.append(f"{parsed.prefix} {parsed.sample_id}")
                    sample_candidates.append(f"{parsed.prefix}{parsed.sample_id}")
                    sample_candidates.append(f"{parsed.prefix}_{parsed.sample_id}")
            sample_candidates.append(src_path.stem)

            for sample_candidate in sample_candidates:
                key = self._normalize_ground_truth_key(f"{sample_candidate} {time_tag}")
                if key in gt_sample_time_map:
                    return gt_sample_time_map[key]
        return ""

    @staticmethod
    def _build_ground_truth_stats_rows(detail_rows: list[list[object]]) -> list[list[object]]:
        grouped: dict[str, list[list[object]]] = {"geral": []}
        for row in detail_rows:
            tempo = str(row[4] or "").strip().lower()
            grouped.setdefault(tempo, []).append(row)
            grouped["geral"].append(row)

        stats_rows: list[list[object]] = []
        for tempo in ("0h", "24h", "geral"):
            rows = grouped.get(tempo, [])
            matched = [
                row for row in rows
                if isinstance(row[7], int) and isinstance(row[8], int)
            ]

            gt_values = np.asarray([row[7] for row in matched], dtype=np.float64) if matched else np.asarray([], dtype=np.float64)
            auto_values = np.asarray([row[8] for row in matched], dtype=np.float64) if matched else np.asarray([], dtype=np.float64)
            signed_errors = np.asarray([row[9] for row in matched], dtype=np.float64) if matched else np.asarray([], dtype=np.float64)
            abs_errors = np.asarray([row[10] for row in matched], dtype=np.float64) if matched else np.asarray([], dtype=np.float64)
            abs_pct_errors = np.asarray([row[11] for row in matched], dtype=np.float64) if matched else np.asarray([], dtype=np.float64)

            def mean_or_blank(values: np.ndarray) -> float | str:
                return "" if values.size == 0 else float(np.mean(values))

            def var_or_blank(values: np.ndarray) -> float | str:
                return "" if values.size < 2 else float(np.var(values, ddof=1))

            stats_rows.append([
                tempo,
                len(rows),
                len(matched),
                mean_or_blank(gt_values),
                mean_or_blank(auto_values),
                mean_or_blank(signed_errors),
                mean_or_blank(abs_errors),
                mean_or_blank(abs_pct_errors),
                var_or_blank(signed_errors),
                var_or_blank(abs_errors),
                var_or_blank(abs_pct_errors),
            ])

        return stats_rows

    def _save_results_to_disk(self) -> Path:
        out_dir = Path(self.output_dir_var.get().strip())
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "resumo_resultados_sem_gabarito.csv"
        xlsx_path = out_dir / "areas_finais_0h_24h.xlsx"

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "grupo",
                    "tempo",
                    "arquivo_origem",
                    "modo_adaptativo",
                    "media_intensidade_cinza",
                    "area_auto_px",
                    "tempo_processamento_s",
                    "overlay_png",
                    "mask_png",
                    "roi_png",
                    "erro",
                ]
            )
            writer.writeheader()

            for group_key in self.group_order:
                for time_tag in self._group_times(group_key):
                    key = (group_key, time_tag)
                    src_path = self.group_files[group_key][time_tag]
                    proc = self.processed_by_group.get(group_key, {}).get(time_tag)
                    err = self.processing_errors.get(key, "")
                    adaptive_info = self.processing_mode_by_item.get(key, {})
                    modo_adaptativo = adaptive_info.get("mode", "")
                    media_intensidade = adaptive_info.get("mean_gray", "")

                    overlay_name = ""
                    mask_name = ""
                    roi_name = ""
                    area_auto = ""
                    proc_time = ""

                    if proc is not None:
                        token = self._safe_filename(f"{group_key}_{time_tag}_{src_path.stem}")
                        overlay_name = f"{token}_overlay.png"
                        mask_name = f"{token}_mask.png"
                        roi_name = f"{token}_roi.png"

                        self._write_image_file(
                            out_dir / overlay_name,
                            cv2.cvtColor(proc.artifacts.contour_overlay_rgb_u8, cv2.COLOR_RGB2BGR)
                        )
                        self._write_image_file(
                            out_dir / mask_name,
                            proc.artifacts.mask_auto.astype(np.uint8) * 255
                        )
                        self._write_image_file(
                            out_dir / roi_name,
                            proc.artifacts.roi_mask.astype(np.uint8) * 255
                        )

                        area_auto = f"{proc.results.area_auto:.0f}"
                        proc_time = f"{proc.results.processing_time_s:.6f}"

                    writer.writerow({
                        "grupo": group_key,
                        "tempo": time_tag,
                        "arquivo_origem": str(src_path),
                        "modo_adaptativo": modo_adaptativo,
                        "media_intensidade_cinza": media_intensidade,
                        "area_auto_px": area_auto,
                        "tempo_processamento_s": proc_time,
                        "overlay_png": overlay_name,
                        "mask_png": mask_name,
                        "roi_png": roi_name,
                        "erro": err,
                    })

        comparison_rows = self._build_ground_truth_comparison_rows()
        stats_rows = self._build_ground_truth_stats_rows(comparison_rows)
        self._write_multi_sheet_xlsx(
            xlsx_path,
            sheets=[
                (
                    "areas_finais",
                    [
                        "amostra",
                        "id_amostra",
                        "arquivo_0h",
                        "area_final_0h_px",
                        "erro_0h",
                        "arquivo_24h",
                        "area_final_24h_px",
                        "erro_24h",
                    ],
                    self._build_area_summary_rows(),
                ),
                (
                    "comparativo_gabarito",
                    [
                        "imagem",
                        "arquivo_origem",
                        "amostra",
                        "id_amostra",
                        "tempo",
                        "modo_adaptativo",
                        "media_intensidade_cinza",
                        "area_gabarito_px",
                        "area_processada_px",
                        "erro_assinado_px",
                        "erro_absoluto_px",
                        "erro_percentual_abs",
                        "erro_processamento",
                    ],
                    comparison_rows,
                ),
                (
                    "estatisticas_erro",
                    [
                        "tempo",
                        "total_linhas",
                        "linhas_comparaveis",
                        "media_area_gabarito_px",
                        "media_area_processada_px",
                        "media_erro_assinado_px",
                        "media_erro_absoluto_px",
                        "media_erro_percentual_abs",
                        "variancia_erro_assinado_px",
                        "variancia_erro_absoluto_px",
                        "variancia_erro_percentual_abs",
                    ],
                    stats_rows,
                ),
            ],
        )

        return out_dir

    def _save_results_clicked(self) -> None:
        if not self.processed_by_group:
            messagebox.showinfo("Aviso", "Nao ha resultados para salvar.")
            return
        if not self._ensure_output_dir_selected():
            return
        out_dir = self._save_results_to_disk()
        self.status_var.set(f"Resultados salvos em: {out_dir}")
        messagebox.showinfo("Salvo", f"Resultados salvos em:\n{out_dir}")

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interface para processamento em lote sem gabarito.")
    parser.add_argument("--folder", type=str, default=default_folder_from_base_path(), help="Pasta inicial com imagens.")
    parser.add_argument("--output", type=str, default="", help="Pasta de saida para salvar resultados.")
    parser.add_argument("--manual-masks", type=str, default="", help="Pasta com mascaras manuais para calculo do Dice.")
    return parser.parse_args()


# Entrada do programa.
def main() -> None:
    args = parse_args()
    ProcessamentoSemGabaritoApp(
        folder_path=str(args.folder).strip(),
        output_dir=str(args.output).strip(),
        manual_mask_dir=str(args.manual_masks).strip(),
    ).run()


if __name__ == "__main__":
    main()