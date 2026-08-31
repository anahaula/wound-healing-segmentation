import argparse
import csv


import hashlib
from dataclasses import dataclass
from pathlib import Path
import queue
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import EllipseSelector, PolygonSelector, RadioButtons, RectangleSelector
import matplotlib.pyplot as plt
import numpy as np

from matlab_style_area_eval import (
    BASE_IMAGE_PATH,
    AreaEvalArtifacts,
    AreaEvalConfig,
    AreaResults,
    run_area_eval,
)


SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
TIME_ORDER = ("0h", "24h", "48h")
GROUP_PATTERN = re.compile(
    r"(?i)(?P<list>ni|i)\s*[-_ ]*\s*(?P<time>0h|24h|48h)\s*[-_ ]*\s*(?P<id>\d+(?:[.,]\d+)*)"
)
OUTPUT_DIR_NAME = "_area_eval_resultados"
MANUAL_AREA_CSV_CANDIDATES = (
    "areas_manuais.csv",
    "area_manual.csv",
    "manual_areas.csv",
    "areas_esperadas.csv",
)


@dataclass
class GroupImage:
    path: Path
    list_tag: str
    time_tag: str
    image_id: str


@dataclass
class ProcessedTimepoint:
    path: Path
    results: AreaResults
    artifacts: AreaEvalArtifacts


def parse_group_image(path: Path) -> GroupImage | None:
    match = GROUP_PATTERN.search(path.stem)
    if match is None:
        return None

    list_tag = match.group("list").lower()
    time_tag = match.group("time").lower()
    image_id = match.group("id").replace(",", ".")
    return GroupImage(path=path, list_tag=list_tag, time_tag=time_tag, image_id=image_id)


def discover_image_groups(folder: Path, progress_callback=None) -> dict[str, dict[str, Path]]:
    groups: dict[str, dict[str, Path]] = {}
    all_paths = sorted(folder.rglob("*"))
    total = len(all_paths)
    if progress_callback is not None:
        progress_callback(0, total)

    for idx, path in enumerate(all_paths, start=1):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            if progress_callback is not None and (idx == total or (idx % 40) == 0):
                progress_callback(idx, total)
            continue

        parsed = parse_group_image(path)
        if parsed is None:
            if progress_callback is not None and (idx == total or (idx % 40) == 0):
                progress_callback(idx, total)
            continue

        group_key = f"{parsed.list_tag} {parsed.image_id}"
        slot = groups.setdefault(group_key, {})
        if parsed.time_tag not in slot:
            slot[parsed.time_tag] = path

        if progress_callback is not None and (idx == total or (idx % 40) == 0):
            progress_callback(idx, total)

    if progress_callback is not None and total == 0:
        progress_callback(0, 0)
    return groups


def mask_to_polygon(mask: np.ndarray) -> np.ndarray:
    bin255 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(bin255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float32)

    contour = max(contours, key=cv2.contourArea)
    eps = max(1.0, 0.01 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, eps, True)
    poly = approx[:, 0, :].astype(np.float32)
    if poly.shape[0] < 3:
        poly = contour[:, 0, :].astype(np.float32)
    if poly.shape[0] > 200:
        step = int(np.ceil(poly.shape[0] / 200.0))
        poly = poly[::step]
    return poly


def polygon_to_mask(shape: tuple[int, int], verts: np.ndarray) -> np.ndarray:
    h, w = shape
    if verts.shape[0] < 3:
        return np.zeros(shape, dtype=bool)

    pts = np.round(verts).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    mask_u8 = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask_u8, [pts], 255)
    return mask_u8 > 0


def resize_mask(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    th, tw = target_shape
    if mask.shape == target_shape:
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST) > 0


def enhance_roi_editor_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image.astype(np.uint8) if image.dtype == np.uint8 else cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        boosted = cv2.addWeighted(eq, 0.82, grad, 0.38, 0.0)
        return cv2.cvtColor(boosted, cv2.COLOR_GRAY2RGB)

    rgb = image[:, :, :3].copy()
    if rgb.dtype != np.uint8:
        rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.2, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    gx = cv2.Sobel(l_eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(l_eq, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    l_mix = cv2.addWeighted(l_eq, 0.78, grad, 0.42, 0.0)
    return cv2.cvtColor(cv2.merge((l_mix, a, b)), cv2.COLOR_LAB2RGB)


def default_folder_from_base_path() -> str:
    base = str(BASE_IMAGE_PATH).strip()
    if not base:
        return ""
    p = Path(base)
    if p.suffix:
        return str(p.parent)
    return str(p)


class AreaEvalApp:
    def __init__(
        self,
        folder_path: str = "",
        areas_csv_path: str = "",
        output_dir: str = "",
    ) -> None:
        self.root = tk.Tk()
        self.root.title("Avaliacao de Area - Revisao em Lote por Grupos")
        self.root.geometry("1440x860")

        self.folder_var = tk.StringVar(value=folder_path)
        self.areas_csv_var = tk.StringVar(value=areas_csv_path)
        self.output_dir_var = tk.StringVar(value=output_dir)
        self.group_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Pronto")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.popup_status_var = tk.StringVar(value="")
        self.popup_progress_var = tk.DoubleVar(value=0.0)
        self.popup_percent_var = tk.StringVar(value="0.0%")

        self.config = AreaEvalConfig()
        self.group_files: dict[str, dict[str, Path]] = {}
        self.group_labels: dict[str, str] = {}
        self.label_to_group: dict[str, str] = {}
        self.group_order: list[str] = []
        self.current_group_key: str | None = None

        self.roi_by_item: dict[tuple[str, str], np.ndarray] = {}
        self.processed_by_group: dict[str, dict[str, ProcessedTimepoint]] = {}
        self.processing_errors: dict[tuple[str, str], str] = {}
        self.pending_reprocess: set[tuple[str, str]] = set()
        self.reprocessed_history: set[tuple[str, str]] = set()
        self.reviewed_groups: set[str] = set()
        self.manual_area_by_path: dict[str, int] = {}
        self.manual_area_by_name: dict[str, int] = {}

        self.ui_busy = False
        self.worker_thread: threading.Thread | None = None
        self.worker_queue: queue.Queue[tuple] = queue.Queue()
        self.progress_popup: tk.Toplevel | None = None

        self.refazer_vars: dict[str, tk.BooleanVar] = {}
        self.refazer_checks: dict[str, ttk.Checkbutton] = {}

        self._build_layout()
        self._set_ui_busy(False)
        self._refresh_metrics()
        self._refresh_figure()
        self.root.after(120, self._prompt_output_dir_on_start)

    def _build_layout(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Pasta de imagens:").grid(row=0, column=0, sticky=tk.W)
        folder_entry = ttk.Entry(top, textvariable=self.folder_var, width=95)
        folder_entry.grid(row=0, column=1, sticky=tk.EW, padx=6)

        self.browse_btn = ttk.Button(top, text="Procurar pasta", command=self._browse_folder)
        self.browse_btn.grid(row=0, column=2, padx=4)

        self.scan_btn = ttk.Button(top, text="Carregar grupos", command=self._scan_folder)
        self.scan_btn.grid(row=0, column=3, padx=4)

        ttk.Label(top, text="CSV areas (imagem,area_manual_px):").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(top, textvariable=self.areas_csv_var, width=95).grid(row=1, column=1, sticky=tk.EW, padx=6, pady=(8, 0))
        self.browse_csv_btn = ttk.Button(top, text="Procurar CSV", command=self._browse_areas_csv)
        self.browse_csv_btn.grid(row=1, column=2, padx=4, pady=(8, 0))

        ttk.Label(top, text="Pasta de saida:").grid(row=2, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(top, textvariable=self.output_dir_var, width=95).grid(row=2, column=1, sticky=tk.EW, padx=6, pady=(8, 0))
        self.browse_output_btn = ttk.Button(top, text="Escolher saida", command=self._browse_output_dir)
        self.browse_output_btn.grid(row=2, column=2, padx=4, pady=(8, 0))

        self.process_all_btn = ttk.Button(top, text="Processar todos os grupos", command=self._start_full_processing)
        self.process_all_btn.grid(row=3, column=1, columnspan=3, sticky=tk.EW, padx=4, pady=(8, 0))

        ttk.Label(top, text="Grupo em revisao:").grid(row=4, column=0, sticky=tk.W, pady=(8, 0))
        self.group_combo = ttk.Combobox(top, textvariable=self.group_var, state="disabled", width=87)
        self.group_combo.grid(row=4, column=1, sticky=tk.EW, padx=6, pady=(8, 0))
        self.group_combo.bind("<<ComboboxSelected>>", self._on_group_selected)

        nav = ttk.Frame(top)
        nav.grid(row=4, column=2, columnspan=2, sticky=tk.E, padx=4, pady=(8, 0))
        self.prev_btn = ttk.Button(nav, text="Grupo anterior", command=self._show_prev_group)
        self.prev_btn.pack(side=tk.LEFT, padx=3)
        self.next_btn = ttk.Button(nav, text="Proximo grupo", command=self._show_next_group)
        self.next_btn.pack(side=tk.LEFT, padx=3)

        mark = ttk.Frame(top)
        mark.grid(row=5, column=0, columnspan=4, sticky=tk.EW, pady=(8, 0))

        ttk.Label(mark, text="Selecionar para salvar:").grid(row=0, column=0, sticky=tk.W)
        for idx, time_tag in enumerate(TIME_ORDER, start=1):
            var = tk.BooleanVar(value=False)
            check = ttk.Checkbutton(mark, text=time_tag.upper(), variable=var, command=self._on_mark_changed)
            check.grid(row=0, column=idx, padx=(8, 0), sticky=tk.W)
            self.refazer_vars[time_tag] = var
            self.refazer_checks[time_tag] = check

        self.redefine_roi_btn = ttk.Button(
            mark,
            text="Redefinir ROI (nao selecionadas)",
            command=self._redefine_roi_for_unselected,
        )
        self.redefine_roi_btn.grid(row=0, column=4, padx=(14, 4), sticky=tk.EW)

        self.clear_roi_btn = ttk.Button(
            mark,
            text="Limpar ROI (nao selecionadas)",
            command=self._clear_roi_for_unselected,
        )
        self.clear_roi_btn.grid(row=0, column=5, padx=4, sticky=tk.EW)

        self.finalize_btn = ttk.Button(
            mark,
            text="Salvar selecionadas + refazer nao selecionadas",
            command=self._finalize_review,
        )
        self.finalize_btn.grid(row=0, column=6, padx=4, sticky=tk.EW)

        self.save_btn = ttk.Button(mark, text="Salvar resultados atuais", command=self._save_results_clicked)
        self.save_btn.grid(row=0, column=7, padx=4, sticky=tk.EW)

        top.columnconfigure(1, weight=1)
        mark.columnconfigure(6, weight=1)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        left = ttk.Frame(body, padding=8)
        right = ttk.Frame(body, padding=8)
        body.add(left, weight=1)
        body.add(right, weight=3)

        ttk.Label(left, text="Resultados do grupo").pack(anchor=tk.W)
        self.metrics_text = tk.Text(left, width=46, height=30, state=tk.DISABLED)
        self.metrics_text.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        ttk.Label(right, text="Imagens do grupo (exibidas em conjunto)").pack(anchor=tk.W)
        self.figure = Figure(figsize=(11.4, 6.9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        bottom = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress_bar = ttk.Progressbar(bottom, mode="determinate", variable=self.progress_var, maximum=100.0)
        self.progress_bar.pack(fill=tk.X)
        ttk.Label(bottom, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, pady=(4, 0))

    def _set_ui_busy(self, busy: bool) -> None:
        self.ui_busy = busy
        btn_state = tk.DISABLED if busy else tk.NORMAL

        for btn in (
            self.browse_btn,
            self.browse_csv_btn,
            self.browse_output_btn,
            self.scan_btn,
            self.process_all_btn,
            self.prev_btn,
            self.next_btn,
            self.redefine_roi_btn,
            self.clear_roi_btn,
            self.finalize_btn,
            self.save_btn,
        ):
            btn.configure(state=btn_state)

        combo_state = "readonly" if (not busy and self.group_order) else "disabled"
        self.group_combo.configure(state=combo_state)

        if busy:
            for check in self.refazer_checks.values():
                check.configure(state=tk.DISABLED)
            self.root.configure(cursor="watch")
        else:
            self.root.configure(cursor="")
            self._sync_check_vars_for_group()
            self._update_navigation_buttons()

    def _open_progress_popup(self, title: str) -> None:
        popup = self.progress_popup
        if popup is None or not popup.winfo_exists():
            popup = tk.Toplevel(self.root)
            popup.geometry("560x150")
            popup.resizable(False, False)
            popup.transient(self.root)

            frame = ttk.Frame(popup, padding=12)
            frame.pack(fill=tk.BOTH, expand=True)

            ttk.Label(frame, text="Progresso do processamento").pack(anchor=tk.W)
            ttk.Label(
                frame,
                textvariable=self.popup_status_var,
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=520,
            ).pack(fill=tk.X, pady=(8, 6))

            row = ttk.Frame(frame)
            row.pack(fill=tk.X)
            ttk.Progressbar(
                row,
                mode="determinate",
                variable=self.popup_progress_var,
                maximum=100.0,
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Label(row, textvariable=self.popup_percent_var, width=8, anchor=tk.E).pack(side=tk.LEFT, padx=(8, 0))
            self.progress_popup = popup
        else:
            popup.deiconify()
            popup.lift()

        popup.title(title)
        self.popup_status_var.set(self.status_var.get())
        current_pct = float(self.progress_var.get())
        self.popup_progress_var.set(current_pct)
        self.popup_percent_var.set(f"{current_pct:.1f}%")
        popup.update_idletasks()

    def _close_progress_popup(self) -> None:
        popup = self.progress_popup
        if popup is None:
            return
        if popup.winfo_exists():
            popup.destroy()
        self.progress_popup = None

    def _set_progress_feedback(self, pct: float, status: str) -> None:
        pct_clamped = min(max(float(pct), 0.0), 100.0)
        self.progress_var.set(pct_clamped)
        self.status_var.set(status)
        self.popup_progress_var.set(pct_clamped)
        self.popup_status_var.set(status)
        self.popup_percent_var.set(f"{pct_clamped:.1f}%")

        popup = self.progress_popup
        if popup is not None and popup.winfo_exists():
            popup.update_idletasks()

    def _browse_folder(self) -> None:
        folder = filedialog.askdirectory(title="Selecione a pasta com as imagens")
        if folder:
            self.folder_var.set(folder)

    def _browse_areas_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Selecione o CSV de areas manuais",
            filetypes=[("CSV", "*.csv"), ("Todos os arquivos", "*.*")],
        )
        if path:
            self.areas_csv_var.set(path)

    def _browse_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="Selecione a pasta de saida dos resultados")
        if folder:
            self.output_dir_var.set(folder)

    def _prompt_output_dir_on_start(self) -> None:
        if self.output_dir_var.get().strip():
            return
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

        self.progress_var.set(0.0)
        self.status_var.set("Escaneando pasta...")
        self.root.update_idletasks()

        def on_scan_progress(done: int, total: int) -> None:
            pct = 100.0 if total <= 0 else (100.0 * float(done) / float(total))
            self.progress_var.set(pct)
            self.status_var.set(f"Separando imagens em grupos: {done}/{total}")
            self.root.update_idletasks()

        groups = discover_image_groups(folder, progress_callback=on_scan_progress)
        self._reset_all_state()

        if not groups:
            self.status_var.set("Nenhum grupo encontrado")
            self._refresh_metrics()
            self._refresh_figure()
            messagebox.showwarning(
                "Aviso",
                "Nenhuma imagem com padrao i|ni + 0h/24h/48h + id foi encontrada na pasta.",
            )
            return

        self.group_files = groups
        self.group_order = sorted(groups.keys())
        loaded_areas, areas_csv_path = self._load_manual_area_mapping(
            folder,
            csv_path_text=self.areas_csv_var.get(),
        )

        labels: list[str] = []
        for group_key in self.group_order:
            times = [t for t in TIME_ORDER if t in groups[group_key]]
            label = f"{group_key} | {'/'.join(times)}"
            labels.append(label)
            self.group_labels[group_key] = label
            self.label_to_group[label] = group_key

        self.group_combo["values"] = labels
        self.group_var.set(labels[0])
        self.current_group_key = None
        self._refresh_metrics()
        self._refresh_figure()
        if self.areas_csv_var.get().strip() and (loaded_areas == 0):
            self.status_var.set(
                f"{len(labels)} grupo(s) encontrado(s). CSV de areas nao carregado: {areas_csv_path}."
            )
            messagebox.showerror(
                "Erro",
                "Nao foi possivel carregar o CSV de areas manuais.\n"
                "Selecione um CSV valido com cabecalho: imagem,area_manual_px.",
            )
            return
        elif loaded_areas > 0 and areas_csv_path is not None:
            self.status_var.set(
                f"{len(labels)} grupo(s) encontrado(s). Areas manuais carregadas ({loaded_areas}) de: {areas_csv_path}. Iniciando processamento em lote..."
            )
        else:
            self.status_var.set(f"{len(labels)} grupo(s) encontrado(s). CSV de areas obrigatorio.")
            messagebox.showerror(
                "Erro",
                "Informe um CSV de areas manuais com cabecalho: imagem,area_manual_px.",
            )
            return
        self._start_full_processing()

    def _reset_all_state(self) -> None:
        self.group_files = {}
        self.group_labels = {}
        self.label_to_group = {}
        self.group_order = []
        self.current_group_key = None

        self.roi_by_item = {}
        self.processed_by_group = {}
        self.processing_errors = {}
        self.pending_reprocess = set()
        self.reprocessed_history = set()
        self.reviewed_groups = set()
        self.manual_area_by_path = {}
        self.manual_area_by_name = {}

        self.group_combo["values"] = []
        self.group_var.set("")
        self.progress_var.set(0.0)

    def _selected_group_key(self) -> str | None:
        label = self.group_var.get().strip()
        if not label:
            return None
        return self.label_to_group.get(label)

    def _group_times(self, group_key: str) -> list[str]:
        group_map = self.group_files.get(group_key, {})
        return [t for t in TIME_ORDER if t in group_map]

    def _current_group_index(self) -> int:
        if self.current_group_key is None:
            return -1
        try:
            return self.group_order.index(self.current_group_key)
        except ValueError:
            return -1

    def _update_navigation_buttons(self) -> None:
        if self.ui_busy or not self.group_order:
            self.prev_btn.configure(state=tk.DISABLED)
            self.next_btn.configure(state=tk.DISABLED)
            return

        idx = self._current_group_index()
        if idx <= 0:
            self.prev_btn.configure(state=tk.DISABLED)
        else:
            self.prev_btn.configure(state=tk.NORMAL)

        if idx < 0 or idx >= (len(self.group_order) - 1):
            self.next_btn.configure(state=tk.DISABLED)
        else:
            self.next_btn.configure(state=tk.NORMAL)

    def _show_group_by_index(self, index: int, mark_reviewed: bool = True) -> None:
        if not self.group_order:
            return
        idx = min(max(0, index), len(self.group_order) - 1)
        self._show_group(self.group_order[idx], mark_reviewed=mark_reviewed)

    def _show_prev_group(self) -> None:
        idx = self._current_group_index()
        if idx > 0:
            self._show_group_by_index(idx - 1, mark_reviewed=True)

    def _show_next_group(self) -> None:
        idx = self._current_group_index()
        if 0 <= idx < len(self.group_order) - 1:
            self._show_group_by_index(idx + 1, mark_reviewed=True)

    def _on_group_selected(self, _event=None) -> None:
        if self.ui_busy:
            return
        key = self._selected_group_key()
        if key is not None:
            self._show_group(key, mark_reviewed=True)

    def _show_group(self, group_key: str, mark_reviewed: bool = True, status_text: str | None = None) -> None:
        if group_key not in self.group_files:
            return

        self.current_group_key = group_key
        if mark_reviewed:
            self.reviewed_groups.add(group_key)

        label = self.group_labels.get(group_key, group_key)
        if self.group_var.get() != label:
            self.group_var.set(label)

        self._sync_check_vars_for_group()
        self._refresh_metrics(group_key)
        self._refresh_figure(group_key)
        self._update_navigation_buttons()

        if status_text is not None:
            self.status_var.set(status_text)
            return

        idx = self._current_group_index() + 1
        total = len(self.group_order)
        reviewed = len(self.reviewed_groups)
        self.status_var.set(f"Revisao: grupo {idx}/{total} | grupos vistos: {reviewed}/{total}")

    def _sync_check_vars_for_group(self) -> None:
        group_key = self.current_group_key
        processed = self.processed_by_group.get(group_key or "", {})

        for time_tag in TIME_ORDER:
            key = (group_key or "", time_tag)
            available = bool(group_key and time_tag in self.group_files.get(group_key, {}) and time_tag in processed)
            checked = bool(available and key not in self.pending_reprocess)
            self.refazer_vars[time_tag].set(checked)

            enabled = (not self.ui_busy) and available
            self.refazer_checks[time_tag].configure(state=(tk.NORMAL if enabled else tk.DISABLED))

    def _on_mark_changed(self) -> None:
        group_key = self.current_group_key
        if group_key is None:
            return

        processed = self.processed_by_group.get(group_key, {})
        for time_tag in TIME_ORDER:
            key = (group_key, time_tag)
            checked = bool(self.refazer_vars[time_tag].get())

            if time_tag not in processed:
                self.pending_reprocess.discard(key)
                continue

            if checked:
                self.pending_reprocess.discard(key)
            else:
                self.pending_reprocess.add(key)

        self._refresh_metrics(group_key)
        self._refresh_figure(group_key)

    @staticmethod
    def _normalize_area_key(raw: str) -> str:
        return str(raw).strip().replace("\\", "/").lower()

    def _load_manual_area_mapping(self, folder: Path, csv_path_text: str = "") -> tuple[int, str | None]:
        self.manual_area_by_path = {}
        self.manual_area_by_name = {}

        csv_path: Path | None = None
        explicit = csv_path_text.strip()
        if explicit:
            p = Path(explicit)
            if p.exists() and p.is_file():
                csv_path = p
            else:
                return 0, explicit
        else:
            for candidate in MANUAL_AREA_CSV_CANDIDATES:
                probe = folder / candidate
                if probe.exists() and probe.is_file():
                    csv_path = probe
                    break

        if csv_path is None:
            return 0, None

        loaded = 0
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return 0, str(csv_path)

            for row in reader:
                area_raw = (
                    row.get("area_manual_px")
                    or row.get("area_manual")
                    or row.get("area_ref")
                    or row.get("area")
                )
                if area_raw is None:
                    continue
                try:
                    area_val = int(float(str(area_raw).strip().replace(",", ".")))
                except ValueError:
                    continue
                if area_val <= 0:
                    continue

                path_raw = (
                    row.get("imagem")
                    or row.get("image")
                    or row.get("arquivo")
                    or row.get("arquivo_origem")
                    or row.get("path")
                    or row.get("file")
                    or row.get("filename")
                )
                if path_raw is None:
                    continue

                key = self._normalize_area_key(path_raw)
                if not key:
                    continue

                self.manual_area_by_path[key] = area_val
                self.manual_area_by_name[Path(key).name] = area_val
                loaded += 1

        return loaded, str(csv_path)

    def _resolve_manual_area_for_path(self, image_path: Path) -> int | None:
        key_full = self._normalize_area_key(str(image_path))
        by_full = self.manual_area_by_path.get(key_full)
        if by_full is not None:
            return int(by_full)

        key_name = self._normalize_area_key(image_path.name)
        by_name = self.manual_area_by_name.get(key_name)
        if by_name is not None:
            return int(by_name)
        return None

    def _all_items(self) -> list[tuple[str, str, Path]]:
        items: list[tuple[str, str, Path]] = []
        for group_key in self.group_order:
            for time_tag in TIME_ORDER:
                path = self.group_files.get(group_key, {}).get(time_tag)
                if path is not None:
                    items.append((group_key, time_tag, path))
        return items

    def _missing_manual_areas(self, items: list[tuple[str, str, Path]]) -> list[str]:
        missing: list[str] = []
        for _group_key, _time_tag, path in items:
            if self._resolve_manual_area_for_path(path) is None:
                missing.append(str(path))
        return missing

    def _roi_for_item(self, group_key: str, time_tag: str) -> np.ndarray | None:
        key = (group_key, time_tag)
        roi = self.roi_by_item.get(key)
        if roi is None:
            existing = self.processed_by_group.get(group_key, {}).get(time_tag)
            if existing is not None:
                roi = existing.artifacts.roi_mask
        if roi is None:
            return None
        return np.asarray(roi).astype(bool)

    def _start_processing_job(
        self,
        mode: str,
        items: list[tuple[str, str, Path]],
        save_after: bool = False,
    ) -> None:
        if self.ui_busy:
            return
        if not items:
            messagebox.showwarning("Aviso", "Nao ha imagens para processar.")
            return

        self.worker_queue = queue.Queue()
        self.progress_var.set(0.0)
        self._set_ui_busy(True)
        title = "Processamento em lote" if mode == "full" else "Reprocessando imagens"
        self._open_progress_popup(title)

        if mode == "full":
            self._set_progress_feedback(0.0, f"Processando todos os grupos (0/{len(items)})...")
        else:
            self._set_progress_feedback(0.0, f"Reprocessando nao selecionadas (0/{len(items)})...")

        def worker() -> None:
            success: list[tuple[str, str]] = []
            failures: list[tuple[str, str, str]] = []
            total = len(items)

            for idx, (group_key, time_tag, path) in enumerate(items, start=1):
                self.worker_queue.put(("item_stage", mode, idx, total, group_key, time_tag, "Preparando imagem", 0.0))

                def on_stage(stage: str, stage_progress: float) -> None:
                    self.worker_queue.put(("item_stage", mode, idx, total, group_key, time_tag, stage, stage_progress))

                try:
                    roi_mask = self._roi_for_item(group_key, time_tag)
                    area_manual = self._resolve_manual_area_for_path(path)
                    if area_manual is None:
                        raise RuntimeError(
                            "Area manual nao encontrada no CSV para a imagem: "
                            f"{path.name}"
                        )
                    results, artifacts = run_area_eval(
                        base_path=str(path),
                        area_manual=area_manual,
                        config=self.config,
                        roi_mask=roi_mask,
                        show=False,
                        verbose=False,
                        return_artifacts=True,
                        progress_callback=on_stage,
                    )
                    success.append((group_key, time_tag))
                    self.worker_queue.put(("item_ok", group_key, time_tag, path, results, artifacts))
                except Exception as exc:
                    failures.append((group_key, time_tag, str(exc)))
                    self.worker_queue.put(("item_error", group_key, time_tag, str(exc)))

                self.worker_queue.put(("item_stage", mode, idx, total, group_key, time_tag, "Finalizado", 1.0))
                self.worker_queue.put(("progress", mode, idx, total, group_key, time_tag))

            self.worker_queue.put(("done", mode, success, failures, save_after))

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
            if tag == "item_ok":
                _, group_key, time_tag, path, results, artifacts = msg
                group_results = self.processed_by_group.setdefault(group_key, {})
                group_results[time_tag] = ProcessedTimepoint(path=path, results=results, artifacts=artifacts)
                self.processing_errors.pop((group_key, time_tag), None)
            elif tag == "item_error":
                _, group_key, time_tag, err = msg
                self.processing_errors[(group_key, time_tag)] = err
            elif tag == "item_stage":
                _, mode, idx, total, group_key, time_tag, stage, stage_progress = msg
                stage_progress = min(max(float(stage_progress), 0.0), 1.0)
                pct = 100.0 * ((float(idx - 1) + stage_progress) / float(max(total, 1)))
                prefix = "Lote" if mode == "full" else "Refazendo"
                self._set_progress_feedback(
                    pct,
                    f"{prefix}: {group_key} {time_tag.upper()} | {stage} ({idx}/{total})",
                )
            elif tag == "progress":
                _, mode, idx, total, group_key, time_tag = msg
                pct = 100.0 * (float(idx) / float(max(total, 1)))
                prefix = "Lote" if mode == "full" else "Refazendo"
                self._set_progress_feedback(
                    pct,
                    f"{prefix}: {group_key} {time_tag.upper()} concluida ({idx}/{total})",
                )
            elif tag == "done":
                _, mode, success, failures, save_after = msg
                try:
                    self._finish_processing_job(mode, success, failures, save_after)
                except Exception as exc:
                    self._set_ui_busy(False)
                    self._close_progress_popup()
                    self.status_var.set(f"Falha ao concluir o processamento: {exc}")
                    messagebox.showerror(
                        "Erro",
                        "O processamento terminou, mas houve uma falha ao atualizar a interface:\n"
                        f"{exc}",
                    )

        if self.ui_busy:
            self.root.after(90, self._poll_worker_queue)

    def _finish_processing_job(
        self,
        mode: str,
        success: list[tuple[str, str]],
        failures: list[tuple[str, str, str]],
        save_after: bool,
    ) -> None:
        self._set_ui_busy(False)
        self.worker_thread = None
        final_status = self.status_var.get()
        self._set_progress_feedback(100.0, final_status)
        self._close_progress_popup()

        if mode == "full":
            self.pending_reprocess = {
                (group_key, time_tag)
                for group_key, processed in self.processed_by_group.items()
                for time_tag in processed
            }
            self.reviewed_groups = set()
            if self.group_order:
                self._show_group(self.group_order[0], mark_reviewed=True)
            else:
                self._refresh_metrics()
                self._refresh_figure()

            if failures:
                self._set_progress_feedback(100.0, f"Processamento em lote concluido com {len(failures)} erro(s).")
                details = "\n".join(f"{g} {t}: {e}" for g, t, e in failures[:12])
                if len(failures) > 12:
                    details += f"\n... +{len(failures) - 12} erro(s)"
                messagebox.showwarning("Processamento concluido com avisos", details)
            else:
                self._set_progress_feedback(100.0, "Processamento em lote concluido. Inicie a revisao dos grupos.")
            return

        for key in success:
            self.pending_reprocess.discard(key)
            self.reprocessed_history.add(key)

        focus_group_key = success[0][0] if success else None
        if self.current_group_key:
            self._sync_check_vars_for_group()
            self._refresh_metrics(self.current_group_key)
            self._refresh_figure(self.current_group_key)

        save_error = ""
        out_dir = None
        if save_after:
            try:
                out_dir = self._save_results_to_disk()
            except Exception as exc:
                save_error = str(exc)

        if failures or save_error:
            self._set_progress_feedback(100.0, "Reprocessamento concluido com avisos.")
            lines = [f"{g} {t}: {e}" for g, t, e in failures[:12]]
            if len(failures) > 12:
                lines.append(f"... +{len(failures) - 12} erro(s)")
            if save_error:
                lines.append(f"Falha ao salvar resultados: {save_error}")
            messagebox.showwarning("Reprocessamento concluido com avisos", "\n".join(lines))
        else:
            if out_dir is not None:
                self._set_progress_feedback(100.0, f"Reprocessamento concluido. Resultados salvos em: {out_dir}")
                messagebox.showinfo("Concluido", f"Reprocessamento concluido e salvo em:\n{out_dir}")
            else:
                self._set_progress_feedback(100.0, "Reprocessamento concluido.")

        if focus_group_key is not None:
            final_status = self.status_var.get()
            self._show_group(focus_group_key, mark_reviewed=False, status_text=final_status)

    def _start_full_processing(self) -> None:
        if not self.group_order:
            messagebox.showinfo("Aviso", "Carregue os grupos antes de processar.")
            return
        if not self._ensure_output_dir_selected():
            return

        csv_path_text = self.areas_csv_var.get().strip()
        if not csv_path_text:
            messagebox.showerror("Erro", "Selecione o CSV de areas manuais (imagem,area_manual_px).")
            return

        items = self._all_items()
        if not items:
            messagebox.showwarning("Aviso", "Nao ha imagens validas para processar.")
            return
        missing = self._missing_manual_areas(items)
        if missing:
            preview = "\n".join(missing[:8])
            if len(missing) > 8:
                preview += f"\n... +{len(missing) - 8} imagem(ns) sem area manual."
            messagebox.showerror(
                "Erro",
                "Existem imagens sem area manual no CSV.\n"
                f"{preview}",
            )
            return

        self.processed_by_group = {}
        self.processing_errors = {}
        self.pending_reprocess = set()
        self.reprocessed_history = set()
        self.reviewed_groups = set()
        self._sync_check_vars_for_group()

        self._start_processing_job(mode="full", items=items, save_after=False)

    def _unselected_times_in_current_group(self) -> list[str]:
        group_key = self.current_group_key
        if group_key is None:
            return []

        processed = self.processed_by_group.get(group_key, {})
        unselected = []
        for time_tag in TIME_ORDER:
            if (not self.refazer_vars[time_tag].get()) and time_tag in processed:
                unselected.append(time_tag)
        return unselected

    def _redefine_roi_for_keys(
        self,
        keys: list[tuple[str, str]],
        title_prefix: str = "",
        require_all: bool = False,
    ) -> tuple[list[tuple[str, str]], bool]:
        updated: list[tuple[str, str]] = []
        aborted = False
        total = len(keys)
        for idx, (group_key, time_tag) in enumerate(keys, start=1):
            proc = self.processed_by_group.get(group_key, {}).get(time_tag)
            if proc is None:
                self.processing_errors[(group_key, time_tag)] = "Sem processamento previo para redefinir ROI."
                continue

            key = (group_key, time_tag)
            seed_roi = self.roi_by_item.get(key)
            if seed_roi is None:
                seed_roi = proc.artifacts.mask_auto
            else:
                seed_roi = resize_mask(seed_roi, proc.artifacts.base_rgb_u8.shape[:2])

            step = f"{title_prefix} ({idx}/{total})" if total > 1 else title_prefix
            step = step.strip()
            header = f"{step}\n" if step else ""
            while True:
                new_roi = self._open_roi_editor(
                    image=proc.artifacts.base_rgb_u8,
                    initial_mask=seed_roi,
                    title=(
                        f"{header}Grupo {group_key} | {time_tag.upper()}\n"
                        "Modo MATLAB: duplo clique fecha poligono + Enter confirma | Modo Geometrico: arraste e solte para confirmar Retangulo/Elipse | Esc cancela"
                    ),
                )
                if new_roi is None:
                    if not require_all:
                        break
                    retry = messagebox.askyesno(
                        "Redefinicao de ROI",
                        f"A ROI de {group_key} {time_tag.upper()} nao foi confirmada.\n"
                        "Deseja tentar novamente?\n\n"
                        "Clique 'Nao' para cancelar toda a fila.",
                    )
                    if retry:
                        continue
                    aborted = True
                    break

                if not np.any(new_roi):
                    messagebox.showerror("Erro", f"A ROI definida para {group_key} {time_tag.upper()} ficou vazia.")
                    continue

                self.roi_by_item[key] = new_roi
                updated.append(key)
                break

            if aborted:
                break

        return updated, aborted

    def _redefine_roi_for_unselected(self) -> None:
        group_key = self.current_group_key
        if group_key is None:
            messagebox.showinfo("Aviso", "Selecione um grupo para redefinir ROI.")
            return

        unselected_times = self._unselected_times_in_current_group()
        if not unselected_times:
            messagebox.showinfo(
                "Aviso",
                "Deixe ao menos uma imagem sem selecao (0h/24h/48h) para redefinir ROI.",
            )
            return

        keys = [(group_key, time_tag) for time_tag in unselected_times]
        updated_keys, _aborted = self._redefine_roi_for_keys(keys)
        updated = len(updated_keys)

        if updated > 0:
            self.status_var.set(f"ROI atualizada para {updated} imagem(ns) nao selecionada(s).")
            self._refresh_metrics(group_key)
        else:
            self.status_var.set("Nenhuma ROI foi alterada.")

    def _clear_roi_for_unselected(self) -> None:
        group_key = self.current_group_key
        if group_key is None:
            return

        unselected_times = self._unselected_times_in_current_group()
        if not unselected_times:
            messagebox.showinfo("Aviso", "Deixe ao menos uma imagem sem selecao para limpar ROI.")
            return

        removed = 0
        for time_tag in unselected_times:
            key = (group_key, time_tag)
            if key in self.roi_by_item:
                del self.roi_by_item[key]
                removed += 1

        if removed > 0:
            self.status_var.set(f"ROI personalizada removida de {removed} imagem(ns).")
        else:
            self.status_var.set("As imagens nao selecionadas ja estavam sem ROI personalizada.")
        self._refresh_metrics(group_key)

    def _finalize_review(self) -> None:
        if not self.group_order:
            messagebox.showinfo("Aviso", "Nao ha grupos carregados.")
            return
        if not self.processed_by_group:
            messagebox.showinfo("Aviso", "Processe os grupos antes de finalizar.")
            return
        if not self._ensure_output_dir_selected():
            return

        missing_groups = [g for g in self.group_order if g not in self.reviewed_groups]
        if missing_groups:
            preview = ", ".join(missing_groups[:4])
            if len(missing_groups) > 4:
                preview += f", ... +{len(missing_groups) - 4}"
            messagebox.showwarning(
                "Revisao incompleta",
                "Passe por todos os grupos antes de finalizar.\n"
                f"Faltam: {preview}",
            )
            return

        pending_roi_group = sorted(self.pending_reprocess)
        if not pending_roi_group:
            out_dir = self._save_results_to_disk()
            self.status_var.set(f"Todas as imagens foram selecionadas para salvar. Resultados salvos em: {out_dir}")
            messagebox.showinfo(
                "Concluido",
                f"Nenhuma imagem ficou no novo grupo para redefinicao de ROI.\nResultados salvos em:\n{out_dir}",
            )
            return

        try:
            out_dir = self._save_results_to_disk()
        except Exception as exc:
            messagebox.showerror("Erro", f"Falha ao salvar resultados selecionados antes da redefinicao:\n{exc}")
            return
        self.status_var.set(
            "Resultados selecionados salvos. Iniciando fila de redefinicao de ROI do novo grupo..."
        )
        self.root.update_idletasks()

        updated_keys, aborted = self._redefine_roi_for_keys(
            pending_roi_group,
            title_prefix="Redefinicao de ROI",
            require_all=True,
        )
        if aborted:
            self.status_var.set(
                "Redefinicao cancelada pelo usuario. Nenhum reprocessamento executado nesta finalizacao."
            )
            messagebox.showinfo(
                "Processo cancelado",
                "A fila de redefinicao de ROI foi cancelada antes de concluir todas as imagens do novo grupo.\n"
                "As imagens selecionadas para salvar ja foram salvas.",
            )
            return
        if self.current_group_key:
            self._refresh_metrics(self.current_group_key)
            self._refresh_figure(self.current_group_key)

        items: list[tuple[str, str, Path]] = []
        for group_key, time_tag in updated_keys:
            path = self.group_files.get(group_key, {}).get(time_tag)
            if path is None:
                self.processing_errors[(group_key, time_tag)] = "Arquivo nao encontrado para reprocessar."
                continue
            items.append((group_key, time_tag, path))

        if not items:
            self.status_var.set(f"Nenhuma ROI foi redefinida. Resultados selecionados salvos em: {out_dir}")
            messagebox.showinfo(
                "Concluido",
                "Nenhuma ROI foi redefinida nas imagens do novo grupo.\n"
                f"Resultados selecionados foram salvos em:\n{out_dir}",
            )
            return

        if len(updated_keys) < len(pending_roi_group):
            messagebox.showinfo(
                "Aviso",
                f"ROIs redefinidas para {len(updated_keys)} de {len(pending_roi_group)} imagem(ns) no novo grupo.\n"
                "Apenas as imagens com ROI redefinida serao reprocessadas agora.",
            )

        if len(items) < len(updated_keys):
            messagebox.showwarning("Aviso", "Algumas imagens redefinidas nao possuem arquivo valido para reprocessar.")

        missing = self._missing_manual_areas(items)
        if missing:
            preview = "\n".join(missing[:8])
            if len(missing) > 8:
                preview += f"\n... +{len(missing) - 8} imagem(ns) sem area manual."
            messagebox.showerror(
                "Erro",
                "Nao e possivel reprocessar sem area manual no CSV para todas as imagens.\n"
                f"{preview}",
            )
            return

        self._start_processing_job(mode="reprocess", items=items, save_after=True)

    def _save_results_clicked(self) -> None:
        if not self.processed_by_group:
            messagebox.showinfo("Aviso", "Nao ha resultados para salvar.")
            return
        if not self._ensure_output_dir_selected():
            return
        try:
            out_dir = self._save_results_to_disk()
        except Exception as exc:
            messagebox.showerror("Erro", f"Falha ao salvar resultados atuais:\n{exc}")
            return
        self.status_var.set(f"Resultados salvos em: {out_dir}")
        messagebox.showinfo("Salvo", f"Resultados salvos em:\n{out_dir}")

    @staticmethod
    def _safe_filename(name: str, max_len: int = 120) -> str:
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
        clean = re.sub(r"_+", "_", clean)
        if not clean:
            return "resultado"
        if len(clean) <= max_len:
            return clean
        digest = hashlib.sha1(clean.encode("utf-8")).hexdigest()[:10]
        head_len = max(1, max_len - 11)
        head = clean[:head_len].rstrip("._-")
        if not head:
            head = "resultado"
        return f"{head}_{digest}"

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

    def _save_results_to_disk(self) -> Path:
        output_text = self.output_dir_var.get().strip()
        if not output_text:
            raise RuntimeError("Pasta de saida nao definida.")

        out_dir = Path(output_text)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "resumo_resultados.csv"

        headers = [
            "grupo",
            "tempo",
            "arquivo_origem",
            "status_revisao",
            "area_manual_px",
            "area_auto_px",
            "erro_abs_px",
            "erro_pct",
            "acerto_pct",
            "area_ratio",
            "tempo_processamento_s",
            "overlay_png",
            "mask_png",
            "roi_png",
            "erro",
        ]

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()

            for group_key in self.group_order:
                group_processed = self.processed_by_group.get(group_key, {})
                for time_tag in self._group_times(group_key):
                    key = (group_key, time_tag)
                    src_path = self.group_files[group_key][time_tag]
                    proc = group_processed.get(time_tag)
                    err = self.processing_errors.get(key, "")

                    overlay_name = ""
                    mask_name = ""
                    roi_name = ""
                    area_manual = ""
                    area_auto = ""
                    erro_abs = ""
                    erro_pct = ""
                    acerto_pct = ""
                    area_ratio = ""
                    proc_time = ""

                    if proc is not None:
                        token = self._safe_filename(f"{group_key}_{time_tag}_{src_path.stem}")
                        err_tag = f"{proc.results.erro_pct:.2f}".replace(".", "_")
                        overlay_name = f"{token}_erro_{err_tag}_overlay.png"
                        mask_name = f"{token}_erro_{err_tag}_mask.png"
                        roi_name = f"{token}_erro_{err_tag}_roi.png"

                        overlay_bgr = cv2.cvtColor(proc.artifacts.contour_overlay_rgb_u8, cv2.COLOR_RGB2BGR)
                        mask_u8 = proc.artifacts.mask_auto.astype(np.uint8) * 255
                        roi_u8 = proc.artifacts.roi_mask.astype(np.uint8) * 255

                        overlay_path = out_dir / overlay_name
                        mask_path = out_dir / mask_name
                        roi_path = out_dir / roi_name

                        if not self._write_image_file(overlay_path, overlay_bgr):
                            raise RuntimeError(f"Falha ao salvar arquivo: {overlay_name}\nCaminho: {overlay_path}")
                        if not self._write_image_file(mask_path, mask_u8):
                            raise RuntimeError(f"Falha ao salvar arquivo: {mask_name}\nCaminho: {mask_path}")
                        if not self._write_image_file(roi_path, roi_u8):
                            raise RuntimeError(f"Falha ao salvar arquivo: {roi_name}\nCaminho: {roi_path}")

                        area_auto = f"{proc.results.area_auto:.0f}"
                        area_manual = f"{proc.results.area_manual:.0f}"
                        erro_abs = f"{proc.results.erro_abs:.0f}"
                        erro_pct = f"{proc.results.erro_pct:.6f}"
                        acerto_pct = f"{proc.results.acerto_pct:.6f}"
                        area_ratio = f"{proc.results.area_ratio:.6f}"
                        proc_time = f"{proc.results.processing_time_s:.6f}"

                    if key in self.pending_reprocess:
                        status_revisao = "nao_selecionada_para_salvar"
                    elif key in self.reprocessed_history:
                        status_revisao = "refeita"
                    else:
                        status_revisao = "selecionada_para_salvar"

                    writer.writerow(
                        {
                            "grupo": group_key,
                            "tempo": time_tag,
                            "arquivo_origem": str(src_path),
                            "status_revisao": status_revisao,
                            "area_manual_px": area_manual,
                            "area_auto_px": area_auto,
                            "erro_abs_px": erro_abs,
                            "erro_pct": erro_pct,
                            "acerto_pct": acerto_pct,
                            "area_ratio": area_ratio,
                            "tempo_processamento_s": proc_time,
                            "overlay_png": overlay_name,
                            "mask_png": mask_name,
                            "roi_png": roi_name,
                            "erro": err,
                        }
                    )

        return out_dir

    def _refresh_metrics(self, group_key: str | None = None) -> None:
        self.metrics_text.configure(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)

        if group_key is None:
            group_key = self.current_group_key
        if group_key is None:
            self.metrics_text.insert(tk.END, "Sem grupo selecionado.")
            self.metrics_text.configure(state=tk.DISABLED)
            return

        lines = [
            "=========== RESULTADOS (GRUPO) ===========",
            f"Grupo: {group_key}",
            f"Grupos revisados: {len(self.reviewed_groups)}/{len(self.group_order)}",
            "Segmentacao ativa: modo area/textura + proximidade do centro",
            (
                f"Centro: janela frac={self.config.center_window_frac:.2f} | "
                f"proc_scale={self.config.proc_scale:.2f}"
            ),
            "",
        ]

        group_processed = self.processed_by_group.get(group_key, {})
        group_times = self._group_times(group_key)
        if not group_times:
            lines.append("Sem imagens no grupo.")
        else:
            for time_tag in group_times:
                key = (group_key, time_tag)
                proc = group_processed.get(time_tag)
                selected_for_save = "SIM" if key not in self.pending_reprocess else "NAO"
                in_new_group = "SIM" if key in self.pending_reprocess else "NAO"
                custom_roi = "SIM" if key in self.roi_by_item else "NAO"

                lines.append(f"[{time_tag.upper()}] {self.group_files[group_key][time_tag].name}")
                lines.append(f"Selecionada para salvar: {selected_for_save}")
                lines.append(f"No novo grupo para ROI: {in_new_group}")
                lines.append(f"ROI personalizada: {custom_roi}")

                if proc is None:
                    err = self.processing_errors.get(key, "Nao processada.")
                    lines.append(f"Status: SEM RESULTADO ({err})")
                    lines.append("")
                    continue

                lines.append(f"Area manual: {proc.results.area_manual:.0f} px")
                lines.append(f"Area automatica: {proc.results.area_auto:.0f} px")
                lines.append(f"Erro absoluto: {proc.results.erro_abs:.0f} px")
                lines.append(f"Erro relativo: {proc.results.erro_pct:.2f} %")
                lines.append(f"Acerto relativo: {proc.results.acerto_pct:.2f} %")
                lines.append(f"Tempo proc.: {proc.results.processing_time_s:.3f} s")
                lines.append(
                    "Gabor/Kmeans: "
                    f"{proc.artifacts.effective_gabor_iterations}/{proc.artifacts.requested_gabor_iterations} | "
                    f"{proc.artifacts.effective_kmeans_iter}/{proc.artifacts.requested_kmeans_iter}"
                )
                lines.append("")

        self.metrics_text.insert(tk.END, "\n".join(lines).rstrip())
        self.metrics_text.configure(state=tk.DISABLED)

    def _refresh_figure(self, group_key: str | None = None) -> None:
        self.figure.clear()
        if group_key is None:
            group_key = self.current_group_key

        if group_key is None or group_key not in self.group_files:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Sem grupo selecionado", ha="center", va="center")
            ax.axis("off")
            self.canvas.draw_idle()
            return

        group_times = self._group_times(group_key)
        if not group_times:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Sem imagens para este grupo", ha="center", va="center")
            ax.axis("off")
            self.canvas.draw_idle()
            return

        group_processed = self.processed_by_group.get(group_key, {})
        ncols = len(group_times)
        for idx, time_tag in enumerate(group_times, start=1):
            ax = self.figure.add_subplot(1, ncols, idx)
            key = (group_key, time_tag)
            proc = group_processed.get(time_tag)

            if proc is not None:
                ax.imshow(proc.artifacts.contour_overlay_rgb_u8)
                title = (
                    f"{time_tag.upper()} | area={proc.results.area_auto:.0f} px | "
                    f"erro={proc.results.erro_pct:.2f}%"
                )
                if key in self.pending_reprocess:
                    title += " | novo grupo ROI"
                ax.set_title(title, fontsize=10)
            else:
                err = self.processing_errors.get(key, "Sem resultado")
                ax.text(0.5, 0.5, f"{time_tag.upper()}\n{err}", ha="center", va="center", wrap=True)
                ax.set_title(time_tag.upper(), fontsize=10)

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
            if verts is None:
                return
            arr = np.asarray(verts, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
                return
            selected["verts"] = arr.tolist()

        def _disconnect_selector() -> None:
            selector = selector_ref.get("selector")
            if selector is None:
                return
            try:
                selector.disconnect_events()
            except Exception:
                pass
            selector_ref["selector"] = None

        def _build_polygon_selector() -> PolygonSelector:
            def on_poly_select(verts):
                _set_selected_verts(verts)

            try:
                selector = PolygonSelector(
                    ax,
                    on_poly_select,
                    useblit=False,
                    props={"color": "#00ffd0", "linewidth": 2.4, "alpha": 0.95},
                    handle_props={"marker": "o", "markersize": 5.5, "mec": "black", "mfc": "#fff176", "alpha": 0.96},
                )
            except TypeError:
                selector = PolygonSelector(
                    ax,
                    on_poly_select,
                    useblit=False,
                    lineprops={"color": "#00ffd0", "linewidth": 2.4, "alpha": 0.95},
                    markerprops={"marker": "o", "markersize": 5.5, "mec": "black", "mfc": "#fff176", "alpha": 0.96},
                )

            if selected["verts"] is not None:
                try:
                    selector.verts = selected["verts"]
                except Exception:
                    pass
            return selector

        def _bounds_from_events(eclick, erelease) -> tuple[float, float, float, float] | None:
            if eclick is None or erelease is None:
                return None
            x0 = eclick.xdata
            y0 = eclick.ydata
            x1 = erelease.xdata
            y1 = erelease.ydata
            if x0 is None or y0 is None or x1 is None or y1 is None:
                return None
            x_min, x_max = sorted((float(x0), float(x1)))
            y_min, y_max = sorted((float(y0), float(y1)))
            if (x_max - x_min) < 1e-3 or (y_max - y_min) < 1e-3:
                return None
            return x_min, y_min, x_max, y_max

        def _bounds_from_extents(selector: object) -> tuple[float, float, float, float] | None:
            if selector is None or not hasattr(selector, "extents"):
                return None
            try:
                extents = selector.extents
            except Exception:
                return None
            if extents is None or len(extents) < 4:
                return None
            x0, x1, y0, y1 = [float(v) for v in extents[:4]]
            x_min, x_max = sorted((x0, x1))
            y_min, y_max = sorted((y0, y1))
            if (x_max - x_min) < 1e-3 or (y_max - y_min) < 1e-3:
                return None
            return x_min, y_min, x_max, y_max

        def _verts_from_bounds(bounds: tuple[float, float, float, float], mode: str) -> np.ndarray | None:
            x_min, y_min, x_max, y_max = bounds
            if mode == "Retangulo":
                return np.asarray(
                    [
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max],
                    ],
                    dtype=np.float32,
                )
            if mode == "Elipse":
                cx = 0.5 * (x_min + x_max)
                cy = 0.5 * (y_min + y_max)
                rx = 0.5 * (x_max - x_min)
                ry = 0.5 * (y_max - y_min)
                if rx < 1e-3 or ry < 1e-3:
                    return None
                theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False, dtype=np.float32)
                return np.column_stack((cx + rx * np.cos(theta), cy + ry * np.sin(theta))).astype(np.float32)
            return None

        def _build_rectangle_selector() -> RectangleSelector:
            def on_rect_select(eclick, erelease):
                bounds = _bounds_from_events(eclick, erelease)
                if bounds is None:
                    bounds = _bounds_from_extents(selector_ref.get("selector"))
                if bounds is None:
                    return
                verts = _verts_from_bounds(bounds, "Retangulo")
                if verts is not None:
                    _set_selected_verts(verts)
                    selected["accepted"] = True
                    plt.close(fig)

            try:
                return RectangleSelector(
                    ax,
                    on_rect_select,
                    useblit=False,
                    props={"edgecolor": "#00ffd0", "facecolor": "none", "linewidth": 2.4, "alpha": 0.98},
                )
            except TypeError:
                return RectangleSelector(
                    ax,
                    on_rect_select,
                    useblit=False,
                    rectprops={"edgecolor": "#00ffd0", "facecolor": "none", "linewidth": 2.4, "alpha": 0.98},
                )

        def _build_ellipse_selector() -> EllipseSelector:
            def on_ellipse_select(eclick, erelease):
                bounds = _bounds_from_events(eclick, erelease)
                if bounds is None:
                    bounds = _bounds_from_extents(selector_ref.get("selector"))
                if bounds is None:
                    return
                verts = _verts_from_bounds(bounds, "Elipse")
                if verts is not None:
                    _set_selected_verts(verts)
                    selected["accepted"] = True
                    plt.close(fig)

            try:
                return EllipseSelector(
                    ax,
                    on_ellipse_select,
                    useblit=False,
                    props={"edgecolor": "#00ffd0", "facecolor": "none", "linewidth": 2.4, "alpha": 0.98},
                )
            except TypeError:
                return EllipseSelector(
                    ax,
                    on_ellipse_select,
                    useblit=False,
                    rectprops={"edgecolor": "#00ffd0", "facecolor": "none", "linewidth": 2.4, "alpha": 0.98},
                )

        def _activate_mode(mode: str) -> None:
            _disconnect_selector()
            mode_ref["mode"] = mode
            if mode == "Retangulo":
                selector_ref["selector"] = _build_rectangle_selector()
            elif mode == "Elipse":
                selector_ref["selector"] = _build_ellipse_selector()
            else:
                selector_ref["selector"] = _build_polygon_selector()
            fig.canvas.draw_idle()

        mode_ax = fig.add_axes([0.835, 0.68, 0.15, 0.23])
        mode_ax.set_title("Modo ROI", fontsize=9)
        mode_selector = RadioButtons(mode_ax, ("MATLAB (poligono)", "Retangulo", "Elipse"), active=0)
        mode_selector.on_clicked(_activate_mode)

        fig.text(
            0.835,
            0.62,
            "MATLAB: desenhe e pressione Enter\nRetangulo/Elipse: clique, arraste e solte para confirmar\nEsc cancela a janela",
            fontsize=8,
            va="top",
        )

        _activate_mode("MATLAB (poligono)")

        def on_key(event):
            if event.key in ("enter", "return"):
                verts = selected["verts"]
                selector = selector_ref.get("selector")
                mode = mode_ref.get("mode", "")
                if mode in ("Retangulo", "Elipse"):
                    bounds = _bounds_from_extents(selector)
                    if bounds is not None:
                        geometric_verts = _verts_from_bounds(bounds, mode)
                        if geometric_verts is not None:
                            verts = geometric_verts
                elif verts is None and selector is not None and hasattr(selector, "verts"):
                    verts = selector.verts
                if verts is not None and len(verts) >= 3:
                    selected["verts"] = np.asarray(verts, dtype=np.float32).tolist()
                    selected["accepted"] = True
                    plt.close(fig)
            elif event.key == "escape":
                selected["accepted"] = False
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
        plt.show(block=True)
        _disconnect_selector()

        if not bool(selected["accepted"]) or selected["verts"] is None:
            return None

        verts = np.asarray(selected["verts"], dtype=np.float32)
        return polygon_to_mask(image.shape[:2], verts)

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interface para processar e revisar grupos 0h/24h/48h em lote.")
    parser.add_argument("--folder", type=str, default=default_folder_from_base_path(), help="Pasta inicial com imagens.")
    parser.add_argument("--areas_csv", type=str, default="", help="CSV com colunas imagem,area_manual_px.")
    parser.add_argument("--output", type=str, default="", help="Pasta de saida para salvar resultados.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = AreaEvalApp(
        folder_path=str(args.folder).strip(),
        areas_csv_path=str(args.areas_csv).strip(),
        output_dir=str(args.output).strip(),
    )
    app.run()


if __name__ == "__main__":
    main()
