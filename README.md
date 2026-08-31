# Wound Healing Segmentation

Automatic image-processing pipeline for wound healing scratch-assay microscopy images.

This repository contains the third and current image-processing pipeline developed during an undergraduate scientific research project focused on the automatic segmentation and quantitative analysis of wound healing assays.

The method was designed to identify the wound region in microscopy images while dealing with variations in image brightness, cell migration, texture, and artifacts commonly observed between images acquired at different experimental time points.

---

## Overview

The proposed pipeline performs automatic processing of wound healing microscopy images and generates segmented wound regions that can be used for quantitative analysis.

The processing workflow includes image preprocessing, adaptive segmentation, morphological operations, and wound-region evaluation.

The current implementation corresponds to **Pipeline 3**, developed as an adaptive alternative to previous image-processing approaches.

---

## Repository Structure

```text
wound-healing-segmentation/
│
├── src/
│   ├── wound_healing_segmentation.py
│   ├── area_eval_interface.py
│   └── matlab_style_area_eval.py
│
├── input/
│   └── README.md
│
├── output/
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

### `src/`

Contains the Python source code required for image processing and quantitative evaluation.

### `input/`

Directory intended for microscopy images to be processed.

### `output/`

Directory intended for generated segmentation results and quantitative outputs.

---

## Requirements

The software was developed in Python.

The required Python packages are listed in:

```text
requirements.txt
```

Python 3.14 was used in the current development environment.

---

## Installation

Clone or download this repository.

Open a terminal inside the project directory:

```powershell
cd wound-healing-segmentation
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

## Running the Software

With the virtual environment activated, run:

```powershell
python .\src\wound_healing_segmentation.py
```

The processing interface will then be initialized.

---

## Processing Pipeline

The general workflow of Pipeline 3 can be summarized as:

```text
Microscopy image
      ↓
Image preprocessing
      ↓
Adaptive parameter selection
      ↓
Wound-region segmentation
      ↓
Morphological refinement
      ↓
Wound mask
      ↓
Quantitative analysis
```

The pipeline was developed to improve segmentation robustness, particularly when analyzing wound healing images acquired after cellular migration has altered the original wound boundaries.

---

## Input Images

The software is intended for microscopy images obtained from wound healing scratch assays.

Experimental images used during the development and validation of this research are **not included in this repository by default**.

Images to be processed can be placed in the `input/` directory or selected through the processing interface, depending on the current software configuration.

---

## Output

Depending on the selected processing options, the software can generate outputs such as:

* segmented wound masks;
* processed microscopy images;
* wound-area measurements;
* quantitative analysis files;
* segmentation evaluation results.

Generated files should be stored in the `output/` directory.

---

## Scientific Context

Scratch assays are commonly used to investigate cellular migration and wound closure in vitro.

Manual wound-area measurement can be time-consuming and susceptible to variability between analyses. This project investigates classical image-processing strategies for automatically identifying wound regions and supporting quantitative evaluation of wound healing experiments.

Pipeline 3 represents the adaptive processing framework developed during the final stage of the research.

---

## Limitations

The current pipeline was developed and evaluated using microscopy images obtained under specific experimental and acquisition conditions.

Image characteristics such as illumination, contrast, cellular density, artifacts, and acquisition settings may affect segmentation performance.

Therefore, additional validation may be required before applying the method to images obtained from other laboratories, microscopy systems, cell lines, or experimental protocols.

---

## Data Availability

The experimental image dataset is not distributed through this repository.

Example images may be added separately when appropriate for public distribution.

---

## Authors

Developed as part of an undergraduate scientific research project at the **Federal University of Uberlândia (UFU)**.

Author information and research contributors will be added to the final public release.

---

## Citation

Citation information for the software and its associated scientific publication will be provided in a future release.

---

## License

License information will be added before the final public release of the repository.
