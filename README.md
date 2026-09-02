# Wound Healing Segmentation

Image-processing methods for automatic segmentation and quantitative analysis of wound healing scratch-assay microscopy images.

This repository contains three image-processing pipelines developed and evaluated during an undergraduate scientific research project focused on the automatic identification of wound regions in microscopy images.

The pipelines represent successive approaches investigated throughout the research, ranging from conventional threshold-based segmentation to a more adaptive image-processing framework designed to handle variations in illumination, texture, cellular migration, and experimental artifacts.

---

## Overview

Automatic analysis of wound healing scratch assays can reduce the time required for manual wound-area measurement and improve the reproducibility of quantitative evaluations.

During this research, three processing pipelines were investigated:

### Pipeline 1 — Conventional Segmentation

The first pipeline was developed in MATLAB and uses a conventional image-processing workflow based on preprocessing, contrast enhancement, threshold-based segmentation, and morphological refinement.

This approach provides relatively low computational complexity but can be more sensitive to local artifacts and variations in wound appearance.

### Pipeline 2 — Texture-Based Segmentation

The second pipeline was also developed in MATLAB and introduces additional image-processing strategies for texture analysis and segmentation.

The workflow includes homomorphic filtering, Gabor-based texture analysis, clustering using k-means, and morphological processing.

This approach was investigated to improve the discrimination between wound and cellular regions in images with more complex textures.

### Pipeline 3 — Adaptive Segmentation

The third pipeline was implemented in Python and corresponds to the current adaptive framework developed during the research.

This method introduces adaptive processing strategies intended to account for variations in image characteristics and experimental time points.

The pipeline was designed particularly to improve segmentation robustness in images where cellular migration, reduced contrast, texture variations, and experimental artifacts make the wound boundaries less evident.

---

## Repository Structure

```text
wound-healing-segmentation/
│
├── src/
│   ├── pipeline_1_matlab/
│   │   └── pipeline_1.m
│   │
│   ├── pipeline_2_matlab/
│   │   └── pipeline_2.m
│   │
│   └── pipeline_3_python/
│       ├── wound_healing_segmentation.py
│       ├── area_eval_interface.py
│       └── matlab_style_area_eval.py
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

---

## Processing Workflow

The three pipelines follow the general workflow:

```text
Microscopy image
       ↓
Image preprocessing
       ↓
Wound-region segmentation
       ↓
Morphological refinement
       ↓
Wound mask
       ↓
Quantitative analysis
```

The specific preprocessing and segmentation strategies differ between the three pipelines.

---

## Pipeline 1

**Language:** MATLAB

Pipeline 1 represents the initial conventional image-processing approach investigated in the project.

The source code is available in:

```text
src/pipeline_1_matlab/
```

Its main processing stages include image preprocessing, contrast enhancement, automatic threshold-based segmentation, and morphological refinement.

---

## Pipeline 2

**Language:** MATLAB

Pipeline 2 was developed to introduce additional texture information into the segmentation procedure.

The source code is available in:

```text
src/pipeline_2_matlab/
```

The method includes preprocessing strategies such as homomorphic filtering, Gabor-based texture analysis, clustering, and morphological operations.

---

## Pipeline 3

**Language:** Python

Pipeline 3 corresponds to the adaptive framework developed during the final stage of the study.

The source code is available in:

```text
src/pipeline_3_python/
```

The method was designed to provide greater robustness to differences between microscopy images and experimental conditions, particularly for images acquired after cellular migration.

---

## Python Requirements

The dependencies required for Pipeline 3 are listed in:

```text
requirements.txt
```

To create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

## Running Pipeline 3

From the repository root, run:

```powershell
python .\src\pipeline_3_python\wound_healing_segmentation.py
```

The processing interface will then be initialized.

---

## MATLAB Pipelines

Pipeline 1 and Pipeline 2 were developed in MATLAB.

Their source files are located in:

```text
src/pipeline_1_matlab/
```

and

```text
src/pipeline_2_matlab/
```

The scripts should be executed from their respective directories using MATLAB.

---

## Experimental Images and Data Availability

The experimental microscopy dataset used for the development and evaluation of the pipelines is **not publicly distributed through this repository**.

Therefore, the original wound healing microscopy images used in the study are not included.

The `input/` directory is provided only as a location where users can place their own compatible microscopy images for processing.

Experimental data must not be committed to the repository.

---

## Output

Depending on the selected pipeline and processing configuration, the software may generate:

* wound segmentation masks;
* processed microscopy images;
* wound-area measurements;
* quantitative results;
* segmentation evaluation data.

Generated files should be stored in the `output/` directory and are not tracked by Git by default.

---

## Scientific Context

Scratch assays are widely used to investigate cellular migration and wound closure in vitro.

Manual wound-area delineation may require substantial analysis time and can introduce variability into quantitative measurements. Automatic image-processing methods provide an alternative for identifying wound regions and assisting quantitative evaluation.

This project investigates the performance and limitations of three classical image-processing strategies developed progressively throughout the undergraduate research project.

---

## Limitations

The pipelines were developed and evaluated using microscopy images obtained under specific experimental and acquisition conditions.

Variations in illumination, image contrast, cell density, acquisition system, experimental artifacts, and wound morphology may influence segmentation performance.

The experimental dataset used for validation originated from a specific laboratory environment, and additional validation is required before generalizing the methods to other microscopy systems, experimental protocols, or datasets.

---

## Data Availability

The source code is publicly available in this repository.

The experimental microscopy images used in the research are not publicly available through this repository.

Users interested in testing the algorithms may use their own compatible wound healing scratch-assay microscopy images.

---

## Authors

Developed as part of an undergraduate scientific research project at the **Federal University of Uberlândia (UFU)**.

Author and contributor information will be included in the final software release.

---

## Citation

Citation information for the software and its associated scientific publication will be provided in the final public release.

---

## License

License information will be provided in the final public release.
