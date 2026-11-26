# readout-space-harmonization

Code and aggregate data for:

> **What affine feature harmonization can and cannot change in a frozen CT foundation model: a readout-space decomposition and five-vendor validation**
> Lu Sun, Yunfei Li, Hongyu Liang, Hao Zhang

The paper asks a narrow question: for a *frozen* CT readout, which components of an affine feature correction (`T(z) = Az + b`) can change the output at all? The answer is forced by linear algebra — the linear part `A` is an exact reparameterization of the head's first layer and adds no information; for the bias-free CT-CLIP head only the patient-independent offset survives, and it acts as a rotation of the normalized score vector — and the repository holds the code and the aggregate numbers behind that argument.

## Contents

```
figures/            plotting code and numerical data for Figures 1-4
  extracted_figure_data.json   aggregate quantities plotted in Figures 2-4
  make_figures_jmi.py          regenerates all four figures
  figure_manifest.json         figure sizes, formats and input digests
JMI_Supplementary_Data.xlsx    the same numbers as supplied with the submission, one sheet per figure or table
```

Run `python3 figures/make_figures_jmi.py` (needs `numpy`, `scipy`, `matplotlib`; no GPU; a few seconds). Outputs are written next to the script.

## Data

No images, patient identifiers or re-identifiable metadata are in this repository. All chest CT volumes, validation metadata and report-derived labels come from the public **CT-RATE** release:

- <https://huggingface.co/datasets/ibrahimhamamci/CT-RATE>
- revision `546640db5d4674f6e6b36d40d2cfcd57c8b39e21`
- licence CC-BY-NC-SA 4.0, plus the provider's terms of use, which prohibit re-identification

The CT-CLIP checkpoint is likewise obtained from the provider. The cohort used in the paper is the official `valid_` split; the two paired strata and the three external cohorts are defined by scanner model, kernel and matrix in the released metadata, so the cohort can be reconstructed from public files alone.

## Status

This repository carries the figure-generation code and the numerical outputs behind Figures 1-4, which are sufficient to regenerate every figure and to check the cohort summaries tabulated in the paper. The correction-fitting, null-space and bootstrap routines are part of the same analysis codebase and are being prepared for release here; they are not included in this version, and the paper's code statement describes only what is released.

## Citation

Cite the journal article once it is published. A `CITATION.cff` and a DOI-archived release will be added at acceptance.

## Licence

MIT for the code in this repository (see `LICENSE`). The CT-RATE data remain under their own licence and are not covered by it.
