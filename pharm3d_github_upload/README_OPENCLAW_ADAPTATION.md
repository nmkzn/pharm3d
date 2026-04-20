# Pharm3D OpenClaw Adaptation

This upload package contains the cleaned source code, OpenClaw adaptation skill, and a single-molecule preprocessing demo.

## Main goal

Convert one raw SMILES molecule into an MLP-compatible tensor.pkl.

## Key folders

- pharm/
- static/files/
- one_molecule_run/
- skills/alpha_pharm3d_preprocess/

## Single molecule pipeline

SMILES -> conformer generation -> alignment / pocket-related preprocessing -> feature matrix -> tensor.pkl -> fixed-index re-flattening -> MLP-compatible tensor

## Example input format

SMILES,Molecule_ID,state

Example:

CC(C)Cc1ccc(C(C)C)cc1,TEST_MOL_001,1

## Important scripts

- one_molecule_run/pipeline_rdkit.py
- one_molecule_run/reflatten.py
- one_molecule_run/reflatten_with_fixed_index.py
- one_molecule_run/compare_tensor_width.py
- one_molecule_run/inspect_tensor_real.py

## OpenClaw adaptation

- skills/alpha_pharm3d_preprocess/SKILL.md

## Verified result

training index rows = 5855
tensor width = 5855
compatible = True

Large model weights, logs, raw generated job outputs, paper PDFs, and local OpenClaw state are intentionally excluded.
