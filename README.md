
# Drift-Sense

AI-ready navigation-error recovery benchmark for synthetic DRAM-style wafer inspection imagery.

## Quick start

```powershell
python -m pip install -r requirements.txt
python generate_dataset.py --architecture DRAM --samples 30
python evaluate.py
python inference.py dataset/reference/ref_0001.png dataset/search/search_0001.png --details
```

The generator uses a 1000x1000 high-magnification reference and a 10000x10000 physical search field downsampled 10x to 1000x1000. Reference and search captures receive independent sensor noise. Edge brightening, blur, scale and rotation variation are applied.

The current baseline is classical multi-scale intensity + edge template matching. In periodic ambiguity, near-equal candidates are resolved by choosing the candidate closest to the search-image center, matching the task specification.

## Inference contract

```powershell
python inference.py REFERENCE_PATH SEARCH_PATH
```

It prints exactly:

```text
x y
```

## Evaluation

`results/predictions.csv` contains per-sample predictions and errors. `results/summary.json` contains aggregate metrics.

## Limitations

This build intentionally establishes a reproducible classical baseline before adding a learned candidate reranker. It should not be presented as a deep-learning final model until a held-out test set is used.
