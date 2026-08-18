
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

The periodic DRAM background is, by construction, identical at every tile occurrence -- exactly like a real memory array, where every unit cell is genuinely indistinguishable from its neighbors. To make the localization ground truth well-defined (and solvable from image content, not just guessed), each sample embeds a unique cross-shaped alignment fiducial into exactly one physical tile and into the matching reference; ground truth is that fiducial's true, geometrically-transformed location. This mirrors how real photolithography reticles/wafers use alignment marks distinct from the repeating array to anchor navigation.

The current baseline is classical multi-scale intensity + edge template matching. It is not fiducial-aware -- it matches whole tiles rather than searching for the landmark specifically -- so periodic ambiguity from the shared background still causes it to prefer wrong tiles some of the time. Near-equal candidates are resolved by choosing the candidate closest to the search-image center as a generic heuristic tie-break, not because that is guaranteed correct.

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
