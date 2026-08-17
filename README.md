# DRAM Modern DL Navigation Recovery

Coarse-to-fine hackathon pipeline:
Synthetic DRAM -> multi-scale candidate generation -> hard negatives -> lightweight DL re-ranking -> coordinate refinement -> evaluation.

Models: `cnn`, `mobilenet`, `attention`.

Install:
`pip install -r requirements.txt`

Quick run:
`python main.py --samples 100 --model mobilenet --epochs 5`

Competition run:
`python main.py --samples 1000 --model mobilenet --epochs 15`

Try attention:
`python main.py --samples 1000 --model attention --epochs 15`
