# AD3: A Multi-Scale Asynchronous Divide-Diffuse-Decode Diffusion Framework for TSP Solving [Scalable Data Science]
Code for the paper AD3: A Multi-Scale Asynchronous Divide-Diffuse-Decode Diffusion Framework for TSP Solving [Scalable Data Science]

## Dependencies
AD3 is built in Python 3.8.13  and Pytorch  1.11.0. 
And Please use the following command to install the requirements:
```bash
pip install -r requirements.txt
```
## ckpt
The models for reproduce could be download from https://anonymous.4open.science/r/HiDiff-N11315
The model files are placed in the "storage/ckpt/" directory by default.

## Evaluate
We provide both single-GPU and multi-GPU configurations in AD3_evaluate_S.py and AD3_evaluate_P.py. We are currently integrating them.
```bash
- `evaluate_single_gpu.py`: single-GPU evaluation script
- `evaluate_multi_gpu.py`: multi-GPU evaluation script
```
The results are recorded in "storage/logs" directory by default.

## Usage
Run from project root with relative paths only:

```bash
python evaluate_single_gpu.py --help
python evaluate_multi_gpu.py --help
```

Then provide your own relative paths for checkpoints/data if needed.
