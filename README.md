# Combining Algorithm Unrolling with Self-Supervised Learning

This project explores the early stages of incorporating self-supervised with algorithm unrolling. Code was written as part of a mathematical-engineering master's thesis (60 ECTS) @ Aalborg University, Denmark.

The implementation of ISTA-Net and MAE (with ViT) is based on the paper ["ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing"](https://arxiv.org/abs/1706.07929) and ["Masked Autoencoders Are Scalable Vision Learners"](https://arxiv.org/abs/2111.06377) respectively.

## Requirements

Created using Python 3.10.6. See requirements.txt for further details.

## Usage

### ISTA-Net and ista2vec

The script is designed such that variables are changed directly in the code. The `training.py` script builds and executes the training loop -- Just provide a list image paths. 

### ISTA-MAE and MAE

The script is designed such that parameters are set in a parameter file, `parameters.py` or `parameters_finetuning.py` for pre-training and fine-tuning respectively. The path to the data needs to be specified in the parameter file as well as a model path for fine-tuning. After setting parameters run one of the training files:

- `pre-train_istamae.py`
- `pre-train_mae.py`
- `train_sr_istamae.py`
- `train_sr_mae.py`

## Credit

Inspired by the original PyTorch implementation ["ISTA-Net-PyTorch"](https://github.com/jianzhangcs/ISTA-Net-PyTorch) by [Jian Zhang](https://github.com/jianzhangcs) and ["Masked Autoencoders: A PyTorch Implementation"](https://github.com/facebookresearch/mae) by [Xinlei Chen](https://github.com/endernewton) and [Kaiming He](https://github.com/KaimingHe).

```
@mastersthesis{jonhardsson2023,
    author       = {Jónhardsson, Magnus and Jørgensen, Mads and Larsen, Andreas},
    school       = {Aalborg University},
    title        = {Combining Algorithm Unrolling with Self-Supervised Learning for Compressed Sensing Image Super-Resolution},
    year         = {2023}
}
```
