# On Utilizing Self-Supervised Method for Training Unrolled Algorithms
This project explores the early stages of incorporating self-supervised with algorithm unrolling. Code was written as part of a mathematical-engineering master's thesis (60 ECTS) @ Aalborg University, Denmark.

## ISTA-Net Adaptation for Image Super-Resolution
Based on the paper ["ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive Sensing"](https://arxiv.org/abs/1706.07929).

### Requirements:
Created using Python 3.10.6. See requirements.txt for further details.

### Usage:
The script is designed such that variables are changed directly in the code. The `training.py` script builds and executes the training loop -- Just provide a list image paths.

### Credit:
Inspired by the original PyTorch implementation ["ISTA-Net-PyTorch"](https://github.com/jianzhangcs/ISTA-Net-PyTorch) by [Jian Zhang](https://github.com/jianzhangcs).

```
@mastersthesis{jonhardsson2023,
    author       = {Jónhardsson, Magnus and Jørgensen, Mads and Larsen, Andreas},
    school       = {Aalborg University},
    title        = {Combining Algorithm Unrolling with Self-Supervised Learning for Compressed Sensing Image Super-Resolution},
    year         = {2023}
}
```