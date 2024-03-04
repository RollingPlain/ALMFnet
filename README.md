# ALMFnet
ALMFnet

# Introduction

This is the implementation of the paper [Learning to Search a Lightweight Generalized Network for Medical Image Fusion](https://ieeexplore.ieee.org/abstract/document/10360160/) (IEEE TCSVT).

## Requirements

* python >= 3.6
* pytorch >= 1.7
* torchvision >= 0.8

* For other packages, please refer to the requirements.txt

## Test

```shell
python eval.py
```

## Search & Train

### step 1

```shell
python train_search_lat.py
```

### step 2

Find the string which descripting the searched architectures in the log file. Copy and paste it into the genotypes.py, the format should consist with the primary architecture string.

### step 3

```shell
python train.py
```

## Citation

If you use any part of this code in your research, please cite our paper:

```
@article{mu2023learning,
  title={Learning to Search a Lightweight Generalized Network for Medical Image Fusion},
  author={Mu, Pan and Wu, Guanyao and Liu, Jinyuan and Zhang, Yuduo and Fan, Xin and Liu, Risheng},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  publisher={IEEE}
}
```
