# ALMFnet Introduction

This is the implementation of the paper [Learning to Search a Lightweight Generalized Network for Medical Image Fusion](https://ieeexplore.ieee.org/abstract/document/10360160/) (IEEE TCSVT).
Our generalized model supports the fusion of medical images combining MRI with PET/CT/SPECT modalities.

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

Find the string that describes the searched architectures by using the trained model. Copy and paste it into the genotypes.py, the format should consist of the primary architecture string.

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
## Any Question

If you have any questions or concerns regarding the code, please feel free to raise them in [Issues](https://github.com/RollingPlain/ALMFnet/issues) or email [Guanyao Wu](rollingplainko@gmail.com).
