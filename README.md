
## DiffDark

### DiffDark: Multi-prior integration driven diffusion model for low-light image enhancement

## How to use our Code?

Here we provide an example for the **Low-light Image Enhancement (LIE)**, but it can be changed to solve other problems by replacing the dataset.

We retrained the model from scratch using a single Nvidia 4090 GPU.

Note that **we didn't tune any parameter**, the last saved checkpoint was used to evaluation.

## Dataset
### Dataset Preparation
We utilize the LOL-v1 dataset, with 485 images for training and 15 images for testing.

Download LOL-v1 dataset: [LOL-v1](https://daooshee.github.io/BMVC2018website/)

### Dataset Directory

Download training and testing datasets and process it in a way such that normal-light images and low-light images are in separately directories, as

```bash
#### training dataset ####
./data/lol/train/at/
./data/lol/train/gt/
./data/lol/train/he/
./data/lol/train/input/
./data/lol/train/reflection/

#### testing dataset ####
./data/val/at/
./data/val/gt/
./data/val/he/
./data/val/input/
./data/val/reflection/
```
Then get into the `./config/lol.yml` directory and modify the dataset paths.

## Train
```bash
attention map.py
histogram equalization.py
The main code for the Retinex decomposition network is located in `./Retinex/train.py`. 
train_diffusion.py
```

## Test
```bash
attention map.py
histogram equalization.py
he main code for the Retinex decomposition network is located in `codes/config/Retinex/predict.py`. 
eval_diffusion.py
```

## Acknowledgement
Our code is adapted from the original [WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion) repository. We thank the authors for sharing their code.

## Citation

If our work is useful for your research, please consider citing:

```
@article{hu2025diffdark,
  title={DiffDark: Multi-prior integration driven diffusion model for low-light image enhancement},
  author={Hu, Renzhi and Luo, Ting and Jiang, Gangyi and Chen, Yeyao and Xu, Haiyong and Liu, Leiming and He, Zhouyan},
  journal={Pattern Recognition},
  pages={111814},
  year={2025},
  publisher={Elsevier}
}
```
---
#### --- Thanks for your interest! --- ####
