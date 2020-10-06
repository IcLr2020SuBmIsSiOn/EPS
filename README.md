# [EPS] Improving Random-Sampling Neural Architecture Search by Evolving the Proxy Search Space 

This repository is for anonymous ICLR2021 submission [link](https://openreview.net/forum?id=qk0FE399OJ)
## Requirements

To install requirements:

```
Python >= 3.7, PyTorch >= 1.0.0, torchvision >= 0.2.1, opencv
```

##Datasets
Instructions for acquiring PTB can be found [here](https://github.com/salesforce/awd-lstm-lm).
NASBench-201 API installation and ImageNet-16-120 dataset can be found [here](https://github.com/D-X-Y/NAS-Bench-201)
Please put all the dataset in the ```./data``` folder.
## Usages 
### Search
#### Search the CNN on NASBench-201 search space:
```
cd search_cnn_nasbench
bash run.sh
```
#### Search the CNN on DARTS search space:
```
cd search_cnn_darts
bash run.sh
```
#### Search the CNN ON ROBUST-DARTS search space: 
```
TBA
```
#### Search the RNN:
```
cd search_rnn_darts
bash run.sh
```
### Evaluation
#### Nasbench-201
Please follow the notebook examples in ```./search_cnn_nasbench/InstructionForSelection.ipynb```.
#### DARTS-CIFAR
```
TBA
```
#### ROBUST-DARTS-CIFAR
```
TBA
```
#### DARTS-PTB
```
TBA
```

## Citation
Our work is based on the code of following papers ([DARTS](https://github.com/quark0/darts), [RandomNAS](https://github.com/liamcli/randomNAS_release), [NASBench-201](https://github.com/D-X-Y/AutoDL-Projects), [SPOS](https://github.com/megvii-model/SinglePathOneShot))
[Robust-DARTS](https://github.com/automl/RobustDARTS). Also if EPS is good for you, please consider cite:
```
@article{liu2018darts,
  title={Darts: Differentiable architecture search},
  author={Liu, Hanxiao and Simonyan, Karen and Yang, Yiming},
  journal={arXiv preprint arXiv:1806.09055},
  year={2018}
}
@article{li2019random,
  title={Random search and reproducibility for neural architecture search},
  author={Li, Liam and Talwalkar, Ameet},
  journal={arXiv preprint arXiv:1902.07638},
  year={2019}
}
@article{guo2019single,
  title={Single path one-shot neural architecture search with uniform sampling},
  author={Guo, Zichao and Zhang, Xiangyu and Mu, Haoyuan and Heng, Wen and Liu, Zechun and Wei, Yichen and Sun, Jian},
  journal={arXiv preprint arXiv:1904.00420},
  year={2019}
}
@article{dong2020bench,
  title={NAS-Bench-102: Extending the Scope of Reproducible Neural Architecture Search},
  author={Dong, Xuanyi and Yang, Yi},
  journal={arXiv preprint arXiv:2001.00326},
  year={2020}
}
@inproceedings{zela2020understanding,
	title={Understanding and Robustifying Differentiable Architecture Search},
	author={Arber Zela and Thomas Elsken and Tonmoy Saikia and Yassine Marrakchi and Thomas Brox and Frank Hutter},
	booktitle={International Conference on Learning Representations},
	year={2020},
	url={https://openreview.net/forum?id=H1gDNyrKDS}
}
@inproceedings{
anonymous2021improving,
title={Improving Random-Sampling Neural Architecture Search by Evolving the Proxy Search Space},
author={Anonymous},
booktitle={Submitted to International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=qk0FE399OJ},
note={under review}
}
```