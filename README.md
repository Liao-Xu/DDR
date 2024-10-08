# Deep Dimension Reduction for Supervised Representation Learning

This repository is the demo implementation of [Deep Dimension Reduction for Supervised Representation Learning]. 
Paper link: https://ieeexplore.ieee.org/document/10456552

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train DDR on toy classification examples, run this command:

```train_toys
python demo_toys.py --save 'Results/toys' --dataset 3
```

To train DDR on toy regression examples, run this command:

```train_toys
python demo_reg_toys.py --save 'Results/reg_toys' --model 2 --scenario 2
```

To train DDR on MNIST dataset and get the evaluation results , run this command:

```train
python train.py --save 'Results/MNIST' --latent_dim 16
```

## Evaluation

To evaluate DDR on MNIST, run:

```eval
python eval.py --path 'Results/MNIST' --latent_dim 16
```


## Pre-trained Models

DDR does not adopt the pre-trained models. However, to save the time and expenses of training, we provide the trained models that can aviod the afresh training. To evaluate DDR on MNIST with trained models, run:

```eval_trained
python eval.py --path 'Results/MNIST_trained_16' --latent_dim 16
```

## Results

Our model DDR achieves the following performance on :

### [Image Classification on MNIST]

| Reduced Dimension|   16   |   32   |   64   |
| ---------------- |--------|--------|--------|
|        DDR       | 99.63% | 99.53% | 99.60% |

With trained models, results can be obtained by running

```eval_trained_all
python eval.py --path 'Results/MNIST_trained_16' --latent_dim 16
python eval.py --path 'Results/MNIST_trained_32' --latent_dim 32
python eval.py --path 'Results/MNIST_trained_64' --latent_dim 64
```
With the afresh training, results can be obtained by running

```train_all
python train.py --save 'Results/MNIST_16' --latent_dim 16
python train.py --save 'Results/MNIST_32' --latent_dim 32
python train.py --save 'Results/MNIST_64' --latent_dim 64
```
## Citation
If you use this implementation in your research, please cite our paper:
```bibtex
@article{huang2024deep,
  author={Huang, Jian and Jiao, Yuling and Liao, Xu and Liu, Jin and Yu, Zhou},
  journal={IEEE Transactions on Information Theory}, 
  title={Deep Dimension Reduction for Supervised Representation Learning}, 
  year={2024},
  volume={70},
  number={5},
  pages={3583-3598},
  keywords={Dimensionality reduction;Representation learning;Estimation;Vectors;Linear programming;Data models;Covariance matrices;Conditional independence;distance covariance;f-divergence;nonparametric estimation;neural networks},
  doi={10.1109/TIT.2023.3340658}}
```
