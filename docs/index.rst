Welcome to DDR's Documentation
==============================

Deep Dimension Reduction for Supervised Representation Learning
---------------------------------------------------------------

Abstract
--------
The goal of supervised representation learning is to construct effective data representations for prediction. Among all the characteristics of an ideal nonparametric representation of high-dimensional complex data, sufficiency, low dimensionality and disentanglement are some of the most essential ones. We propose a deep dimension reduction approach to learning representations with these characteristics.

The proposed approach is a nonparametric generalization of the sufficient dimension reduction method. We formulate the ideal representation learning task as that of finding a nonparametric representation that minimizes an objective function characterizing conditional independence and promoting disentanglement at the population level. We then estimate the target representation at the sample level nonparametrically using deep neural networks.

We show that the estimated deep nonparametric representation is consistent in the sense that its excess risk converges to zero. Our extensive numerical experiments using simulated and real benchmark data demonstrate that the proposed methods have better performance than several existing dimension reduction methods and the standard deep learning models in the context of classification and regression.

Installation
------------
To install DDR and its requirements, run this command in your terminal:

.. code-block:: bash

   pip install -r requirements.txt

Usage
-----
Here are some examples of how to use DDR:

For toy classification examples:

.. code-block:: bash

   python demo_toys.py --save 'Results/toys' --dataset 3

For toy regression examples:

.. code-block:: bash

   python demo_reg_toys.py --save 'Results/reg_toys' --model 2 --scenario 2

To train DDR on MNIST dataset:

.. code-block:: bash

   python train.py --save 'Results/MNIST' --latent_dim 16

Evaluation
----------
To evaluate DDR on MNIST:

.. code-block:: bash

   python eval.py --path 'Results/MNIST' --latent_dim 16

Pre-trained Models
------------------
While DDR does not adopt pre-trained models, we provide trained models to save time and computational resources. To evaluate DDR on MNIST with trained models:

.. code-block:: bash

   python eval.py --path 'Results/MNIST_trained_16' --latent_dim 16

Results
-------
DDR achieves the following performance on Image Classification on MNIST:

+-------------------+--------+--------+--------+
| Reduced Dimension |   16   |   32   |   64   |
+===================+========+========+========+
|        DDR        | 99.63% | 99.53% | 99.60% |
+-------------------+--------+--------+--------+

For more detailed results and experiments, please refer to the :ref:`results` section.

Citation
--------
If you use DDR in your research, please cite our paper:

.. code-block:: bibtex

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

Table of Contents
-----------------
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   methods
   results
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
