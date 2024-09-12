Welcome to DDR's Documentation
==============================

.. toctree::
   :maxdepth: 1
   :caption: Deep Dimension Reduction for Supervised Representation Learning

   installation
   usage
   demo

Introduction
------------

DDR (Deep Dimension Reduction) is a novel approach to supervised representation learning that aims to construct effective data representations for prediction tasks. Our method focuses on three key characteristics of an ideal nonparametric representation of high-dimensional complex data:

1. Sufficiency
2. Low dimensionality
3. Disentanglement

DDR is a nonparametric generalization of the sufficient dimension reduction method. It finds a nonparametric representation by minimizing an objective function that characterizes conditional independence and promotes disentanglement at the population level. The target representation is then estimated at the sample level using deep neural networks.

Flowchart of DDR
----------------

Here's a high-level flowchart illustrating the DDR process:

.. figure:: _static/ddr_flowchart.png
   :width: 600
   :alt: DDR Flowchart

   Figure 1: Flowchart of the Deep Dimension Reduction (DDR) process

This flowchart outlines the key steps in the DDR process, from input data to the final low-dimensional representation used for prediction tasks.


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