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
