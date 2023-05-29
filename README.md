# Interpretable weighting framework
 
This is the code for the paper: Investigating the Weighting Mechanism Using an Interpretable Weighting Framework<br>

Setups
-------  
The requiring environment is as bellow:<br>
* Linux<br>
* python 3.8<br>
* pytorch 1.9.0<br>
* torchvision 0.10.0<br>

Running Interpretable weighting framework on benchmark datasets (CIFAR-10 and CIFAR-100).
-------  
Here are two examples for training imbalanced and noisy data:<br>
ResNet32 on CIFAR10-LT with imbalanced factor of 10:<br>

`python main.py --dataset cifar10 --imbalanced_factor 10`

ResNet32 on noisy CIFAR10 with 20\% pair-flip noise:<br>
`python main.py --dataset cifar10 --corruption_type flip2 --corruption_ratio 0.2`

The default sample weighting network in the code is Neural Regression Tree (NRT) with pruning. You can also use MLP as the sample weighting network. Both the two networks are in the file ``model.py".
