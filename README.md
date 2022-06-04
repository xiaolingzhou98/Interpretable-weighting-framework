# Explainable weighting framework
 
This is the code for the paper: Investigating Weighting Mechanism through an Explainable Weighting Framework<br>

Set ups
-------  

Linux<br>
python 3.8<br>
pytorch 1.9.0<br>
torchvision 0.10.0<br>

Running Explainable weighting framework on benchmark datasets (CIFAR-10 and CIFAR-100).
-------  

ResNet32 on CIFAR10-LT with imbalanced factor of 10:<br>

`python main.py --dataset cifar10 --imbalanced_factor 10`

ResNet32 on noisy CIFAR10 with 20\% pair-flip noise:<br>
`python main.py --dataset cifar10 --corruption_type flip2 --corruption_ratio 0.2`