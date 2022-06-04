(

CUDA_VISIBLE_DEVICES=3 python main.py --corruption_ratio 0.2 >log1.txt 2>&1 && \
CUDA_VISIBLE_DEVICES=3 python main.py --corruption_ratio 0.4 >log2.txt 2>&1 && \
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --corruption_ratio 0.2 >log3.txt 2>&1 && \
CUDA_VISIBLE_DEVICES=3 python main.py --dataset cifar100 --corruption_ratio 0.4 >log4.txt 2>&1 \


) &