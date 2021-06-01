#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 1 &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 2 &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 3 &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 4 &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 5 &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 6 &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 7 &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 8 &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 9 &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 10 &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 11 &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 12 &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 13 &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 14 &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 15 &
