#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 16 &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 17 &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 18 &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 19 &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 20 &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 21 &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 22 &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 23 &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 24 &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 25 &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 26 &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 27 &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 28 &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 29 &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17 --data_name CIFAR10 --trail 30 & 