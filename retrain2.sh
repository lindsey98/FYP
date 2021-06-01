#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17_add1 --data_name CIFAR10 --trail 1 --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17_add1  --data_name CIFAR10 --trail 2  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17_add1  --data_name CIFAR10 --trail 3  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17_add1  --data_name CIFAR10 --trail 4  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17_add1  --data_name CIFAR10 --trail 5  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add2 --data_name CIFAR10 --trail 1 --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add2  --data_name CIFAR10 --trail 2  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add2  --data_name CIFAR10 --trail 3  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add2  --data_name CIFAR10 --trail 4  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add2  --data_name CIFAR10 --trail 5  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add3 --data_name CIFAR10 --trail 1 --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add3  --data_name CIFAR10 --trail 2  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add3  --data_name CIFAR10 --trail 3  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add3  --data_name CIFAR10 --trail 4  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add3  --data_name CIFAR10 --trail 5  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &


