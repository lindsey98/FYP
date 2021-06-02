#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17_add1 --data_name CIFAR10 --trail 11 --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17_add1  --data_name CIFAR10 --trail 12  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17_add1  --data_name CIFAR10 --trail 13  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17_add1  --data_name CIFAR10 --trail 14  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=0 python -m ResidualLoss.train_test --model_name CIFAR17_add1  --data_name CIFAR10 --trail 15  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add2 --data_name CIFAR10 --trail 11 --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add2  --data_name CIFAR10 --trail 12  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add2  --data_name CIFAR10 --trail 13  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add2  --data_name CIFAR10 --trail 14  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add2  --data_name CIFAR10 --trail 15  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add3 --data_name CIFAR10 --trail 11 --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add3  --data_name CIFAR10 --trail 12  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add3  --data_name CIFAR10 --trail 13  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add3  --data_name CIFAR10 --trail 14  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &
CUDA_VISIBLE_DEVICES=1 python -m ResidualLoss.train_test --model_name CIFAR17_add3  --data_name CIFAR10 --trail 15  --retrain True --weights checkpoints/CIFAR17-CIFAR10-model1//999.pt &


