#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python ufc_generation/ufc_cifar.py \
    --iteration 1000 \
    --r-bn 1 \
    --batch-size 100 \
    --lr 0.25 \
    --exp-name generated_results \
    --wandb-name cifar100-ipc10 \
    --store-best-images \
    --syn-data-path syn/ \
    --init_path init_images/c100/ \
    --ipc 10 \
    --dataset cifar100

# validation with static labeling
python ufc_validation/val_static.py \
    --epochs 400 --batch-size 64 --ipc 10 \
    --syn-data-path syn/cifar100-ipc10/generated_results \
    --output-dir syn/cifar100-ipc10/generated_results \
    --wandb-name cifar100-ipc10 \
    --dataset cifar100 --networks resnet18

# validation with dynamic labeling
python ufc_validation/val_dyn.py \
    --epochs 80 --batch-size 64 --ipc 10 \
    --syn-data-path syn/cifar100-ipc10/generated_results\
    --output-dir syn/cifar100-ipc10 \
    --wandb-name cifar100-ipc10 \
    --dataset cifar100 --networks resnet18

python ufc_generation/ufc_cifar.py \
    --iteration 1000 \
    --r-bn 1 \
    --batch-size 100 \
    --lr 0.25 \
    --exp-name generated_results \
    --wandb-name cifar100-ipc50 \
    --store-best-images \
    --syn-data-path syn/ \
    --init_path init_images/c100/ \
    --ipc 50 \
    --dataset cifar100

# validation with static labeling
python ufc_validation/val_static.py \
    --epochs 400 --batch-size 64 --ipc 50 \
    --syn-data-path syn/cifar100-ipc50/generated_results\
    --output-dir syn/cifar100-ipc50 \
    --wandb-name cifar100-ipc50 \
    --dataset cifar100 --networks resnet18

# validation with dynamic labeling
python ufc_validation/val_dyn.py \
    --epochs 80 --batch-size 64 --ipc 50 \
    --syn-data-path syn/cifar100-ipc50/generated_results\
    --output-dir syn/cifar100-ipc50 \
    --wandb-name cifar100-ipc50 \
    --dataset cifar100 --networks resnet18
