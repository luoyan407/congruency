#!/bin/bash
DATASET='cifar10'
export CUDA_VISIBLE_DEVICES=0,1,2,3
#-----------Baseline model: EfficientNet--------------
python train_cifar.py \
	-a 'efficientnet-b1' \
	--train-batch 256 \
	--dataset ${DATASET} \
	--schedule 100 200 \
	--lr 0.05 \
	--wd 1e-5 \
	--gamma 0.1 \
	--datapath '/home/yluo/project/dataset/cifar' \
	--checkpoint "checkpoints/${DATASET}/efficientnet-b1-dcl-5-1" \
	--mtype 'dcl' \
	--dcl_refsize 1 \
	--dcl_window 5 \
	--dcl_QP_margin 0.1
python train_cifar.py \
	-a 'efficientnet-b1' \
	--train-batch 256 \
	--dataset ${DATASET} \
	--schedule 100 200 \
	--lr 0.05 \
	--wd 1e-5 \
	--gamma 0.1 \
	--datapath '/home/yluo/project/dataset/cifar' \
	--checkpoint "checkpoints/${DATASET}/efficientnet-b1-gem-1" \
	--mtype 'gem' \
	--gem_memsize 1 
python train_cifar.py \
	-a 'efficientnet-b1' \
	--train-batch 256 \
	--dataset ${DATASET} \
	--schedule 100 200 \
	--lr 0.05 \
	--wd 1e-5 \
	--gamma 0.1 \
	--datapath '/home/yluo/project/dataset/cifar' \
	--checkpoint "checkpoints/${DATASET}/efficientnet-b1-baseline" \
	--mtype 'baseline'
#-----------Baseline model: ResNet--------------
python train_cifar.py \
	-a resnext \
	--train-batch 128 \
	--dataset ${DATASET} \
	--depth 29 \
	--cardinality 16 \
	--widen-factor 4 \
	--schedule 150 225 \
	--wd 5e-4 \
	--gamma 0.1 \
	--datapath '/home/yluo/project/dataset/cifar' \
	--checkpoint "checkpoints/${DATASET}/resnext_dcl_inf_1" \
	--mtype 'dcl' \
	--dcl_refsize 1 \
	--dcl_window 0
python train_cifar.py \
	-a resnext \
	--train-batch 128 \
	--dataset ${DATASET} \
	--depth 29 \
	--cardinality 16 \
	--widen-factor 4 \
	--schedule 150 225 \
	--wd 5e-4 \
	--gamma 0.1 \
	--datapath '/home/yluo/project/dataset/cifar' \
	--checkpoint "checkpoints/${DATASET}/resnext_gem_1" \
	--mtype 'gem' \
	--gem_memsize 1
python train_cifar.py \
	-a resnext \
	--train-batch 128 \
	--dataset ${DATASET} \
	--depth 29 \
	--cardinality 16 \
	--widen-factor 4 \
	--schedule 150 225 \
	--wd 5e-4 \
	--gamma 0.1 \
	--datapath '/home/yluo/project/dataset/cifar' \
	--checkpoint "checkpoints/${DATASET}/resnext_baseline" \
	--mtype 'baseline'