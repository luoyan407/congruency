: "
CUDA_VISIBLE_DEVICES=0,1,2 python train_cifar.py \
	-a resnext \
	--train-batch 128 \
	--dataset 'cifar100' \
	--depth 29 \
	--cardinality 16 \
	--widen-factor 4 \
	--schedule 150 225 \
	--wd 5e-4 \
	--gamma 0.1 \
	--datapath '/home/yluo/project/dataset/cifar' \
	--checkpoint 'checkpoints/cifar100/resnext_gem_1' \
	--mtype 'gem' \
	--gem_memsize 1
CUDA_VISIBLE_DEVICES=0,1,2 python train_cifar.py \
	-a resnext \
	--train-batch 128 \
	--dataset 'cifar100' \
	--depth 29 \
	--cardinality 16 \
	--widen-factor 4 \
	--schedule 150 225 \
	--wd 5e-4 \
	--gamma 0.1 \
	--datapath '/home/yluo/project/dataset/cifar' \
	--checkpoint 'checkpoints/cifar100/resnext_baseline' \
	--mtype 'baseline'
"
CUDA_VISIBLE_DEVICES=0,1,2 python train_cifar.py \
	-a resnext \
	--train-batch 128 \
	--dataset 'cifar100' \
	--depth 29 \
	--cardinality 16 \
	--widen-factor 4 \
	--schedule 150 225 \
	--wd 5e-4 \
	--gamma 0.1 \
	--datapath '/home/yluo/project/dataset/cifar' \
	--checkpoint 'checkpoints/cifar100/resnext_dcl_inf_1' \
	--mtype 'dcl' \
	--dcl_refsize 1 \
	--dcl_window 0 \