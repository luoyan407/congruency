: "
CUDA_VISIBLE_DEVICES=0,1,2 python train_timgnet.py \
	--pretrained_model resnet101 \
	--train-batch 64 \
	--epochs 30 \
	--schedule 10 20 \
	--gamma 0.1 \
	--lr 1e-03 \
	--n_classes 200 \
	--data '/home/yluo/project/dataset/tinyimagenet/images/' \
	--checkpoint 'checkpoints/tinyimagenet/ablation_study/resnet101-GEM-1' \
	--mtype 'gem' \
	--gem_memsize 1 
CUDA_VISIBLE_DEVICES=0,1,2 python train_timgnet.py \
	--pretrained_model resnet101 \
	--train-batch 64 \
	--epochs 30 \
	--schedule 10 20 \
	--gamma 0.1 \
	--lr 1e-03 \
	--n_classes 200 \
	--data '/home/yluo/project/dataset/tinyimagenet/images/' \
	--checkpoint 'checkpoints/tinyimagenet/ablation_study/resnet101-baseline' \
	--mtype 'baseline'
"
CUDA_VISIBLE_DEVICES=0,1,2 python train_timgnet.py \
	--pretrained_model resnet101 \
	--train-batch 64 \
	--epochs 30 \
	--schedule 10 20 \
	--gamma 0.1 \
	--lr 1e-03 \
	--n_classes 200 \
	--data '/home/yluo/project/dataset/tinyimagenet/images/' \
	--checkpoint 'checkpoints/tinyimagenet/ablation_study/resnet101-DCL-60-1' \
	--mtype 'dcl' \
	--dcl_refsize 1 \
	--dcl_window 50 