: "
CUDA_VISIBLE_DEVICES=0,1,2 python -W ignore train_imgnet.py \
    -a resnet50 \
    --lr 0.1 \
    --lr-decay-epoch 30 \
    -b 512 \
    --epochs 90 \
    --n-classes 1000 \
    --workers 16 \
    --checkpoint-path 'checkpoints/imagenet/resnet50_gem_1' \
    --mtype 'gem' \
    --gem-memsize 1 \
    '/home/yluo/project/dataset/imagenet/images/'
CUDA_VISIBLE_DEVICES=0,1,2 python -W ignore train_imgnet.py \
    -a resnet50 \
    --lr 0.1 \
    --lr-decay-epoch 30 \
    -b 512 \
    --epochs 90 \
    --n-classes 1000 \
    --workers 16 \
    --checkpoint-path 'checkpoints/imagenet/resnet50_baseline' \
    --mtype 'baseline' \
    '/home/yluo/project/dataset/imagenet/images/'
"
CUDA_VISIBLE_DEVICES=0,1,2 python -W ignore train_imgnet.py \
    -a resnet50 \
    --lr 0.1 \
    --lr-decay-epoch 30 \
    -b 512 \
    --epochs 90 \
    --n-classes 1000 \
    --workers 16 \
    --checkpoint-path 'checkpoints/imagenet/resnet50_dcl_1_1' \
    --mtype 'dcl' \
    --dcl-refsize 1 \
    --dcl-window 1 \
    '/home/yluo/project/dataset/imagenet/images/'