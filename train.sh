python train.py --batch 8 --experiment_name Color2Embed_noPreTrain_autoCaption --n_aug 1 --auto_caption --ckpt experiments/Color2Embed_noPreTrain_autoCaption/218750.pt

#python train.py --batch 4 --experiment_name Color2Embed_autoCaption_small_lr --n_aug 4 --load_only_colorUNet --ckpt experiments/240000.pt --auto_caption

#python train.py --batch 1 --experiment_name Color2Embed_autoCaption_small_lr --n_aug 4 --auto_caption


## you can reload the model for continuing training.
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 train.py --batch 8 --experiment_name Color2Embed_1 --ckpt experiments/Color2Embed_1/005000.pt --datasets ./train_datasets/ImageNet_train_lmdb
