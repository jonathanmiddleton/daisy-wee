export RUN_ID=1
torchrun --standalone --nproc_per_node=8 \
    train.py config/pretrain.yml
