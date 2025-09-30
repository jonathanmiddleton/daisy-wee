export RUN_ID=1
export OMP_NUM_THREADS=8
torchrun --standalone --nproc_per_node=8 \
    train.py config/pretrain_350m.yml
