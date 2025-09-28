export RUN_ID=2
torchrun --standalone --nproc_per_node=8 \
  train.py config/instruct_sft.yml \
  --init-checkpoint=logs/000_xxxxxxxx/state_step005960.pt
