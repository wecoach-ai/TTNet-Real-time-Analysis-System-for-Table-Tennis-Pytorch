#!/bin/bash

python main.py \
  --dataset-dir 'E:\Work\DatasetHandler\data' \
  --working-dir 'E:\Temporary' \
  --saved_fn 'ttnet_3rd_phase' \
  --no-val \
  --batch_size 8 \
  --num_workers 4 \
  --lr 0.0001 \
  --lr_type 'step_lr' \
  --lr_step_size 10 \
  --lr_factor 0.2 \
  --gpu_idx 0 \
  --global_weight 1. \
  --seg_weight 1. \
  --event_weight 1. \
  --local_weight 1. \
  --pretrained_path ../checkpoints/ttnet_2nd_phase/ttnet_2nd_phase_epoch_30.pth \
  --smooth-labelling

if [ $? -eq 0 ]; then
  message="<!channel> Third Phase of training is completed without any error."
else
  message="<!channel> Third Phase of training failed."
fi

curl --request POST --header 'Content-type: application/json' --data '{"text":"$message"}' --location $SLACK_WEBHOOK_URL