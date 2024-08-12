#!/bin/bash

python main.py \
  --dataset-dir '' \
  --working-dir '' \
  --saved_fn 'ttnet_2nd_phase' \
  --no-val \
  --batch_size 8 \
  --num_workers 4 \
  --lr 0.001 \
  --lr_type 'step_lr' \
  --lr_step_size 10 \
  --lr_factor 0.1 \
  --gpu_idx 0 \
  --global_weight 0. \
  --seg_weight 0. \
  --event_weight 2. \
  --local_weight 1. \
  --pretrained_path '' \
  --overwrite_global_2_local \
  --freeze_seg \
  --freeze_global \
  --smooth-labelling

if [ $? -eq 0 ]; then
  message="<!channel> Second Phase of training is completed without any error."
else
  message="<!channel> Second Phase of training failed."
fi

curl --request POST --header 'Content-type: application/json' --data "{\"text\":\"$message\"}" --location $SLACK_WEBHOOK_URL
