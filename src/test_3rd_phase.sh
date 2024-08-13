#!/bin/bash

python test.py \
  --working-dir '../' \
  --dataset-dir 'E:\Work\DatasetHandler\data' \
  --saved_fn 'ttnet_3rd_phase' \
  --gpu_idx 0 \
  --batch_size 1 \
  --pretrained_path ../checkpoints/ttnet_3rd_phase/ttnet_3rd_phase_epoch_30.pth \
  --seg_thresh 0.5 \
  --event_thresh 0.5 \
  --smooth-labelling \
  --save_test_output

if [ $? -eq 0 ]; then
  message="<!channel> Testing of 3rd phase is completed without any error."
else
  message="<!channel> Testing of 3rd phase failed."
fi

curl --request POST --header 'Content-type: application/json' --data "{\"text\":\"$message\"}" --location $SLACK_WEBHOOK_URL
