#!/bin/bash

# image_folder: A folder containing all faces extracted from test image s
# original_images_root: A folder containing all original test images
# location_json: A json file of all face locations

set -e

python test_lav.py \
  --weights_path Lav/training/checkpoints/two_stream_epoch_24.pth \
  --image_folder ./face/test \
  --original_images_root ./ddl_data/test \
  --location_json ./face/face_locations.json  

python test_mesorch.py \
  --weights_path Mesorch/ckpts/checkpoint-40.pth \
  --model Mesorch \
  --image_folder ./ddl_data/test

python merge.py \
  --mask_dir_1 ./results/lav_outputs \
  --mask_dir_2 ./results/mesorch_outputs \
  --output_dir ./results/submission
