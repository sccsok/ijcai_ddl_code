# -*- coding: utf-8 -*-
# @Date     : 2025/03/05
# @Author   : Chao Shuai
# @File     : test_ngpus.py

# To train the detector: dual_stream
CUDA_VISIBLE_DEVICES='6,7' python -m torch.distributed.launch --nproc_per_node 2 --nnode 1 \
    --master_addr=127.0.0.1 --master_port 3362 \
    train.py \
   --detector_path config/detector/dual_stream.yaml

# To train the detector: mesorch
# The code has some problems, so we don't use it.
# CUDA_VISIBLE_DEVICES='6,7' python -m torch.distributed.launch --nproc_per_node 2 --nnode 1 \
#     --master_addr=127.0.0.1 --master_port 3362 \
#     train.py \
#    --detector_path /data1/home/shuaichao/ijcai_ddl_code/model_training/training/config/detector/mesorch.yaml

