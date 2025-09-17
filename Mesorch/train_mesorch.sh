base_dir="./output_dir_mesorch"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=6,7 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=2 \
./train.py \
    --model Mesorch \
    --conv_pretrain True \
    --seg_pretrain_path ./pretrained/mit_b3.pth \
    --start_epoch 0 \
    --world_size 12 \
    --find_unused_parameters \
    --batch_size 32 \
    --data_path ./datasets/ijcai_metadata_mesorch_train.json \
    --epochs 150 \
    --lr 1e-4 \
    --image_size 512 \
    --if_resizing \
    --min_lr 5e-7 \
    --weight_decay 0.05 \
    --test_data_path ./datasets/ijcai_metadata_mesorch_valid.json \
    --warmup_epochs 2 \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --accum_iter 2 \
    --seed 42 \
    --test_period 2 \
    --num_workers 12 \
    --if_not_amp \
2> ${base_dir}/error.log 1>${base_dir}/logs.log