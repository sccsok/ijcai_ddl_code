base_dir="/data1/home/shuaichao/ijcai_ddl_code/Mesorch/output_dir_test"
mkdir -p ${base_dir}

CUDA_VISIBLE_DEVICES=6,7 \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=2 \
./test.py \
    --model Mesorch \
    --world_size 12 \
    --test_data_json "./ijcai_ddl.json" \
    --checkpoint_path "./ckpts/" \
    --test_batch_size 64 \
    --image_size 512 \
    --if_resizing \
    --output_dir ${base_dir}/ \
    --log_dir ${base_dir}/ \
    --save_dir "/data1/home/shuaichao/ijcai_ddl_code/Mesorch/ijcai_result" \
    --threshold 0.0001 \
2> ${base_dir}/error.log 1>${base_dir}/logs.log