expert='E1'
OMP_NUM_THREADS=1 torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint="gpu11:1234" \
    --nnodes=1 \
    --nproc-per-node=4 \
    --rdzv_id=1 \
    test_gpu.py --Using_deep \
    --batch_size 256 \
    --model Unet_drop_fs4 \
    --epochs 200 \
    --blr 1e-4  \
    --weight_decay 1e-8 \
    --dist_eval \
    --lr_policy plateau \
    --expert $expert
