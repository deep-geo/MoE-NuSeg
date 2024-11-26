export PYTHON_PATH='./train_swin_unet.py'
export MODEL_TYPE='Swin_UNet'
export ALPHA='0.3'
export BETA='0.35'
export GAMMA='0.35'
export SHARING_RATIO='0.5'
export DATASET='Lizard'  # 'Lizard'   "Radiology", "Histology", "Thyroid"
export NUM_EPOCH=500
export LR=0.001
export RANDOM_SEED=44
export BATCH_SIZE=2

python $PYTHON_PATH --model_type=$MODEL_TYPE --alpha=$ALPHA --beta=$BETA --gamma=$GAMMA --sharing_ratio=$SHARING_RATIO --dataset=$DATASET --lr=$LR --num_epoch=$NUM_EPOCH --random_seed=$RANDOM_SEED --batch_size=$BATCH_SIZE
