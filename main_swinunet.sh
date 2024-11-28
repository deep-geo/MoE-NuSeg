export PYTHON_PATH='./train_swin_unet.py'
export MODEL_TYPE='Swin_UNet'
export ALPHA='0.3'
export BETA='0.35'
export GAMMA='0.35'
export DATASET='Histology'  # 'Lizard'   "Radiology", "Histology", "Thyroid"
export NUM_EPOCH=300
export LR=0.001
export RANDOM_SEED=42
export BATCH_SIZE=2

python $PYTHON_PATH --model_type=$MODEL_TYPE --alpha=$ALPHA --beta=$BETA --gamma=$GAMMA --dataset=$DATASET --lr=$LR --num_epoch=$NUM_EPOCH --random_seed=$RANDOM_SEED --batch_size=$BATCH_SIZE
