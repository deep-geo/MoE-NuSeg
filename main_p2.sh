export PYTHON_PATH='./train_step2.py'
export MODEL_TYPE='MoE_p2'
export ALPHA='0.3'
export BETA='0.35'
export GAMMA='0.35'
#export SHARING_RATIO='0.5'
export DATASET='Histology'  # "Radiology", "Histology", "Thyroid", "Lizard"
export NUM_EPOCH=300
export LR=0.001
export RANDOM_SEED=42
export BATCH_SIZE=2

python train_step2.py --model_type=$MODEL_TYPE --alpha=$ALPHA --beta=$BETA --gamma=$GAMMA --dataset=$DATASET --lr=$LR --num_epoch=$NUM_EPOCH --random_seed=$RANDOM_SEED --batch_size=$BATCH_SIZE

