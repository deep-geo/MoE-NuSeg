export PYTHON_PATH='./train_step1.py'
export MODEL_TYPE='MoE_p1'
export ALPHA='0.3'
export BETA='0.35'
export GAMMA='0.35'
export DATASET='Radiology'  #'Lizard'  # "Radiology", "Histology", "Thyroid"
export NUM_EPOCH=150
export LR=0.001
export RANDOM_SEED=42
export BATCH_SIZE=2

python train_step1.py --model_type=$MODEL_TYPE --alpha=$ALPHA --beta=$BETA --gamma=$GAMMA --dataset=$DATASET --lr=$LR --num_epoch=$NUM_EPOCH --random_seed=$RANDOM_SEED --batch_size=$BATCH_SIZE


