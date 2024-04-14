export PYTHON_PATH='./train_step1.py'
export MODEL_TYPE='MoE_p1'
export ALPHA='0.3'
export BETA='0.35'
export GAMMA='0.35'
export DATASET='Lizard'  #'Lizard'  # "Radiology", "Histology", "Thyroid", "Lizard"
export NUM_EPOCH=1
export LR=0.001
export RANDOM_SEED=46
export BATCH_SIZE=2

/root/autodl-tmp/venvs/ray/bin/python3.10 train_step1.py --model_type=$MODEL_TYPE --alpha=$ALPHA --beta=$BETA --gamma=$GAMMA --dataset=$DATASET --lr=$LR --num_epoch=$NUM_EPOCH --random_seed=$RANDOM_SEED --batch_size=$BATCH_SIZE
