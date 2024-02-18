CONFIG=$1
GPUS=$2
NNODES=${NNODES:}
NODE_RANK=${NODE_RANK:}
PORT=${PORT: }
MASTER_ADDR=${MASTER_ADDR:}
CUDA_VISIBLE_DEVICES= \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}





