CONFIG=
CHECKPOINT=
GPUS=
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:}
PORT=${PORT:}
MASTER_ADDR=${MASTER_ADDR:}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
