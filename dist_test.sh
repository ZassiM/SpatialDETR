#!/usr/bin/env bash
ARGS="args.toml"
GPUS=$1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

#modify args for pytorch launcher
sed -i 's/launcher = "none"/launcher = "pytorch"/gI' $ARGS

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py

#reset the pytorch launcher arg
sed -i 's/launcher = "pytorch"/launcher = "none"/gI' $ARGS
