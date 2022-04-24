#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT_PATH=$2
#CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}
for i in $(find ${CHECKPOINT_PATH} -name '*.pth');
#for file in `ls ${CHECKPOINT_PATH}`
do
    echo "test with ${i}"
    PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        $(dirname "$0")/test.py $CONFIG $i --launcher pytorch ${@:4}

done