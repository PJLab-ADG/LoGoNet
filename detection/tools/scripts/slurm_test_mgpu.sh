#!/usr/bin/env bash

set -x

PARTITION=$1
GPUS=$2
GPUS_PER_NODE=$GPUS

PY_ARGS=${@:3}
JOB_NAME=eval
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
SRUN_ARGS=${SRUN_ARGS:-""}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    -x SH-IDC1-10-140-1-123 \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u test.py --launcher slurm --tcp_port $PORT ${PY_ARGS}
