#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
GPUS=$3
PY_ARGS=${@:4}

GPUS_PER_NODE=${GPUS}
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
    -x SH-IDC1-10-140-1-173,SH-IDC1-10-140-1-74 \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u train.py --launcher slurm --tcp_port $PORT ${PY_ARGS}
