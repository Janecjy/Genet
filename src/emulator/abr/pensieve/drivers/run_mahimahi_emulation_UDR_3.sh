#!/bin/bash
set -e

GENET_BASE_PATH=$1
VIDEO_SIZE_DIR=pensieve/data/video_sizes
ACTOR_PATH=$2
UP_LINK_SPEED_FILE=${GENET_BASE_PATH}/src/emulator/abr/pensieve/data/12mbps
TRACE_DIR=$3
SUMMARY_DIR_NAME=$4
PORT_ID=$5
AGENT_ID=$6
ADAPTOR_INPUT=$7
ADAPTOR_HIDDEN_SIZE=$8
CONTEXT_WINDOW=${9:-1}  # New parameter with default value 1
EXTRA_ARG=${10}
ORIGINAL_MODEL_PATH=${GENET_BASE_PATH}/fig_reproduce/data/all_models/udr_3/nn_model_ep_58000.ckpt
CONFIG_FILE=${GENET_BASE_PATH}/config/abr/udr3_emu_par.json

trap "pkill -f abr_server" SIGINT
trap "pkill -f abr_server" EXIT

delay=40
up_pkt_loss=0
down_pkt_loss=0
buf_th=60
trace_files=`ls ${TRACE_DIR}`

echo -e "model_checkpoint_path: $2\n all_model_checkpoint_paths: $2" > ${GENET_BASE_PATH}/fig_reproduce/model/checkpoint

for trace_file in ${trace_files} ; do
    mm-delay ${delay} mm-loss uplink ${up_pkt_loss} mm-loss downlink ${down_pkt_loss} \
    mm-link ${UP_LINK_SPEED_FILE} ${TRACE_DIR}${trace_file} -- \
    bash -c "cd ${GENET_BASE_PATH}/src/emulator/abr && python -m pensieve.virtual_browser.virtual_browser --ip \${MAHIMAHI_BASE} --port ${PORT_ID} --abr RL \
    --video-size-file-dir ${VIDEO_SIZE_DIR} --summary-dir /mydata/results/${SUMMARY_DIR_NAME}/UDR-3_${AGENT_ID}_${buf_th}_${delay} \
    --trace-file ${trace_file} --actor-path ${ACTOR_PATH} --abr-server-port=8322 --num-epochs=1 --run_time=0 \
    --original-model-path ${ORIGINAL_MODEL_PATH} --adaptor-input ${ADAPTOR_INPUT} --adaptor-hidden-size ${ADAPTOR_HIDDEN_SIZE} --context-window ${CONTEXT_WINDOW} ${EXTRA_ARG}"
done
