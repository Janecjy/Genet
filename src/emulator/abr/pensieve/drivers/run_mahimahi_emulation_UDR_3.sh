#!/bin/bash
set -e

VIDEO_SIZE_DIR=pensieve/data/video_sizes
ACTOR_PATH=$1
UP_LINK_SPEED_FILE=/users/janechen/Genet/src/emulator/abr/pensieve/data/12mbps
TRACE_DIR=$2
SUMMARY_DIR_NAME=$3
PORT_ID=$4
AGENT_ID=$5
ADAPTOR_INPUT=$6
ADAPTOR_HIDDEN_SIZE=$7
EXTRA_ARG=$8
ORIGINAL_MODEL_PATH=/users/janechen/Genet/fig_reproduce/data/all_models/udr_3/nn_model_ep_58000.ckpt
CONFIG_FILE=/users/janechen/Genet/config/abr/udr3_emu_par.json

trap "pkill -f abr_server" SIGINT
trap "pkill -f abr_server" EXIT

delay=40
up_pkt_loss=0
down_pkt_loss=0
buf_th=60
trace_files=`ls ${TRACE_DIR}`

echo -e "model_checkpoint_path: $1\n all_model_checkpoint_paths: $1" > ~/Genet/fig_reproduce/model/checkpoint

for trace_file in ${trace_files} ; do
    mm-delay ${delay} mm-loss uplink ${up_pkt_loss} mm-loss downlink ${down_pkt_loss} \
    mm-link ${UP_LINK_SPEED_FILE} ${TRACE_DIR}${trace_file} -- \
    bash -c "python -m pensieve.virtual_browser.virtual_browser --ip \${MAHIMAHI_BASE} --port ${PORT_ID} --abr RL \
    --video-size-file-dir ${VIDEO_SIZE_DIR} --summary-dir /mydata/results/${SUMMARY_DIR_NAME}/UDR-3_${AGENT_ID}_${buf_th}_${delay} \
    --trace-file ${trace_file} --actor-path ${ACTOR_PATH} --abr-server-port=8322 --num-epochs=1 --run_time=0 \
    --original-model-path ${ORIGINAL_MODEL_PATH} --adaptor-input ${ADAPTOR_INPUT} --adaptor-hidden-size ${ADAPTOR_HIDDEN_SIZE} ${EXTRA_ARG}"
done
