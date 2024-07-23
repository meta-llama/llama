#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3.1 Community License Agreement.

set -e

read -p "Enter the URL from email: " PRESIGNED_URL
ALL_MODELS_LIST="meta-llama-3.1-405b,meta-llama-3.1-70b,meta-llama-3.1-8b,meta-llama-guard-3-8b,prompt-guard"
printf "\n **** Model list ***\n"
for MODEL in ${ALL_MODELS_LIST//,/ }
do
    printf " -  ${MODEL}\n"
done
read -p "Choose the model to download: " SELECTED_MODEL
printf "\n Selected model: ${SELECTED_MODEL} \n"

SELECTED_MODELS=""
if [[ $SELECTED_MODEL == "meta-llama-3.1-405b" ]]; then
    MODEL_LIST="meta-llama-3.1-405b-instruct-mp16,meta-llama-3.1-405b-instruct-mp8,meta-llama-3.1-405b-instruct-fb8,meta-llama-3.1-405b-mp16,meta-llama-3.1-405b-mp8,meta-llama-3.1-405b-fp8"
elif [[ $SELECTED_MODEL == "meta-llama-3.1-70b" ]]; then
    MODEL_LIST="meta-llama-3.1-70b-instruct,meta-llama-3.1-70b"
elif [[ $SELECTED_MODEL == "meta-llama-3.1-8b" ]]; then
    MODEL_LIST="meta-llama-3.1-8b-instruct,meta-llama-3.1-8b"
elif [[ $SELECTED_MODEL == "meta-llama-guard-3-8b" ]]; then
    MODEL_LIST="meta-llama-guard-3-8b-int8-hf,meta-llama-guard-3-8b"
elif [[ $SELECTED_MODEL == "prompt-guard" ]]; then
    SELECTED_MODELS="prompt-guard"
    MODEL_LIST=""
fi

if [[ -z "$SELECTED_MODELS" ]]; then
    printf "\n **** Available models to download: ***\n"
    for MODEL in ${MODEL_LIST//,/ }
    do
        printf " -  ${MODEL}\n"
    done
    read -p "Enter the list of models to download without spaces or press Enter for all: " SELECTED_MODELS
fi

TARGET_FOLDER="."             # where all files should end up
mkdir -p ${TARGET_FOLDER}

if [[ $SELECTED_MODELS == "" ]]; then
    SELECTED_MODELS=${MODEL_LIST}
fi

if [[ $SELECTED_MODEL == "meta-llama-3.1-405b" ]]; then
    printf "\nModel requires significant storage and computational resources, occupying approximately 750GB of disk storage space and necessitating two nodes on MP16 for inferencing.\n"
    read -p "Enter Y to continue: " ACK
    if [[ $ACK != 'Y' ]]; then
        printf "Exiting..."
        exit 1
    fi
fi

printf "Downloading LICENSE and Acceptable Usage Policy\n"
wget --continue ${PRESIGNED_URL/'*'/"LICENSE"} -O ${TARGET_FOLDER}"/LICENSE"
wget --continue ${PRESIGNED_URL/'*'/"USE_POLICY.md"} -O ${TARGET_FOLDER}"/USE_POLICY.md"

for m in ${SELECTED_MODELS//,/ }
do

    ADDITIONAL_FILES=""
    TOKENIZER_MODEL=1
    if [[ $m == "meta-llama-3.1-405b-instruct-mp16" ]]; then
        PTH_FILE_COUNT=15
        MODEL_PATH="Meta-Llama-3.1-405B-Instruct-MP16"
    elif [[ $m == "meta-llama-3.1-405b-instruct-mp8" ]]; then
        PTH_FILE_COUNT=7
        MODEL_PATH="Meta-Llama-3.1-405B-Instruct-MP8"
    elif [[ $m == "meta-llama-3.1-405b-instruct-fp8" ]]; then
        PTH_FILE_COUNT=7
        MODEL_PATH="Meta-Llama-3.1-405B-Instruct"
        ADDITIONAL_FILES="fp8_scales_0.pt,fp8_scales_1.pt,fp8_scales_2.pt,fp8_scales_3.pt,fp8_scales_4.pt,fp8_scales_5.pt,fp8_scales_6.pt,fp8_scales_7.pt"
    elif [[ $m == "meta-llama-3.1-405b-mp16" ]]; then
        PTH_FILE_COUNT=15
        MODEL_PATH="Meta-Llama-3.1-405B-MP16"
    elif [[ $m == "meta-llama-3.1-405b-mp8" ]]; then
        PTH_FILE_COUNT=7
        MODEL_PATH="Meta-Llama-3.1-405B-MP8"
    elif [[ $m == "meta-llama-3.1-405b-fp8" ]]; then
        PTH_FILE_COUNT=7
        MODEL_PATH="Meta-Llama-3.1-405B"
    elif [[ $m == "meta-llama-3.1-70b-instruct" ]]; then
        PTH_FILE_COUNT=7
        MODEL_PATH="Meta-Llama-3.1-70B-Instruct"
    elif [[ $m == "meta-llama-3.1-70b" ]]; then
        PTH_FILE_COUNT=7
        MODEL_PATH="Meta-Llama-3.1-70B"
    elif [[ $m == "meta-llama-3.1-8b-instruct" ]]; then
        PTH_FILE_COUNT=0
        MODEL_PATH="Meta-Llama-3.1-8B-Instruct"
    elif [[ $m == "meta-llama-3.1-8b" ]]; then
        PTH_FILE_COUNT=0
        MODEL_PATH="Meta-Llama-3.1-8B"
    elif [[ $m == "meta-llama-guard-3-8b-int8-hf" ]]; then
        PTH_FILE_COUNT=-1
        MODEL_PATH="Meta-Llama-Guard-3-8B-INT8-HF"
        ADDITIONAL_FILES="generation_config.json,model-00001-of-00002.safetensors,model-00002-of-00002.safetensors,model.safetensors.index.json,special_tokens_map.json,tokenizer_config.json,tokenizer.json"
        TOKENIZER_MODEL=0
    elif [[ $m == "meta-llama-guard-3-8b" ]]; then
        PTH_FILE_COUNT=0
        MODEL_PATH="Meta-Llama-Guard-3-8B"
    elif [[ $m == "prompt-guard" ]]; then
        PTH_FILE_COUNT=-1
        MODEL_PATH="Prompt-Guard"
        ADDITIONAL_FILES="model.safetensors,special_tokens_map.json,tokenizer_config.json,tokenizer.json"
        TOKENIZER_MODEL=0
    fi

    printf "\n***Downloading ${MODEL_PATH}***\n"
    mkdir -p ${TARGET_FOLDER}"/${MODEL_PATH}"

    if [[ $TOKENIZER_MODEL == 1 ]]; then
        printf "Downloading tokenizer\n"
        wget --continue ${PRESIGNED_URL/'*'/"${MODEL_PATH}/tokenizer.model"} -O ${TARGET_FOLDER}"/${MODEL_PATH}/tokenizer.model"
    fi


    if [[ $PTH_FILE_COUNT -ge 0 ]]; then
        for s in $(seq -f "0%g" 0 ${PTH_FILE_COUNT})
        do
            printf "Downloading consolidated.${s}.pth\n"
            wget --continue ${PRESIGNED_URL/'*'/"${MODEL_PATH}/consolidated.${s}.pth"} -O ${TARGET_FOLDER}"/${MODEL_PATH}/consolidated.${s}.pth"
        done
    fi

    for ADDITIONAL_FILE in ${ADDITIONAL_FILES//,/ }
    do
        printf "Downloading $ADDITIONAL_FILE...\n"
        wget --continue ${PRESIGNED_URL/'*'/"${MODEL_PATH}/${ADDITIONAL_FILE}"} -O ${TARGET_FOLDER}"/${MODEL_PATH}/${ADDITIONAL_FILE}"
    done

    if [[ $m != "prompt-guard" &&  $m != "meta-llama-guard-3-8b-int8-hf" ]]; then
        printf "Downloading params.json...\n"
        wget --continue ${PRESIGNED_URL/'*'/"${MODEL_PATH}/params.json"} -O ${TARGET_FOLDER}"/${MODEL_PATH}/params.json"
    fi
done
