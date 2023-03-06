# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

#!/bin/bash

set -e

PRESIGNED_URL=""             # replace with presigned url from email
MODEL_SIZE="7B,13B,30B,65B"  # edit this list with the model sizes you wish to download
TARGET_FOLDER=""             # where all files should end up

declare -A N_SHARD_DICT

N_SHARD_DICT["7B"]="0"
N_SHARD_DICT["13B"]="1"
N_SHARD_DICT["30B"]="3"
N_SHARD_DICT["65B"]="7"

echo "Downloading tokenizer"
curl -f ${PRESIGNED_URL/'*'/"tokenizer.model"} -o ${TARGET_FOLDER}"/tokenizer.model"
curl -f ${PRESIGNED_URL/'*'/"tokenizer_checklist.chk"} -o ${TARGET_FOLDER}"/tokenizer_checklist.chk"

(cd ${TARGET_FOLDER} && md5sum -c tokenizer_checklist.chk)

for i in ${MODEL_SIZE//,/ }
do
    echo "Downloading ${i}"
    mkdir -p ${TARGET_FOLDER}"/${i}"
    for s in $(seq -f "0%g" 0 ${N_SHARD_DICT[$i]})
    do
        curl -f ${PRESIGNED_URL/'*'/"${i}/consolidated.${s}.pth"} -o ${TARGET_FOLDER}"/${i}/consolidated.${s}.pth"
    done
    curl -f ${PRESIGNED_URL/'*'/"${i}/params.json"} -o ${TARGET_FOLDER}"/${i}/params.json"
    curl -f ${PRESIGNED_URL/'*'/"${i}/checklist.chk"} -o ${TARGET_FOLDER}"/${i}/checklist.chk"
    echo "Checking checksums"
    (cd ${TARGET_FOLDER}"/${i}" && md5sum -c checklist.chk)
done
