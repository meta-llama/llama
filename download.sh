# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

PRESIGNED_URL="https://agi.gpt4.org/llama/LLaMA/*" # edit this with the presigned url
MODEL_SIZE="7B,13B,30B,65B"             # edit this list with the model sizes you wish to download
TARGET_FOLDER="/data"             # where all files should end up 

declare -A N_SHARD_DICT

N_SHARD_DICT["7B"]="0"
N_SHARD_DICT["13B"]="1"
N_SHARD_DICT["30B"]="3"
N_SHARD_DICT["65B"]="7"

echo "Downloading tokenizer"
if cd ${TARGET_FOLDER} && [[ ! -f tokenizer.model ]] && [[ ! -f tokenizer_checklist.chk ]] && ! md5sum -c tokenizer_checklist.chk; then
    wget ${PRESIGNED_URL/'*'/"tokenizer.model"} -O ${TARGET_FOLDER}"/tokenizer.model"
    wget ${PRESIGNED_URL/'*'/"tokenizer_checklist.chk"} -O ${TARGET_FOLDER}"/tokenizer_checklist.chk"
    (cd ${TARGET_FOLDER} && md5sum -c tokenizer_checklist.chk)
else
    echo "Skipping downloading tokenizer, already exists and checksum matches"
fi

for i in ${MODEL_SIZE//,/ }
do

    echo "Downloading ${i}"
    mkdir -p ${TARGET_FOLDER}"/${i}"

    file_name="${TARGET_FOLDER}/${i}/checklist.chk"
    echo "Downloading ${file_name}"
    if ! [[ -f "${file_name}" ]]; then
        wget ${PRESIGNED_URL/'*'/"${i}/checklist.chk"} -O ${TARGET_FOLDER}"/${i}/checklist.chk"
    else
        echo "Skipping downloading ${file_name}, already exists"
    fi
    for s in $(seq -f "0%g" 0 ${N_SHARD_DICT})
    do
        echo $s
        file_name="consolidated.${s}.pth"
        echo $file_name
        checklist_file="${TARGET_FOLDER}/${i}/checklist.chk"
        echo "${checklist_file##*/}"
        checksum=$(grep "${file_name##*/}" "${checklist_file}" | cut -d' ' -f1)

        if cd "${TARGET_FOLDER}/${i}" && ! [[ -f "${file_name}" ]] || ! [[ $(md5sum "${file_name}" | cut -d' ' -f1) == "${checksum}" ]]; then
            wget ${PRESIGNED_URL/'*'/"${i}/consolidated.${s}.pth"} -O ${TARGET_FOLDER}"/${i}/consolidated.${s}.pth"
        else
            echo "Skipping downloading ${file_name}, already exists and checksum matches"
        fi
    done
    file_name="params.json"
    if cd ${TARGET_FOLDER}/${i} && ! [[ -f "${file_name}" ]]; then
        wget ${PRESIGNED_URL/'*'/"${i}/params.json"} -O ${TARGET_FOLDER}"/${i}/params.json"
    else
        echo "Skipping downloading ${file_name}, already exists"
    fi
    (cd ${TARGET_FOLDER}"/${i}" && md5sum -c checklist.chk)
done