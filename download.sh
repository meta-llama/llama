# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

PRESIGNED_URL="${1:-${PRESIGNED_URL:-}}"        # get it from email and hardcode it here, pass as env PRESIGNED_URL or as parameter
MODEL_SIZE="${MODEL_SIZE:-7B,13B,30B,65B}"      # edit this list with the model sizes you wish to download
TARGET_FOLDER="${TARGET_FOLDER:-model-weights}" # read an env with TARGET_FOLDER where all files should end up or hard code it here

declare -A N_SHARD_DICT

N_SHARD_DICT["7B"]="0"
N_SHARD_DICT["13B"]="1"
N_SHARD_DICT["30B"]="3"
N_SHARD_DICT["65B"]="7"

echo "Downloading tokenizer"
mkdir -p "${TARGET_FOLDER}"
CHK_FILE="tokenizer_checklist.chk"
[ -f "${TARGET_FOLDER}/${CHK_FILE}" ] && MD5SUM_RESULT=$(cd "${TARGET_FOLDER}" && md5sum -c "${CHK_FILE}")
if [ $? -ne 0 ]; then
    wget ${PRESIGNED_URL/'*'/"${CHK_FILE}"} -O "${TARGET_FOLDER}/${CHK_FILE}"
    wget ${PRESIGNED_URL/'*'/"tokenizer.model"} -O ${TARGET_FOLDER}"/tokenizer.model"
    (cd "${TARGET_FOLDER}" && md5sum -c "${CHK_FILE}")
    [ $? -ne 0 ] && exit 1
fi
echo $MD5SUM_RESULT

CHK_FILE="checklist.chk"
for i in ${MODEL_SIZE//,/ }; do
    echo "Downloading ${i}"
    mkdir -p "${TARGET_FOLDER}/${i}"    
    echo "If you tried to download it before, please, wait while we check the integrity of already downloaded weights..."
    [ -f "${TARGET_FOLDER}/${i}/${CHK_FILE}" ] && MD5SUM_RESULT=$(cd "${TARGET_FOLDER}/${i}" && md5sum -c "${CHK_FILE}")
    if [ $? -ne 0 ]; then
        wget ${PRESIGNED_URL/'*'/"${i}/${CHK_FILE}"} -O ${TARGET_FOLDER}"/${i}/${CHK_FILE}"
        wget ${PRESIGNED_URL/'*'/"${i}/params.json"} -O ${TARGET_FOLDER}"/${i}/params.json"
        for s in $(seq -f "0%g" 0 ${N_SHARD_DICT[$i]}); do
            DOWNLOADED=$(echo $MD5SUM_RESULT | grep -e consolidated.${s}.pth -e OK)
            [ -z $DOWNLOADED ] && wget ${PRESIGNED_URL/'*'/"${i}/consolidated.${s}.pth"} -O ${TARGET_FOLDER}"/${i}/consolidated.${s}.pth"
        done
        echo "Checking checksums"
        (cd "${TARGET_FOLDER}/${i}" && md5sum -c "${CHK_FILE}")
    else
        echo -e $MD5SUM_RESULT
    fi
done
