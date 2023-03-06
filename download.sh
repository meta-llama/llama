# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

PRESIGNED_URL="https://urldefense.com/v3/__https://dobf1k6cxlizq.cloudfront.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kb2JmMWs2Y3hsaXpxLmNsb3VkZnJvbnQubmV0LyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2NzgzNzE3NDV9fX1dfQ__&Signature=oPbHryWP2OHHN582Ek-6Iyti9x5CkYh*Zsdhnjhl0OCftgTa8A5mkHbCuHl1Oh82gfFT-sESWOXmDlkcAICbNiRJAEHfPc8WmJkweKeobM4c9vcRlez4CaFfk0Rr2EVhOfmx-D6fq0lz01E9rWKQ8KfYpgrWSx-5NDBsh8ceeR8S2lvobLb2ktYdb-cvvgEv6-wHiRSEhN34rx5nn2PU8FCa0ffZITvoT5o7bb5nDkvmXiU1y53esLr7bEY9DnmMq1kQkB9D6*H0XtDP6V7GKfUDttsTcBoMLoC-oluRldNyWA81Mdrakw5hPzJIKwOhbNS8he4q88NfRv0y17yASQ__&Key-Pair-Id=K231VYXPC1TA1R__;Kn5-!!GNU8KkXDZlD12Q!9rgXzak5bg08bePGoqViteDU9T3rSYetPC8Ae93BvNGxLXEWyQYK5FohmxauiaM97dsZ5fIQVWuFzWhhelFmfstrh9aBXb2EmM0$"             # replace with presigned url from email
MODEL_SIZE="7B,13B,30B,65B"  # edit this list with the model sizes you wish to download
TARGET_FOLDER=""             # where all files should end up

declare -A N_SHARD_DICT

N_SHARD_DICT["7B"]="0"
N_SHARD_DICT["13B"]="1"
N_SHARD_DICT["30B"]="3"
N_SHARD_DICT["65B"]="7"

echo "Downloading tokenizer"
wget ${PRESIGNED_URL/'*'/"tokenizer.model"} -O ${TARGET_FOLDER}"/tokenizer.model"
wget ${PRESIGNED_URL/'*'/"tokenizer_checklist.chk"} -O ${TARGET_FOLDER}"/tokenizer_checklist.chk"

(cd ${TARGET_FOLDER} && md5sum -c tokenizer_checklist.chk)

for i in ${MODEL_SIZE//,/ }
do
    echo "Downloading ${i}"
    mkdir -p ${TARGET_FOLDER}"/${i}"
    for s in $(seq -f "0%g" 0 ${N_SHARD_DICT[$i]})
    do
        wget ${PRESIGNED_URL/'*'/"${i}/consolidated.${s}.pth"} -O ${TARGET_FOLDER}"/${i}/consolidated.${s}.pth"
    done
    wget ${PRESIGNED_URL/'*'/"${i}/params.json"} -O ${TARGET_FOLDER}"/${i}/params.json"
    wget ${PRESIGNED_URL/'*'/"${i}/checklist.chk"} -O ${TARGET_FOLDER}"/${i}/checklist.chk"
    echo "Checking checksums"
    (cd ${TARGET_FOLDER}"/${i}" && md5sum -c checklist.chk)
done
