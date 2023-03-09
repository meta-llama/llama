PRESIGNED_URL="https://dobf1k6cxlizq.cloudfront.net/*?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9kb2JmMWs2Y3hsaXpxLmNsb3VkZnJvbnQubmV0LyoiLCJDb25kaXRpb24iOnsiRGF0ZUxlc3NUaGFuIjp7IkFXUzpFcG9jaFRpbWUiOjE2NzgzNjcyNDF9fX1dfQ__&Signature=UxwcuJVF8dSHJuIj8-rcml9vD-U5k1lM6A6muHDJCoqqofb6fqb~WEnVukc-YwUHzPrK0V~Kqs29fmx3M8BfCA34tAhbPlmsPDNRXshYKse6g32h4fx7Dvvk23So4XoTKwaKe74nj4BYN8Gk7GEtQuIx5ZlssvweRD31ASiP-Oe0htGM0QjlTt3Nn~siitbsaPvIOYWGlKMtHODc-fwtZWM8j3E0RVHi5KYqrrpHRBfEsQJswUybbLZPNQUJPdjbI~nh8DG-RbYp-D71wCEVCz9VPp115XEQTCHj2oxuGvmfNlCwswkDGQyDhLMoFbsKcsrUHA0HrIBpTK8Dy61ZlQ__&Key-Pair-Id=K231VYXPC1TA1R"             # replace with presigned url from email
MODEL_SIZE="65"  # edit this list with the model sizes you wish to download
TARGET_FOLDER="download"             # where all files should end up

declare -A N_SHARD_DICT

N_SHARD_DICT["7"]="0"
N_SHARD_DICT["13"]="1"
N_SHARD_DICT["30"]="3"
N_SHARD_DICT["65"]="7"

echo "Downloading tokenizer"
wget ${PRESIGNED_URL/'*'/"tokenizer.model"} -O ${TARGET_FOLDER}"/tokenizer.model"
wget ${PRESIGNED_URL/'*'/"tokenizer_checklist.chk"} -O ${TARGET_FOLDER}"/tokenizer_checklist.chk"

(cd ${TARGET_FOLDER} && md5 tokenizer_checklist.chk)

for i in ${MODEL_SIZE//,/ }
do
    echo "Downloading ${i}B"
    mkdir -p ${TARGET_FOLDER}"/${i}B"
    for s in $(seq -f "0%g" 0 ${N_SHARD_DICT[$i]})
    do
        echo "Downloading shard ${s}"
        wget ${PRESIGNED_URL/'*'/"${i}B/consolidated.${s}.pth"} -O ${TARGET_FOLDER}"/${i}B/consolidated.${s}.pth"
    done
    wget ${PRESIGNED_URL/'*'/"${i}B/params.json"} -O ${TARGET_FOLDER}"/${i}B/params.json"
    wget ${PRESIGNED_URL/'*'/"${i}B/checklist.chk"} -O ${TARGET_FOLDER}"/${i}B/checklist.chk"
    echo "Checking checksums"
    (cd ${TARGET_FOLDER}"/${i}B" && md5 checklist.chk)
done