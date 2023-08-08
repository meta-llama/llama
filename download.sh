#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

PRESIGNED_URL=""              # URL from email
TARGET_FOLDER="."             # where all files should end up

# Download a file from the presigned URL, saving it to the target folder.
# Resumes downloading partially downloaded files.
# $1 - partial path of file to download
# $2 - target folder
download () {
    local filename="$1"
    local target_dir="$2"
    [[ -d "$target_dir" ]] || mkdir -p "$target_dir"
    wget --continue "${PRESIGNED_URL/'*'/"$filename"}" -O "${target_dir}/${filename}"
}

# $1 - checklist file
md5check () {
    if [[ ! -f "$1" ]]; then
        echo "Missing checklist file: $1" >&2
        return 1
    fi
    if [[ "$OSTYPE" == "darwin"* ]]; then
        failed=0
        while read -r hash file; do
            if [[ -n "$hash" ]]; then
                printf "%s: " "$file"
                if [[ "$(md5 -q "$file")" != "$hash" ]]; then
                    echo "FAILED"
                    (( failed++ ))
                else
                    echo "OK"
                fi
            fi
        done < "$1"
        if (( failed > 0 )); then
            echo "$failed files failed checksum" >&2
            return 1
        else
            return 0
        fi
    else
        md5sum -c "$1"
        return $?
    fi
}

# If the user didn't modify the script to include the URL, ask for it
if [[ -z "$PRESIGNED_URL" ]]; then
    read -p "Enter the URL from email: " PRESIGNED_URL
    echo ""
fi

# Check if the PRESIGNED_URL has the form `https://..*...` (i.e. with one wildcard '*' in the path)
if ! echo "$PRESIGNED_URL" | grep -q -E '^https://[^*]+[*][^*]*'; then
    echo "Invalid URL: $PRESIGNED_URL" >&2
    echo "Expected: The URL from the email, containing one asterisk '*'." >&2
    exit 1
fi

ALL_MODELS="7B,13B,70B,7B-chat,13B-chat,70B-chat"
read -p "Enter the list of models to download without spaces (${ALL_MODELS}), or press Enter for all: " MODEL_SIZE

if [[ $MODEL_SIZE == "" ]]; then
    MODEL_SIZE="$ALL_MODELS"
fi

mkdir -p "$TARGET_FOLDER"

echo "Downloading LICENSE and Acceptable Usage Policy"
download "LICENSE" "$TARGET_FOLDER"
download "USE_POLICY.md" "$TARGET_FOLDER"

echo "Downloading tokenizer"
download "tokenizer.model" "$TARGET_FOLDER"
download "tokenizer_checklist.chk" "$TARGET_FOLDER"
(cd "$TARGET_FOLDER" && md5check tokenizer_checklist.chk)

for m in ${MODEL_SIZE//,/ }; do
    case $m in
        "7B")
            SHARD=0; MODEL_PATH="llama-2-7b" ;;
        "7B-chat")
            SHARD=0; MODEL_PATH="llama-2-7b-chat" ;;
        "13B")
            SHARD=1; MODEL_PATH="llama-2-13b" ;;
        "13B-chat")
            SHARD=1; MODEL_PATH="llama-2-13b-chat" ;;
        "70B")
            SHARD=7; MODEL_PATH="llama-2-70b" ;;
        "70B-chat")
            SHARD=7; MODEL_PATH="llama-2-70b-chat" ;;
        *)
            echo "Invalid model size: $m" >&2
            echo "Expected: one of $ALL_MODELS" >&2
            continue
            ;;
    esac

    echo "Downloading $MODEL_PATH"
    mkdir -p "${TARGET_FOLDER}/${MODEL_PATH}"

    for s in $(seq -f "0%g" 0 ${SHARD}); do
        download "${MODEL_PATH}/consolidated.${s}.pth" "$TARGET_FOLDER"
    done

    download "${MODEL_PATH}/params.json" "$TARGET_FOLDER"
    download "${MODEL_PATH}/checklist.chk" "$TARGET_FOLDER"
    echo "Checking checksums"
    (cd "${TARGET_FOLDER}/${MODEL_PATH}" && md5check checklist.chk)
done
