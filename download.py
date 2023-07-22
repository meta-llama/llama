#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import requests
import hashlib

presigned_url = input("Enter the URL from email: ")
print("")
model_size = input("Enter the list of models to download without spaces (7B,13B,70B,7B-chat,13B-chat,70B-chat), or press Enter for all: ")
target_folder = ".\models"             # where all files should end up
os.makedirs(target_folder, exist_ok=True)

if model_size == "":
    model_size = "7B,13B,70B,7B-chat,13B-chat,70B-chat"

print("Downloading LICENSE and Acceptable Usage Policy")
license_response = requests.get(presigned_url.replace('*', "LICENSE"))
with open(os.path.join(target_folder, "LICENSE"), 'wb') as license_file:
    license_file.write(license_response.content)
policy_response = requests.get(presigned_url.replace('*', "USE_POLICY.md"))
with open(os.path.join(target_folder, "USE_POLICY.md"), 'wb') as policy_file:
    policy_file.write(policy_response.content)

print("Downloading tokenizer")
tokenizer_model_response = requests.get(presigned_url.replace('*', "tokenizer.model"))
with open(os.path.join(target_folder, "tokenizer.model"), 'wb') as tokenizer_model_file:
    tokenizer_model_file.write(tokenizer_model_response.content)
tokenizer_checklist_response = requests.get(presigned_url.replace('*', "tokenizer_checklist.chk"))
with open(os.path.join(target_folder, "tokenizer_checklist.chk"), 'wb') as tokenizer_checklist_file:
    tokenizer_checklist_file.write(tokenizer_checklist_response.content)

def check_md5(file_path, checksum):
    with open(file_path, 'rb') as f:
        data = f.read()
        md5 = hashlib.md5(data).hexdigest()
        return md5 == checksum

def check_checksums(folder_path, checklist_path):
    with open(checklist_path) as f:
        for line in f:
            checksum, file_name = line.strip().split()
            file_path = os.path.join(folder_path, file_name)
            if check_md5(file_path, checksum):
                print(f"{file_name}: OK")
            else:
                print(f"{file_name}: FAILED")

check_checksums(target_folder, os.path.join(target_folder, "tokenizer_checklist.chk"))

for model in model_size.split(','):
    if model == "7B":
        shard = 0
        model_path = "llama-2-7b"
    elif model == "7B-chat":
        shard = 0
        model_path = "llama-2-7b-chat"
    elif model == "13B":
        shard = 1
        model_path = "llama-2-13b"
    elif model == "13B-chat":
        shard = 1
        model_path = "llama-2-13b-chat"
    elif model == "70B":
        shard = 7
        model_path = "llama-2-70b"
    elif model == "70B-chat":
        shard = 7
        model_path = "llama-2-70b-chat"

    print(f"Downloading {model_path}")
    os.makedirs(os.path.join(target_folder, model_path), exist_ok=True)

    for s in range(shard + 1):
        consolidated_response = requests.get(presigned_url.replace('*', f"{model_path}/consolidated.{s:02d}.pth"))
        with open(os.path.join(target_folder, model_path, f"consolidated.{s:02d}.pth"), 'wb') as consolidated_file:
            consolidated_file.write(consolidated_response.content)

    params_response = requests.get(presigned_url.replace('*', f"{model_path}/params.json"))
    with open(os.path.join(target_folder, model_path, "params.json"), 'wb') as params_file:
        params_file.write(params_response.content)
    checklist_response = requests.get(presigned_url.replace('*', f"{model_path}/checklist.chk"))
    with open(os.path.join(target_folder, model_path, "checklist.chk"), 'wb') as checklist_file:
        checklist_file.write(checklist_response.content)
    
    print("Checking checksums")
    check_checksums(os.path.join(target_folder, model_path), os.path.join(target_folder, model_path, "checklist.chk"))

