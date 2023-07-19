#!/usr/bin/env python3

"""
Filename: download.py
Author: Adnan Boz (adnanboz)
Date: July 19, 2023
Description: python translation of the download.sh script for Meta llama 2
"""

import os
import urllib.request
import hashlib
import sys
import ssl
from getpass import getpass
from urllib.error import HTTPError

try:
	import certifi
except ImportError:
	raise ImportError("The certifi module is required. Please install it with 'pip install certifi'")

ssl._create_default_https_context = ssl._create_unverified_context

def checksum(file_path, checksum_file_path):
	hasher = hashlib.md5()
	with open(file_path, 'rb') as afile:
		buf = afile.read()
		hasher.update(buf)
	with open(checksum_file_path, 'r') as cfile:
		checksum = cfile.readline().split()[0]
	if hasher.hexdigest() != checksum:
		print("Checksum does not match.")
		return False
	return True

def download_file(url, filename):
	print(f"Downloading file from {url}")
	urllib.request.urlretrieve(url, filename)
	print(f"Downloaded file saved as {filename}")

# Click on the link in the email, then the browser will redirect you to the actual presigned link after a question. Copy the presigned link from the address bar.
PRESIGNED_URL = getpass("Enter the presigned URL (Click on the link in the email, then the browser will redirect you to the actual presigned link after a question. Copy the presigned URL from the address bar): ")

if "%2A" in PRESIGNED_URL:
	print("Please paste the text of the URL not the encoded link. i.e. the text should have a * but the URL you provided has encoded %2A")
	sys.exit()

MODEL_SIZE = input("Enter the list of models to download without spaces (7B,13B,70B,7B-chat,13B-chat,70B-chat), or press Enter for all: ")

TARGET_FOLDER = "."
os.makedirs(TARGET_FOLDER, exist_ok=True)

if not MODEL_SIZE:
	MODEL_SIZE = "7B,13B,70B,7B-chat,13B-chat,70B-chat"

print("Downloading LICENSE and Acceptable Usage Policy")
urllib.request.urlretrieve(PRESIGNED_URL.replace('*', 'LICENSE'), f"{TARGET_FOLDER}/LICENSE")
urllib.request.urlretrieve(PRESIGNED_URL.replace('*', 'USE_POLICY.md'), f"{TARGET_FOLDER}/USE_POLICY.md")

print("Downloading tokenizer")
urllib.request.urlretrieve(PRESIGNED_URL.replace('*', 'tokenizer.model'), f"{TARGET_FOLDER}/tokenizer.model")
urllib.request.urlretrieve(PRESIGNED_URL.replace('*', 'tokenizer_checklist.chk'), f"{TARGET_FOLDER}/tokenizer_checklist.chk")

checksum(f"{TARGET_FOLDER}/tokenizer.model", f"{TARGET_FOLDER}/tokenizer_checklist.chk")

for m in MODEL_SIZE.split(','):
	if m == "7B":
		SHARD = 0
		MODEL_PATH = "llama-2-7b"
	elif m == "7B-chat":
		SHARD = 0
		MODEL_PATH = "llama-2-7b-chat"
	elif m == "13B":
		SHARD = 1
		MODEL_PATH = "llama-2-13b"
	elif m == "13B-chat":
		SHARD = 1
		MODEL_PATH = "llama-2-13b-chat"
	elif m == "70B":
		SHARD = 7
		MODEL_PATH = "llama-2-70b"
	elif m == "70B-chat":
		SHARD = 7
		MODEL_PATH = "llama-2-70b-chat"

	print(f"Downloading total {SHARD} shard(s) of {MODEL_PATH}")
	os.makedirs(f"{TARGET_FOLDER}/{MODEL_PATH}", exist_ok=True)

	for s in range(SHARD+1):
		s = f"{s:02}"
		download_file(PRESIGNED_URL.replace('*', f"{MODEL_PATH}/consolidated.{s}.pth"), f"{TARGET_FOLDER}/{MODEL_PATH}/consolidated.{s}.pth")

	download_file(PRESIGNED_URL.replace('*', f"{MODEL_PATH}/params.json"), f"{TARGET_FOLDER}/{MODEL_PATH}/params.json")
	download_file(PRESIGNED_URL.replace('*', f"{MODEL_PATH}/checklist.chk"), f"{TARGET_FOLDER}/{MODEL_PATH}/checklist.chk")

	print("Checking checksums")
	checksum(f"{TARGET_FOLDER}/{MODEL_PATH}/consolidated.00.pth", f"{TARGET_FOLDER}/{MODEL_PATH}/checklist.chk")
