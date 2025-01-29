:: Copyright (c) Meta Platforms, Inc. and affiliates.
:: This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

:: Prerequisites:
:: 0. Windows machine
:: 1. wget installed
:: 2. You've broken up your presigned url into its URL variables (separated by "?" and "&")

:: RUN AT YOUR OWN RISK: I removed the md5sum dependency because I couldn't get the windows version to work right with this script.

:: Example:
:: download.bat YOUR_POLICY YOUR_SIGNATURE YOUR_KPID 7B .

:: Model size options = 7B,13B,70B,7B-chat,13B-chat,70B-chat
:: Unlike the bash script, this only takes one model at a time

:: First argument = Policy field from presigned url
:: Second argument = Signature field from presigned url
:: Third argument = Key-Pair-Id field from presigned url
:: Fourth argument = model size (only one at a time)
:: Fifth argument = target folder

:: WARNING: This will NOT work in PowerShell
:: WARNING: You must run cmd.exe as Administrator. I have a cmd shortcut on my desktop that I right click then click "Run as Administrator"

@ECHO Off

SET POLICY=%1
SET SIGNATURE=%2
SET KEY_PAIR_ID=%3
SET MODEL_SIZE=%4
SET TARGET_FOLDER=%5

IF %MODEL_SIZE%=="" SET MODEL_SIZE=7B
IF %TARGET_FOLDER%=="" SET TARGET_FOLDER=.

ECHO "Downloading LICENSE and Acceptable Usage Policy"
wget "https://download.llamameta.net/LICENSE?Policy=%POLICY%&Signature=%SIGNATURE%&Key-Pair-Id=%KEY_PAIR_ID%" -O "%TARGET_FOLDER%/LICENSE"
wget "https://download.llamameta.net/USE_POLICY.md?Policy=%POLICY%&Signature=%SIGNATURE%&Key-Pair-Id=%KEY_PAIR_ID%" -O "%TARGET_FOLDER%/USE_POLICY.md"

ECHO "Downloading tokenizer"
wget "https://download.llamameta.net/tokenizer.model?Policy=%POLICY%&Signature=%SIGNATURE%&Key-Pair-Id=%KEY_PAIR_ID%" -O "%TARGET_FOLDER%/tokenizer.model"
wget "https://download.llamameta.net/tokenizer_checklist.chk?Policy=%POLICY%&Signature=%SIGNATURE%&Key-Pair-Id=%KEY_PAIR_ID%" -O "%TARGET_FOLDER%/tokenizer_checklist.chk"

IF %MODEL_SIZE%==7B SET /A SHARD=0
IF %MODEL_SIZE%==7B SET MODEL_PATH=llama-2-7b

if %MODEL_SIZE%==7B-chat SET /A SHARD=0
if %MODEL_SIZE%==7B-chat SET MODEL_PATH=llama-2-7b-chat

if %MODEL_SIZE%==13B SET /A SHARD=1
if %MODEL_SIZE%==13B SET MODEL_PATH=llama-2-13b

if %MODEL_SIZE%==13B-chat SET /A SHARD=1
if %MODEL_SIZE%==13B-chat SET MODEL_PATH=llama-2-13b-chat

if %MODEL_SIZE%==70B SET /A SHARD=7
if %MODEL_SIZE%==70B SET MODEL_PATH=llama-2-70b

if %MODEL_SIZE%==70B-chat SET /A SHARD=7
if %MODEL_SIZE%==70B-chat SET MODEL_PATH=llama-2-70b-chat

ECHO "Downloading model to %MODEL_PATH% with shard %SHARD% because MODEL_SIZE is %MODEL_SIZE%"
MKDIR "%TARGET_FOLDER%/%MODEL_PATH%"

FOR %%s IN (0, 1, %SHARD%) DO (
    wget "https://download.llamameta.net/%MODEL_PATH%/consolidated.%s%.pth?Policy=%POLICY%&Signature=%SIGNATURE%&Key-Pair-Id=%KEY_PAIR_ID%"  -O "%TARGET_FOLDER%/%MODEL_PATH%/consolidated.%s%.pth"
)

wget "https://download.llamameta.net/%MODEL_PATH%/params.json?Policy=%POLICY%&Signature=%SIGNATURE%&Key-Pair-Id=%KEY_PAIR_ID%"  -O "%TARGET_FOLDER%/%MODEL_PATH%/params.json"
wget "https://download.llamameta.net/%MODEL_PATH%/checklist.chk?Policy=%POLICY%&Signature=%SIGNATURE%&Key-Pair-Id=%KEY_PAIR_ID%"  -O "%TARGET_FOLDER%/%MODEL_PATH%/checklist.chk"

ECHO DONE