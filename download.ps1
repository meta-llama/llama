function Check-MD5Sum {
    param (
        [Parameter(Mandatory=$true)]
        [string]$ChecklistPath
    )
    $checklist = Get-Content $ChecklistPath
    $checklist | ForEach-Object {
        $md5sum = $_.split('  ')[0]
        $filename = $_.split('  ')[1]
        if (Test-Path $filename) {
            $hash = Get-FileHash -Algorithm MD5 -Path $filename | Select-Object -ExpandProperty Hash
            if ($md5sum -eq $hash) {
                Write-Host "${filename}: OK"
            }
            else {
                Write-Host "${filename}: FAILED"
            }
        }
        else {
            Write-Host "${filename}: NOT FOUND"
        }
    }
}


$PRESIGNED_URL = Read-Host -Prompt "Enter the URL from email "
$MODEL_SIZE=Read-Host -Prompt "Enter the list of models to download without spaces (7B,13B,70B,7B-chat,13B-chat,70B-chat), or press Enter for all "
$TARGET_FOLDER="."             # where all files should end up
mkdir ${TARGET_FOLDER} -ea 0

if ($MODEL_SIZE -eq "") {
    $MODEL_SIZE = "7B,13B,70B,7B-chat,13B-chat,70B-chat"
}

Write-Host "Downloading LICENSE and Acceptable Usage Policy"
Invoke-WebRequest -UserAgent "Wget/1.13.4" -Uri ($PRESIGNED_URL -replace '\*', 'LICENSE') -OutFile "$TARGET_FOLDER/LICENSE" 
Invoke-WebRequest -UserAgent "Wget/1.13.4" -Uri ($PRESIGNED_URL -replace '\*', 'USE_POLICY.md') -OutFile "$TARGET_FOLDER/USE_POLICY.md"

Write-Host "Downloading tokenizer"
Invoke-WebRequest -UserAgent "Wget/1.13.4" -Uri ($PRESIGNED_URL -replace '\*', 'tokenizer.model') -OutFile "${TARGET_FOLDER}/tokenizer.model"
Invoke-WebRequest -UserAgent "Wget/1.13.4" -Uri ($PRESIGNED_URL -replace '\*', 'tokenizer_checklist.chk') -OutFile "${TARGET_FOLDER}/tokenizer_checklist.chk"
Push-Location ${TARGET_FOLDER} && Check-MD5Sum -ChecklistPath tokenizer_checklist.chk && Pop-Location

foreach ($m in $MODEL_SIZE.Split(',')) {
    if ($m -eq "7B") {
        $SHARD = 0
        $MODEL_PATH = "llama-2-7b"
    } elseif ($m -eq "7B-chat") {
        $SHARD = 0
        $MODEL_PATH = "llama-2-7b-chat"
    } elseif ($m -eq "13B") {
        $SHARD = 1
        $MODEL_PATH = "llama-2-13b"
    } elseif ($m -eq "13B-chat") {
        $SHARD = 1
        $MODEL_PATH = "llama-2-13b-chat"
    } elseif ($m -eq "70B") {
        $SHARD = 7
        $MODEL_PATH = "llama-2-70b"
    } elseif ($m -eq "70B-chat") {
        $SHARD = 7
        $MODEL_PATH = "llama-2-70b-chat"
    }

    Write-Host "Downloading ${MODEL_PATH}"
    New-Item -ItemType Directory -Path "${TARGET_FOLDER}/${MODEL_PATH}" -Force | Out-Null

    for ($s = 0; $s -le $SHARD; $s++) {
        $url=($PRESIGNED_URL -replace '\*', "${MODEL_PATH}/consolidated.0${s}.pth")
        Invoke-WebRequest -UserAgent "Wget/1.13.4" -Uri $url -OutFile "${TARGET_FOLDER}/${MODEL_PATH}/consolidated.0${s}.pth"
    }
    $url=($PRESIGNED_URL -replace '\*', "${MODEL_PATH}/params.json") 
    Invoke-WebRequest -UserAgent "Wget/1.13.4" -Uri $url -OutFile "${TARGET_FOLDER}/${MODEL_PATH}/params.json"
    $url=($PRESIGNED_URL -replace '\*', "${MODEL_PATH}/checklist.chk")
    Invoke-WebRequest -UserAgent "Wget/1.13.4" -Uri $url -OutFile "${TARGET_FOLDER}/${MODEL_PATH}/checklist.chk"
    Write-Host "Checking checksums"
    Push-Location  "${TARGET_FOLDER}/${MODEL_PATH}" && Check-MD5Sum -ChecklistPath checklist.chk && Pop-Location
}
