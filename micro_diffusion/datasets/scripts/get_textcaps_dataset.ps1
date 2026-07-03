param(
    [Parameter(Mandatory = $true)]
    [string]$Datadir,

    [Parameter(Mandatory = $true)]
    [int]$NumGpus,

    [string]$Mode = "default",

    [ValidateSet("all", "download", "convert", "precompute")]
    [string]$Stage = "all",

    [string]$RawDataDir = ""
)

$ErrorActionPreference = "Stop"

$batchSize = 32 # use batch size of 8 for <16GB GPU memory
$convertRetries = if ($env:TEXTCAPS_CONVERT_RETRIES) { [int]$env:TEXTCAPS_CONVERT_RETRIES } else { 5 }
$downloadTimeout = if ($env:TEXTCAPS_DOWNLOAD_TIMEOUT) { [int]$env:TEXTCAPS_DOWNLOAD_TIMEOUT } else { 3600 }

if ($Mode -eq "region") {
    $mdsDir = Join-Path $Datadir "mds_region"
    $latentsDir = Join-Path $Datadir "mds_latents_sdxl1_dfnclipH14_region"
    $convertArgs = @("--save_text_bboxes")
    $precomputeArgs = @("--save_text_region_masks")
} else {
    $mdsDir = Join-Path $Datadir "mds"
    $latentsDir = Join-Path $Datadir "mds_latents_sdxl1_dfnclipH14"
    $convertArgs = @()
    $precomputeArgs = @()
}

function Invoke-TextCapsConvert {
    $rawArgs = @()
    if ($RawDataDir -ne "") {
        $rawArgs = @("--raw_data_dir", $RawDataDir)
    }
    for ($attempt = 1; $attempt -le $convertRetries; $attempt++) {
        Write-Host "TextCaps convert attempt $attempt/$convertRetries"
        python micro_diffusion/datasets/prepare/textcaps/convert.py `
            --local_mds_dir $mdsDir `
            --download_timeout $downloadTimeout `
            @rawArgs `
            @convertArgs
        if ($LASTEXITCODE -eq 0) {
            return
        }
        if ($attempt -eq $convertRetries) {
            throw "TextCaps convert failed after $convertRetries attempts."
        }
        $sleepSeconds = $attempt * 30
        Write-Host "TextCaps convert failed; retrying in ${sleepSeconds}s..."
        Start-Sleep -Seconds $sleepSeconds
    }
}

if ($Stage -eq "download") {
    Write-Host "Download TextCaps raw files manually into a directory, then pass that directory as the fifth argument to convert."
    Write-Host "Required files: TextCaps_0.1_train.json, TextCaps_0.1_val.json,"
    Write-Host "TextVQA_Rosetta_OCR_v0.2_train.json, TextVQA_Rosetta_OCR_v0.2_val.json,"
    Write-Host "and either train_val_images.zip or an extracted train_images/ directory."
    Write-Host "Example convert command after manual download:"
    Write-Host "powershell -ExecutionPolicy Bypass -File micro_diffusion/datasets/scripts/get_textcaps_dataset.ps1 $Datadir $NumGpus $Mode convert C:\path\to\textcaps\raw"
    exit 0
}

if ($Stage -eq "all" -or $Stage -eq "convert") {
    Invoke-TextCapsConvert
}

if ($Stage -eq "convert") {
    exit 0
}

if ($Stage -eq "all" -or $Stage -eq "precompute") {
    python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()"

    if ($NumGpus -gt 1) {
        $accelerateArgs = @("--multi_gpu", "--num_processes", "$NumGpus")
    } else {
        $accelerateArgs = @("--num_processes", "1")
    }

    accelerate launch @accelerateArgs `
        micro_diffusion/datasets/prepare/textcaps/precompute.py `
        --datadir $mdsDir `
        --savedir $latentsDir `
        --vae stabilityai/stable-diffusion-xl-base-1.0 `
        --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 `
        --batch_size $batchSize `
        @precomputeArgs
}
