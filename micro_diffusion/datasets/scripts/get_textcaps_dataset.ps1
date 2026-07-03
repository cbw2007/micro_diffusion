param(
    [Parameter(Mandatory = $true)]
    [string]$Datadir,

    [Parameter(Mandatory = $true)]
    [int]$NumGpus,

    [string]$Mode = "default",

    [ValidateSet("all", "convert", "precompute")]
    [string]$Stage = "all"
)

$ErrorActionPreference = "Stop"

$batchSize = 32 # use batch size of 8 for <16GB GPU memory

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

if ($Stage -eq "all" -or $Stage -eq "convert") {
    python micro_diffusion/datasets/prepare/textcaps/convert.py `
        --local_mds_dir $mdsDir `
        @convertArgs
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
