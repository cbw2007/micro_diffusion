#!/bin/bash

# Get user input for data directory and dataset size
datadir=$1
num_gpus=$2
mode=${3:-default}
stage=${4:-all}

num_proc=16
batch_size=32 # use batch size of 8 for <16GB GPU memory

if [ "$mode" = "region" ]; then
    mds_dir="${datadir}/mds_region/"
    latents_dir="${datadir}/mds_latents_sdxl1_dfnclipH14_region/"
    convert_args="--save_text_bboxes"
    precompute_args="--save_text_region_masks"
else
    mds_dir="${datadir}/mds/"
    latents_dir="${datadir}/mds_latents_sdxl1_dfnclipH14/"
    convert_args=""
    precompute_args=""
fi

if [ "$stage" = "all" ] || [ "$stage" = "convert" ]; then
    # Textcaps is fairly small so we download all of it during conversion to MDS.
    python micro_diffusion/datasets/prepare/textcaps/convert.py --local_mds_dir "$mds_dir" $convert_args
fi

if [ "$stage" = "convert" ]; then
    exit 0
fi

if [ "$stage" = "all" ] || [ "$stage" = "precompute" ]; then
    # Precompute latents across one or more GPUs from an existing MDS directory.
    python -c "from streaming.base.util import clean_stale_shared_memory; clean_stale_shared_memory()"
    if [ "$num_gpus" -gt 1 ]; then
        accelerate_args="--multi_gpu --num_processes $num_gpus"
    else
        accelerate_args="--num_processes 1"
    fi

    accelerate launch $accelerate_args micro_diffusion/datasets/prepare/textcaps/precompute.py --datadir "$mds_dir" \
        --savedir "$latents_dir" --vae stabilityai/stable-diffusion-xl-base-1.0 \
        --text_encoder openclip:hf-hub:apple/DFN5B-CLIP-ViT-H-14-378 --batch_size $batch_size $precompute_args
else
    echo "Unknown stage: $stage. Use one of: all, convert, precompute."
    exit 1
fi
