#!/bin/bash
set -e

# Get user input for data directory and dataset size
datadir=$1
num_gpus=$2
mode=${3:-default}
stage=${4:-all}
raw_data_dir=${5:-}

num_proc=16
batch_size=32 # use batch size of 8 for <16GB GPU memory
convert_retries=${TEXTCAPS_CONVERT_RETRIES:-5}
download_timeout=${TEXTCAPS_DOWNLOAD_TIMEOUT:-3600}

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

run_convert() {
    raw_args=()
    if [ -n "$raw_data_dir" ]; then
        raw_args=(--raw_data_dir "$raw_data_dir")
    fi
    attempt=1
    while [ "$attempt" -le "$convert_retries" ]; do
        echo "TextCaps convert attempt ${attempt}/${convert_retries}"
        if python micro_diffusion/datasets/prepare/textcaps/convert.py --local_mds_dir "$mds_dir" \
            --download_timeout "$download_timeout" "${raw_args[@]}" $convert_args; then
            return 0
        fi
        if [ "$attempt" -eq "$convert_retries" ]; then
            echo "TextCaps convert failed after ${convert_retries} attempts."
            return 1
        fi
        sleep_seconds=$((attempt * 30))
        echo "TextCaps convert failed; retrying in ${sleep_seconds}s..."
        sleep "$sleep_seconds"
        attempt=$((attempt + 1))
    done
}

if [ "$stage" = "download" ]; then
    echo "Download TextCaps raw files manually into a directory, then pass that directory as the fifth argument to convert."
    echo "Required files: TextCaps_0.1_train.json, TextCaps_0.1_val.json,"
    echo "TextVQA_Rosetta_OCR_v0.2_train.json, TextVQA_Rosetta_OCR_v0.2_val.json,"
    echo "and either train_val_images.zip or an extracted train_images/ directory."
    echo "Example convert command after manual download:"
    echo "bash micro_diffusion/datasets/scripts/get_textcaps_dataset.sh $datadir $num_gpus $mode convert /path/to/textcaps/raw"
    exit 0
fi

if [ "$stage" = "all" ] || [ "$stage" = "convert" ]; then
    # Convert TextCaps to MDS. If raw_data_dir is set, this uses local files only.
    run_convert
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
    echo "Unknown stage: $stage. Use one of: all, download, convert, precompute."
    exit 1
fi
