#!/bin/bash -l

# this script is intended to be run from the project root
model_variant="best_model.pt"

dataset=${1//-/_}
task_name=$2
shift 2

dataset_dir="data/tokenized_datasets"

case $dataset in
mimic*)
    model="layer_3_do_0.3"
    ;;
*)
    echo "Wrong dataset name: '$dataset', available are: 'mimic'."
    exit 1
    ;;
esac
dataset_dir="data/tokenized_datasets/$dataset"

clear
if [[ ! -d $dataset_dir ]]; then
    echo "Dataset directory not found: $dataset_dir"
    exit 1
fi

res_name_prefix=""
rep_num=1
for arg in "$@"; do
    if [[ $arg == +dataset_kwargs.sample_idx=* ]]; then
        res_name_prefix="${arg#+dataset_kwargs.sample_idx=}/"
    fi
    if [[ $arg == rep_num=* ]]; then
        rep_num="${arg#rep_num=}"
    fi
done

singularity_preamble="
export PATH=\$HOME/.local/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat/:/.singularity.d/libs/

# Install ethos
cd /ethos
pip install \
    --no-deps \
    --no-index \
    --no-build-isolation \
    --user \
    -e  \
    . 1>/dev/null

# Use other tmp dir to avoid /tmp filling up and preserve the cache across the runs
export TORCHINDUCTOR_CACHE_DIR=/ethos/torchinductor_cache
"

script_body="
set -e

echo RUNNING: dataset=${dataset}, task=${task_name}, model_variant=${model_variant}
ethos_infer \
    task=${task_name} \
    model_fp=${dataset_dir}/models/${model}/${model_variant} \
    input_dir=${dataset_dir}/test \
    output_dir=results/${task_name}/${dataset}_${model}_${model_variant%_*} \
    output_fn=${res_name_prefix}rep_size_${rep_num}_\$(date +%Y-%m-%d_%H-%M-%S) \
    $* \
    n_gpus=\${NUM_GPUS}
"

module load singularity 2>/dev/null

if command -v singularity >/dev/null; then
    export NUM_GPUS=${SLURM_GPUS_ON_NODE}
    singularity exec \
        --contain \
        --nv \
        --writable-tmpfs \
        --bind "$(pwd)":/ethos \
        --bind /mnt:/mnt \
        ethos.sif \
        bash -c "${singularity_preamble}${script_body}"
else
    NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    export NUM_GPUS
    bash -c "${script_body}"
fi
