# Federated Timeline Synthesis

The project was built on top of the [ETHOS-ARES](https://github.com/ipolharvard/ethos-ares) repository.

<img src="figures/framework.png" alt="FTS framework">

## Installation

[Optional] We strongly encourage the use of a virtual environment, for example, Conda:
To create a new conda env:

```bash
conda create --name ethos python=3.12
conda activate ethos
```

Fetch the project and set it up in the development mode (`-e`) and install all necessary
dependencies for running notebooks and scripts by executing:

```bash
cd fts_paper
pip install -e .[jupyter]
```

## Paper reproducibility

We provide the complete code necessary to reproduce all experiments presented in the paper.

Additionally, all precomputed inference results of our experiments are available in `results.tar.gz`
[[Google Drive (5GB)]](https://drive.google.com/file/d/1QuhRea5urY5DXNt-41-u2mVfs6HBVXZK/view?usp=sharing).
Once extracted in the project's root directory (`tar zxvf results.tar.gz`), it is possible to generate all
the figures in the paper using the notebooks in the `notebooks` directory. The test set, which was used to produce
the results, can be easily recreated by running the MEDS extraction and tokenization pipelines
(see below). MEDS guarantees the same data split if run on the same data with the same configuration
(see `scripts/meds`).

We do not publish the tokenized dataset or the pretrained models due to restrictions on MIMIC
derivatives.

## Pre-tokenization step

The tokenization step uses an intermediate
format [MEDS](https://github.com/Medical-Event-Data-Standard/meds), extracted via
the [MEDS_transforms](https://github.com/mmcdermott/MEDS_transforms) pipeline. Scripts for running
this pipeline are located in `scripts/meds`.

Below is an example command to run the extraction step.

```bash
export N_WORKERS=7

# cd scripts/meds
# Please define output_dir, strogly suggested is <PROJECT_ROOT>/data
# output_dir="../../data"
bash run_mimic.sh \
    "$MIMIC_IV_DIR" \  # Please define MIMIC_IV_DIR
    "$output_dir/mimic-2.2-premeds" \
    "$output_dir/mimic-2.2-meds"
```

Note, that using 7 workers for the tokenization requires around 250GB of RAM peak usage. You can
reduce the memory requirement by reducing the number of workers.

The number and size of the splits can be adjusted in the `scripts/meds/mimic/configs/extract_MIMIC.yaml`
file. Note, that keys: `train`, `tuning` and `held_out` have to be always present in the config file,
but can be set to null.

```yaml
split_and_shard_subjects:
    ...
    split_fracs:
        orig: 0.9
        test: 0.05
        val1: 0.025
        val2: 0.025
        train: null
        tuning: null
        held_out: null
```

If something does not work, and the extraction pipeline failed. Rerun the extraction script or remove the generated files and rerun. If this does not help, please open a GitHub issue.

Once the data extraction is complete, you can tokenize using the `ethos_tokenize` command, see the instructions below.

## Package Scripts

After installing the package, the below commands will become available for running from the command line.

1. `ethos_tokenize` - tokenizes data in the MEDS format into Patient Health Timelines.

Example of tokenization with parameters used for MIMIC-IV.

```bash
ethos_tokenize -m worker='range(0,7)' \  # spawns 7 workers
    input_dir=$input_dir/orig \  # `input_dir` is the path to MEDS_dir/data
    output_dir=$output_dir \  # can be <project_root>/data/tokenized_datasets/mimic
    out_fn=orig

ethos_tokenize -m worker='range(0,2)' \
    input_dir=$input_dir/test \
    vocab=$output_dir/orig \  # uses vocab created on the `orig` split
    output_dir=$output_dir \
    out_fn=test
```

See the full example in `scripts/run_tokenization.sh`.

2. `ethos_train` - runs the model training.

Example of training a model in the 8GPU setting.

```bash
torchrun --no_python --standalone --nproc_per_node=8 ethos_train \
  data_fp=$data_path/train \
  val_size=6 \ # uses the last 6M tokens of train as the validation dataset
  batch_size=$BATCH_SIZE \
  max_epochs=300 \
  out_dir="$data_path/models/${model_name}" # the path to save model checkpoints
```

See the full example in `scripts/run_training.sh`.

3. `ethos_infer` - runs the inference of a chosen downstream tasks.

Example of running the zero-shot inference for the 30-day readmission task with 32 repetitions per sample in the 8GPU setting.

```bash
ethos_infer \
    task=readmission \ # see ethos.inference.constants for all available tasks
    model_fp=$model_dir/$model/best_model.pt \
    input_dir=$dataset_dir/test \
    output_dir=results/$task_name/$dataset_$model \
    output_fn=rep_size_32_\$(date +%Y-%m-%d_%H-%M-%S) \
    rep_num=32 \
    n_gpus=8
```

Example of running the zero-shot inference for the ICU admission task on the random 40% of the whole test set.

```bash
ethos_infer \
    task=icu_admission \
    model_fp=$model_dir/$model/best_model.pt \
    input_dir=$dataset_dir/test \
    output_dir=results/$task_name/$dataset_$model \
    output_fn=rep_size_8\$(date +%Y-%m-%d_%H-%M-%S) \
    rep_num=8 \
    subset=0.4
```

Example of generating a synthetic dataset with the same demographic properties as the `train` dataset. Requires the `ethos_synth` step to convert it into a fully-fledged dataset that can be used for training.

```bash
ethos_infer \
    task=synthetic \
    model_fp=$model_dir/$model/best_model.pt \
    input_dir=$dataset_dir/train \
    output_dir=results/$task_name/$dataset_$model \
    output_fn=synthetic_\$(date +%Y-%m-%d_%H-%M-%S) \
    save_generated_tokens=true \ # it is crucial to save the trajectories (this will be default in the future)
    n_gpus=8
```

See the full example in `scripts/run_inference.sh`.

4. `ethos_synth` - converts timelines generated in the inference step (the `synthetic` task) into a Patient Helath Timelines dataset, that can be used for training.

Example of creating a standalone `big_synth` dataset based on `big`, meaning it will have the same vocabulary, patient static information and other properties as `big`.

```bash
ethos_synth \
    input_dir=results/synthetic/mimic_synth_layer_3_do_0.3_big_best_mpew57w3/ \
    dataset_dir=$out_dir/big \
    output_dir=$out_dir/big_synth
```

Example of adding synthetic `small` data to the `big+small_synth` dataset, presumably it already contains `big`.

```bash
ethos_synth \
    input_dir=results/synthetic/mimic_synth_layer_3_do_0.3_small_best_masd123/ \
    dataset_dir=$out_dir/big+small_synth
```
