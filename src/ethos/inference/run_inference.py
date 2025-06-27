import json
import time
from multiprocessing import Manager, Process, set_start_method
from pathlib import Path
from queue import Empty

import hydra
import numpy as np
import torch as th
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from .constants import Task
from .inference import get_dataset_cls, spawn_inference_worker
from .utils import evaluate_dataset_subset, format_big_number, producer, write_results_to_parquet


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    task = Task(cfg.task)
    input_dir = Path(cfg.input_dir)

    model_checkpoint = th.load(cfg.model_fp, map_location="cpu", mmap=True, weights_only=False)

    model_config = model_checkpoint["model_config"]
    n_positions = (
        model_config.decoder.n_positions
        if model_config.is_encoder_decoder
        else model_config.n_positions
    )
    dataset_kwargs = {
        "input_dir": input_dir,
        "n_positions": n_positions,
        "is_encoder_decoder": model_config.is_encoder_decoder,
    } | dict(cfg.dataset_kwargs or {})

    dataset_cls = get_dataset_cls(task)
    start_time = time.time()
    dataset = dataset_cls(**dataset_kwargs)
    logger.info(f"{dataset} initialized in {time.time() - start_time:.0f}s.")

    if len(stop_stokens := dataset.stop_stokens) > 10:
        stop_stokens = stop_stokens[:10] + ["..."]
    logger.info(f"Stop tokens: {', '.join(stop_stokens)}")
    logger.info(f"Time limit: {dataset.time_limit}")

    n_samples, subset_suffix = evaluate_dataset_subset(dataset, cfg.subset)
    logger.info(
        f"Number of samples: {n_samples:,} ({n_samples / len(dataset):.2%})."
        + (f" Full dataset size: {len(dataset):,}." if subset_suffix else "")
        + f" Number of repetitions: {cfg.rep_num}"
    )
    result_dir = Path(cfg.output_dir + subset_suffix)

    if cfg.temperature != 1.0:
        result_dir = result_dir.with_name(f"{result_dir.name}_temp{cfg.temperature}")

    if "wandb_path" in model_checkpoint:
        run_id = model_checkpoint["wandb_path"].split("/")[-1]
        result_dir = result_dir.with_name(f"{result_dir.name}_{run_id}")

    if not (metadata_fp := result_dir / "metadata.json").exists():
        result_dir.resolve().mkdir(parents=True, exist_ok=True)
        metadata = {
            k: str(cfg.get(k))
            for k in [
                "task",
                "model_fp",
                "input_dir",
                "temperature",
                "subset",
                "dataset_kwargs",
                "modality_embeddings_dir",
            ]
        }
        metadata["wandb_path"] = model_checkpoint.get("wandb_path")
        metadata["model"] = model_config.to_dict()
        with metadata_fp.open("w") as f:
            json.dump(metadata, f)
        logger.info(f"Metadata saved to '{metadata_fp}'")

    if cfg.output_fn is not None:
        result_dir /= cfg.output_fn

    result_dir.resolve().mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to '{result_dir}'")

    np.random.seed(cfg.seed)
    indices = np.random.choice(np.arange(len(dataset)), n_samples, replace=False)
    chunk_num = n_samples // cfg.chunksize
    subsets = [subset_indices for subset_indices in np.array_split(indices, chunk_num)]

    if cfg.device == "cuda":
        num_proc = cfg.n_jobs * cfg.n_gpus
    elif cfg.device == "cpu":
        num_proc = cfg.n_jobs
    else:
        raise ValueError(f"Unknown device: {cfg.device}, must be 'cpu' or 'cuda'")
    if num_proc > len(subsets):
        logger.warning(
            f"Number of processes ({num_proc}) is larger than the number of subsets "
            f"({len(subsets)}). Lunching only {len(subsets)} processes."
        )
        num_proc = len(subsets)

    set_start_method("spawn")
    with Manager() as mgr:
        job_queue = mgr.Queue(maxsize=num_proc * 2)
        progress_queue = mgr.Queue()

        processes = [Process(target=producer, args=(subsets, job_queue, num_proc), name="producer")]
        processes.extend(
            Process(
                target=spawn_inference_worker,
                args=(
                    job_queue,
                    cfg.model_fp,
                    task,
                    dataset_kwargs,
                    progress_queue,
                    cfg.temperature,
                    cfg.rep_num,
                    "cpu" if cfg.device == "cpu" else f"cuda:{i % cfg.n_gpus}",
                    cfg.no_compile,
                    cfg.save_generated_tokens,
                ),
                name=f"Process_{i}",
            )
            for i in range(num_proc)
        )

        for p in processes:
            p.start()

        results, generated_tokens = [], 0
        total_samples = n_samples * cfg.rep_num
        progress_bar = tqdm(total=total_samples, desc="Progress", unit="samples", smoothing=0.1)
        try:
            for _ in range(total_samples):
                results.append(progress_queue.get(timeout=cfg.timeout))
                generated_tokens += results[-1]["token_dist"]
                progress_bar.set_postfix_str(
                    "total generated tokens: {}, {} tokens/s".format(
                        format_big_number(generated_tokens),
                        format_big_number(generated_tokens / progress_bar.format_dict["elapsed"]),
                    )
                )
                progress_bar.update()

                if len(results) >= cfg.result_chunk_size:
                    write_results_to_parquet(result_dir, results, progress_bar.format_dict["n"])
                    results = []

        except Empty:
            logger.error("Progress queue timed out.")
            for p in processes:
                if p.is_alive():
                    p.terminate()

        for p in processes:
            p.join()

    if results:
        write_results_to_parquet(result_dir, results, progress_bar.format_dict["n"])

    logger.info("Workers finished.")


if __name__ == "__main__":
    main()
