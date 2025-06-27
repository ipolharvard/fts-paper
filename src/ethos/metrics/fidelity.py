import itertools
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score

from ..datasets.base import InferenceDataset

SEED = 1337
np.random.seed(SEED)


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)  # Convert NumPy floats to Python floats
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)  # Convert NumPy integers to Python integers
    return obj  # Return as-is if not a NumPy type


def fetch_all_codes(data, dataset_type="readmission"):
    if dataset_type == "readmission":
        indices = data.start_indices.tolist()

        # entire patient visit, spuer long running time
        # start_idxs = [data.patient_offset_at_idx[idx].item() for idx in indices]
        # return [data.tokens[start_idx:idx + 1] for start_idx, idx in zip(start_idxs, indices)]

        # patient visit length limited to timeline length and exclude the demographic tokens
        return [
            InferenceDataset.__getitem__(data, idx).tolist()[data.context_size :] for idx in indices
        ]
    elif dataset_type == "timeline":
        patient_offsets = data.patient_offsets.numpy()

        # entire patient visit, spuer long running time
        # end_offsets = list(patient_offsets[1:]) + [len(data.tokens)]
        # return [data.tokens[start:end] for start, end in zip(patient_offsets, end_offsets)]

        # patient visit length limited to timeline length and exclude the demographic tokens
        return [data[idx][0].tolist()[data.context_size :] for idx in patient_offsets]
    else:
        raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def data_to_matrix(all_codes, code_vocab_size, mode="binary", dataset_type="readmission"):
    """Converts data to binary or count matrix based on dataset_type.

    Args:
        all_codes: Dataset object.
        code_vocab_size: Number of unique codes.
        mode: 'binary' or 'count'.
        dataset_type: 'readmission' or 'timeline'.

    Returns:
        np.ndarray: Binary or count matrix.
    """
    matrix = np.zeros((len(all_codes), code_vocab_size), dtype=int)
    for i, codes in enumerate(all_codes):
        unique_codes = set(codes) if mode == "binary" else codes
        if mode == "binary":
            matrix[i, list(unique_codes)] = 1
        elif mode == "count":
            for code in unique_codes:
                matrix[i, code] += 1
        else:
            raise ValueError("Mode must be 'binary' or 'count'")

    return matrix


def transform_data_matrix(all_codes, code_vocab_size, matrix_type, dataset_type="readmission"):
    """Transforms data into binary, count, or probability matrix.

    Args:
        all_codes: Dataset object.
        code_vocab_size: Vocabulary size of codes.
        matrix_type: 'binary', 'count', or 'probability'.
        dataset_type: 'readmission' or 'timeline'.

    Returns:
        np.ndarray: Transformed matrix.
    """
    if matrix_type == "binary":
        return data_to_matrix(all_codes, code_vocab_size, mode="binary", dataset_type=dataset_type)
    elif matrix_type == "count":
        return data_to_matrix(all_codes, code_vocab_size, mode="count", dataset_type=dataset_type)
    elif matrix_type == "probability":
        count_matrix = data_to_matrix(
            all_codes, code_vocab_size, mode="count", dataset_type=dataset_type
        )
        row_sums = count_matrix.sum(axis=1, keepdims=True)
        return count_matrix / (row_sums + 1e-5)  # Avoid division by zero
    else:
        raise ValueError(f"Unsupported matrix type: {matrix_type}")


# Function to calculate mean lengths
def calculate_mean_lengths(all_codes, dataset_type="readmission"):
    visit_lengths = [len(codes) for codes in all_codes]
    return {
        "mean_visit_length": np.mean(visit_lengths),
        "std_visit_length": np.std(visit_lengths),
    }


# Function to calculate unigram and bigram probabilities
def calculate_code_probabilities(all_codes, dataset_type="readmission"):
    unigram_counts = {}
    flattened_codes = list(itertools.chain.from_iterable(all_codes))
    unigram_counts = Counter(flattened_codes)

    total_unigrams = sum(unigram_counts.values())
    unigram_probs = {code: count / total_unigrams for code, count in unigram_counts.items()}

    return unigram_probs


# Function to compare probabilities
def compare_probabilities(real_probs, synthetic_probs, label, save_dir):
    real_values = []
    synthetic_values = []

    for key in set(real_probs.keys()).union(synthetic_probs.keys()):
        real_values.append(real_probs.get(key, 0))
        synthetic_values.append(synthetic_probs.get(key, 0))

    r2 = r2_score(real_values, synthetic_values)

    # Calculate deviation from the reference line y=x
    deviation = np.abs(np.array(real_values) - np.array(synthetic_values))

    # Normalize the deviation to range [0, 1] for consistent coloring
    deviation_normalized = (deviation - deviation.min()) / (deviation.max() - deviation.min())
    deviation_normalized = 1 - deviation_normalized

    # Create a scatter plot with colormap
    # plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        real_values,
        synthetic_values,
        c=deviation_normalized,
        cmap="viridis",
        alpha=0.7,
        s=20,
        label="Data points",
    )

    # Add diagonal reference line (y=x)
    max_val = max(max(real_values), max(synthetic_values)) * 1.1
    plt.plot(
        [0, max_val],
        [0, max_val],
        color="red",
        linestyle="--",
        linewidth=1,
        label="Reference line (y=x)",
    )

    # Add colorbar for reference
    plt.colorbar(scatter)
    # cbar.set_label("Gradient Metric (e.g., Density or Order)", fontsize=10)

    # Add gridlines and beautify
    plt.grid(visible=True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.xlabel("Real Data", fontsize=12)
    plt.ylabel("Synthetic Data", fontsize=12)
    plt.title(f"{label} Comparison ($R^2 = {r2:.2f}$)", fontsize=12)
    plt.xlim([0, max_val])
    plt.ylim([0, max_val])
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Save the beautified plot
    plot_path = os.path.join(save_dir, f"{label.replace(' ', '_')}_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return r2


def fidelity_from_halo_evaluation(
    real_codes, synthetic_codes, save_dir, dataset_type="readmission"
):
    # Generate statistics for real and synthetic data

    real_lengths = calculate_mean_lengths(real_codes, dataset_type)
    synthetic_lengths = calculate_mean_lengths(synthetic_codes, dataset_type)

    real_unigrams = calculate_code_probabilities(real_codes, dataset_type)
    synthetic_unigrams = calculate_code_probabilities(synthetic_codes, dataset_type)

    # Compare probabilities and save plots
    r2_unigrams = compare_probabilities(
        real_unigrams, synthetic_unigrams, "Unigram Probabilities", save_dir
    )

    # Save metrics to JSON
    metrics = {
        "real_visit_mean_length": real_lengths["mean_visit_length"],
        "real_visit_std_length": real_lengths["std_visit_length"],
        "synthetic_visit_mean_length": synthetic_lengths["mean_visit_length"],
        "synthetic_visit_std_length": synthetic_lengths["std_visit_length"],
        "r2_unigrams": r2_unigrams,
    }
    return metrics


def compute_prevalence(data_matrix, matrix_type):
    """Computes prevalence for the given matrix type.

    Args:
        data_matrix: Transformed data matrix (binary, count, or probability).
        matrix_type: Type of matrix ('binary', 'count', or 'probability').

    Returns:
        numpy.ndarray: Prevalence values for each code.
    """
    if matrix_type == "binary":
        # Proportion of patients with each code
        return np.mean(data_matrix > 0, axis=0)
    elif matrix_type in ["count", "probability"]:
        # Mean frequency or normalized probabilities
        return np.mean(data_matrix, axis=0)
    else:
        raise ValueError(f"Unsupported matrix type: {matrix_type}")


def fidelity_evaluation(real_data, synthetic_data, matrix_type):
    """Evaluates fidelity metrics using the specified matrix type.

    Args:
        real_data: Real dataset matrix.
        synthetic_data: Synthetic dataset matrix.
        matrix_type: The type of matrix used ('binary', 'count', or 'probability').

    Returns:
        dict: Fidelity metrics.
    """
    results = {}

    # Prevalence-based metrics
    real_prevalence = compute_prevalence(real_data, matrix_type)
    synthetic_prevalence = compute_prevalence(synthetic_data, matrix_type)
    results[f"{matrix_type}_mmd"] = np.abs(real_prevalence - synthetic_prevalence).max()
    # R² score
    results[f"{matrix_type}_r2_score"] = r2_score(real_prevalence, synthetic_prevalence)

    return results
