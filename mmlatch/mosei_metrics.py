import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os
import pickle


def save_comparison_data(filepath, predictions, targets, masks_txt, masks_au, masks_vi):
    """
    Saves predictions, targets, and masks to a pickle file.

    Args:
        filepath (str): Path to save the pickle file.
        predictions (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth targets.
        masks_txt (torch.Tensor): Text masks.
        masks_au (torch.Tensor): Audio masks.
        masks_vi (torch.Tensor): Visual masks.
    """
    data = {
        'predictions': predictions.cpu().numpy(),
        'targets': targets.cpu().numpy(),
        'masks_txt': masks_txt.cpu().numpy(),
        'masks_au': masks_au.cpu().numpy(),
        'masks_vi': masks_vi.cpu().numpy(),
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Comparison data saved to {filepath}")

def load_comparison_data(filepath):
    """
    Loads predictions, targets, and masks from a pickle file.

    Args:
        filepath (str): Path to the pickle file.

    Returns:
        dict: Dictionary containing predictions, targets, and masks.
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Comparison data loaded from {filepath}")
    return data

# mosei_metrics.py

def calculate_mask_metrics(mask1, mask2):
    """
    Calculates MAE, MSE, Cosine Similarity, and KL-Divergence between two masks.

    Args:
        mask1 (numpy.ndarray): First mask array.
        mask2 (numpy.ndarray): Second mask array.

    Returns:
        dict: Dictionary containing the calculated metrics.
    """
    metrics = {}
    
    # Ensure masks are flattened for comparison
    mask1_flat = mask1.flatten()
    mask2_flat = mask2.flatten()
    
    # Mean Absolute Error
    mae = np.mean(np.abs(mask1_flat - mask2_flat))
    metrics['MAE'] = mae
    
    # Mean Squared Error
    mse = np.mean((mask1_flat - mask2_flat) ** 2)
    metrics['MSE'] = mse
    
    # Cosine Similarity
    cos_sim = 1 - cosine(mask1_flat, mask2_flat)
    metrics['Cosine_Similarity'] = cos_sim
    
    # KL-Divergence
    # Add a small epsilon to avoid division by zero and log(0)
    epsilon = 1e-10
    mask1_probs = mask1_flat + epsilon
    mask2_probs = mask2_flat + epsilon
    mask1_probs /= np.sum(mask1_probs)
    mask2_probs /= np.sum(mask2_probs)
    kl_div = entropy(mask1_probs, mask2_probs)
    metrics['KL_Divergence'] = kl_div
    
    return metrics

def compare_masks(data_new, data_comparison, modality='txt'):
    """
    Compares masks between two datasets for a specific modality.

    Args:
        data_new (dict): Dictionary containing 'masks_txt', 'masks_au', or 'masks_vi'.
        data_comparison (dict): Dictionary containing 'masks_txt', 'masks_au', or 'masks_vi'.
        modality (str): 'txt', 'au', or 'vi'.

    Returns:
        dict: Dictionary containing average metrics and per mask metrics.
    """
    metrics_all = {}
    
    mask_new = data_new[f'masks_{modality}']
    mask_comparison = data_comparison[f'masks_{modality}']
    
    num_masks = mask_new.shape[1]  # Assuming mask shape is (num_samples, num_masks_per_sample)
    
    # Calculate average value per mask position
    avg_new = np.mean(mask_new, axis=0)
    avg_comparison = np.mean(mask_comparison, axis=0)
    avg_diff = avg_new - avg_comparison
    metrics_all['Average_Value_New'] = avg_new
    metrics_all['Average_Value_Comparison'] = avg_comparison
    metrics_all['Average_Value_Difference'] = avg_diff
    
    # Initialize lists to store per mask metrics
    mae_list = []
    mse_list = []
    cos_sim_list = []
    kl_div_list = []
    
    for i in range(num_masks):
        mask1 = mask_new[:, i]
        mask2 = mask_comparison[:, i]
        mask_metrics = calculate_mask_metrics(mask1, mask2)
        mae_list.append(mask_metrics['MAE'])
        mse_list.append(mask_metrics['MSE'])
        cos_sim_list.append(mask_metrics['Cosine_Similarity'])
        kl_div_list.append(mask_metrics['KL_Divergence'])
    
    # Calculate average metrics across all mask positions
    metrics_all['Average_MAE'] = np.mean(mae_list)
    metrics_all['Average_MSE'] = np.mean(mse_list)
    metrics_all['Average_Cosine_Similarity'] = np.mean(cos_sim_list)
    metrics_all['Average_KL_Divergence'] = np.mean(kl_div_list)
    
    # Calculate per target output average (assuming 'target' refers to mask positions)
    metrics_all['Per_Mask_MAE'] = mae_list
    metrics_all['Per_Mask_MSE'] = mse_list
    metrics_all['Per_Mask_Cosine_Similarity'] = cos_sim_list
    metrics_all['Per_Mask_KL_Divergence'] = kl_div_list
    
    return metrics_all

def plot_target_histogram(targets_new, targets_comparison, output_path):
    """
    Plots histograms of targets for two datasets.

    Args:
        targets_new (numpy.ndarray): Targets from the new dataset.
        targets_comparison (numpy.ndarray): Targets from the comparison dataset.
        output_path (str): Path to save the histogram image.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(targets_new, bins=7, alpha=0.5, label='New Data', color='blue', edgecolor='black')
    plt.hist(targets_comparison, bins=7, alpha=0.5, label='Comparison Data', color='orange', edgecolor='black')
    plt.title('Histogram of Targets')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(output_path)
    plt.close()
    print(f"Histogram of targets saved to {output_path}")


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = test_truth_emo > 0
    predicted_label = test_preds_emo > 0
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n / p) + tn) / (2 * n)


def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
    test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

    mae = np.mean(
        np.absolute(test_preds - test_truth)
    )  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)

    f_score = f1_score((test_preds >= 0), (test_truth >= 0), average="weighted")
    binary_truth = test_truth >= 0
    binary_preds = test_preds >= 0

    f_score_neg = f1_score((test_preds > 0), (test_truth > 0), average="weighted")
    binary_truth_neg = test_truth > 0
    binary_preds_neg = test_preds > 0

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    f_score_non_zero = f1_score(
        (test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average="weighted"
    )
    binary_truth_non_zero = test_truth[non_zeros] > 0
    binary_preds_non_zero = test_preds[non_zeros] > 0

    return {
        "mae": mae,
        "corr": corr,
        "acc_7": mult_a7,
        "acc_5": mult_a5,
        "f1_pos": f_score,  # zeros are positive
        "bin_acc_pos": accuracy_score(binary_truth, binary_preds),  # zeros are positive
        "f1_neg": f_score_neg,  # zeros are negative
        "bin_acc_neg": accuracy_score(
            binary_truth_neg, binary_preds_neg
        ),  # zeros are negative
        "f1": f_score_non_zero,  # zeros are excluded
        "bin_acc": accuracy_score(
            binary_truth_non_zero, binary_preds_non_zero
        ),  # zeros are excluded
    }


def eval_mosei_senti_old(results, truths, exclude_zero=False):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    non_zeros = np.array(
        [i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)]
    )

    test_preds_a7 = np.clip(test_preds, a_min=-3.0, a_max=3.0)
    test_truth_a7 = np.clip(test_truth, a_min=-3.0, a_max=3.0)
    test_preds_a5 = np.clip(test_preds, a_min=-2.0, a_max=2.0)
    test_truth_a5 = np.clip(test_truth, a_min=-2.0, a_max=2.0)

    mae = np.mean(
        np.absolute(test_preds - test_truth)
    )  # Average L1 distance between preds and truths
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score(
        (test_preds[non_zeros] >= 0), (test_truth[non_zeros] >= 0), average="weighted"
    )
    binary_truth = test_truth[non_zeros] >= 0
    binary_preds = test_preds[non_zeros] >= 0
    f_score_neg = f1_score(
        (test_preds[non_zeros] > 0), (test_truth[non_zeros] > 0), average="weighted"
    )
    binary_truth_neg = test_truth[non_zeros] > 0
    binary_preds_neg = test_preds[non_zeros] > 0

    return {
        "mae": mae,
        "corr": corr,
        "acc_7": mult_a7,
        "acc_5": mult_a5,
        "f1": f_score,
        "f1_neg": f_score_neg,
        "bin_acc_neg": accuracy_score(binary_truth_neg, binary_preds_neg),
    }


def print_metrics(metrics):
    for k, v in metrics.items():
        print("{}:\t{}".format(k, v))


def save_metrics(metrics, results_file):
    with open(results_file, "w") as fd:
        for k, v in metrics.items():
            print("{}:\t{}".format(k, v), file=fd)


def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)


def eval_iemocap(results, truths, single=-1):
    emos = ["neutral", "happy", "sad", "angry"]
    test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
    test_truth = truths.view(-1, 4).cpu().detach().numpy()

    results = {}
    for emo_ind in range(4):
        test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
        test_truth_i = test_truth[:, emo_ind]
        f1 = f1_score(test_truth_i, test_preds_i, average="weighted")
        acc = accuracy_score(test_truth_i, test_preds_i)
        results["{}_acc".format(emos[emo_ind])] = acc
        results["{}_f1".format(emos[emo_ind])] = f1

    return results
