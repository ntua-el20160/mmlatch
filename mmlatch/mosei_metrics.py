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


import torch

import pickle
import torch

def save_comparison_data_pickle(filepath, predictions, targets, masks_txt, masks_au, masks_vi):
    """
    Saves predictions, targets, and masks to a pickle file.

    Args:
        filepath (str): Path to save the pickle file.
        predictions (List[torch.Tensor]): Model predictions.
        targets (List[torch.Tensor]): Ground truth targets.
        masks_txt (List[torch.Tensor]): Text masks.
        masks_au (List[torch.Tensor]): Audio masks.
        masks_vi (List[torch.Tensor]): Visual masks.
    """
    data = {
        'predictions': predictions.cpu().numpy(),  # Concatenate if applicable
        'targets': targets.cpu().numpy(),          # Concatenate if applicable
        'masks_txt': [mask.cpu().numpy() for mask in masks_txt],     # List of NumPy arrays
        'masks_au': [mask.cpu().numpy() for mask in masks_au],       # List of NumPy arrays
        'masks_vi': [mask.cpu().numpy() for mask in masks_vi],       # List of NumPy arrays
    }
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"Comparison data saved to {filepath}")


def load_comparison_data_pickle(filepath):
    """
    Loads predictions, targets, and masks from a pickle file.

    Args:
        filepath (str): Path to the pickle file.

    Returns:
        dict: A dictionary containing predictions, targets, and masks.
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

# Usage
#data = load_comparison_data_pickle('comparison_data.pkl')
#predictions = data['predictions']
#targets = data['targets']
#masks_txt = data['masks_txt']
#masks_au = data['masks_au']
#masks_vi = data['masks_vi']



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

def compare_masks(data_new, data_comparison_link):
    """Function to compare masks between two models."""
    metrics_all = {} #stores each of the 4 metrics per modality
   
    #seperates the masks by the target that their input should have been
    masks_per_target_new = {}
    masks_per_target_comp = {}

    #mean value of masks (2d array) per modality
    mean_mask_new = {}

    #the difference of mean between the new data to the comparison data per modality 
    diff_mean_mask_new = {}

    #similarly to before but grouped per target
    mean_mask_new_target = {}
    diff_mean_mask_new_target = {}
    

    targets = data_new['targets']
    data = load_comparison_data_pickle(data_comparison_link)

    for modality in ["txt", "au", "vi"]:
        print(f"Comparing masks for modality '{modality}'")
        masks_transformed_comp = []
        masks_transformed_new = []

        #get the masks for the new data and the comparison data
        mask_new = data_new.get(f'masks_{modality}', [])
        mask_comparison = data.get(f'masks_{modality}', [])


        # Check if both mask lists have the same length
        if len(mask_new) != len(mask_comparison):
            raise ValueError(
            f"Number of masks for modality '{modality}' does not match between datasets. "
            f"data_new has {len(mask_new)} masks, "
            f"data_comparison has {len(mask_comparison)} masks."
        )

        for i in range(len(mask_new)):
            #calculate the metrics for each mask
            mask_metrics = calculate_mask_metrics(mask_new[i], mask_comparison[i])
            
            # store the metrics for each mask
            for metric_name, metric_value in mask_metrics.items():
                key = f'{modality}_{metric_name}'
                if key not in metrics_all:
                    metrics_all[key] = []
                metrics_all[key].append(metric_value) 
            
            # Organize the new and comparison masks per target
            key_target = f'{modality}_{targets[i]}'

            # Initialize lists if keys do not exist
            if key_target not in masks_per_target_new:
                masks_per_target_new[key_target] = []
            if key_target not in masks_per_target_comp:
                masks_per_target_comp[key_target] = []
            
            mask_flat_new = np.mean(mask_new[i], axis=(0, 1))
            mask_flat_comp = np.mean(mask_comparison[i], axis=(0, 1))
            masks_transformed_new.append(mask_flat_new)
            masks_transformed_comp.append(mask_flat_comp)
            masks_per_target_new[key_target].append(mask_flat_new)
            masks_per_target_comp[key_target].append(mask_flat_comp)

        # Compute the mean and difference mask for this modality
        mean_mask_new[modality] = np.mean(masks_transformed_new, axis=0)
        diff_mean_mask_new[modality] = np.mean(masks_transformed_new, axis=0) - np.mean(masks_transformed_comp, axis=0)

    # the mean and difference mask for this modality per target
    for key,value in masks_per_target_new.items():
        mean_mask_new_target[key] = np.mean(value, axis=0)
        diff_mean_mask_new_target[key] = np.mean(value, axis=0) - np.mean(masks_per_target_comp[key], axis=0)
    
    average_metrics = {key: np.mean(values) for key, values in metrics_all.items()}

        
    return average_metrics,mean_mask_new,diff_mean_mask_new,mean_mask_new_target,diff_mean_mask_new_target
    
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from typing import Dict, Any

def plot_masks(mask_dict,description,save_directory):

    # Create the save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    for key, mask in mask_dict.items():
        # Plotting the mask
        plt.figure(figsize=(6, 5))
        plt.imshow(mask, cmap='viridis', aspect='auto')
        plt.colorbar()
        
        # Split the key to extract modality and target if applicable
        if '_' in key:
            modality, target = key.split('_', 1)
            title = f"{description.replace('_', ' ').capitalize()} - Modality: {modality}, Target: {target}"
        else:
            modality = key
            title = f"{description.replace('_', ' ').capitalize()} - Modality: {modality}"
        
        plt.title(title)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.tight_layout()
        plt.show()  # Display the plot

    # Save the mask dictionary using pickle
    save_path = os.path.join(save_directory, f"{description}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(mask_dict, f)
    print(f"Mask data saved to {save_path}")


def prediction_count(data_new, data_comparison_link):
    
    data = load_comparison_data_pickle(data_comparison_link)
    predictions_new = data_new['predictions']
    predictions_comparison = data['predictions']
    targets = data_new['targets']
    predictions_distr_new = {}
    predictions_distr_comparison = {}
    targets_distr = {}
    for i in range(1,8):
        predictions_distr_new[i] = 0
        predictions_distr_comparison[i] = 0
        targets_distr[i] = 0
    

    for i in range(len(predictions_new)):
        targets_distr[targets[i]] += 1
        predictions_distr_new[predictions_new[i]] += 1
        predictions_distr_comparison[predictions_comparison[i]] += 1
    return predictions_distr_new,predictions_distr_comparison,targets_distr

import matplotlib.pyplot as plt
import pickle
import os
from typing import Dict

def save_histogram_data(predictions_distr_new,predictions_distr_comparison,targets_distr,save_directory,experiment_name):
    """
    Plots histograms for prediction distributions and saves the distribution dictionaries.

    Args:
        predictions_distr_new (Dict[int, int]): Distribution of new predictions.
        predictions_distr_comparison (Dict[int, int]): Distribution of comparison predictions.
        targets_distr (Dict[int, int]): Distribution of targets.
        save_directory (str): Directory where distribution data will be saved.
    """
    import os

    # Create the save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Define the bins (assuming predictions and targets are integers from 1 to 7)
    bins = range(1, 9)  # To include 7 as the last bin

    # Plotting New Predictions
    plt.figure(figsize=(8, 6))
    plt.hist(predictions_distr_new.keys(), bins=bins, weights=predictions_distr_new.values(), alpha=0.5, label='New Predictions', color='blue', edgecolor='black')
    plt.hist(predictions_distr_comparison.keys(), bins=bins, weights=predictions_distr_comparison.values(), alpha=0.5, label='Comparison Predictions', color='green', edgecolor='black')
    plt.hist(targets_distr.keys(), bins=bins, weights=targets_distr.values(), alpha=0.5, label='Targets', color='red', edgecolor='black')
    plt.xlabel('Prediction/Target Class')
    plt.ylabel('Count')
    plt.title('Prediction and Target Distributions')
    plt.legend(loc='upper right')
    plt.xticks(bins[:-1])  # Set x-ticks to class labels
    plt.tight_layout()
    plt.show()  # Display the histogram

    # Save the distribution dictionaries using pickle
    distribution_data = {
        'predictions_distr_new': predictions_distr_new,
        'predictions_distr_comparison': predictions_distr_comparison,
        'targets_distr': targets_distr
    }
    save_path = os.path.join(save_directory, f"prediction_target_distributions_{experiment_name}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(distribution_data, f)
    print(f"Histogram distribution data saved to {save_path}")


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
