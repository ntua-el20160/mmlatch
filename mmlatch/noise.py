import numpy as np
import copy

def add_gaussian_noise(features, sigma=0.01):
    """
    Adds gasussian noise to the feature vectors of a modality.
    sigma = noise_level
    """
    noise = np.random.normal(0, sigma, features.shape)
    return features + noise

def add_dropout_noise(features, rate=0.5):
    """
    Adds dropout noise to the feature vectors of a modality.
    rate = noise_level
    """
    mask = np.random.binomial(1, 1-rate, features.shape)
    return features * mask

def shuffle_modalities(dataset, modality_to_shuffle="all", shuffle_prob=0.1):
    """
    Shuffles modalities between samples to create out-of-context combinations.

    Args:
        dataset (list): List of samples, where each sample is a dict with keys "text", "audio", and "visual".
        modality_to_shuffle (str or list): Modality to shuffle. Options: "all", "text", "audio", "visual", or a list of modalities.
        shuffle_prob (float): Probability of shuffling a modality for a given sample.

    Returns:
        list: Dataset with shuffled modalities.
    """

    # Determine which modalities to shuffle
    if modality_to_shuffle == "all":
        modalities = ["text", "audio", "visual"]
    else:
        modalities = [modality_to_shuffle]

    # Shuffle each modality
    for modality in modalities:
        # Collect all features for this modality across the dataset
        modality_features = [sample[modality] for sample in dataset]

        # Shuffle the features
        np.random.shuffle(modality_features)

        # Apply shuffled features to samples based on shuffle_prob
        for i, sample in enumerate(dataset):
            if np.random.rand() < shuffle_prob:
                original = sample[modality]
                new_feat = modality_features[i]
                orig_len = original.shape[0]
                new_len = new_feat.shape[0]

                # If new feature is shorter, pad it with zeros.
                if new_len < orig_len:
                    pad_width = ((0, orig_len - new_len),) + ((0, 0),) * (new_feat.ndim - 1)
                    new_feat = np.pad(new_feat, pad_width, mode='constant')
                # If new feature is longer, cut it to match the original length.
                elif new_len > orig_len:
                    new_feat = new_feat[:orig_len]

                sample[modality] = new_feat

    return dataset

def add_noise(dataset, noise_type='none', noise_modality='all', noise_level=0):
    """
    Adds noise to the selected dataset (test/train)
    noise_type = 'none', 'gaussian', 'dropout', 'shuffle'
    noise_modality = 'all', 'text', 'audio', 'visual' (note that 'text'=='glove')
    modifies dataset by reference
    """
    if noise_type == 'none':
        return dataset

    ds = copy.deepcopy(dataset)
    modalities = ['text', 'audio', 'visual']
    if noise_type == 'gaussian':
        for i in range(len(ds)):
            if noise_modality == 'all':
                for modality in modalities:
                    ds[i][modality] = add_gaussian_noise(ds[i][modality], noise_level)
            else:
                ds[i][noise_modality] = add_gaussian_noise(ds[i][noise_modality], noise_level)

    elif noise_type == 'dropout':
        for i in range(len(ds)):
            if noise_modality == 'all':
                for modality in modalities:
                    ds[i][modality] = add_dropout_noise(ds[i][modality], noise_level)
            else:
                ds[i][noise_modality] = add_dropout_noise(ds[i][noise_modality], noise_level)

    elif noise_type == 'shuffle':
        ds = shuffle_modalities(ds, modality_to_shuffle=noise_modality, shuffle_prob=noise_level)
    
    elif noise_type == 'all':
        noise_level_gaussian  = noise_level[0]
        noise_level_dropout = noise_level[1]
        noise_level_shuffle = noise_level[2]
        for i in range(len(ds)):
            if noise_modality == 'all':
                for modality in modalities:
                    ds[i][modality] = add_gaussian_noise(ds[i][modality], noise_level_gaussian)
                    ds[i][modality] = add_dropout_noise(ds[i][modality], noise_level_dropout)
            else:
                ds[i][noise_modality] = add_gaussian_noise(ds[i][noise_modality], noise_level_gaussian)
                ds[i][noise_modality] = add_dropout_noise(ds[i][noise_modality], noise_level_dropout)
        ds = shuffle_modalities(ds, modality_to_shuffle=noise_modality, shuffle_prob=noise_level_shuffle)

    return ds