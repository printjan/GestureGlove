# src/data_fusion_project/processing/splits.py
"""
Train/validation/test splitting helpers for :class:`GestureDataset`.

For gesture recognition the most honest evaluation is *leave-session-out*: no window from a
session appears in both train and test. This prevents the CNN from exploiting
session-specific quirks (mounting, drift, person) and reflects real deployment, where the
model sees a brand-new recording. A simple stratified random split is also provided for
quick experiments.

Both helpers return index arrays so the caller stays in control of how the
:class:`GestureDataset` arrays are sliced.
"""

from __future__ import annotations

import numpy as np

from data_fusion_project.core.logger_setup import get_logger

logger = get_logger(__name__)


# ======================================================================================================================
# Index-based splits
# ======================================================================================================================
def leave_sessions_out(groups: np.ndarray, test_fraction: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits sample indices by session so that whole sessions go entirely to train or test.
    :param: groups (np.ndarray): session id per sample, shape (N,).
    :param: test_fraction (float): approximate fraction of *sessions* held out for test.
    :param: seed (int): RNG seed for reproducible session assignment.
    :return: indices (tuple): (train_idx, test_idx) integer arrays.
    """
    unique = np.array(sorted(set(groups.tolist())), dtype=object)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    n_test = max(1, int(round(len(unique) * test_fraction)))
    test_sessions = set(unique[order[:n_test]].tolist())

    test_mask = np.array([g in test_sessions for g in groups])
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]
    logger.info("Leave-sessions-out: %d train / %d test samples (%d/%d sessions held out).",
                len(train_idx), len(test_idx), n_test, len(unique))
    return train_idx, test_idx


def stratified_split(y: np.ndarray, test_fraction: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits sample indices randomly while preserving the per-class ratio.
    :param: y (np.ndarray): integer labels, shape (N,).
    :param: test_fraction (float): fraction of samples per class held out for test.
    :param: seed (int): RNG seed for reproducibility.
    :return: indices (tuple): (train_idx, test_idx) integer arrays.
    """
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_test = max(1, int(round(len(idx) * test_fraction)))
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def chronological_split(y: np.ndarray, test_fraction: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits sample indices chronologically per class (first part for train, last part for test)
    to prevent temporal leakage on overlapping sliding windows.
    :param: y (np.ndarray): integer labels, shape (N,).
    :param: test_fraction (float): fraction of samples per class held out for test.
    :return: indices (tuple): (train_idx, test_idx) integer arrays.
    """
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        # Since samples are loaded sequentially in dataset.py, they are already sorted chronologically
        n_test = max(1, int(round(len(idx) * test_fraction)))
        n_train = len(idx) - n_test
        train_idx.extend(idx[:n_train].tolist())
        test_idx.extend(idx[n_train:].tolist())
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


# ======================================================================================================================
# Three-way disjoint splits (Train / Validation / Test)
# ======================================================================================================================
def leave_sessions_out_three_way(
    groups: np.ndarray,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits sample indices by session into three disjoint subsets (train, val, test).
    
    :param: groups (np.ndarray): session id per sample, shape (N,).
    :param: test_fraction (float): fraction of sessions held out for final test.
    :param: val_fraction (float): fraction of sessions held out for val.
    :param: seed (int): RNG seed for reproducible session assignment.
    :return: indices (tuple): (train_idx, val_idx, test_idx) disjoint integer arrays.
    """
    unique = np.array(sorted(set(groups.tolist())), dtype=object)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    
    n_test = max(1, int(round(len(unique) * test_fraction)))
    n_val = max(1, int(round(len(unique) * val_fraction)))
    
    if n_test + n_val >= len(unique):
        # Fallback for datasets with very few unique sessions
        n_test = max(1, len(unique) // 3)
        n_val = max(1, len(unique) // 3)
        
    test_sessions = set(unique[order[:n_test]].tolist())
    val_sessions = set(unique[order[n_test : n_test + n_val]].tolist())
    
    test_mask = np.array([g in test_sessions for g in groups])
    val_mask = np.array([g in val_sessions for g in groups])
    
    test_idx = np.where(test_mask)[0]
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~test_mask & ~val_mask)[0]
    
    logger.info("Leave-sessions-out (3-way): %d train / %d val / %d test samples (%d/%d/%d sessions split).",
                len(train_idx), len(val_idx), len(test_idx),
                len(unique) - n_test - n_val, n_val, n_test)
    return train_idx, val_idx, test_idx


def stratified_split_three_way(
    y: np.ndarray,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits sample indices randomly into three disjoint subsets while preserving per-class ratios.
    
    :param: y (np.ndarray): integer labels, shape (N,).
    :param: test_fraction (float): fraction of samples per class held out for test.
    :param: val_fraction (float): fraction of samples per class held out for val.
    :param: seed (int): RNG seed for reproducibility.
    :return: indices (tuple): (train_idx, val_idx, test_idx) disjoint integer arrays.
    """
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        
        n_test = max(1, int(round(len(idx) * test_fraction)))
        n_val = max(1, int(round(len(idx) * val_fraction)))
        
        test_idx.extend(idx[:n_test].tolist())
        val_idx.extend(idx[n_test : n_test + n_val].tolist())
        train_idx.extend(idx[n_test + n_val:].tolist())
        
    return np.array(sorted(train_idx)), np.array(sorted(val_idx)), np.array(sorted(test_idx))


def chronological_split_three_way(
    y: np.ndarray,
    test_fraction: float = 0.2,
    val_fraction: float = 0.1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits sample indices chronologically per class into three sequential disjoint subsets:
    [Train (first part) | Validation (middle part) | Test (last part)].
    
    :param: y (np.ndarray): integer labels, shape (N,).
    :param: test_fraction (float): fraction of samples held out for final test.
    :param: val_fraction (float): fraction of samples held out for validation.
    :return: indices (tuple): (train_idx, val_idx, test_idx) disjoint integer arrays.
    """
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        n_test = max(1, int(round(len(idx) * test_fraction)))
        n_val = max(1, int(round(len(idx) * val_fraction)))
        n_train = len(idx) - n_test - n_val
        
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val:].tolist())
        
    return np.array(sorted(train_idx)), np.array(sorted(val_idx)), np.array(sorted(test_idx))
