from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd


class PurgedKFold:
    """K‑fold cross‑validation with purging and an optional embargo.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds to create.  Must be at least 2.
    embargo_pct : float, default 0.0
        Fraction of the dataset to embargo after each test fold.  An
        embargo of 0.01 corresponds to skipping 1% of the data after
        the end of the test window for the training set.  Set to 0 to
        disable the embargo.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.0) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not (0.0 <= embargo_pct < 1.0):
            raise ValueError("embargo_pct must be in [0, 1)")
        self.n_splits = int(n_splits)
        self.embargo_pct = float(embargo_pct)

    def split(self, events: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices with purging and embargo.

        Parameters
        ----------
        events : DataFrame
            Event table with at least ``t0`` and ``t1`` columns.  The
            ordering of rows defines the chronological order of events.

        Yields
        ------
        train_indices : ndarray of int
            Indices of events for training, purged of overlaps with the
            current test fold and subject to an optional embargo.
        test_indices : ndarray of int
            Indices of events comprising the current test fold.
        """
        if "t0" not in events.columns or "t1" not in events.columns:
            raise KeyError("events must contain 't0' and 't1' columns")
        n_events = len(events)
        indices = np.arange(n_events)
        # determine fold sizes for approximate equal splits
        fold_sizes = np.full(self.n_splits, n_events // self.n_splits, dtype=int)
        fold_sizes[: n_events % self.n_splits] += 1
        # cumulative boundaries to slice indices
        boundaries = np.cumsum(fold_sizes)
        start = 0
        for fold_size, stop in zip(fold_sizes, boundaries):
            test_idx = indices[start:stop]
            start = stop
            # compute time boundaries for test
            test_start = events.iloc[test_idx]["t0"].min()
            test_end = events.iloc[test_idx]["t1"].max()
            # compute embargo period length
            if self.embargo_pct > 0:
                span = events["t1"].max() - events["t0"].min()
                embargo = span * self.embargo_pct
            else:
                embargo = pd.Timedelta(0)
            # build train indices skipping overlapping and embargoed events
            train_mask = np.ones(n_events, dtype=bool)
            train_mask[test_idx] = False
            for i in indices:
                if not train_mask[i]:
                    continue
                ev_t0 = events.iloc[i]["t0"]
                ev_t1 = events.iloc[i]["t1"]
                # drop events overlapping test or within embargo window
                if (ev_t0 <= test_end + embargo) and (ev_t1 >= test_start - embargo):
                    train_mask[i] = False
            train_idx = indices[train_mask]
            yield train_idx, test_idx


def combinatorial_purged_cv(
    events: pd.DataFrame,
    n_splits: int = 5,
    test_fold_size: int = 1,
    embargo_pct: float = 0.0,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Generate combinatorial purged cross‑validation splits.

    This function enumerates all combinations of ``test_fold_size`` out
    of ``n_splits`` possible folds.  For each combination, the union of
    the selected folds constitutes the test set.  The remaining folds
    are used for training, with purging and embargo applied as in
    :class:`PurgedKFold`.  The number of combinations grows
    combinatorially, so choose small values for ``n_splits`` and
    ``test_fold_size``.

    Parameters
    ----------
    events : DataFrame
        Event table with ``t0`` and ``t1`` columns.
    n_splits : int, default 5
        Total number of folds to divide the data into.
    test_fold_size : int, default 1
        Number of folds to include in the test set for each split.
    embargo_pct : float, default 0.0
        Fraction of the dataset to embargo after the test window.

    Yields
    ------
    train_indices : ndarray of int
        Indices for training.
    test_indices : ndarray of int
        Indices for testing.
    """
    if test_fold_size < 1 or test_fold_size >= n_splits:
        raise ValueError("test_fold_size must be between 1 and n_splits - 1")
    # build simple k‑fold boundaries
    n_events = len(events)
    indices = np.arange(n_events)
    fold_sizes = np.full(n_splits, n_events // n_splits, dtype=int)
    fold_sizes[: n_events % n_splits] += 1
    boundaries = np.cumsum(fold_sizes)
    # compute fold index arrays
    fold_indices: List[np.ndarray] = []
    start = 0
    for fold_size, stop in zip(fold_sizes, boundaries):
        fold_indices.append(indices[start:stop])
        start = stop
    # generate combinations of test folds
    import itertools

    for combo in itertools.combinations(range(n_splits), test_fold_size):
        test_idx = np.concatenate([fold_indices[i] for i in combo])
        # compute time boundaries for test set
        test_start = events.iloc[test_idx]["t0"].min()
        test_end = events.iloc[test_idx]["t1"].max()
        # embargo period
        if embargo_pct > 0:
            span = events["t1"].max() - events["t0"].min()
            embargo = span * embargo_pct
        else:
            embargo = pd.Timedelta(0)
        train_mask = np.ones(n_events, dtype=bool)
        train_mask[test_idx] = False
        for i in indices:
            if not train_mask[i]:
                continue
            ev_t0 = events.iloc[i]["t0"]
            ev_t1 = events.iloc[i]["t1"]
            if (ev_t0 <= test_end + embargo) and (ev_t1 >= test_start - embargo):
                train_mask[i] = False
        train_idx = indices[train_mask]
        yield train_idx, test_idx
