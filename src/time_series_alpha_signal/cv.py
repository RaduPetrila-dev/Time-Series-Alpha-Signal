"""Cross-validation splitters with purging and embargo.

This module provides two time-series-aware CV strategies that prevent
information leakage between train and test folds:

* :class:`PurgedKFold` -- standard k-fold with purging and optional embargo.
* :func:`combinatorial_purged_cv` -- enumerates all C(n, k) combinations
  of test folds (CPCV) with the same purging logic.

Both implementations follow the methodology described in:

    Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*.
    Wiley. Chapter 7.

Key concepts
------------
**Purging**
    Training observations whose lifespan ``[t0, t1]`` overlaps with any
    test observation are removed to prevent lookahead bias.

**Embargo**
    An additional buffer after the test window during which training
    observations are also excluded.  Expressed as a fraction of the
    total dataset span.
"""

from __future__ import annotations

import itertools
import logging
from typing import Iterator, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_events(events: pd.DataFrame) -> None:
    """Check that *events* has the required ``t0`` and ``t1`` columns.

    Raises
    ------
    KeyError
        If either column is missing.
    ValueError
        If the DataFrame is empty.
    """
    missing = {"t0", "t1"} - set(events.columns)
    if missing:
        raise KeyError(f"events DataFrame is missing columns: {missing}")
    if events.empty:
        raise ValueError("events DataFrame is empty.")


def _compute_fold_boundaries(
    n_events: int,
    n_splits: int,
) -> list[np.ndarray]:
    """Split ``range(n_events)`` into *n_splits* approximately equal folds.

    Parameters
    ----------
    n_events : int
        Total number of observations.
    n_splits : int
        Number of folds.

    Returns
    -------
    list of ndarray
        Each element contains the integer indices for one fold.
    """
    indices = np.arange(n_events)
    fold_sizes = np.full(n_splits, n_events // n_splits, dtype=int)
    fold_sizes[: n_events % n_splits] += 1
    boundaries = np.cumsum(fold_sizes)

    folds: list[np.ndarray] = []
    start = 0
    for stop in boundaries:
        folds.append(indices[start:stop])
        start = stop
    return folds


# ---------------------------------------------------------------------------
# Vectorised purge + embargo
# ---------------------------------------------------------------------------


def _purge_train_indices(
    events: pd.DataFrame,
    test_indices: np.ndarray,
    embargo_pct: float,
) -> np.ndarray:
    """Remove training observations that overlap with the test window.

    The purge is **vectorised**: instead of looping over every event in
    Python, we compute the test time boundaries once and use boolean
    array operations to build the training mask.

    An observation at row *i* is purged if its lifespan
    ``[t0_i, t1_i]`` overlaps with the expanded test window
    ``[test_start - embargo, test_end + embargo]``.

    Parameters
    ----------
    events : DataFrame
        Event table with ``t0`` (event start) and ``t1`` (event end)
        columns.
    test_indices : ndarray of int
        Row positions of the test observations.
    embargo_pct : float
        Fraction of the total dataset span to embargo after the test
        window.

    Returns
    -------
    ndarray of int
        Row positions of observations safe for training.
    """
    n_events = len(events)

    # Time boundaries of the test set
    test_t0 = events.iloc[test_indices]["t0"]
    test_t1 = events.iloc[test_indices]["t1"]
    test_start = test_t0.min()
    test_end = test_t1.max()

    # Embargo buffer
    if embargo_pct > 0:
        total_span = events["t1"].max() - events["t0"].min()
        embargo = total_span * embargo_pct
    else:
        embargo = pd.Timedelta(0)

    # Vectorised overlap check
    all_t0 = events["t0"].values
    all_t1 = events["t1"].values
    expanded_start = test_start - embargo
    expanded_end = test_end + embargo

    overlaps = (all_t0 <= expanded_end) & (all_t1 >= expanded_start)

    # Build mask: exclude test indices and overlapping events
    train_mask = np.ones(n_events, dtype=bool)
    train_mask[test_indices] = False
    train_mask[overlaps] = False

    train_indices = np.where(train_mask)[0]

    n_purged = n_events - len(test_indices) - len(train_indices)
    if n_purged > 0:
        logger.debug(
            "Purged %d observations (%.1f%% of training candidates).",
            n_purged,
            100.0 * n_purged / max(1, n_events - len(test_indices)),
        )

    return train_indices


# ---------------------------------------------------------------------------
# Event table construction helper
# ---------------------------------------------------------------------------


def make_event_table(
    index: pd.DatetimeIndex,
    horizon: int = 1,
) -> pd.DataFrame:
    """Build a simple event table from a datetime index.

    Each event spans from ``index[i]`` to ``index[i + horizon]``.  The
    last *horizon* rows are dropped because their end date would fall
    outside the index.

    This is a convenience function for constructing the ``events``
    DataFrame expected by :class:`PurgedKFold` and
    :func:`combinatorial_purged_cv` when you only have a price index
    and a fixed holding period.

    Parameters
    ----------
    index : DatetimeIndex
        Chronologically sorted datetime index (e.g. from a price
        DataFrame).
    horizon : int, default 1
        Number of index steps each event spans.

    Returns
    -------
    DataFrame
        Columns ``t0`` and ``t1`` with datetime values.

    Examples
    --------
    >>> idx = pd.bdate_range("2020-01-01", periods=10)
    >>> make_event_table(idx, horizon=2)
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    t0 = index[:-horizon]
    t1 = index[horizon:]
    return pd.DataFrame({"t0": t0, "t1": t1}).reset_index(drop=True)


# ---------------------------------------------------------------------------
# PurgedKFold
# ---------------------------------------------------------------------------


class PurgedKFold:
    """K-fold cross-validation with purging and optional embargo.

    Observations whose lifespan overlaps with the test fold (plus an
    optional embargo buffer) are removed from the training set to
    prevent information leakage.

    Parameters
    ----------
    n_splits : int, default 5
        Number of folds.  Must be >= 2.
    embargo_pct : float, default 0.0
        Fraction of the total dataset time span to embargo after the
        test window.  0.01 corresponds to 1% of the span.

    Raises
    ------
    ValueError
        If *n_splits* < 2 or *embargo_pct* is outside ``[0, 1)``.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.bdate_range("2020-01-01", periods=100)
    >>> events = make_event_table(idx, horizon=5)
    >>> cv = PurgedKFold(n_splits=5, embargo_pct=0.01)
    >>> for train_idx, test_idx in cv.split(events):
    ...     assert len(np.intersect1d(train_idx, test_idx)) == 0
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.0) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        if not (0.0 <= embargo_pct < 1.0):
            raise ValueError(
                f"embargo_pct must be in [0, 1), got {embargo_pct}"
            )
        self.n_splits = int(n_splits)
        self.embargo_pct = float(embargo_pct)

    def __repr__(self) -> str:
        return (
            f"PurgedKFold(n_splits={self.n_splits}, "
            f"embargo_pct={self.embargo_pct})"
        )

    def get_n_splits(self) -> int:
        """Return the number of folds."""
        return self.n_splits

    def split(
        self,
        events: pd.DataFrame,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate purged train/test index pairs.

        Parameters
        ----------
        events : DataFrame
            Must contain ``t0`` (event start) and ``t1`` (event end)
            columns.

        Yields
        ------
        train_indices : ndarray of int
            Training row positions (purged + embargoed).
        test_indices : ndarray of int
            Test row positions.
        """
        _validate_events(events)

        folds = _compute_fold_boundaries(len(events), self.n_splits)

        for fold_idx, test_indices in enumerate(folds):
            train_indices = _purge_train_indices(
                events, test_indices, self.embargo_pct
            )
            logger.debug(
                "Fold %d/%d: train=%d, test=%d",
                fold_idx + 1,
                self.n_splits,
                len(train_indices),
                len(test_indices),
            )
            yield train_indices, test_indices


# ---------------------------------------------------------------------------
# Combinatorial purged CV
# ---------------------------------------------------------------------------


def combinatorial_purged_cv(
    events: pd.DataFrame,
    n_splits: int = 5,
    test_fold_size: int = 1,
    embargo_pct: float = 0.0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    r"""Generate combinatorial purged cross-validation splits (CPCV).

    Enumerates all :math:`\binom{n\_splits}{test\_fold\_size}`
    combinations of test folds.  For each combination the union of
    selected folds forms the test set; remaining observations (minus
    purged and embargoed rows) form the training set.

    Parameters
    ----------
    events : DataFrame
        Event table with ``t0`` and ``t1`` columns.
    n_splits : int, default 5
        Total number of folds.
    test_fold_size : int, default 1
        Number of folds in each test set.
    embargo_pct : float, default 0.0
        Embargo fraction (see :class:`PurgedKFold`).

    Yields
    ------
    train_indices : ndarray of int
        Training row positions.
    test_indices : ndarray of int
        Test row positions.

    Raises
    ------
    ValueError
        If *test_fold_size* is not in ``[1, n_splits - 1]``.

    Notes
    -----
    The number of splits grows as :math:`\binom{n}{k}`.  For
    ``n_splits=10, test_fold_size=2`` this produces 45 splits, which
    is manageable.  Larger values grow quickly.
    """
    if test_fold_size < 1 or test_fold_size >= n_splits:
        raise ValueError(
            f"test_fold_size must be in [1, n_splits-1], "
            f"got {test_fold_size} with n_splits={n_splits}"
        )

    _validate_events(events)

    folds = _compute_fold_boundaries(len(events), n_splits)
    n_combos = len(list(itertools.combinations(range(n_splits), test_fold_size)))
    logger.info(
        "CPCV: %d splits, test_fold_size=%d -> %d combinations.",
        n_splits,
        test_fold_size,
        n_combos,
    )

    for combo_idx, combo in enumerate(
        itertools.combinations(range(n_splits), test_fold_size)
    ):
        test_indices = np.concatenate([folds[i] for i in combo])
        train_indices = _purge_train_indices(
            events, test_indices, embargo_pct
        )
        logger.debug(
            "CPCV combo %d/%d %s: train=%d, test=%d",
            combo_idx + 1,
            n_combos,
            combo,
            len(train_indices),
            len(test_indices),
        )
        yield train_indices, test_indices
