"""Meta-labelling model training and cross-validated evaluation.

This module trains a logistic regression classifier to predict
**meta-labels** -- binary indicators of whether a base signal's
directional forecast was correct.  The idea, from Lopez de Prado
(2018, Chapter 3), is to separate the *side* decision (long/short)
from the *sizing* decision (how much to bet).

The workflow is:

1. Generate triple-barrier labels from the price series.
2. Compute realised returns for each event.
3. Derive meta-labels (1 = base signal was correct, 0 = wrong).
4. Build features (momentum, volatility, mean-reversion).
5. Fit a logistic regression per CV fold and report accuracy.

The classifier uses ``scikit-learn``'s :class:`LogisticRegression`
rather than ``statsmodels.Logit`` for consistency with the broader
ML ecosystem and simpler handling of regularisation and edge cases.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from .cv import PurgedKFold
from .labels import daily_volatility, meta_label, triple_barrier_labels

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaModelResult:
    """Structured result from :func:`train_meta_model`.

    Attributes
    ----------
    cv_accuracy_mean : float
        Mean cross-validated accuracy across folds.
    cv_accuracy_std : float
        Standard deviation of fold accuracies.
    fold_accuracies : list[float]
        Per-fold accuracy values.
    n_samples : int
        Number of valid samples used for training.
    n_positive : int
        Number of positive meta-labels (base signal correct).
    n_negative : int
        Number of negative meta-labels (base signal wrong).
    """

    cv_accuracy_mean: float
    cv_accuracy_std: float
    fold_accuracies: list[float]
    n_samples: int
    n_positive: int
    n_negative: int

    def to_dict(self) -> dict[str, float | int | list[float]]:
        """Convert to a JSON-serialisable dictionary."""
        return {
            "cv_accuracy_mean": self.cv_accuracy_mean,
            "cv_accuracy_std": self.cv_accuracy_std,
            "fold_accuracies": self.fold_accuracies,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
        }


# ---------------------------------------------------------------------------
# Realised returns (vectorised)
# ---------------------------------------------------------------------------


def _compute_realized_returns(
    prices: pd.Series,
    events: pd.DataFrame,
) -> pd.Series:
    """Compute the simple return between event start and end.

    Parameters
    ----------
    prices : Series
        Full price series.
    events : DataFrame
        Must contain ``t1`` column.  Index = event start ``t0``.

    Returns
    -------
    Series
        Realised returns aligned to the event index.
    """
    t0_prices = prices.reindex(events.index)

    # For t1, use reindex with nearest-backward fill for dates that
    # fall outside the price index.
    t1_dates = events["t1"]
    t1_prices = prices.reindex(t1_dates)

    # Where t1 is not in the index, use the last available price
    t1_prices_filled = t1_prices.values.copy()
    missing = np.isnan(t1_prices_filled) | pd.isna(t1_dates.values)
    if missing.any():
        t1_prices_filled[missing] = float(prices.iloc[-1])

    t0_vals = t0_prices.values.astype(np.float64)
    t1_vals = np.array(t1_prices_filled, dtype=np.float64)

    # Guard against division by zero
    safe_t0 = np.where(t0_vals == 0, np.nan, t0_vals)
    returns = (t1_vals - t0_vals) / safe_t0

    return pd.Series(returns, index=events.index, dtype=float)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def build_features(
    prices: pd.Series,
    lookback: int = 20,
    vol_span: int = 50,
) -> pd.DataFrame:
    """Construct features for the meta-labelling classifier.

    Features (all lagged by one period to avoid lookahead):

    * ``momentum`` -- percentage change over *lookback* periods.
    * ``volatility`` -- exponentially-weighted rolling std of returns.
    * ``mean_rev`` -- z-score of price relative to its rolling mean.

    Parameters
    ----------
    prices : Series
        Price series.
    lookback : int, default 20
        Momentum lookback window.
    vol_span : int, default 50
        EWM span for volatility feature.

    Returns
    -------
    DataFrame
        Feature matrix indexed by datetime.  All features are lagged
        by one period.

    Raises
    ------
    ValueError
        If *lookback* < 1.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")

    returns = prices.pct_change()

    momentum = prices.pct_change(lookback).shift(1)
    volatility = returns.ewm(span=vol_span, adjust=False).std().shift(1)

    rolling_mean = prices.rolling(lookback).mean()
    rolling_std = prices.rolling(lookback).std()
    mean_rev = ((prices - rolling_mean) / rolling_std.replace(0, np.nan)).shift(1)

    features = pd.DataFrame(
        {
            "momentum": momentum,
            "volatility": volatility,
            "mean_rev": mean_rev,
        },
        index=prices.index,
    )

    return features


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_meta_model(
    prices: pd.Series,
    lookback: int = 20,
    vol_span: int = 50,
    pt_sl: Tuple[float, float] = (1.0, 1.0),
    horizon: int = 5,
    n_splits: int = 5,
    embargo_pct: float = 0.0,
) -> dict[str, float]:
    """Train and evaluate a logistic meta-model on a single asset.

    Builds features and meta-labels from the price series, then fits
    a logistic regression using purged k-fold cross-validation.
    Returns classification accuracy statistics.

    Parameters
    ----------
    prices : Series
        Price series indexed by datetime.
    lookback : int, default 20
        Momentum feature lookback.
    vol_span : int, default 50
        EWM span for volatility estimation.
    pt_sl : tuple of float, default (1.0, 1.0)
        ``(profit_taking_mult, stop_loss_mult)`` for triple-barrier.
    horizon : int, default 5
        Vertical barrier in index steps.
    n_splits : int, default 5
        Number of purged CV folds.
    embargo_pct : float, default 0.0
        Embargo fraction between folds.

    Returns
    -------
    dict
        JSON-serialisable dictionary with ``cv_accuracy_mean``,
        ``cv_accuracy_std``, ``fold_accuracies``, ``n_samples``,
        ``n_positive``, ``n_negative``.  If insufficient data, values
        are ``NaN``.
    """
    # Lazy import: sklearn is only needed here
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required for train_meta_model. "
            "Install with: pip install scikit-learn"
        ) from exc

    # -- Labels ----------------------------------------------------------
    vol = daily_volatility(prices, span=vol_span)
    events = triple_barrier_labels(
        prices, vol=vol, pt_sl=pt_sl, horizon=horizon, vol_span=vol_span
    )

    if events.empty:
        logger.warning("No valid events generated. Returning NaN metrics.")
        return MetaModelResult(
            cv_accuracy_mean=float("nan"),
            cv_accuracy_std=float("nan"),
            fold_accuracies=[],
            n_samples=0,
            n_positive=0,
            n_negative=0,
        ).to_dict()

    realized_returns = _compute_realized_returns(prices, events)
    base_labels = events["y"]
    meta_labels = meta_label(base_labels, realized_returns)

    # -- Features --------------------------------------------------------
    features = build_features(prices, lookback=lookback, vol_span=vol_span)
    X_df = features.reindex(meta_labels.index)

    # Drop rows with any NaN in features or labels
    mask = X_df.notna().all(axis=1) & meta_labels.notna()
    n_valid = int(mask.sum())

    if n_valid < n_splits * 2:
        logger.warning(
            "Only %d valid samples (need >= %d for %d folds). "
            "Returning NaN metrics.",
            n_valid,
            n_splits * 2,
            n_splits,
        )
        return MetaModelResult(
            cv_accuracy_mean=float("nan"),
            cv_accuracy_std=float("nan"),
            fold_accuracies=[],
            n_samples=n_valid,
            n_positive=int((meta_labels[mask] == 1).sum()),
            n_negative=int((meta_labels[mask] == 0).sum()),
        ).to_dict()

    X = X_df.loc[mask].values.astype(np.float64)
    y = meta_labels.loc[mask].values.astype(int)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    logger.info(
        "Meta-model: %d samples, %d positive (%.1f%%), %d features.",
        n_valid,
        n_pos,
        100.0 * n_pos / n_valid,
        X.shape[1],
    )

    # -- CV events table -------------------------------------------------
    filtered_index = meta_labels.index[mask]
    cv_events = pd.DataFrame({
        "t0": filtered_index,
        "t1": events.loc[filtered_index, "t1"],
    })

    # -- Purged cross-validation -----------------------------------------
    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    accuracies: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(cv_events)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Skip degenerate folds (single class in training)
        if len(np.unique(y_train)) < 2:
            logger.debug(
                "Fold %d: single class in training set, skipping.", fold_idx
            )
            continue

        # Standardise features (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit logistic regression with L2 regularisation
        try:
            model = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        except Exception as exc:
            logger.warning(
                "Fold %d: model fitting failed (%s). Skipping.",
                fold_idx,
                exc,
            )
            continue

        acc = float((preds == y_test).mean())
        accuracies.append(acc)
        logger.debug(
            "Fold %d: train=%d, test=%d, accuracy=%.3f",
            fold_idx,
            len(train_idx),
            len(test_idx),
            acc,
        )

    if len(accuracies) == 0:
        logger.warning("All folds failed. Returning NaN metrics.")
        return MetaModelResult(
            cv_accuracy_mean=float("nan"),
            cv_accuracy_std=float("nan"),
            fold_accuracies=[],
            n_samples=n_valid,
            n_positive=n_pos,
            n_negative=n_neg,
        ).to_dict()

    result = MetaModelResult(
        cv_accuracy_mean=float(np.mean(accuracies)),
        cv_accuracy_std=float(np.std(accuracies, ddof=0)),
        fold_accuracies=accuracies,
        n_samples=n_valid,
        n_positive=n_pos,
        n_negative=n_neg,
    )

    logger.info(
        "Meta-model CV: accuracy=%.3f +/- %.3f (%d folds).",
        result.cv_accuracy_mean,
        result.cv_accuracy_std,
        len(accuracies),
    )

    return result.to_dict()
