from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.api import Logit

from .cv import PurgedKFold
from .labels import daily_volatility, triple_barrier_labels, meta_label


def _compute_realized_returns(prices: pd.Series, events: pd.DataFrame) -> pd.Series:
    """Compute the realised returns for each event.

    Given a price series and an events table returned by
    :func:`~time_series_alpha_signal.labels.triple_barrier_labels`,
    calculate the simple return between the event start ``t0`` and the
    event end ``t1``.  If ``t1`` is missing the return is set to zero.

    Parameters
    ----------
    prices : Series
        Price series indexed by datetime.
    events : DataFrame
        Events table with columns ``t1`` and ``y`` indexed by event
        start times.

    Returns
    -------
    Series
        Realised returns aligned to the event index.
    """
    realized = []
    for t0, t1 in zip(events.index, events["t1"]):
        if pd.isna(t1):
            realized.append(0.0)
            continue
        try:
            p0 = float(prices.loc[t0])
            p1 = float(prices.loc[t1])
        except KeyError:
            # if t1 is outside price index use last available price
            p1 = float(prices.iloc[-1])
            p0 = float(prices.loc[t0])
        realized.append((p1 - p0) / p0)
    return pd.Series(realized, index=events.index, dtype=float)


def _build_features(prices: pd.Series, lookback: int) -> pd.Series:
    """Construct a simple momentum feature for classification.

    The momentum is defined as the percentage change over ``lookback``
    periods, lagged by one period so that the feature at time ``t0``
    does not look ahead.  Missing values are left as NaN and should
    be filtered prior to model training.

    Parameters
    ----------
    prices : Series
        Price series indexed by datetime.
    lookback : int
        Number of periods over which to compute the momentum.

    Returns
    -------
    Series
        Lagged momentum feature.
    """
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    momentum = prices.pct_change(lookback)
    return momentum.shift(1)


def train_meta_model(
    prices: pd.Series,
    lookback: int = 20,
    vol_span: int = 50,
    pt_sl: Tuple[float, float] = (1.0, 1.0),
    horizon: int = 5,
    n_splits: int = 5,
    embargo_pct: float = 0.0,
) -> Dict[str, float]:
    """Train and evaluate a logistic meta‑model on a single asset.

    This function builds a momentum feature and meta‑labels from the
    price series, then fits a logistic regression model using
    purged k‑fold cross‑validation【95993860354040†L629-L645】.  The meta‑label is 1
    when the underlying triple‑barrier label correctly predicts the
    direction of the realised return and 0 otherwise.  The momentum
    feature is the percentage price change over ``lookback`` periods
    lagged by one period.  Returns from the model include the mean
    and standard deviation of the cross‑validated accuracy as well as
    the number of valid samples.

    Parameters
    ----------
    prices : Series
        Price series indexed by datetime.
    lookback : int, default 20
        Lookback window for the momentum feature.
    vol_span : int, default 50
        Span parameter for the volatility estimate used in the
        triple‑barrier method.
    pt_sl : tuple of float, default (1.0, 1.0)
        Profit‑taking and stop‑loss multipliers for the triple‑barrier
        method.
    horizon : int, default 5
        Vertical barrier horizon (number of periods) for the
        triple‑barrier method.
    n_splits : int, default 5
        Number of folds for purged cross‑validation.
    embargo_pct : float, default 0.0
        Fraction of the data to embargo after each test fold.

    Returns
    -------
    dict
        Dictionary containing the mean and standard deviation of the
        cross‑validated accuracy and the number of samples used.  If
        there are insufficient data points to train the model the
        returned metrics are NaN.
    """
    # compute volatility and apply triple barrier labelling
    vol = daily_volatility(prices, span=vol_span)
    events = triple_barrier_labels(prices, vol=vol, pt_sl=pt_sl, horizon=horizon)
    # compute realised returns and meta‑labels
    realized_returns = _compute_realized_returns(prices, events)
    base_labels = events["y"]
    meta_labels = meta_label(base_labels, realized_returns)
    # build momentum feature
    feature = _build_features(prices, lookback)
    # align feature to event index
    X_series = feature.reindex(meta_labels.index)
    # drop missing values
    mask = (~X_series.isna()) & (~meta_labels.isna())
    if mask.sum() < n_splits:
        return {"cv_accuracy_mean": float("nan"), "cv_accuracy_std": float("nan"), "n_samples": int(mask.sum())}
    X = X_series[mask].to_numpy().reshape(-1, 1)
    y = meta_labels[mask].to_numpy().astype(int)
    # build events DataFrame for CV splits
    # Only keep rows where mask is True
    filtered_index = meta_labels.index[mask]
    cv_events = pd.DataFrame({"t0": filtered_index, "t1": events.loc[filtered_index, "t1"]})
    # purge cross‑validation
    cv = PurgedKFold(n_splits=n_splits, embargo_pct=embargo_pct)
    accuracies: List[float] = []
    for train_idx, test_idx in cv.split(cv_events):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        # add intercept term explicitly
        X_train_with_const = np.column_stack([np.ones(len(X_train)), X_train])
        X_test_with_const = np.column_stack([np.ones(len(X_test)), X_test])
        # fit logistic regression; suppress convergence warnings by setting disp=False
        try:
            model = Logit(y_train, X_train_with_const).fit(disp=False)
            preds = model.predict(X_test_with_const)
        except Exception:
            # if fit fails (e.g. perfect separation) treat all predictions as majority class
            preds = np.full(len(y_test), y_train.mean())
        pred_labels = (preds >= 0.5).astype(int)
        acc = float((pred_labels == y_test).mean())
        accuracies.append(acc)
    return {
        "cv_accuracy_mean": float(np.mean(accuracies)),
        "cv_accuracy_std": float(np.std(accuracies, ddof=0)),
        "n_samples": int(mask.sum()),
    }
