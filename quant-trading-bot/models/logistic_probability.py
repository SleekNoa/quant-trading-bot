"""
Logistic Probability Model â€” directional signal quality estimator.
===================================================================
Replaces the Brownian-motion estimator that produced 0% output due to
numerical edge-cases in the gap-drift formula.

Why logistic regression?
    â€¢ Works on moderate samples (500-1000 bars) that quant systems typically have
    â€¢ Outputs calibrated probabilities natively (predict_proba)
    â€¢ Regularisation (CV-selected C, fallback C=0.5) prevents overfitting
    â€¢ Time-series CV is used when enough samples are available
    â€¢ ``balanced`` class weights handle uneven up/down distributions
    â€¢ Fast enough to retrain on every run without latency impact

Features
--------
    ma_spread       : (short_ma - long_ma) / (20-bar vol Ã— close)
                      Normalised trend separation â€” the primary crossover signal
    momentum_5d     : 5-day percentage price return
                      Short-term directional momentum
    rsi_norm        : RSI / 100  (inline Wilder smoothing if 'rsi' not in df)
                      Overbought/oversold condition
    volume_ratio    : volume / 20-bar mean volume
                      Participation strength (high vol = conviction)
    close_position  : (close - 20-bar low) / (20-bar high - 20-bar low)
                      Position within recent range (0 = bottom, 1 = top)
    bar_return      : log(close / prev_close)
                      Most recent impulse direction

Reference
---------
    Chan, E. (2013). Algorithmic Trading. Wiley.
        "A well-regularised classifier trained on price features is
         more robust than heuristic probability estimates."

Usage
-----
    from models.logistic_probability import LogisticProbabilityModel
    model = LogisticProbabilityModel()
    model.train(df)          # fit on historical bars
    prob = model.predict(df) # P(next bar closes UP)  0.0 â€“ 1.0
    gate = model.gate(prob, crossover=1)  # â†’ "BUY" | "SELL" | "HOLD"
"""

from __future__ import annotations

from config.settings import (
    PROB_BUY_THRESHOLD,     # e.g. 60   â†’ means 0.60
    PROB_SELL_THRESHOLD,    # e.g. 45   â†’ means 0.45
    PROB_TRAIN_BARS,
)

import numpy as np
import pandas as pd
from utils.logger import logger

try:
    from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.exceptions import NotFittedError
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_FEATURE_NAMES = [
    "ma_spread",
    "momentum_5d",
    "rsi_norm",
    "volume_ratio",
    "close_position",
    "bar_return",
]

_MIN_TRAIN_SAMPLES = 50  # absolute floor for reliable logistic regression
_MIN_CV_SAMPLES = 200  # minimum samples before using time-series CV


# â”€â”€ Core model class â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class LogisticProbabilityModel:
    """
    6-feature logistic regression estimator of next-bar directional
    probability. Designed as a drop-in upgrade for any probability gate
    that previously used the Brownian-motion crossover formula.
    """

    def __init__(
        self,
        train_bars:      int   = PROB_TRAIN_BARS,
        buy_threshold:   float = PROB_BUY_THRESHOLD / 100.0,
        sell_threshold:  float = PROB_SELL_THRESHOLD / 100.0,
    ):
        """
        Parameters
        ----------
        train_bars      : how many bars to use for training (from tail of df)
        buy_threshold   : minimum P(up) to emit BUY  (default 0.55 = 55%)
        sell_threshold  : maximum P(up) to emit SELL (default 0.45 = 45%)
        """
        self.train_bars     = train_bars
        self.buy_threshold  = buy_threshold
        self.sell_threshold = sell_threshold
        self.trained        = False
        self._last_error: str | None = None

        self._pipe = None
        self._pipe_lr = None
        self._pipe_cv = None
        if SKLEARN_AVAILABLE:
            self._pipe_lr = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",   # handles class imbalance automatically
                    solver="lbfgs",
                    C=0.5,                     # fallback L2 regularisation when CV is not used
                    random_state=42,
                )),
            ])

            self._pipe_cv = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegressionCV(
                    Cs=20,
                    cv=TimeSeriesSplit(n_splits=5),
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight="balanced",
                    scoring="neg_log_loss",
                    n_jobs=-1,
                    l1_ratios=(0.0,),
                    use_legacy_attributes=False,
                )),
            ])

            # Default to CV when available (train() falls back if samples < _MIN_CV_SAMPLES)
            self._pipe = self._pipe_cv
    # â”€â”€ Feature construction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the 6 features from any OHLCV + indicator DataFrame.
        All NaN rows are dropped before returning.
        """
        close = df["close"].astype(float)
        f = pd.DataFrame(index=df.index)

        # â”€â”€ Shared vol denominator (used by features 2 and 6) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        vol_20 = close.pct_change().rolling(20).std().replace(0, np.nan)  # â† ADD THIS LINE

        # 1. MA spread (normalised by rolling volatility Ã— price)
        if "short_ma" in df.columns and "long_ma" in df.columns:
            denom = (vol_20 * close).replace(0, np.nan)
            f["ma_spread"] = (
                                     df["short_ma"].astype(float) - df["long_ma"].astype(float)
                             ) / denom
        else:
            ema_short = close.ewm(span=10, min_periods=10).mean()
            ema_long = close.ewm(span=30, min_periods=30).mean()
            denom = (vol_20 * close).replace(0, np.nan)
            f["ma_spread"] = (ema_short - ema_long) / denom

        # 2. 5-day price momentum  â† CHANGED: divide by vol_20
        f["momentum_5d"] = close.pct_change(5) / vol_20

        # 3. RSI (normalised 0 â€“ 1); computed inline if column absent
        if "rsi" in df.columns:
            f["rsi_norm"] = df["rsi"].astype(float) / 100.0
        else:
            delta = close.diff()
            gain = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
            loss = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
            rs = gain / loss.replace(0, np.nan)
            f["rsi_norm"] = (100 - 100 / (1 + rs)) / 100.0

        # 4. Volume ratio (current bar vs 20-bar mean)
        if "volume" in df.columns:
            vol_mean = df["volume"].astype(float).rolling(20).mean().replace(0, np.nan)
            f["volume_ratio"] = df["volume"].astype(float) / vol_mean
        else:
            f["volume_ratio"] = 1.0

        # 5. Close position within 20-bar high/low range
        if "high" in df.columns and "low" in df.columns:
            high_20 = df["high"].astype(float).rolling(20).max()
            low_20 = df["low"].astype(float).rolling(20).min()
            rng = (high_20 - low_20).replace(0, np.nan)
            f["close_position"] = (close - low_20) / rng
        else:
            f["close_position"] = 0.5

        # 6. Single-bar log return  â† CHANGED: divide by vol_20
        f["bar_return"] = np.log(close / close.shift(1).replace(0, np.nan)) / vol_20

        return f.dropna()

    # â”€â”€ Training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def train(self, df: pd.DataFrame, log_selection: bool = True) -> bool:
        """
        Fit the model on df (or the last self.train_bars rows of df).

        Returns True on success, False otherwise.
        Sets self.trained accordingly.
        """
        if not SKLEARN_AVAILABLE:
            self._last_error = (
                "scikit-learn not installed â€” run: pip install scikit-learn"
            )
            return False

        # Slice to training window
        train_df = (
            df.iloc[-self.train_bars :].copy()
            if len(df) > self.train_bars
            else df.copy()
        )

        features = self._build_features(train_df)

        # Target: did close rise on the NEXT bar?
        next_close = train_df["close"].astype(float).shift(-1)
        target     = (next_close > train_df["close"].astype(float)).astype(int)
        target     = target.reindex(features.index).dropna()
        features   = features.reindex(target.index)

        if len(features) < _MIN_TRAIN_SAMPLES:
            self._last_error = (
                f"Training failed: only {len(features)} samples after "
                f"feature construction (need â‰¥{_MIN_TRAIN_SAMPLES})"
            )
            return False

        if target.nunique() < 2:
            self._last_error = "Training failed: only one class in target window"
            return False

        if SKLEARN_AVAILABLE and self._pipe_cv and len(features) >= _MIN_CV_SAMPLES:
            self._pipe = self._pipe_cv
            if log_selection:
                logger.info(
                    f"[prob] LogisticRegressionCV selected (samples={len(features)})"
                )
        else:
            self._pipe = self._pipe_lr
            if log_selection:
                logger.info(
                    f"[prob] LogisticRegression selected (samples={len(features)})"
                )

        if self._pipe is None:
            self._last_error = "Training failed: sklearn pipeline unavailable"
            return False
        try:
            self._pipe.fit(features[_FEATURE_NAMES], target)
            self.trained     = True
            self._last_error = None
            return True

        except Exception as exc:
            self._last_error = str(exc)
            self.trained     = False
            return False

    # â”€â”€ Inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def predict(self, df: pd.DataFrame) -> float:
        """
        Return P(next bar closes UP) for the last row of df.

        Falls back to 0.5 (neutral) when:
            â€¢ model not yet trained
            â€¢ scikit-learn unavailable
            â€¢ feature computation fails
        """
        if not self.trained or not SKLEARN_AVAILABLE:
            return 0.5

        features = self._build_features(df)
        if features.empty:
            return 0.5

        try:
            row  = features[_FEATURE_NAMES].iloc[[-1]]
            prob = float(self._pipe.predict_proba(row)[0][1])
            return float(np.clip(prob, 0.0, 1.0))

        except (NotFittedError, Exception):
            return 0.5

    def gate(self, up_prob: float, crossover: int) -> str:
        """
        Translate raw probability + crossover direction to a trade decision.

        Parameters
        ----------
        up_prob    : P(next bar up) from predict()
        crossover  : +1 = bullish crossover, -1 = bearish, 0 = no crossover

        Returns
        -------
        "BUY" | "SELL" | "HOLD"
        """
        if crossover == 1 and up_prob >= self.buy_threshold:
            return "BUY"
        if crossover == -1 and up_prob <= self.sell_threshold:
            return "SELL"
        return "HOLD"

    # â”€â”€ Diagnostics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @property
    def is_trained(self) -> bool:
        return self.trained

    @property
    def last_error(self) -> str:
        return self._last_error or ""

    def summary(self) -> str:
        sklearn_status = "available" if SKLEARN_AVAILABLE else "MISSING (pip install scikit-learn)"
        train_status   = "trained" if self.trained else f"NOT trained â€” {self._last_error}"
        return (
            f"LogisticProbabilityModel  |  sklearn={sklearn_status}  "
            f"|  status={train_status}  "
            f"|  buy_thresh={self.buy_threshold:.2f}  "
            f"|  sell_thresh={self.sell_threshold:.2f}"
        )


# â”€â”€ Module-level singleton â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Trained once per run and cached.  Call reset_model() before a new ticker.

_MODEL: LogisticProbabilityModel | None = None


def get_or_train_model(
    df:             pd.DataFrame,
    train_bars:      int   = PROB_TRAIN_BARS,
    buy_threshold:  float = 0.55,
    sell_threshold: float = 0.45,
    log_selection:  bool  = True,
) -> LogisticProbabilityModel:
    """
    Return a trained singleton.  Trains on first call, reuses on subsequent calls.

    Parameters
    ----------
    df              : full historical DataFrame (model trains on last train_bars rows)
    train_bars      : training window
    buy_threshold   : as fraction (e.g. 0.55)
    sell_threshold  : as fraction (e.g. 0.45)
    """
    global _MODEL

    if _MODEL is None or not _MODEL.is_trained:
        _MODEL = LogisticProbabilityModel(
            train_bars=train_bars,
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
        )
        _MODEL.train(df, log_selection=log_selection)

    return _MODEL


def reset_model() -> None:
    """
    Clear the cached singleton.
    Call this before switching to a new ticker in multi-ticker mode so the
    new ticker gets its own freshly trained model.
    """
    global _MODEL
    _MODEL = None








