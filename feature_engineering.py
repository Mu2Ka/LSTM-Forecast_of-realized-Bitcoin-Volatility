"""Построение признаков из часовых рыночных данных."""

import numpy as np
import pandas as pd


def build_hourly_features(hourly_df):
    """Создать признаки только из текущей закрытой свечи и прошлого."""
    features = {}
    groups = {
        "market": [],
        "lags": [],
        "rolling": [],
        "exponential": [],
        "technical": [],
        "statistical": [],
        "calendar": [],
    }

    def add(group, name, values):
        features[name] = values
        groups[group].append(name)

    def safe_divide(numerator, denominator):
        return numerator / denominator.replace(0, np.nan)

    close = hourly_df["close"].astype(float)
    open_price = hourly_df["open"].astype(float)
    high = hourly_df["high"].astype(float)
    low = hourly_df["low"].astype(float)
    volume = hourly_df["volume"].astype(float)
    quote_volume = hourly_df["quote_volume"].astype(float)
    trades = hourly_df["trades"].astype(float)
    taker_volume = hourly_df["taker_buy_volume"].astype(float)
    taker_quote_volume = hourly_df["taker_buy_quote_volume"].astype(float)
    returns = np.log(close).diff()
    rv_24h_past = np.sqrt(
        returns.pow(2).rolling(24, min_periods=24).sum()
    )

    for name, values in {
        "market_open": open_price,
        "market_high": high,
        "market_low": low,
        "market_close": close,
        "market_volume": volume,
        "market_quote_volume": quote_volume,
        "market_trades": trades,
        "market_taker_buy_volume": taker_volume,
        "market_taker_buy_quote_volume": taker_quote_volume,
        "market_log_return_1h": returns,
        "market_rv_past_24h": rv_24h_past,
        "market_candle_range_ratio": safe_divide(high - low, close),
        "market_candle_body_ratio": safe_divide(close - open_price, open_price),
        "market_taker_buy_ratio": safe_divide(taker_volume, volume),
        "market_is_gap": hourly_df["is_gap"].astype(float),
    }.items():
        add("market", name, values)

    for window in [24, 168, 336]:
        add(
            "market",
            f"market_gap_count_{window}h",
            hourly_df["is_gap"].rolling(window, min_periods=window).sum(),
        )

    lag_sources = {
        "close": close,
        "return": returns,
        "volume": volume,
        "rv24": rv_24h_past,
    }
    for source_name, source in lag_sources.items():
        for lag in [1, 2, 3, 6, 12]:
            add("lags", f"lag_{source_name}_{lag}h", source.shift(lag))
        for lag in [24, 48, 72, 168, 336, 504, 672]:
            add(
                "lags",
                f"seasonal_lag_{source_name}_{lag}h",
                source.shift(lag),
            )

    rolling_windows = [3, 6, 12, 24, 48, 72, 168, 336]
    for window in rolling_windows:
        rolling_return = returns.rolling(window, min_periods=window)
        q25 = rolling_return.quantile(0.25)
        q75 = rolling_return.quantile(0.75)
        minimum = rolling_return.min()
        maximum = rolling_return.max()
        if window == 3:
            mean_1 = rolling_return.mean()
            mean_2 = returns.pow(2).rolling(window, min_periods=window).mean()
            mean_3 = returns.pow(3).rolling(window, min_periods=window).mean()
            mean_4 = returns.pow(4).rolling(window, min_periods=window).mean()
            variance_population = mean_2 - mean_1.pow(2)
            fourth_central_moment = (
                    mean_4
                    - 4 * mean_1 * mean_3
                    + 6 * mean_1.pow(2) * mean_2
                    - 3 * mean_1.pow(4)
            )
            rolling_kurtosis = (
                    fourth_central_moment / variance_population.pow(2).replace(0, np.nan)
                    - 3
            )
        else:
            rolling_kurtosis = rolling_return.kurt()

        rolling_statistics = {
            "mean": rolling_return.mean(),
            "median": rolling_return.median(),
            "std": rolling_return.std(),
            "min": minimum,
            "max": maximum,
            "range": maximum - minimum,
            "var": rolling_return.var(),
            "q25": q25,
            "q75": q75,
            "iqr": q75 - q25,
            "skew": rolling_return.skew(),
            "kurt": rolling_kurtosis,
        }
        for statistic_name, values in rolling_statistics.items():
            add(
                "rolling",
                f"roll_return_{statistic_name}_{window}h",
                values,
            )

        for source_name, source in {
            "close": close,
            "volume": volume,
            "rv24": rv_24h_past,
        }.items():
            rolling_source = source.rolling(window, min_periods=window)
            add(
                "rolling",
                f"roll_{source_name}_mean_{window}h",
                rolling_source.mean(),
            )
            add(
                "rolling",
                f"roll_{source_name}_std_{window}h",
                rolling_source.std(),
            )

    ema_spans = [6, 12, 24, 48, 72, 168]
    close_ema = {}
    for span in ema_spans:
        close_ema[span] = close.ewm(
            span=span, adjust=False, min_periods=span
        ).mean()
        return_ema = returns.ewm(
            span=span, adjust=False, min_periods=span
        ).mean()
        volume_ema = volume.ewm(
            span=span, adjust=False, min_periods=span
        ).mean()
        exp_volatility = np.sqrt(
            returns.pow(2)
            .ewm(span=span, adjust=False, min_periods=span)
            .mean()
        )

        add("exponential", f"ema_close_{span}h", close_ema[span])
        add(
            "exponential",
            f"close_to_ema_{span}h",
            safe_divide(close, close_ema[span]),
        )
        add("exponential", f"ema_return_{span}h", return_ema)
        add("exponential", f"ema_volume_{span}h", volume_ema)
        add("exponential", f"exp_volatility_{span}h", exp_volatility)

    for short_span, long_span in [(6, 24), (12, 48), (24, 72), (72, 168)]:
        add(
            "exponential",
            f"ema_difference_{short_span}_{long_span}h",
            close_ema[short_span] - close_ema[long_span],
        )

    price_change = close.diff()
    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)
    for period in [7, 14, 21, 30]:
        average_gain = gains.ewm(
            alpha=1 / period, adjust=False, min_periods=period
        ).mean()
        average_loss = losses.ewm(
            alpha=1 / period, adjust=False, min_periods=period
        ).mean()
        relative_strength = safe_divide(average_gain, average_loss)
        add(
            "technical",
            f"rsi_{period}h",
            100 - 100 / (1 + relative_strength),
        )

    for fast, slow, signal in [(6, 24, 9), (12, 26, 9), (24, 72, 18)]:
        fast_ema = close.ewm(
            span=fast, adjust=False, min_periods=fast
        ).mean()
        slow_ema = close.ewm(
            span=slow, adjust=False, min_periods=slow
        ).mean()
        macd = fast_ema - slow_ema
        signal_line = macd.ewm(
            span=signal, adjust=False, min_periods=signal
        ).mean()
        prefix = f"macd_{fast}_{slow}_{signal}"
        add("technical", prefix, macd)
        add("technical", f"{prefix}_signal", signal_line)
        add("technical", f"{prefix}_histogram", macd - signal_line)

    for period in [12, 24, 48]:
        middle = close.rolling(period, min_periods=period).mean()
        std = close.rolling(period, min_periods=period).std()
        upper = middle + 2 * std
        lower = middle - 2 * std
        width = safe_divide(upper - lower, middle)
        position = safe_divide(close - lower, upper - lower)
        add("technical", f"bollinger_middle_{period}h", middle)
        add("technical", f"bollinger_upper_{period}h", upper)
        add("technical", f"bollinger_lower_{period}h", lower)
        add("technical", f"bollinger_width_{period}h", width)
        add("technical", f"bollinger_position_{period}h", position)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    for period in [7, 14, 21, 30]:
        add(
            "technical",
            f"atr_{period}h",
            true_range.rolling(period, min_periods=period).mean(),
        )

    for period in [6, 12, 24, 48, 168]:
        add(
            "technical",
            f"roc_{period}h",
            safe_divide(close, close.shift(period)) - 1,
        )
        add(
            "technical",
            f"momentum_{period}h",
            close - close.shift(period),
        )

    for period in [14, 24, 48]:
        rolling_low = low.rolling(period, min_periods=period).min()
        rolling_high = high.rolling(period, min_periods=period).max()
        price_range = rolling_high - rolling_low
        stochastic_k = 100 * safe_divide(close - rolling_low, price_range)
        stochastic_d = stochastic_k.rolling(3, min_periods=3).mean()
        williams_r = -100 * safe_divide(rolling_high - close, price_range)
        add("technical", f"stochastic_k_{period}h", stochastic_k)
        add("technical", f"stochastic_d_{period}h", stochastic_d)
        add("technical", f"williams_r_{period}h", williams_r)

    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    add("technical", "obv", obv)

    typical_price = (high + low + close) / 3
    for period in [12, 24, 48, 168]:
        weighted_price = (typical_price * volume).rolling(
            period, min_periods=period
        ).sum()
        rolling_volume = volume.rolling(period, min_periods=period).sum()
        add(
            "technical",
            f"vwap_{period}h",
            safe_divide(weighted_price, rolling_volume),
        )

    for window in [24, 48, 168, 336]:
        for lag in [1, 6, 24]:
            add(
                "statistical",
                f"autocorr_return_w{window}_lag{lag}",
                returns.rolling(window, min_periods=window).corr(
                    returns.shift(lag)
                ),
            )

    for window in rolling_windows:
        for source_name, source in {"close": close, "volume": volume}.items():
            rolling_source = source.rolling(window, min_periods=window)
            coefficient_variation = safe_divide(
                rolling_source.std(), rolling_source.mean().abs()
            )
            add(
                "statistical",
                f"coefficient_variation_{source_name}_{window}h",
                coefficient_variation,
            )

    positive = returns.gt(0).astype(float)
    negative = returns.lt(0).astype(float)
    zero = returns.eq(0).astype(float)
    for window in [24, 72, 168, 336]:
        probabilities = [
            state.rolling(window, min_periods=window).mean()
            for state in [positive, negative, zero]
        ]
        entropy = sum(
            -probability * np.log(probability.where(probability > 0))
            for probability in probabilities
        ).fillna(0)
        add("statistical", f"sign_entropy_{window}h", entropy)

    confirmed_local_maximum = (
            (close.shift(1) > close.shift(2))
            & (close.shift(1) > close)
    ).astype(float)
    confirmed_local_minimum = (
            (close.shift(1) < close.shift(2))
            & (close.shift(1) < close)
    ).astype(float)
    for window in [24, 72, 168, 336]:
        add(
            "statistical",
            f"local_maxima_count_{window}h",
            confirmed_local_maximum.rolling(window, min_periods=window).sum(),
        )
        add(
            "statistical",
            f"local_minima_count_{window}h",
            confirmed_local_minimum.rolling(window, min_periods=window).sum(),
        )

    for window in [24, 72, 168, 336]:
        positive_semivariance = (
            returns.pow(2)
            .where(returns > 0)
            .rolling(window, min_periods=max(3, window // 4))
            .mean()
        )
        negative_semivariance = (
            returns.pow(2)
            .where(returns < 0)
            .rolling(window, min_periods=max(3, window // 4))
            .mean()
        )
        add(
            "statistical",
            f"positive_semivariance_{window}h",
            positive_semivariance,
        )
        add(
            "statistical",
            f"negative_semivariance_{window}h",
            negative_semivariance,
        )

    for window in [24, 72, 168, 336]:
        add(
            "statistical",
            f"volatility_of_volatility_{window}h",
            rv_24h_past.rolling(window, min_periods=window).std(),
        )

        forecast_time = hourly_df.index + pd.Timedelta(1, unit="h")
    hour = pd.Series(forecast_time.hour, index=hourly_df.index, dtype=float)
    day_of_week = pd.Series(
        forecast_time.dayofweek, index=hourly_df.index, dtype=float
    )
    month = pd.Series(forecast_time.month, index=hourly_df.index, dtype=float)
    day_of_year = pd.Series(
        forecast_time.dayofyear, index=hourly_df.index, dtype=float
    )

    add("calendar", "hour", hour)
    add("calendar", "day_of_week", day_of_week)
    add("calendar", "month", month)
    add("calendar", "is_weekend", day_of_week.ge(5).astype(float))
    add("calendar", "hour_sin", np.sin(2 * np.pi * hour / 24))
    add("calendar", "hour_cos", np.cos(2 * np.pi * hour / 24))
    add("calendar", "day_of_week_sin", np.sin(2 * np.pi * day_of_week / 7))
    add("calendar", "day_of_week_cos", np.cos(2 * np.pi * day_of_week / 7))
    add("calendar", "month_sin", np.sin(2 * np.pi * (month - 1) / 12))
    add("calendar", "month_cos", np.cos(2 * np.pi * (month - 1) / 12))
    add("calendar", "day_of_year_sin", np.sin(2 * np.pi * day_of_year / 365.25))
    add("calendar", "day_of_year_cos", np.cos(2 * np.pi * day_of_year / 365.25))

    feature_frame = pd.DataFrame(features, index=hourly_df.index)
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    feature_frame = feature_frame.astype("float32")
    return feature_frame, groups
