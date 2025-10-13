import numpy as np


import numpy as np
from scipy.stats import skew, kurtosis


def extract_features_from_data(data):
    """
    Extracts statistical, derivative, trend, peak, time-based, ratio, and AUC features
    from BME688 sensor data including gas_resistance, temperature, pressure, and humidity.

    data: list of dicts, each with keys:
        "gas_resistance", "temperature", "pressure", "humidity"
    """

    print(data)
    
    x = np.arange(len(data))  # time in seconds
    
    # Create arrays for each variable
    y_gas = np.array([d["gas_resistance"] for d in data])
    y_temp = np.array([d["temperature"] for d in data])
    y_pres = np.array([d["pressure"] for d in data])
    y_hum = np.array([d["humidity"] for d in data])

    def compute_features(y, prefix):
        features = {}
        # --- Basic Statistics ---
        features[f'{prefix}_mean'] = np.mean(y)
        features[f'{prefix}_std'] = np.std(y)
        features[f'{prefix}_min'] = np.min(y)
        features[f'{prefix}_max'] = np.max(y)
        features[f'{prefix}_median'] = np.median(y)
        features[f'{prefix}_range'] = np.ptp(y)
        features[f'{prefix}_q25'] = np.percentile(y, 25)
        features[f'{prefix}_q75'] = np.percentile(y, 75)
        features[f'{prefix}_skew'] = skew(y)
        features[f'{prefix}_kurtosis'] = kurtosis(y)

        # --- Derivatives ---
        dy = np.diff(y) / np.diff(x)
        features[f'{prefix}_dy_mean'] = np.mean(dy)
        features[f'{prefix}_dy_std'] = np.std(dy)
        features[f'{prefix}_dy_min'] = np.min(dy)
        features[f'{prefix}_dy_max'] = np.max(dy)

        # Second derivative
        ddy = np.diff(dy) / np.diff(x[:-1])
        features[f'{prefix}_ddy_mean'] = np.mean(ddy)
        features[f'{prefix}_ddy_std'] = np.std(ddy)

        # --- Trend ---
        slope = np.polyfit(x, y, 1)[0]
        features[f'{prefix}_slope'] = slope

        # --- Peaks ---
        peaks = ((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])).sum()
        features[f'{prefix}_peaks'] = peaks

        # --- Area under curve ---
        features[f'{prefix}_auc'] = np.trapezoid(y, x)
        # --- Time-based Features ---
        # Smooth signal slightly to reduce noise effect
        from scipy.ndimage import uniform_filter1d
        y_smooth = uniform_filter1d(y, size=3)
        dy = np.diff(y_smooth) / np.diff(x)

        # Steady-state detection — last time when |dy| < 1% of range
        threshold = 0.03 * np.ptp(y_smooth)
        steady_idx = np.where(np.abs(dy) < threshold)[0]
        if len(steady_idx) > 0:
            features[f'{prefix}_steady_time'] = steady_idx[-1]  # last stable moment
        else:
            features[f'{prefix}_steady_time'] = len(y) - 1

        # Rise/fall time (10% to 90% of total range)
        y_min, y_max = np.min(y_smooth), np.max(y_smooth)
        y_10 = y_min + 0.1 * (y_max - y_min)
        y_90 = y_min + 0.9 * (y_max - y_min)

        if y_smooth[0] < y_smooth[-1]:
            # Rising curve
            t10 = np.argmax(y_smooth >= y_10)
            t90 = np.argmax(y_smooth >= y_90)
        else:
            # Falling curve
            t10 = np.argmax(y_smooth <= y_90)
            t90 = np.argmax(y_smooth <= y_10)

        features[f'{prefix}_rise_time'] = abs(t90 - t10)

        # Area ratio (rise vs decay)
        # Detect whether it’s a peak (rising then falling) or a dip (falling then rising)
        if y_smooth[0] < y_smooth[-1]:
            mid_idx = np.argmax(y_smooth)  # peak for rising curve
        else:
            mid_idx = np.argmin(y_smooth)  # valley for falling curve

        # area_rise = np.trapezoid(y_smooth[:mid_idx+1], x[:mid_idx+1])
        # area_decay = np.trapezoid(y_smooth[mid_idx:], x[mid_idx:])
        # features[f'{prefix}_area_ratio'] = abs(area_rise / area_decay) if area_decay != 0 else 0
        area_rise = np.trapezoid(y_smooth[:mid_idx+1], x[:mid_idx+1])
        area_decay = np.trapezoid(y_smooth[mid_idx:], x[mid_idx:])
        eps = 1e-3 * (y_max - y_min)  # small fraction of range
        area_rise += eps
        area_decay += eps
        features[f'{prefix}_area_ratio'] = abs(area_rise / area_decay)
        

        return features

    # --- Compute features for all sensors ---
    features = {}
    features.update(compute_features(y_gas, "gas"))
    features.update(compute_features(y_temp, "temp"))
    features.update(compute_features(y_pres, "pres"))
    features.update(compute_features(y_hum, "hum"))

    # --- Ratio Features ---
    features["gas_std_over_mean"] = features["gas_std"] / (features["gas_mean"] + 1e-9)
    features["gas_over_temp_corr"] = np.corrcoef(y_gas, y_temp)[0, 1]
    features["gas_over_hum_corr"] = np.corrcoef(y_gas, y_hum)[0, 1]
    features["gas_over_pres_corr"] = np.corrcoef(y_gas, y_pres)[0, 1]

    print(features)
    return features


# def extract_features_from_data(data):
#     x = np.array([])
#     y = np.array([])
#     for idx, value in enumerate(data):
#         y = np.append(y, value["gas_resistance"])
#         x  = np.append(x, idx)

#     print(x)
#     print(y)
#     features = {}
#     # Statistical
#     features['mean'] = np.mean(y)
#     features['std'] = np.std(y)
#     features['min'] = np.min(y)
#     features['max'] = np.max(y)
#     features['median'] = np.median(y)
#     features['range'] = np.ptp(y)
#     features['q25'] = np.percentile(y, 25)
#     features['q75'] = np.percentile(y, 75)
#     features['skew'] = skew(y)
#     features['kurtosis'] = kurtosis(y)
    
#     # Derivatives
#     dy = np.diff(y) / np.diff(x)
#     features['dy_mean'] = np.mean(dy)
#     features['dy_std'] = np.std(dy)
#     features['dy_min'] = np.min(dy)
#     features['dy_max'] = np.max(dy)
    
#     ddy = np.diff(dy) / np.diff(x[:-1])
#     features['ddy_mean'] = np.mean(ddy)
#     features['ddy_std'] = np.std(ddy)
    
#     # Trend
#     slope = np.polyfit(x, y, 1)[0]
#     features['slope'] = slope
    
#     # Peak count
#     peaks = ((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:])).sum()
#     features['peaks'] = peaks
    
#     # Area under curve
#     features['auc'] = np.trapezoid(y, x)
    
#     print(features)
#     return features


# def calculate_derivative(data):
#     features = {}
#     results = []

#     x = np.array([])
#     y = np.array([])
#     for idx, value in enumerate(data):
#         y = np.append(y, value["gas_resistance"])
#         x  = np.append(x, idx)
        

#     features['mean'] = np.mean(y)
#     features['std'] = np.std(y)
#     features['min'] = np.min(y)
#     features['max'] = np.max(y)
#     features['median'] = np.median(y)
#     features['range'] = np.ptp(y)

#     dy_dx = np.gradient(y, x)
#     features["dy_dx"] = dy_dx.tolist()

#     # d2y_dx2 = np.gradient(dy_dx, x)


#     return features