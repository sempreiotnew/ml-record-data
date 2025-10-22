import numpy as np
from scipy.stats import skew, kurtosis
from scipy.ndimage import uniform_filter1d

# def extract_features_from_data(data, label=None):
#     """
#     Extract compound-discriminative features from BME688 sensor data.
#     Works for gas_resistance, temperature, pressure, and humidity.
#     """
#     x = np.arange(len(data))

#     y_gas = np.array([d["gas_resistance"] for d in data])
#     y_temp = np.array([d["temperature"] for d in data])
#     y_pres = np.array([d["pressure"] for d in data])
#     y_hum = np.array([d["humidity"] for d in data])

#     def compute_features(y, prefix):
#         features = {}

#         # --- Smoothed signal ---
#         y_smooth = uniform_filter1d(y, size=3)

#         # --- Basic Stats ---
#         features[f'{prefix}_mean'] = np.mean(y_smooth)
#         features[f'{prefix}_std'] = np.std(y_smooth)
#         features[f'{prefix}_min'] = np.min(y_smooth)
#         features[f'{prefix}_max'] = np.max(y_smooth)
#         features[f'{prefix}_median'] = np.median(y_smooth)
#         features[f'{prefix}_range'] = np.ptp(y_smooth)
#         features[f'{prefix}_q25'] = np.percentile(y_smooth, 25)
#         features[f'{prefix}_q75'] = np.percentile(y_smooth, 75)
#         features[f'{prefix}_skew'] = skew(y_smooth)
#         features[f'{prefix}_kurtosis'] = kurtosis(y_smooth)

#         # --- Derivatives ---
#         dy = np.diff(y_smooth) / np.diff(x)
#         features[f'{prefix}_dy_mean'] = np.mean(dy)
#         features[f'{prefix}_dy_std'] = np.std(dy)
#         features[f'{prefix}_dy_min'] = np.min(dy)
#         features[f'{prefix}_dy_max'] = np.max(dy)

#         # Second derivative
#         ddy = np.diff(dy) / np.diff(x[:-1])
#         features[f'{prefix}_ddy_mean'] = np.mean(ddy)
#         features[f'{prefix}_ddy_std'] = np.std(ddy)

#         # --- Trend ---
#         slope = np.polyfit(x, y_smooth, 1)[0]
#         features[f'{prefix}_slope'] = slope

#         # --- Peak features ---
#         peaks = ((y_smooth[1:-1] > y_smooth[:-2]) & (y_smooth[1:-1] > y_smooth[2:])).nonzero()[0]
#         features[f'{prefix}_peaks_count'] = len(peaks)
#         if len(peaks) > 0:
#             features[f'{prefix}_peak_max'] = np.max(y_smooth[peaks])
#             features[f'{prefix}_peak_width'] = peaks[-1] - peaks[0]  # rough width
#         else:
#             features[f'{prefix}_peak_max'] = 0
#             features[f'{prefix}_peak_width'] = 0

#         # --- Rise/fall times ---
#         y_min, y_max = np.min(y_smooth), np.max(y_smooth)
#         y_10 = y_min + 0.1*(y_max - y_min)
#         y_90 = y_min + 0.9*(y_max - y_min)
#         if y_smooth[0] < y_smooth[-1]:
#             t10 = np.argmax(y_smooth >= y_10)
#             t90 = np.argmax(y_smooth >= y_90)
#         else:
#             t10 = np.argmax(y_smooth <= y_90)
#             t90 = np.argmax(y_smooth <= y_10)
#         features[f'{prefix}_rise_time'] = abs(t90 - t10)

#         # --- Area under curve and area ratio ---
#         mid_idx = np.argmax(y_smooth) if y_smooth[0] < y_smooth[-1] else np.argmin(y_smooth)
#         area_rise = np.trapz(y_smooth[:mid_idx+1], x[:mid_idx+1])
#         area_decay = np.trapz(y_smooth[mid_idx:], x[mid_idx:])
#         eps = 1e-3 * (y_max - y_min)
#         features[f'{prefix}_area_ratio'] = abs((area_rise+eps)/(area_decay+eps))
#         features[f'{prefix}_auc'] = np.trapz(y_smooth, x)

#         # --- Frequency domain features ---
#         fft_coeff = np.fft.fft(y_smooth)
#         fft_power = np.abs(fft_coeff)**2
#         features[f'{prefix}_fft_power_mean'] = np.mean(fft_power)
#         features[f'{prefix}_fft_power_max'] = np.max(fft_power)
#         features[f'{prefix}_fft_entropy'] = -np.sum((fft_power/np.sum(fft_power))*np.log2((fft_power/np.sum(fft_power))+1e-9))

#         return features

#     # --- Compute for all sensors ---
#     features = {}
#     features.update(compute_features(y_gas, "gas"))
#     features.update(compute_features(y_temp, "temp"))
#     features.update(compute_features(y_pres, "pres"))
#     features.update(compute_features(y_hum, "hum"))

#     # --- Cross-sensor ratios ---
#     features["gas_temp_ratio"] = np.mean(y_gas)/ (np.mean(y_temp)+1e-9)
#     features["gas_hum_ratio"] = np.mean(y_gas)/ (np.mean(y_hum)+1e-9)
#     features["gas_pres_ratio"] = np.mean(y_gas)/ (np.mean(y_pres)+1e-9)

#     if label:
#         features["label"] = label

#     return features


import numpy as np
from scipy.stats import skew, kurtosis
from scipy.ndimage import uniform_filter1d

def extract_features_from_data(data, label=None, normalize=True, log_transform=True):
    """
    Extract compound-discriminative features from BME688 sensor data.
    Handles sensor-to-sensor deviation via per-sensor baseline normalization.
    Measures drop_duration, drop_ratio, and recovery_slope from peak to recovery.
    """
    x = np.arange(len(data))

    # --- Extract raw sensor arrays ---
    y_gas = np.array([d["gas_resistance"] for d in data])
    y_temp = np.array([d["temperature"] for d in data])
    y_pres = np.array([d["pressure"] for d in data])
    y_hum = np.array([d["humidity"] for d in data])

    # --- Normalize for sensor deviation (gas only) ---
    if normalize:
        baseline = np.median(y_gas[:10])  # clean-air baseline from start
        y_gas = (y_gas - baseline) / (baseline + 1e-8)
    
    # --- Optional log transform to linearize exponential responses ---
    if log_transform:
        y_gas = np.log1p(np.abs(y_gas)) * np.sign(y_gas)

    def compute_features(y, prefix):
        features = {}

        # --- Smoothed signal ---
        y_smooth = uniform_filter1d(y, size=3)

        # --- Basic Stats ---
        features[f'{prefix}_mean'] = np.mean(y_smooth)
        features[f'{prefix}_std'] = np.std(y_smooth)
        features[f'{prefix}_min'] = np.min(y_smooth)
        features[f'{prefix}_max'] = np.max(y_smooth)
        features[f'{prefix}_median'] = np.median(y_smooth)
        features[f'{prefix}_range'] = np.ptp(y_smooth)
        features[f'{prefix}_q25'] = np.percentile(y_smooth, 25)
        features[f'{prefix}_q75'] = np.percentile(y_smooth, 75)
        features[f'{prefix}_skew'] = skew(y_smooth)
        features[f'{prefix}_kurtosis'] = kurtosis(y_smooth)

        # --- Derivatives ---
        dy = np.diff(y_smooth) / np.diff(x)
        features[f'{prefix}_dy_mean'] = np.mean(dy)
        features[f'{prefix}_dy_std'] = np.std(dy)
        features[f'{prefix}_dy_min'] = np.min(dy)
        features[f'{prefix}_dy_max'] = np.max(dy)

        # Second derivative
        ddy = np.diff(dy) / np.diff(x[:-1])
        features[f'{prefix}_ddy_mean'] = np.mean(ddy)
        features[f'{prefix}_ddy_std'] = np.std(ddy)

        # --- Trend ---
        slope = np.polyfit(x, y_smooth, 1)[0]
        features[f'{prefix}_slope'] = slope

        # --- Peak features ---
        peaks = ((y_smooth[1:-1] > y_smooth[:-2]) & (y_smooth[1:-1] > y_smooth[2:])).nonzero()[0]
        features[f'{prefix}_peaks_count'] = len(peaks)
        if len(peaks) > 0:
            features[f'{prefix}_peak_max'] = np.max(y_smooth[peaks])
            features[f'{prefix}_peak_width'] = peaks[-1] - peaks[0]  # rough width
        else:
            features[f'{prefix}_peak_max'] = 0
            features[f'{prefix}_peak_width'] = 0

        # --- Rise/fall times ---
        y_min, y_max = np.min(y_smooth), np.max(y_smooth)
        y_10 = y_min + 0.1*(y_max - y_min)
        y_90 = y_min + 0.9*(y_max - y_min)
        if y_smooth[0] < y_smooth[-1]:
            t10 = np.argmax(y_smooth >= y_10)
            t90 = np.argmax(y_smooth >= y_90)
        else:
            t10 = np.argmax(y_smooth <= y_90)
            t90 = np.argmax(y_smooth <= y_10)
        features[f'{prefix}_rise_time'] = abs(t90 - t10)

        # --- Area under curve and area ratio ---
        mid_idx = np.argmax(y_smooth) if y_smooth[0] < y_smooth[-1] else np.argmin(y_smooth)
        area_rise = np.trapezoid(y_smooth[:mid_idx+1], x[:mid_idx+1])
        area_decay = np.trapezoid(y_smooth[mid_idx:], x[mid_idx:])
        eps = 1e-5 * (y_max - y_min)
        features[f'{prefix}_area_ratio'] = abs((area_rise+eps)/(area_decay+eps))
        features[f'{prefix}_auc'] = np.trapezoid(y_smooth, x)

        # --- Drop / Recovery Analysis (standard for gas sensors) ---
        peak_idx = np.argmax(y_smooth)
        min_idx = np.argmin(y_smooth[peak_idx:]) + peak_idx
        recovery_idx = min_idx + np.argmax(y_smooth[min_idx:] >= 0.9 * y_smooth[peak_idx])

        drop_duration = recovery_idx - peak_idx
        drop_ratio = y_smooth[min_idx] / (y_smooth[peak_idx] + 1e-9)
        recovery_slope = (y_smooth[recovery_idx] - y_smooth[min_idx]) / (drop_duration + 1e-9)

        features[f'{prefix}_drop_duration'] = drop_duration
        features[f'{prefix}_drop_ratio'] = drop_ratio
        features[f'{prefix}_recovery_slope'] = recovery_slope

        # --- Frequency domain features ---
        fft_coeff = np.fft.fft(y_smooth)
        fft_power = np.abs(fft_coeff)**2
        features[f'{prefix}_fft_power_mean'] = np.mean(fft_power)
        features[f'{prefix}_fft_power_max'] = np.max(fft_power)
        features[f'{prefix}_fft_entropy'] = -np.sum(
            (fft_power/np.sum(fft_power)) * np.log2((fft_power/np.sum(fft_power))+1e-8)
        )

        return features

    # --- Compute for all sensors ---
    features = {}
    features.update(compute_features(y_gas, "gas"))
    features.update(compute_features(y_temp, "temp"))
    features.update(compute_features(y_pres, "pres"))
    features.update(compute_features(y_hum, "hum"))

    # --- Cross-sensor ratios ---
    features["gas_temp_ratio"] = np.mean(y_gas)/ (np.mean(y_temp)+1e-8)
    features["gas_hum_ratio"] = np.mean(y_gas)/ (np.mean(y_hum)+1e-8)
    features["gas_pres_ratio"] = np.mean(y_gas)/ (np.mean(y_pres)+1e-8)

    if label:
        features["label"] = label

    return features
