# features.py
import numpy as np
import pandas as pd
from typing import Any, Tuple, List
from scipy.signal import find_peaks
import matplotlib
import matplotlib.pyplot as plt
import json
# matplotlib.use('Agg')  # non-GUI backend


SEGMENT_SIZE = 100

def safe_to_float_array(x: Any) -> np.ndarray:
    if x is None:
        return np.array([], dtype=float)
    if isinstance(x, (list, tuple, pd.Series, np.ndarray)):
        s = pd.Series(x)
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    if isinstance(x, str):
        sep = ';' if ';' in x else (',' if ',' in x else None)
        if sep:
            parts = [p.strip() for p in x.split(sep) if p.strip() != ""]
            if not parts:
                return np.array([], dtype=float)
            return pd.to_numeric(pd.Series(parts), errors="coerce").to_numpy(dtype=float)
        try:
            return np.array([float(x)], dtype=float)
        except Exception:
            return np.array([np.nan], dtype=float)
    try:
        return np.array([float(x)], dtype=float)
    except Exception:
        return np.array([np.nan], dtype=float)


def safe_to_float_array(arr: Any) -> np.ndarray:
    return np.array(arr, dtype=float)

def compute_standard_deviation(gas_arr : np.ndarray):
    gas = np.asarray(gas_arr, dtype=float)
    n = gas.size
    
    if n < 2:
        return np.array([], dtype=float)
    
    stds = np.empty(n - 1, dtype=float)
    
    for i in range(1, n):
        stds[i-1] = np.nanstd([gas[i-1], gas[i]])
    
    return stds

def compute_variance_standard_deviation(gas_arr: np.ndarray, segment_size: int = SEGMENT_SIZE, plot = True):
    variance = []
    standard_deviation = []
    mean = []
    segment_centers = []

    # Compute variance and std for each segment
    for start in range(0, len(gas_arr), segment_size):
        end = min(start + segment_size, len(gas_arr))
        segment = gas_arr[start:end]
        if len(segment) < 2:  # skip too small segments
            continue
        var = np.var(segment, ddof=1)
        std = np.std(segment, ddof=1)
        mn = np.mean(segment)
        variance.append(var)
        standard_deviation.append(std)
        segment_centers.append(start + len(segment)//2)  # middle of the segment
        mean.append(mn)

    variance = np.array(variance)
    standard_deviation = np.array(standard_deviation)
    mean = np.array(mean)

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(14,10), sharex=True)  # 2 rows, 1 column, shared X-axis

        # --- Plot 1: Original curve with segment points ---
        axes[0].plot(gas_arr, color='gray', alpha=0.5, label="Gas Sensor Signal")
        axes[0].scatter(segment_centers, gas_arr[segment_centers], color='purple', label="Segment center")
        axes[0].set_ylabel("Gas Resistance")
        axes[0].set_title("Gas Sensor Signal with Segment Centers")
        axes[0].legend()
        axes[0].grid(True)

        # --- Plot 2: Variance and Standard Deviation ---
        # axes[1].plot(segment_centers, variance, 'o-', color='blue', label="Variance per segment")
        # axes[1].plot(segment_centers, standard_deviation, 's-', color='red', label="Standard Deviation per segment")
        axes[1].plot(segment_centers, variance, 'o-', color='blue', label="Variance per segment")
        axes[1].plot(segment_centers, standard_deviation, 's-', color='red', label="Std per segment")

        axes[1].set_xlabel("Measurement Index")
        axes[1].set_ylabel("Value")
        axes[1].set_title("Variance and Standard Deviation per Segment")
        axes[1].legend()
        axes[1].grid(True)
        axes[1].plot(segment_centers, variance, 'o-', color='blue', label="Variance per segment")
        # optional: log scale to make small deviations visible
        axes[1].set_yscale('log')
        

        plt.tight_layout()  # nice spacing
        plt.show()

    return variance, standard_deviation, mean


        



def calculate_area_fixed_segments(gas_arr: np.ndarray, segment_size: int = SEGMENT_SIZE, fixed_baseline: float = 1e5, plot = True):
    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(gas_arr, label="Gas Sensor", color="gray")

    drop_areas = []

    # Split the signal into fixed segments
    for start in range(0, len(gas_arr), segment_size):
        end = min(start + segment_size, len(gas_arr))
        x_region = np.arange(start, end)
        y_region = gas_arr[start:end]

        # Area = curve down to fixed baseline
        drop_values = y_region - fixed_baseline
        total_area = np.trapezoid(drop_values)  # integrate
        drop_areas.append(total_area)

        # Fill area under curve to baseline
        if plot : plt.fill_between(x_region, fixed_baseline, y_region, color="orange", alpha=0.3)

        # Annotate area
        mid_x = x_region[len(x_region)//2]
        mid_y = fixed_baseline + np.max(drop_values)/2
        if plot: plt.text(mid_x, mid_y, f"{total_area:.0f}", ha='center', va='bottom', fontsize=8, color='black')

    if plot:
        plt.axhline(fixed_baseline, color="green", linestyle="--", label=f"Baseline ({fixed_baseline})")
        plt.xlabel("Sample Index")
        plt.ylabel("Gas Resistance")
        plt.title(f"Gas Sensor Signal with Drop Areas ({segment_size}-measurement segments)")
        plt.legend()
        plt.show()

    return drop_areas

def find_peaks_segment(
    gas_arr: np.ndarray, 
    fixed_baseline: float = 1e5, 
    segment_size: int = 500,
    plot: bool = True
):
    data_min = np.min(gas_arr)
    data_max = np.max(gas_arr)
    rangePeak = 0.1 * (data_max - data_min)
    rangeValley = 0.05 * (data_max - data_min)

    all_peaks = []
    all_valleys = []

    # Process in segments
    n_segments = int(np.ceil(len(gas_arr) / segment_size))

    for i in range(n_segments):
        start = i * segment_size
        end = min((i + 1) * segment_size, len(gas_arr))
        segment = gas_arr[start:end]

        # --- Find peaks in this segment ---
        peaks, _ = find_peaks(
            segment,
        )
        peaks += start
        all_peaks.extend(peaks)

        # --- Find valleys in this segment ---
        valleys, _ = find_peaks(
            -segment,
            # prominence=rangeValley,
        )
        valleys += start
        all_valleys.extend(valleys)

    all_peaks = np.array(all_peaks)
    all_valleys = np.array(all_valleys)

    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(gas_arr, label="Gas Sensor", color="gray")
        plt.plot(all_peaks, gas_arr[all_peaks], "x", color="blue", label="Peaks")
        plt.plot(all_valleys, gas_arr[all_valleys], "x", color="red", label="Valleys")
        plt.axhline(fixed_baseline, color="green", linestyle="--", label=f"Baseline ({fixed_baseline})")

        # --- Mark segment areas ---
        for i in range(n_segments):
            start = i * segment_size
            end = min((i + 1) * segment_size, len(gas_arr))
            plt.axvspan(start, end, color="yellow", alpha=0.1)

        plt.xlabel("Sample Index")
        plt.ylabel("Gas Resistance")
        plt.title(f"Gas Sensor Signal with Segments and Peaks/Valleys")
        plt.legend()
        plt.show()

    return all_peaks, all_valleys



def calculate_velocities(gas_arr: np.ndarray, peaks: np.ndarray, valleys: np.ndarray, sample_interval_ms: int = 817, plot: bool = True):
    """
    Calculate drop and recovery velocities from gas sensor data and plot markers.

    Args:
        gas_arr: np.ndarray -> array of gas resistance values
        peaks: np.ndarray -> indices of peaks
        valleys: np.ndarray -> indices of valleys
        sample_interval_ms: int -> interval between samples in ms (default: 817ms)
        plot: bool -> whether to plot the results

    Returns:
        results: list of dict with velocities and details
    """
    results = []
    dt = sample_interval_ms / 1000.0  # convert ms → seconds

    # --- loop through valleys and find nearest peak before and after ---
    for v in valleys:
        valley_val = gas_arr[v]

        # --- Drop (previous peak → valley) ---
        drop_velocity = None
        prev_peaks = peaks[peaks < v]
        if len(prev_peaks) > 0:
            p = prev_peaks[-1]
            peak_val = gas_arr[p]
            time_elapsed = (v - p) * dt
            if time_elapsed > 0:
                drop_velocity = (peak_val - valley_val) / time_elapsed

        # --- Recovery (valley → next peak) ---
        recovery_velocity = None
        next_peaks = peaks[peaks > v]
        if len(next_peaks) > 0:
            p = next_peaks[0]
            peak_val = gas_arr[p]
            time_elapsed = (p - v) * dt
            if time_elapsed > 0:
                recovery_velocity = (peak_val - valley_val) / time_elapsed

        results.append({
            "valley_index": v,
            "valley_value": valley_val,
            "drop_velocity": drop_velocity,
            "recovery_velocity": recovery_velocity
        })

    # --- Plotting ---
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(gas_arr, label="Gas Sensor", color="gray")
        plt.plot(peaks, gas_arr[peaks], "x", color="blue", label="Peaks")
        plt.plot(valleys, gas_arr[valleys], "o", color="red", label="Valleys")

        # annotate velocities
        for res in results:
            v = res["valley_index"]
            val = res["valley_value"]

            if res["drop_velocity"] is not None:
                plt.annotate(f"Drop v={res['drop_velocity']:.2f}",
                             (v, val),
                             textcoords="offset points", xytext=(-40, -20),
                             ha='center', color="red", fontsize=8,
                             arrowprops=dict(arrowstyle="->", color="red"))

            if res["recovery_velocity"] is not None:
                plt.annotate(f"Rec v={res['recovery_velocity']:.2f}",
                             (v, val),
                             textcoords="offset points", xytext=(40, 20),
                             ha='center', color="green", fontsize=8,
                             arrowprops=dict(arrowstyle="->", color="green"))

        plt.xlabel("Sample Index")
        plt.ylabel("Gas Resistance")
        plt.title("Gas Sensor Velocities (Drop & Recovery)")
        plt.legend()
        plt.grid(True)
        plt.show()

    return results



import numpy as np
import matplotlib.pyplot as plt

def calculate_slopes(
    gas_arr: np.ndarray, 
    peaks: np.ndarray, 
    valleys: np.ndarray, 
    sample_interval_ms: int = 817, 
    plot: bool = True,
    segment_size: int = 500  # NEW parameter
):
    """
    Calculate slopes (ratios) for drops and recoveries in gas sensor data per segment.

    Args:
        gas_arr: np.ndarray -> array of gas resistance values
        peaks: np.ndarray -> indices of peaks
        valleys: np.ndarray -> indices of valleys
        sample_interval_ms: int -> interval between samples in ms (default: 817ms)
        plot: bool -> whether to plot the slopes and ratios
        segment_size: int -> number of samples per segment (default: 500)

    Returns:
        slopes: list of dict with slopes and ratio for each valley
    """
    dt = sample_interval_ms / 1000.0  # convert ms → seconds
    slopes = []

    n_segments = int(np.ceil(len(gas_arr) / segment_size))

    for seg in range(n_segments):
        start = seg * segment_size
        end = min((seg + 1) * segment_size, len(gas_arr))

        seg_valleys = valleys[(valleys >= start) & (valleys < end)]
        seg_peaks = peaks[(peaks >= start) & (peaks < end)]

        for v in seg_valleys:
            valley_val = gas_arr[v]

            # --- Drop slope (previous peak → valley) ---
            drop_slope, rec_slope = None, None
            prev_peaks = peaks[peaks < v]
            if len(prev_peaks) > 0:
                p = prev_peaks[-1]
                peak_val = gas_arr[p]
                dx = (v - p) * dt
                if dx > 0:
                    drop_slope = (valley_val - peak_val) / dx

            # --- Recovery slope (valley → next peak) ---
            next_peaks = peaks[peaks > v]
            if len(next_peaks) > 0:
                p = next_peaks[0]
                peak_val = gas_arr[p]
                dx = (p - v) * dt
                if dx > 0:
                    rec_slope = (peak_val - valley_val) / dx

            # --- Ratio (recovery / drop) ---
            ratio = None
            if drop_slope is not None and rec_slope is not None:
                ratio = rec_slope / abs(drop_slope)

            slopes.append({
                "valley_index": v,
                "valley_value": valley_val,
                "drop_slope": drop_slope,
                "recovery_slope": rec_slope,
                "ratio": ratio
            })

    # --- Plotting ---
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(gas_arr, label="Gas Sensor Data", color="blue")

        for s in slopes:
            v = s["valley_index"]
            drop, rec, ratio = s["drop_slope"], s["recovery_slope"], s["ratio"]

            # highlight valley
            plt.scatter(v, s["valley_value"], color="red", marker="v", s=80)

            if drop is not None and rec is not None:
                color = "green" if ratio > 0.5 else "orange" if ratio > 0 else "purple"
                plt.text(v, s["valley_value"], f"R={ratio:.2f}", fontsize=9, color=color)

        plt.title(f"Gas Sensor Curve with Slopes Ratios (Segment size={segment_size})")
        plt.xlabel("Sample Index")
        plt.ylabel("Gas Resistance")
        plt.legend()
        plt.grid(True)
        plt.show()

    return slopes



def compute_derivatives_inflection(gas_arr: np.ndarray, millis: np.ndarray, segment_size: int = SEGMENT_SIZE, plot=True):

    
    results = []
    

    if segment_size == 0:
        start = 0
        end = int(min(gas_arr))
        x = millis[start:end] / 1000.0
        y = gas_arr[start:end]

        # derivatives
        dy_dx = np.gradient(y, x)
        d2y_dx2 = np.gradient(dy_dx, x)

        # inflection points
        sign_changes = np.where(np.diff(np.sign(d2y_dx2)))[0]
        inflection_points_x = x[sign_changes]
        inflection_points_y = y[sign_changes]

        # slopes at inflection points (dy/dx)
        slopes = dy_dx[sign_changes]

        results.append({
            'x': x.tolist(),
            'y': y.tolist(),
            'dy_dx': dy_dx.tolist(),
            'd2y_dx2': d2y_dx2.tolist(),
            # 'inflection_points_x': inflection_points_x,
            # 'inflection_points_y': inflection_points_y,
            # 'slopes_at_inflections': slopes
        })
        return np.array(results, dtype=list)    

    if plot : plt.figure(figsize=(12,6))

    n_segments = int(np.ceil(len(gas_arr) / segment_size))
    for i in range(n_segments):
        start = i * segment_size
        end = min((i + 1) * segment_size, len(gas_arr))
        x = millis[start:end] / 1000.0
        y = gas_arr[start:end]

        # derivatives
        dy_dx = np.gradient(y, x)
        d2y_dx2 = np.gradient(dy_dx, x)

        # inflection points
        sign_changes = np.where(np.diff(np.sign(d2y_dx2)))[0]
        inflection_points_x = x[sign_changes]
        inflection_points_y = y[sign_changes]

        # slopes at inflection points (dy/dx)
        slopes = dy_dx[sign_changes]

        results.append({
            'x': x.tolist(),
            'y': y.tolist(),
            'dy_dx': dy_dx.tolist(),
            'd2y_dx2': d2y_dx2.tolist(),
            # 'inflection_points_x': inflection_points_x,
            # 'inflection_points_y': inflection_points_y,
            # 'slopes_at_inflections': slopes
        })

        if plot:
            plt.plot(x, y, color='blue', alpha=0.6, label='Gas Sensor' if i==0 else "")
            plt.plot(x, dy_dx, color='orange', alpha=0.6, label='1st derivative' if i==0 else "")
            plt.plot(x, d2y_dx2, color='red', alpha=0.6, label='2nd derivative' if i==0 else "")
            # plt.scatter(inflection_points_x, inflection_points_y, color='green', marker='x', s=60, label='Inflection' if i==0 else "")
            # # mark slopes at inflection points
            # for xi, yi, slope in zip(inflection_points_x, inflection_points_y, slopes):
            #     plt.text(xi, yi, f"{slope:.0f}", color='purple', fontsize=8, ha='center', va='bottom')

    if plot:
        plt.xlabel("Time (s)")
        plt.ylabel("Gas Resistance / Derivatives")
        plt.title("Gas Sensor Signal with Derivatives and Inflection Points")
        plt.legend()
        plt.grid(True)
        plt.show()
        

    return results


def calculate_velocity_new(gas_arr: Any, millis: Any, segment : int = SEGMENT_SIZE):
    
    
    final_idx = np.argmin(gas_arr)
    inital_idx = np.argmax(gas_arr[:final_idx])

    
    # initial = np.max(gas_arr[inital_idx:final_idx])
    # final = np.min(gas_arr)
    

    gas_initial = gas_arr[inital_idx]
    gas_final = gas_arr[final_idx]

    prev_gas = 0
    points = []

    for idx, gas in enumerate(gas_arr[inital_idx:final_idx]):
        print(gas)
        peaks, _ = find_peaks(
            segment,
        )
        print(peaks)
        if idx != 0:
            calculated = (prev_gas - gas) / 817
            points.append(calculated)

        prev_gas = gas        
        
    print(points)

    points_rec = []
    prev_rec_gas = 0
    for idx, gas in enumerate(gas_arr[final_idx:len(gas_arr)]):
        if idx != 0:
            calculated = (prev_rec_gas - gas) / 817
            points_rec.append(calculated)

        prev_rec_gas = gas            

    print(points_rec)

    drop_percentage = ((gas_initial - gas_final) / gas_initial) * 100


    return points


def extract_features(
    sensor_id,    
    gas_arr: Any,
    temp_arr: Any = None,
    pressure_arr: Any = None,
    humidity_arr: Any = None,
    millis_arr: Any = None,
    threshold_pct: float = 0.20,
) -> Tuple[np.ndarray, List[str]]:

    gas = safe_to_float_array(gas_arr)
    temp = safe_to_float_array(temp_arr)
    press = safe_to_float_array(pressure_arr)
    hum = safe_to_float_array(humidity_arr)
    millis = safe_to_float_array(millis_arr)  # in milliseconds
    

    def _stats(a: np.ndarray):
        if a.size == 0 or np.all(np.isnan(a)):
            return (np.nan, np.nan, np.nan)
        return (
            float(np.nanmin(a)),
            float(np.nanmax(a)),
            float(np.nanmean(a)),
        )

    gas_min, gas_max, gas_mean = _stats(gas)
    t_min, t_max, t_mean = _stats(temp)
    p_min, p_max, p_mean = _stats(press)
    h_min, h_max, h_mean = _stats(hum)
    
    # gas_std = compute_standard_deviation(gas_arr=gas_arr)
    # area = calculate_area_fixed_segments(gas_arr=gas, plot=True)
    # result = compute_derivatives_inflection(gas_arr=gas_arr, millis=millis, plot=False, segment_size=100)
    velocity = calculate_velocity_new(gas_arr=gas_arr, millis=millis)
    # variance, standard_deviation, mean = compute_variance_standard_deviation(gas_arr=gas_arr, plot=False)
    #_segment(gas_arr=gas_arr, plot=True)
    
    # calculate_velocities(gas_arr=gas, peaks=peaks, valleys=valleys, plot=True)
    # slopes = calculate_slopes(gas_arr=gas, peaks=peaks, valleys=valleys, plot=True)

    # print(gas)
    # print(variance)
    # print(standard_deviation)
    # print(mean)

    feature_names = [
        "sensor_id",
        "gas_min", "gas_max", "gas_mean",
        "temperature_min", "temperature_max", "temperature_mean", 
        "pressure_min", "pressure_max", "pressure_mean", 
        "humidity_min", "humidity_max", "humidity_mean", 
        "result"
    ]
    
    vals = [
        sensor_id,
        gas_min, gas_max, gas_mean,
        t_min, t_max, t_mean, 
        p_min, p_max, p_mean, 
        h_min, h_max, h_mean,
        #result
        # gas_std,
        # area,
    ]
    

    return np.array(vals, dtype=object), feature_names

if __name__ == "__main__":
    import pandas as pd
    import sys

    # if len(sys.argv) < 2:
    #     print("Usage: python features.py <csv_file>")
    #     sys.exit(1)

    # csv_file = sys.argv[1]

    # Load CSV
    df = pd.read_csv("predict_data_20251003_223733.csv")
    
    # Keep everything that is NOT in bad_ids
    df = df[~df["id"].isin([765269850, 765286747, 765279836, 765271387, 765283665, 765268824, 765287253])]
    # df = df[200:len(df) - 100]

    # Expected columns: gas, temperature, pressure, humidity, millis
    gas = []
    temperature = [] 
    pressure = [] 
    humidity = [] 
    millis = []
    sensor_id = df["id"].values if "id" in df else []

    for id in sensor_id:
        df_sensor = df[df["id"] == id]  # filter rows for the current sensor_id
        
        gas = df_sensor["gas_resistance"].values if "gas_resistance" in df_sensor else []
        temp = df_sensor["temperature"].values if "temperature" in df_sensor else []
        press = df_sensor["pressure"].values if "pressure" in df_sensor else []
        hum = df_sensor["humidity"].values if "humidity" in df_sensor else []
        millis = df_sensor["millis"].values if "millis" in df_sensor else []
        features, names = extract_features(
                        sensor_id=id,
                        gas_arr=gas,
                        temp_arr=temp,
                        pressure_arr=press,
                        humidity_arr=hum,
                        millis_arr=millis,
                        threshold_pct=0.20
                )

        results = features[13][0]
        rows = []


        row_dict = {
            "sensor_id": int(id),
            "gas_min": features[1],
            "gas_max": features[2],
            "gas_mean": features[3],
            "results": results
        }

        rows.append(row_dict)

        # Save to JSON
        try:
            with open('cigarro_.json', 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = []

        existing_data.append(row_dict)

        with open('training_data.json', 'w') as f:
            json.dump(existing_data, f, indent=2)
        
