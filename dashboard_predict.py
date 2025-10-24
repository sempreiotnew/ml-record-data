import threading
import pandas as pd
import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from datetime import datetime
from serial_reader import start_time, serial_buffer, serial_reader_function, save_annotations
import joblib
from feature_extractor import extract_features_from_data

# ---------------- CONFIG ----------------
clf = joblib.load("random_forest_model.pkl")  # pre-trained model
SLIDDING_WINDOW = 50
THRESOLD_PREDICTION = 20
CONFIDENCE_THRESHOLD = 0.7
sensor_prediction_history = {}
sensor_current_label = {}
sensor_current_prob = {}
baselines = {}
active_drops = {}  # track active drop sessions (sid -> start_idx)
last_recovery_index = {}  # track last index at which we recovered to avoid reusing old drops

label_colors = {
    "cigarro": "#E60A0A",
    "ar": "#0D9501",
    "alcool": "#1D05B5",
    "gas": "#D46205",
    "unknow": "#505050"
}

# ---------------- DASH APP ----------------
app = Dash(__name__)

app.layout = html.Div([
    html.H2("BME688 Leonardo's Data", style={"color": "white", "textAlign": "center"}),

    html.Div(id="sensor-predictions", style={
        "display": "grid",
        "gridTemplateColumns": "repeat(4, 1fr)",
        "gap": "10px",
        "marginBottom": "20px"
    }),

    dcc.Graph(id="live-graph", style={"height": "70vh"}),
    dcc.Interval(id="interval-refresh", interval=1000, n_intervals=0),
], style={"backgroundColor": "#111", "padding": "20px"})


# ---------------- DASH UPDATE ----------------
@app.callback(
    [Output("sensor-predictions", "children"),
     Output("live-graph", "figure")],
    Input("interval-refresh", "n_intervals"),
)
def update_dashboard(n):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    sensor_ids = list(serial_buffer.keys())
    if not sensor_ids:
        return ["No sensors"], go.Figure()

    dfs = {sid: pd.DataFrame(list(serial_buffer[sid])) for sid in sensor_ids}

    # --- baseline setup ---
    for sid in sensor_ids:
        df = dfs[sid]
        if df.empty:
            continue
        if sid not in baselines:
            baselines[sid] = np.median(df["gas_resistance"].head(20))

    prediction_cards = []
    rows = int(np.ceil(len(sensor_ids) / 2))
    cols = 2 if len(sensor_ids) > 1 else 1
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Sensor {sid}" for sid in sensor_ids])

    # --- Process each sensor ---
    for idx, sid in enumerate(sensor_ids):
        df = dfs[sid]
        if df.empty:
            continue

        baseline = baselines.get(sid)
        if baseline is None:
            continue

        gas = df["gas_resistance"].values
        drop_threshold = baseline * 0.9
        drop_indices = np.where(gas < drop_threshold)[0]

        predicted_label = sensor_current_label.get(sid)
        slide_df = pd.DataFrame()
        drop_percentage = (1 - gas[-1] / baseline) * 100
        print(f"{sid} - {drop_percentage:.1f}%")

        if drop_percentage <= -10:
            print(f"[{sid}] Extreme negative spike → reset baseline")
            baselines[sid] = gas[-1]
            drop_percentage = 0
            # clear any active drop / prediction state
            if sid in active_drops:
                active_drops.pop(sid)
            sensor_current_label.pop(sid, None)
            sensor_current_prob.pop(sid, None) if sid in sensor_current_prob else None
            last_recovery_index[sid] = len(df) - 1        

        # --- Sliding / Drop logic with robust reset + new-drop gating ---
        # 1) If currently sliding and recovered -> clear all sliding/prediction state and record recovery index
        if sid in active_drops and drop_percentage <= 5:
            print(f"[{sid}] Recovery detected → fully reset state")
            # set last recovery index to last sample index so future drops must start after this
            last_recovery_index[sid] = len(df) - 1
            # clear sliding & prediction state
            active_drops.pop(sid, None)
            sensor_current_label.pop(sid, None)
            sensor_current_prob.pop(sid, None) if sid in sensor_current_prob else None
            # ensure no slide window used this iteration
            slide_df = pd.DataFrame()
            predicted_label = None

        # 2) If not currently sliding, look for a new drop index that is after last_recovery_index (if any)
        if sid not in active_drops:
            if len(drop_indices) > 0:
                # filter drop indices to be strictly after last recovery (if exists)
                last_rec = last_recovery_index.get(sid, -1)
                valid_indices = [i for i in drop_indices if i > last_rec]
                if len(valid_indices) > 0:
                    start_idx = max(valid_indices[0] - 5, 0)
                    active_drops[sid] = start_idx
                    # when we start a fresh drop, clear previous probabilities so UI shows fresh prediction
                    sensor_current_prob[sid] = 0
                    print(f"[{sid}] New drop started at index {start_idx} (last_rec={last_rec})")

        # 3) If sliding active, build the sliding window anchored on that start index
        if sid in active_drops:
            start_idx = active_drops[sid]
            end_idx = len(df)
            # create sliding window that moves to the most recent data but keeps event anchored
            if end_idx - start_idx >= SLIDDING_WINDOW:
                slide_df = df.iloc[end_idx - SLIDDING_WINDOW:end_idx].copy()
            else:
                slide_df = df.iloc[start_idx:end_idx].copy()

        # --- Prediction (only if slide_df has data) ---
        probs = np.array([0, 0])
        if not slide_df.empty and len(slide_df) >= THRESOLD_PREDICTION:
            probs, predicted_label = get_prediction(slide_df)
            sensor_current_label[sid] = predicted_label
            sensor_current_prob[sid] = probs.max()
            print(f"[{sid}] Sliding prediction → {predicted_label} ({probs.max()*100:.1f}%)")
        else:
            # do not trigger prediction when there's no slide window
            predicted_label = sensor_current_label.get(sid, None)

        # --- Prediction card ---
        if predicted_label:
            prob_display = (sensor_current_prob.get(sid, probs).max() * 100) if isinstance(sensor_current_prob.get(sid, probs), np.ndarray) or isinstance(sensor_current_prob.get(sid, probs), float) else (probs.max() * 100)
            # ensure prob_display is a float
            try:
                prob_display = float(prob_display)
            except Exception:
                prob_display = probs.max() * 100
            prediction_cards.append(
                html.Div([
                    html.H4(f"Sensor {sid}", style={"margin": "0"}),
                    html.P(f"{predicted_label} ({prob_display:.1f}%)", style={"margin": "0"})
                ], style={
                    "padding": "10px",
                    "color": "white",
                    "borderRadius": "6px",
                    "textAlign": "center",
                    "backgroundColor": label_colors.get(predicted_label, "#505050")
                })
            )
        else:
            prediction_cards.append(
                html.Div([
                    html.H4(f"Sensor {sid}", style={"margin": "0"}),
                    html.P("Monitoring...", style={"margin": "0"})
                ], style={
                    "padding": "10px",
                    "color": "white",
                    "borderRadius": "6px",
                    "textAlign": "center",
                    "backgroundColor": "#222"
                })
            )

        # --- Plot full signal ---
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['gas_resistance'],
                mode='lines',
                name=f'Sensor {sid}',
                line=dict(color="#00FFFF", width=2)
            ),
            row=idx // cols + 1,
            col=idx % cols + 1
        )

        # --- Plot sliding window ---
        if not slide_df.empty:
            color = label_colors.get(predicted_label, "#FF69B4")
            fig.add_trace(
                go.Scatter(
                    x=slide_df['timestamp'],
                    y=slide_df['gas_resistance'],
                    mode='lines',
                    name=f'{sid} (window)',
                    line=dict(color=color, width=4)
                ),
                row=idx // cols + 1,
                col=idx % cols + 1
            )

    fig.update_layout(template="plotly_dark", hovermode="x unified", height=1200)
    fig.update_yaxes(title_text="Resistance (Ω, log)", type="log")

    return prediction_cards, fig


# ---------------- HELPERS ----------------
def get_sliding_data(df):
    if len(df) < SLIDDING_WINDOW:
        return df.copy()
    else:
        return df.tail(SLIDDING_WINDOW).copy()


def get_prediction(slide_df):
    data_list = slide_df.to_dict('records')
    features_dict = extract_features_from_data(data_list, label=None)

    features_df = pd.DataFrame([features_dict])
    missing_cols = set(clf.feature_names_in_) - set(features_df.columns)
    for col in missing_cols:
        features_df[col] = 0
    features_df = features_df[clf.feature_names_in_]

    probs = clf.predict_proba(features_df)[0]
    predicted_label = clf.classes_[np.argmax(probs)]
    return probs, predicted_label


# ---------------- MAIN ----------------
if __name__ == "__main__":
    threading.Thread(target=serial_reader_function, daemon=True).start()
    app.run(debug=True, use_reloader=False)
