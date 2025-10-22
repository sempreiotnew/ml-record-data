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
baselines = {}  # store baseline per sensor

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
            baselines[sid] = np.median(df["gas_resistance"].head(20))  # first 20 samples as baseline

    # --- Prepare prediction board & figure ---
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
        drop_threshold = baseline * 0.8  # 20% drop threshold
        drop_indices = np.where(gas < drop_threshold)[0]

        predicted_label = sensor_current_label.get(sid)
        slide_df = pd.DataFrame()

        # --- Detect drop and prepare sliding window ---
        if len(drop_indices) > 0:
            if predicted_label == None:
                start_idx = max(drop_indices[0] - 5, 0)  # include 5 samples before drop
                end_idx = min(start_idx + SLIDDING_WINDOW, len(df))
                slide_df = df.iloc[start_idx:end_idx].copy()
            else:
                slide_df = get_sliding_data(df=df)

        # if len(slide_df) >= THRESOLD_PREDICTION:
        #     probs, predicted_label = get_prediction(slide_df)
        #     print(f"[{sid}] Drop detected ({(1 - gas[-1]/baseline)*100:.1f}% drop) → Pred: {predicted_label}")
        # --- Perform prediction if we have enough samples ---
        probs = np.array([0, 0])
        if not slide_df.empty and len(slide_df) >= THRESOLD_PREDICTION:
            probs, predicted_label = get_prediction(slide_df)
            sensor_current_label[sid] = predicted_label  # 🔹 store it persistently
            print(f"[{sid}] Drop detected ({(1 - gas[-1]/baseline)*100:.1f}% drop) → Pred: {predicted_label}")
        else:
            # keep last prediction
            predicted_label = sensor_current_label.get(sid, None)        
        
        # --- Prediction card ---
        if predicted_label:
            prediction_cards.append(
                html.Div([
                    html.H4(f"Sensor {sid}", style={"margin": "0"}),
                    html.P(f"{predicted_label} ({probs.max() * 100:.1f}%)", style={"margin": "0"})
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

        # --- Plot sliding window if active ---
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
    """Return the last SLIDDING_WINDOW samples."""
    if len(df) < SLIDDING_WINDOW:
        return df.copy()
    else:
        return df.tail(SLIDDING_WINDOW).copy()


def get_prediction(slide_df):
    """Run model prediction on sliding window."""
    data_list = slide_df.to_dict('records')
    features_dict = extract_features_from_data(data_list, label=None)

    features_df = pd.DataFrame([features_dict])

    # Match training features
    missing_cols = set(clf.feature_names_in_) - set(features_df.columns)
    for col in missing_cols:
        features_df[col] = 0
    features_df = features_df[clf.feature_names_in_]

    # Predict
    probs = clf.predict_proba(features_df)[0]
    predicted_label = clf.classes_[np.argmax(probs)]
    return probs, predicted_label


# ---------------- MAIN ----------------
if __name__ == "__main__":
    threading.Thread(target=serial_reader_function, daemon=True).start()
    app.run(debug=True, use_reloader=False)