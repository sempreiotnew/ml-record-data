import threading
import pandas as pd
import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from serial_reader import start_time, serial_buffer, serial_reader_function, save_annotations

# ---------------- CONFIG ----------------
SLIDDING_WINDOW = 50
SMOOTH_WINDOW = 50  # controls smoothing intensity; higher = flatter

# ---------------- HELPERS ----------------
def smooth_std(series: pd.Series, window: int = SMOOTH_WINDOW):
    """
    Apply rolling mean and std to flatten the curve.
    Returns a smoothed signal where noise is reduced.
    """
    std_series = series.rolling(window=window, min_periods=1).std()
    mean_series = series.rolling(window=window, min_periods=1).mean()
    smoothed = mean_series - std_series / 2
    return smoothed

# ---------------- DASH APP ----------------
app = Dash(__name__)

app.layout = html.Div([
    html.H2("BME688 Leonardo's Data", style={"color": "white", "textAlign": "center"}),

    html.Div([
        dcc.Input(
            id="user-input",
            type="text",
            placeholder="Digite a descrição",
            style={"marginRight": "10px", "width": "300px", "height": "40px"}
        ),
        html.Button(
            "Salvar",
            id="save-btn",
            n_clicks=0,
            disabled=True,
            style={
                "backgroundColor": "#008CBA",
                "color": "white",
                "padding": "8px 16px",
                "border": "none",
                "borderRadius": "6px"
            }
        ),
        html.Div(id="saved-output", style={"color": "white", "marginTop": "10px"}),
        dcc.Store(id="stored-variable")
    ], style={"marginBottom": "30px", "textAlign": "center"}),

    html.Div(id="sensor-predictions",
             style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "10px"}),

    dcc.Graph(id="live-graph", style={"height": "80vh"}),

    dcc.Interval(id="interval-refresh", interval=1000, n_intervals=0),
], style={"backgroundColor": "#111", "padding": "20px"})

# ---------------- USER INPUT HANDLER ----------------
@app.callback(
    Output("save-btn", "disabled"),
    Input("user-input", "value")
)
def toggle_button_state(value):
    return not (value and value.strip())

@app.callback(
    [Output("stored-variable", "data"),
     Output("saved-output", "children")],
    Input("save-btn", "n_clicks"),
    State("user-input", "value"),
    prevent_initial_call=True
)
def save_user_input(n_clicks, value):
    if value:
        save_annotations(f"{datetime.now()} - {value}")
        return value, f"✅ Descrição salva: {value}"
    return None, "No data saved"

# ---------------- MAIN DASH CALLBACK ----------------
@app.callback(
    [Output("sensor-predictions", "children"),
     Output("live-graph", "figure")],
    [Input("interval-refresh", "n_intervals")],
)
def update_dashboard(n):
    sensor_ids = list(serial_buffer.keys())
    if not sensor_ids:
        return ["No sensors"], go.Figure()

    dfs = {sid: pd.DataFrame(list(serial_buffer[sid])) for sid in sensor_ids}
    cards = []
    rows = int(np.ceil(len(sensor_ids) / 2))
    cols = 2 if len(sensor_ids) > 1 else 1
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"Sensor {sid}" for sid in sensor_ids])

    for idx, sid in enumerate(sensor_ids):
        df = dfs[sid]
        if df.empty:
            continue

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['elapsed_s'] = (df['timestamp'] - start_time).dt.total_seconds()

        row = idx // cols + 1
        col = idx % cols + 1

        # 🔹 Raw curve (full)
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['gas_resistance'],
                mode='lines',
                name=f'{sid} (raw)',
                line=dict(color="#00FFFF", width=1)
            ),
            row=row,
            col=col
        )

        # 🔹 Smoothed curve (full)
        df['smoothed'] = smooth_std(df['gas_resistance'], SMOOTH_WINDOW)
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['smoothed'],
                mode='lines',
                name=f'{sid} (smoothed)',
                line=dict(color="#FF69B4", width=1)
            ),
            row=row,
            col=col
        )

    fig.update_layout(template="plotly_dark", hovermode="x unified", height=1200)
    fig.update_yaxes(title_text="Gas Resistance (Ω, log)", type="log")

    return cards, fig

# ---------------- MAIN ----------------
if __name__ == "__main__":
    threading.Thread(target=serial_reader_function, daemon=True).start()
    app.run(debug=True, use_reloader=False)
