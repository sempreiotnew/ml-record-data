import threading
import pandas as pd
import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from datetime import datetime
from serial_reader import start_time, serial_buffer, serial_reader_function, save_annotations
import joblib
from feature_extractor import extract_features_from_data
# Load your pre-trained model
clf = joblib.load("random_forest_model.pkl")  # Make sure you trained this before

SLIDDING_WINDOW = 50
THRESOLD_PREDICTION = 20
CONFIDENCE_THRESHOLD = 0.7
sensor_prediction_history = {}  
sensor_current_label = {}


label_colors = {
    "cigarro": "#E60A0A",
    "ar": "#0D9501",
    "alcool" : "#1D05B5",
    "gas" : "#D46205",
    "unknow": "#505050"
}

# ---------------- DASH APP ----------------
app = Dash(__name__)

app.layout = html.Div([
    html.H2("BME688 Leonardo's Data", style={"color":"white","textAlign":"center"}),

    # 🔹 Board showing predictions
    html.Div(id="sensor-predictions", style={
        "display":"grid",
        "gridTemplateColumns":"repeat(4, 1fr)",
        "gap":"10px",
        "marginBottom":"20px"
    }),

    # 🔹 Graph
    dcc.Graph(id="live-graph", style={"height":"70vh"}),
    dcc.Interval(id="interval-refresh", interval=1000, n_intervals=0),
], style={"backgroundColor":"#111","padding":"20px"})


# ---------------- DASH UPDATE ----------------
@app.callback(
    [Output("sensor-predictions","children"),
     Output("live-graph","figure")],
    Input("interval-refresh","n_intervals"),
)
def update_dashboard(n):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    sensor_ids = list(serial_buffer.keys())
    for sensor in sensor_ids:
        if len(sensor) < 9:
            print("------------------------------------------------------------")
            print("WARNING - THERE IS A SENSOR_ID CUT, LETS GET THE KEYS AGAIN")
            print("------------------------------------------------------------")
            sensor_ids = list(serial_buffer.keys())

    dfs = {sid: pd.DataFrame(list(serial_buffer[sid])) for sid in sensor_ids}

    if not sensor_ids:
        return ["No sensors"], go.Figure()

    # 🔹 Prepare prediction board
    prediction_cards = []

    # 🔹 Prepare graph
    rows = int(np.ceil(len(sensor_ids)/2))
    cols = 2 if len(sensor_ids) > 1 else 1
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Sensor {sid}" for sid in sensor_ids])

    for idx, sid in enumerate(sensor_ids):
        df = dfs[sid]
        if df.empty:
            continue

        df = df.copy()

        # 🔹 Sliding window
        slide_df = get_sliding_data(df)
        predicted_label = None
        # 🔹 Prediction on sliding window
        if not slide_df.empty and len(slide_df) > THRESOLD_PREDICTION :
            
            probs, predicted_label = get_prediction(slide_df=slide_df)
            
            prediction_cards.append(
                html.Div([
                    html.H4(f"Sensor {sid}", style={"margin":"0"}),
                    html.P(f"{predicted_label} ({probs.max()*100:.1f}%)", style={"margin":"0"},)
                ], style={
                    "padding":"10px",
                    "backgroundColor":"#222",
                    "color":"white",
                    "borderRadius":"6px",
                    "textAlign":"center",
                    "backgroundColor" : label_colors.get(predicted_label, "#22222")
                    
                })
            )
        else:
            prediction_cards.append(
                html.Div([
                    html.H4(f"Sensor {sid}", style={"margin":"0"}),
                    html.P(f"Thinking ", style={"margin":"0"},)
                ], style={
                    "padding":"10px",
                    "backgroundColor":"#222",
                    "color":"white",
                    "borderRadius":"6px",
                    "textAlign":"center",
                    "backgroundColor" : label_colors.get( "#22222")
                    
                })
            )

        # 🔹 Draw full curve (cyan)
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['gas_resistance'],
                mode='lines',
                name=f'Sensor {sid}',
            ),
            row=idx//cols+1,
            col=idx%cols+1
        )

        # 🔹 Draw sliding window (pink)
        color = label_colors.get(predicted_label, "#FF69B4")
        if not slide_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=slide_df['timestamp'],
                    y=slide_df['gas_resistance'],
                    mode='lines',
                    name=f'{sid} (window)',
                    # line=dict(color="#FF69B4", width=4)
                    line=dict(color=color, width=4)
                ),
                row=idx//cols+1,
                col=idx%cols+1
            )

    fig.update_layout(template="plotly_dark", hovermode="x unified", height=1200)
    fig.update_yaxes(title_text="(Ω, log)", type="log")

    return prediction_cards, fig


def get_sliding_data(df):
    """Return the last SLIDDING_WINDOW samples for highlighting."""
    if len(df) < SLIDDING_WINDOW:
        return df.copy()
    else:
        return df.tail(SLIDDING_WINDOW).copy()

def get_prediction(slide_df):
# Extract features from sliding window
    data_list = slide_df.to_dict('records')  # convert sliding window to list of dicts
    features_dict = extract_features_from_data(data_list, label=None)
    
    # Convert to DataFrame
    features_df = pd.DataFrame([features_dict])
    
    # Ensure columns match the trained model
    missing_cols = set(clf.feature_names_in_) - set(features_df.columns)
    for col in missing_cols:
        features_df[col] = 0
    features_df = features_df[clf.feature_names_in_]

    # Predict
    probs = clf.predict_proba(features_df)[0]
    predicted_label = clf.classes_[np.argmax(probs)]
    print(f"Classes: {clf.classes_} Probability: {probs} ----- {predicted_label} - ")
    return probs, predicted_label


# ---------------- MAIN ----------------
if __name__ == "__main__":
    threading.Thread(target=serial_reader_function, daemon=True).start()
    app.run(debug=True, use_reloader=False)
