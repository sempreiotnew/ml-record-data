import threading
import pandas as pd
import numpy as np
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from datetime import datetime
import numpy as np
import pandas as pd
from serial_reader import start_time, serial_buffer, serial_reader_function, save_annotations
# ---------------- DASH APP ----------------
app = Dash(__name__)

app.layout = html.Div([
    html.H2("BME688 Leonardo's Data", style={"color":"white","textAlign":"center"}),
    # 🔹 Input + Button to store data
    html.Div([
        dcc.Input(
            id="user-input",
            type="text",
            placeholder="Digite a descrição",
            style={"marginRight": "10px", "width": "300px", "height" : "40px"}
        ),
        html.Button(
            "Salvar",
            id="save-btn",
            n_clicks=0,
            disabled=True,
            style={"backgroundColor": "#008CBA", "color": "white", "padding": "8px 16px", "border": "none", "borderRadius": "6px"}
        ),
        html.Div(id="saved-output", style={"color": "white", "marginTop": "10px"}),
        dcc.Store(id="stored-variable")
    ], style={"marginBottom": "30px", "textAlign": "center"}),
    html.Div(id="sensor-predictions", style={"display":"grid","gridTemplateColumns":"repeat(4, 1fr)","gap":"10px"}),
    dcc.Graph(id="live-graph", style={"height":"80vh"}),
    dcc.Interval(id="interval-refresh", interval=1000, n_intervals=0),
], style={"backgroundColor":"#111","padding":"20px"})


# ---------------- USER INPUT HANDLER ----------------
@app.callback(
    Output("save-btn", "disabled"),
    Input("user-input", "value")
)
def toggle_button_state(value):
    """Enable button only when input is not empty."""
    if value and value.strip():
        return False
    return True


@app.callback(
    [Output("stored-variable", "data"),
     Output("saved-output", "children")],
    Input("save-btn", "n_clicks"),
    State("user-input", "value"),
    prevent_initial_call=True
)
def save_user_input(n_clicks, value):
    """Save the input value when button is clicked."""
    if value:
        save_annotations(f"{datetime.now()} - {value}")
        return value, f"✅ Descrição salva: {value}"
    return None, "No data saved"

@app.callback(
    [Output("sensor-predictions","children"),
     Output("live-graph","figure")],
    [Input("interval-refresh","n_intervals")],
)
def update_dashboard(n):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # with serial_lock:
    sensor_ids = list(serial_buffer.keys())
    #To avoid cut sensor_id and get the correct sensor_id list
    for sensor in sensor_ids:
        if(len(sensor) < 9):
            print("------------------------------------------------------------")
            print("WARNING - THERE IS A SENSOR_ID CUT, LETS GET THE KEYS AGAIN")
            print("------------------------------------------------------------")
            sensor_ids = list(serial_buffer.keys())
    dfs = {sid: pd.DataFrame(list(serial_buffer[sid])) for sid in sensor_ids}



    if not sensor_ids:
        return ["No sensors"], go.Figure()

    cards = []


    rows = int(np.ceil(len(sensor_ids)/2))
    cols = 2 if len(sensor_ids) > 1 else 1
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"Sensor {sid}" for sid in sensor_ids])
    for idx, sid in enumerate(sensor_ids):
        df = dfs[sid]
        if df.empty:
            continue

        row = idx // cols + 1
        col = idx % cols + 1

        # inside your callback, for each df:
        df = dfs[sid].copy()
        if df.empty:
            continue

        # make sure timestamp column is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # optional: show relative seconds from start_time
        df['elapsed_s'] = (df['timestamp'] - start_time).dt.total_seconds()

        # line=dict(color="#00FFFF") -> TO CHANGE THE LINES COLOR INCLUDING THIS PROPERTY
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['gas_resistance'], mode='lines', name=f'Sensor ID {sid}'), row=row, col=col)
        

    fig.update_layout(template="plotly_dark", hovermode="x unified", height=1200)
    fig.update_yaxes(title_text="(Ω, log)", type="log")

    return cards, fig

# ---------------- MAIN ----------------
if __name__ == "__main__":
    threading.Thread(target=serial_reader_function, daemon=True).start()
    app.run(debug=True, use_reloader=False)
