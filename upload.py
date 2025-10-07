import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, State, Output, no_update
import plotly.graph_objs as go
import io, base64
import os
from features import extract_features
from helper import build_table, build_layout

app = Dash(__name__)
app.title = "Sensor Dashboard"

df_global = pd.DataFrame()
df_selected_data = None 
# Layout (unchanged, just added export button)
app.layout = build_layout()

# ----------------------------
# CSV parsing and dropdown update (unchanged)
# ----------------------------
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return df
        else:
            return None
    except Exception as e:
        print(e)
        return None

@app.callback(
    Output('sensor-dropdown', 'options'),
    Output('sensor-dropdown', 'value'),
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_dropdown(contents, filename):
    global df_global
    if contents is None:
        return [], None, ""
    df_global = parse_contents(contents, filename)
    print(df_global)
    print(" --------------")
    if df_global is None:
        return [], None, "Failed to load file."
    if 'id' not in df_global.columns:
        return [], None, "CSV missing 'id' column."
    ids = df_global['id'].unique()
    options = [{'label': str(i), 'value': i} for i in ids]
    first_id = ids[0] if len(ids) > 0 else None
    return options, first_id, f"File '{filename}' uploaded successfully. Found {len(ids)} unique sensors."

# ----------------------------
# Helper to extract valid selection info
# ----------------------------
def build_valid_selection_payload(selectedData):
    if not selectedData or not isinstance(selectedData, dict):
        return None

    pts = selectedData.get('points') or []
    indices = []
    for p in pts:
        try:
            if ('curveNumber' not in p) or (p.get('curveNumber') == 0):
                if 'pointIndex' in p:
                    indices.append(int(p['pointIndex']))
        except Exception:
            continue

    if indices:
        indices = sorted(list(dict.fromkeys(indices)))
        return {'indices': indices}

    rng = selectedData.get('range')
    if rng and isinstance(rng, dict) and rng.get('x'):
        try:
            x0 = float(rng['x'][0])
            x1 = float(rng['x'][1])
            return {'x_range': [min(x0, x1), max(x0, x1)]}
        except Exception:
            return None

    return None

# ----------------------------
# Store selected points
# ----------------------------
@app.callback(
    Output('selected-points-store', 'data'),
    Input('gas-graph', 'selectedData'),
    State('sensor-dropdown', 'value')
)
def store_selected_points(selectedData, current_sensor):
    payload = build_valid_selection_payload(selectedData)
    if payload is None:
        return no_update
    return {'sensor': current_sensor, **payload}

# ----------------------------
# Graph update callback (index-based version)
# ----------------------------
@app.callback(
    Output('gas-graph', 'figure'),
    Input('sensor-dropdown', 'value'),
    Input('selected-points-store', 'data')
)
def update_graph(selected_id, stored_selected):
    fig = go.Figure()
    fig.update_layout(dragmode='select')
    fig.update_layout(
        title=f"Gas Resistance (ID {selected_id})" if selected_id is not None else "Gas Resistance",
        plot_bgcolor='rgba(40,40,40,0.8)',
        paper_bgcolor='rgba(50,50,50,0.8)',
        font=dict(color='white'),
        xaxis=dict(title='Sample Index', gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        hovermode="x unified",
        height=750
    )

    if selected_id is None or df_global.empty:
        return fig

    # Filter by sensor ID
    df_filtered = df_global[df_global['id'] == selected_id].copy()
    df_filtered['gas_resistance'] = pd.to_numeric(df_filtered.get('gas_resistance'), errors='coerce')
    df_filtered = df_filtered.dropna(subset=['gas_resistance'])
    if df_filtered.empty:
        return fig

    # Reset index to ensure sequential numbering
    df_filtered = df_filtered.reset_index(drop=True)

    # Use index as "time" (sample count)
    df_filtered['time'] = df_filtered.index

    # Add main gas resistance curve
    fig.add_trace(go.Scatter(
        x=df_filtered['time'],
        y=df_filtered['gas_resistance'],
        mode='lines+markers',
        name='Gas Resistance',
        line=dict(color='red'),
        marker=dict(color='red')
    ))

    # Handle selected region
    if stored_selected and stored_selected.get('sensor') == selected_id:
        df_selected = pd.DataFrame()
        if 'indices' in stored_selected:
            idxs = [i for i in stored_selected['indices'] if 0 <= i < len(df_filtered)]
            if idxs:
                df_selected = df_filtered.iloc[idxs].copy()
        elif 'x_range' in stored_selected:
            x_min, x_max = stored_selected['x_range']
            df_selected = df_filtered[
                (df_filtered['time'] >= x_min) & (df_filtered['time'] <= x_max)
            ].copy()

        if not df_selected.empty:
            df_selected = df_selected.sort_values('time').reset_index(drop=True)
            y = df_selected['gas_resistance'].values
            t = df_selected['time'].values

            # Find min and max indices in selected region
            min_idx = int(np.argmin(y))
            if min_idx < len(y) - 1:
                max_idx = int(np.argmax(y[min_idx:])) + min_idx
            else:
                max_idx = min_idx

            # Highlight selected region and points
            fig.add_trace(go.Scatter(
                x=t,
                y=y,
                mode='lines',
                name='Selected Area',
                line=dict(color='yellow', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=[t[min_idx]], y=[y[min_idx]],
                mode='markers', name='Drop Min',
                marker=dict(color='blue', size=12, symbol='triangle-down')
            ))
            fig.add_trace(go.Scatter(
                x=[t[max_idx]], y=[y[max_idx]],
                mode='markers', name='Recovery Max',
                marker=dict(color='green', size=12, symbol='triangle-up')
            ))

    return fig


# Update metrics
@app.callback(
    Output('selected-data-output', 'children'),
    Input('sensor-dropdown', 'value'),
    Input('selected-points-store', 'data')
)
def update_metrics(selected_id, stored_selected):
    global df_selected_data
    # If no data or no sensor selected -> show default
    if selected_id is None or df_global.empty:
        return "No data selected."

    # If there is no stored valid selection for current sensor, show message
    if not stored_selected or stored_selected.get('sensor') != selected_id:
        return "No data selected."

    # Build filtered dataframe
    df_filtered = df_global[df_global['id'] == selected_id].copy()
    df_filtered['millis'] = pd.to_numeric(df_filtered.get('millis'), errors='coerce')
    df_filtered['gas_resistance'] = pd.to_numeric(df_filtered.get('gas_resistance'), errors='coerce')
    df_filtered = df_filtered.dropna(subset=['millis', 'gas_resistance'])
    if df_filtered.empty:
        return "No numeric data for this sensor."

    df_filtered = df_filtered.reset_index(drop=True)
    # KEEP absolute time in seconds
    df_filtered['time'] = df_filtered['millis'] / 1000.0
    # Compose df_selected exactly using saved indices if available, otherwise using x-range
    if 'indices' in stored_selected:
        idxs = stored_selected['indices']
        idxs = [i for i in idxs if 0 <= i < len(df_filtered)]
        if not idxs:
            return "Selected points do not match any data."
        df_selected = df_filtered.iloc[idxs].copy().sort_values('time').reset_index(drop=True)
        # baseline candidates: up to 5 points before the first selected index, if any
        first_global_idx = idxs[0]
        baseline_candidates = df_filtered.iloc[max(0, first_global_idx - 5): first_global_idx]['gas_resistance'].values
    elif 'x_range' in stored_selected:
        x_min, x_max = stored_selected['x_range']
        df_selected = df_filtered[(df_filtered['time'] >= x_min) & (df_filtered['time'] <= x_max)].copy().sort_values('time').reset_index(drop=True)
        if df_selected.empty:
            return "Selected points do not match any data."
        # baseline candidates: up to 5 points BEFORE the selected region if available
        sel_idx_mask = (df_filtered['time'] >= x_min) & (df_filtered['time'] <= x_max)
        global_indices = np.where(sel_idx_mask)[0]
        if len(global_indices) > 0:
            first_global_idx = int(global_indices[0])
            
    else:
        return "No points selected."

    # Sort and arrays
    df_selected = df_selected.sort_values('time').reset_index(drop=True)
    y = df_selected['gas_resistance'].values.astype(float)
    t = df_selected['time'].values.astype(float)
    millis = df_selected['millis'].values.astype(int)

    df_selected_data = df_selected.copy()
    if len(y) == 0:
        return "Selected points do not match any data."


    return html.Div([
        html.H4("Selected data:"),
        #html.Ul(feature_items)
    ]), build_table(df_selected)    


@app.callback(
    Output("download-xlsx", "data"),
    Input("export-button", "n_clicks"),
    State("df-selected-store", "data"),  # read df_selected from store
    prevent_initial_call=True
)
def export_to_xlsx(n_clicks, data):
    # Convert DataFrame to Excel bytes
    global df_selected_data
    
    print(len(df_selected_data))
    df_sorted = df_selected_data.sort_values('id')
    print(len(df_sorted))


if __name__ == '__main__':
    app.run(debug=True, port=8051)
