from dash import Dash, dcc, html, Input, State, Output, no_update
import plotly.graph_objs as go

# Build table of selected points (cap rows to 200)
def build_table(df_sel, max_rows=200):
    #header = [html.Tr([html.Th("idx"), html.Th("millis"), html.Th("time_s"), html.Th("gas_resistance")], style={'color':'white'})]
    header = [html.Tr([html.Th("idx"), html.Th("sensor_id"), html.Th("gas_resistance"), html.Th("date_time")], style={'color':'white'})]
    rows = []
    #nrows = min(len(df_sel), max_rows)
    nrows = len(df_sel)
    for i in range(nrows):
        rows.append(html.Tr([
            html.Td(str(i)),
            #html.Td(str(int(df_sel['millis'].iloc[i]))),
            #html.Td(f"{float(df_sel['time'].iloc[i]):.3f}"),
            html.Td(str((df_sel['id'].iloc[i]))),
            html.Td(f"{float(df_sel['gas_resistance'].iloc[i]):.2f}"),
            html.Td(str((df_sel['date_time'].iloc[i]))),
            #html.Td(f"{float(df_sel['temperature'].iloc[i]):.3f}")
        ]))
    #if len(df_sel) > max_rows:
    #    rows.append(html.Tr([html.Td(f"... {len(df_sel)-max_rows} more rows", colSpan=4, style={'color':'white'})]))
    #table_style = {'border': '1px solid #444', 'borderCollapse': 'collapse', 'color':'white'}
    table_style = {
        'borderCollapse': 'collapse',
        'width': '100%',
        'border': '1px solid #2a2a2a',
        'color': '#f5f5f5',
        'backgroundColor': '#1e1e1e',
        'fontFamily': 'Segoe UI, Roboto, sans-serif',
        'fontSize': '14px',
        'textAlign': 'center',
        'borderRadius': '8px',
        'overflow': 'hidden',
        'boxShadow': '0 2px 10px rgba(0, 0, 0, 0.3)'
    }
    return html.Table(header + rows, style=table_style)

def build_layout():
    return html.Div([
    html.H1("Sensor Data Dashboard", style={'textAlign': 'center', 'color': 'white'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select a CSV File', style={'color': 'white', 'textDecoration': 'underline', 'cursor' : 'pointer'})]),
        style={'width': '50%', 'height': '60px', 'lineHeight': '60px',
               'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
               'textAlign': 'center', 'margin': '10px auto', 'color': 'white',
               'backgroundColor': 'rgba(50,50,50,0.7)'},
        multiple=False
    ),
    html.Div([
        dcc.Store(id='uploaded-file-store'),  # store last uploaded contents+filename
        html.Button("Reload File", id="reload-button", n_clicks=0,
            style={
                'margin': '10px',
                "margin-top" : "20px",
                "color" : "white",
                "background-color" : "blue",
                "border-radius" : "20px",
                "padding" : "12px",
                "cursor" : "pointer"
            }),
    ], style={
        "display" : "flex",
        "justify-content" : "center"
    }),
    html.Div(id='output-data-upload', style={'color': 'white', 'textAlign': 'center'}),
    html.Div([
        html.Label("Select Sensor ID:", style={'color': 'white'}),
        dcc.Dropdown(id='sensor-dropdown', placeholder="Select an ID", style={'color': 'black'})
    ], style={'width': '30%', 'margin': '20px auto'}),
    html.Div([
        dcc.Store(id='selected-points-store'),
        dcc.Store(id="df-selected-store"),
        dcc.Graph(id='gas-graph', config={'displayModeBar': True, 'modeBarButtonsToAdd': ['select2d', 'lasso2d']}),
    ]),
    html.Div([
        dcc.Input(id='my-input', type='text', placeholder='Type something...'),
        html.Button('Submit', id='my-button', n_clicks=0, disabled=True),
        html.Div(id='output')
    ], style={
        'display': 'flex',
        'flexDirection': 'column',  # stack input + button
        'alignItems': 'center',     # horizontal center
        'gap': '10px'               # space between elements
    }),
    html.Div([
        html.Button("Export XLSX file", id="export-button", n_clicks=0, disabled=True,
        style={
            'margin': '10px', 
            "margin-top" : "20px",
            "color" : "white", 
            "background-color" : "green",
            "border-radius" : "20px",
            "padding" : "12px",
            "cursor" : "pointer"
        })
    ], style={
        "display" : "flex",
        "justify-content" : "center"
    }),
    dcc.Download(id="download-xlsx"),
    html.Div(id='selected-data-output', style={'color': 'white', 'margin': '20px'}),
    
    ], style={'backgroundColor': 'rgba(20,20,20,0.9)', 'minHeight': '100vh', 'padding': '20px'})


