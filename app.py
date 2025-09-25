# app.py

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import io
import base64

# Importar módulos del proyecto
from modules.config import Config
from modules.data_processor import load_csv_data, parse_spanish_dates
from modules.sidebar import create_sidebar
# Iremos importando más funciones de visualizer a medida que las migremos
from modules.visualizer import display_welcome_tab_dash 

# Inicializar la aplicación Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Layout principal
app.layout = dbc.Container(
    [
        # Almacenes de datos en el navegador
        dcc.Store(id='store-stations-data'),
        dcc.Store(id='store-precip-data'),
        
        # Fila del Título
        dbc.Row([
            dbc.Col(html.Img(src=app.get_asset_url('CuencaVerde_Logo.jpg'), height="50px"), width="auto"),
            dbc.Col(html.H1(Config.APP_TITLE), className="mt-2"),
        ]),
        
        html.Hr(),
        
        # Fila Principal con Sidebar y Contenido
        dbc.Row(
            [
                dbc.Col(id='sidebar-column', width=12, lg=3, className="bg-light"),
                dbc.Col(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(label="Bienvenida", tab_id="tab-bienvenida"),
                                dbc.Tab(label="Distribución Espacial", tab_id="tab-distribucion", disabled=True),
                                dbc.Tab(label="Gráficos", tab_id="tab-graficos", disabled=True),
                                # Agregaremos más pestañas aquí
                            ],
                            id="tabs",
                            active_tab="tab-bienvenida",
                        ),
                        html.Div(id="tab-content", className="p-4"),
                    ],
                    width=12, lg=9
                )
            ]
        )
    ],
    fluid=True,
)

# Callback para actualizar el contenido de las pestañas
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    if active_tab == "tab-bienvenida":
        return display_welcome_tab_dash()
    # En el futuro, añadiremos más condiciones aquí
    # elif active_tab == "tab-distribucion":
    #     return display_spatial_distribution_tab(...)
    return html.P("Esta pestaña está en construcción.")

# Callback para cargar y procesar los datos
@app.callback(
    Output('store-stations-data', 'data'),
    Input('upload-mapa', 'contents'),
    State('upload-mapa', 'filename')
)
def Cargar_datos_estaciones(contents, filename):
    if contents:
        df = load_csv_data_from_contents(contents, filename)
        if df is not None:
            return df.to_json(date_format='iso', orient='split')
    return dash.no_update

# Callback para generar la barra lateral (depende de si los datos están cargados)
@app.callback(
    Output('sidebar-column', 'children'),
    Input('store-stations-data', 'data')
)
def update_sidebar(stations_json):
    if stations_json:
        df_stations = pd.read_json(stations_json, orient='split')
        return create_sidebar(df_stations)
    # Barra lateral por defecto si no hay datos
    return create_sidebar(pd.DataFrame({
        Config.REGION_COL: [],
        Config.MUNICIPALITY_COL: []
    }))

# Función auxiliar para decodificar archivos subidos
def load_csv_data_from_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), sep=';')
    except Exception as e:
        print(e)
        df = pd.read_csv(io.StringIO(decoded.decode('latin-1')), sep=';')
    
    df.columns = df.columns.str.strip().str.lower()
    return df

if __name__ == '__main__':
    app.run_server(debug=True)
