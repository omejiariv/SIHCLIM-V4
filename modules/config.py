# modules/config.py

import streamlit as st
import pandas as pd
import os

# Define la ruta base del proyecto de forma robusta
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    # --- Nombres de Columnas de Datos ---
    STATION_NAME_COL = 'nom_est'
    PRECIPITATION_COL = 'precipitation'
    LATITUDE_COL = 'latitud_geo'
    LONGITUDE_COL = 'longitud_geo'
    YEAR_COL = 'año'
    MONTH_COL = 'mes'
    DATE_COL = 'fecha_mes_año'
    ENSO_ONI_COL = 'anomalia_oni'
    ORIGIN_COL = 'origen'
    ALTITUDE_COL = 'alt_est'
    MUNICIPALITY_COL = 'municipio'
    REGION_COL = 'depto_region'
    PERCENTAGE_COL = 'porc_datos'
    CELL_COL = 'celda_xy'
    ET_COL = 'et_mmy'
    ELEVATION_COL = 'elevation_dem' # <-- LÍNEA AÑADIDA

    # --- Índices climáticos ---
    SOI_COL = 'soi'
    IOD_COL = 'iod'

    # --- Rutas de Archivos ---
    LOGO_PATH = os.path.join(BASE_DIR, "data", "CuencaVerde_Logo.jpg")
    LOGO_DROP_PATH = os.path.join(BASE_DIR, "data", "CuencaVerde_Logo.jpg")
    GIF_PATH = os.path.join(BASE_DIR, "data", "PPAM.gif")

    # --- Mensajes de la UI ---
    APP_TITLE = "Sistema de información de las lluvias y el Clima en el norte de la región Andina"
    WELCOME_TEXT = """
    <p style="text-align: center; font-style: italic; font-size: 1.1em;">
    "El futuro, también depende del pasado y de nuestra capacidad presente para anticiparlo". — omr.
    </p>
    <hr>
    <p>
    Esta plataforma interactiva está diseñada para la visualización y análisis de datos históricos de
    precipitación y su relación con el fenómeno ENSO en el norte de la región Andina.
    </p>

    <h4>¿Cómo empezar?</h4>
    <ol>
        <li>
            <b>Cargue sus archivos:</b> Si es la primera vez que usa la aplicación, el panel de la izquierda le
            solicitará cargar los archivos de estaciones, precipitación y el shapefile de municipios.
            La aplicación recordará estos archivos en su sesión.
        </li>
        <li>
            <b>Filtre los datos:</b> Una vez cargados los datos, utilice el <b>Panel de Control</b> en la barra
            lateral para filtrar las estaciones por ubicación (región, municipio), altitud,
            porcentaje de datos disponibles, y para seleccionar el período de análisis (años y meses).
        </li>
        <li>
            <b>Explore las pestañas:</b> Cada pestaña ofrece una perspectiva diferente de los datos.
            Navegue a través de ellas para descubrir:
            <ul>
                <li><b>Distribución Espacial:</b> Mapas interactivos de las estaciones.</li>
                <li><b>Gráficos:</b> Series de tiempo anuales, mensuales, comparaciones y distribuciones.</li>
                <li><b>Mapas Avanzados:</b> Animaciones y mapas de interpolación.</li>
                <li><b>Análisis de Anomalías:</b> Desviaciones de la precipitación respecto a la media histórica.</li>
                <li><b>Tendencias y Pronósticos:</b> Análisis de tendencias a largo plazo y modelos de pronóstico.</li>
            </ul>
            <p>
            Utilice el botón <b>🔄 Limpiar Filtros</b> en el panel lateral para reiniciar su selección en cualquier
            momento.
            </p>
        </li>
    </ol>
    """

    @staticmethod
    def initialize_session_state():
        """Inicializa todas las variables necesarias en el estado de la sesión de Streamlit."""
        state_defaults = {
            'data_loaded': False,
            'analysis_mode': "Usar datos originales",
            'select_all_checkbox': True,
            'filtered_station_options': [],
            'station_multiselect': [],
            'df_monthly_processed': pd.DataFrame(),
            'gdf_stations': None,
            'df_precip_anual': None,
            'gdf_municipios': None,
            'df_long': None,
            'df_enso': None,
            'min_data_perc_slider': 0,
            'altitude_multiselect': [],
            'regions_multiselect': [],
            'municipios_multiselect': [],
            'celdas_multiselect': [],
            'exclude_na': False,
            'exclude_zeros': False,
            'uploaded_forecast': None,
            'sarima_forecast': None,
            'prophet_forecast': None,
            'year_range': (1970, 2021),
            'meses_nombres': ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'],
            'meses_numeros': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'gif_reload_key': 0
        }
        for key, value in state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
