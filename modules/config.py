# modules/config.py

import streamlit as st
import pandas as pd
import os

class Config:
    # --- Configuración de la Aplicación ---
    APP_TITLE = "Sistema de Información de Lluvias y Clima en el norte de la región Andina"
    LOGO_PATH = "assets/CuencaVerde_Logo.jpg" 
    GIF_PATH = "assets/PPAM.gif"
    WELCOME_TEXT = """
    "El futuro, también depende del pasado y de nuestra capacidad presente para anticiparlo" -- omr.

    Esta plataforma interactiva está diseñada para la visualización y análisis de datos históricos de precipitación y su relación con el fenómeno ENSO en el norte de la región Andina.

    #### ¿Cómo empezar?
    1. **Cargar Archivos:** En el panel de la izquierda, suba los archivos de estaciones, precipitación y el shapefile de municipios.
    2. **Aplicar Filtros:** Utilice el **Panel de Control** para filtrar las estaciones y seleccionar el período de análisis.
    3. **Explorar Análisis:** Navegue a través de las pestañas para visualizar los datos.
    """

    # --- Nombres de Columnas Estándar (deben coincidir con la lógica de data_processor.py) ---
    DATE_COL = 'fecha_mes_año'
    PRECIPITATION_COL = 'precipitation'
    STATION_NAME_COL = 'nom_est'
    ALTITUDE_COL = 'alt_est'
    LATITUDE_COL = 'latitud_wgs84'
    LONGITUDE_COL = 'longitud_wgs84'
    MUNICIPALITY_COL = 'municipio'
    REGION_COL = 'depto_region'
    PERCENTAGE_COL = 'porc_datos'
    YEAR_COL = 'año'
    MONTH_COL = 'mes'
    ORIGIN_COL = 'origin'
    CELL_COL = 'celda_xy'
    ET_COL = 'et_mmy' # Evapotranspiración
    ELEVATION_COL = 'elevation_dem' # Usado para KED desde DEM
    
    # Índices Climáticos
    ENSO_ONI_COL = 'anomalia_oni'
    SOI_COL = 'soi'
    IOD_COL = 'iod'
    
    # --- Configuración para DEM ---
    # 💥 CORRECCIÓN DEM_SERVER_URL 💥
    DEM_SERVER_URL = "https://your-server-domain/dem.tif" # Debe ser reemplazada por tu URL real
    
    @staticmethod
    def initialize_session_state():
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'gdf_stations' not in st.session_state:
            st.session_state.gdf_stations = None
        if 'df_long' not in st.session_state:
            st.session_state.df_long = None
        if 'df_enso' not in st.session_state:
            st.session_state.df_enso = None
        if 'gdf_municipios' not in st.session_state:
            st.session_state.gdf_municipios = None
        if 'df_monthly_processed' not in st.session_state:
            st.session_state.df_monthly_processed = pd.DataFrame()
        if 'meses_numeros' not in st.session_state:
            st.session_state.meses_numeros = list(range(1, 13))
        if 'year_range' not in st.session_state:
            st.session_state.year_range = (2000, 2020)
        if 'dem_source' not in st.session_state:
            st.session_state.dem_source = "No usar DEM"
        if 'dem_raster' not in st.session_state:
            st.session_state.dem_raster = None
        if 'sarima_forecast' not in st.session_state:
            st.session_state.sarima_forecast = None
        if 'prophet_forecast' not in st.session_state:
            st.session_state.prophet_forecast = None
        if 'gif_reload_key' not in st.session_state:
            st.session_state.gif_reload_key = 0
