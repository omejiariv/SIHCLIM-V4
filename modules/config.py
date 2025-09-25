# modules/config.py

import streamlit as st
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
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
    ELEVATION_COL = 'elevation_dem'
    
    SOI_COL = 'soi'
    IOD_COL = 'iod'

    LOGO_PATH = os.path.join(BASE_DIR, "data", "CuencaVerde_Logo.jpg")
    GIF_PATH = os.path.join(BASE_DIR, "data", "PPAM.gif")

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
            <b>Cargar Archivos:</b> En el panel de la izquierda, sube los archivos de estaciones, precipitación y el shapefile de municipios.
        </li>
        <li>
            <b>Aplicar Filtros:</b> Utiliza el <b>Panel de Control</b> para filtrar las estaciones y seleccionar el período de análisis.
        </li>
        <li>
            <b>Explorar Análisis:</b> Navega a través de las pestañas para visualizar los datos.
        </li>
    </ol>
    """

    @staticmethod
    def initialize_session_state():
        state_defaults = {
            'data_loaded': False,
            'analysis_mode': "Usar datos originales",
            'select_all_checkbox': True,
            'gif_reload_key': 0
        }
        for key, value in state_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
