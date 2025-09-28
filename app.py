import streamlit as st
import pandas as pd
import geopandas as gpd
import altair as alt

from modules.config import Config
from modules.data_processor import load_and_process_all_data
from modules.visualizer import (
    display_welcome_tab,
    display_map_analysis_tab,
    display_temporal_analysis_tab,
    display_enso_tab,
    display_correlation_tab,
    display_forecast_tab,
    display_stats_tab,
    display_advanced_maps_tab,
    display_about_tab,
)

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="SIHCLIM-V4: Sistema de Información Hidroclimática",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurar el estilo base
st.markdown("""
    <style>
    .reportview-container .main {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #27818E;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1a5c66;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    .stFileUploader {
        margin-top: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Inicializa las variables de estado de sesión necesarias."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df_long' not in st.session_state:
        st.session_state.df_long = None
    if 'gdf_stations' not in st.session_state:
        st.session_state.gdf_stations = None
    if 'gdf_municipios' not in st.session_state:
        st.session_state.gdf_municipios = None
    if 'df_enso' not in st.session_state:
        st.session_state.df_enso = None
    if 'gif_reload_key' not in st.session_state:
        # Clave para forzar la recarga del GIF
        st.session_state.gif_reload_key = 0


def sidebar_controls():
    """Dibuja los controles de carga de datos en la barra lateral."""
    
    st.sidebar.title("🛠️ Controles de la Aplicación")

    st.sidebar.header("1. Carga de Datos Base")
    st.sidebar.markdown("Cargue los 3 archivos necesarios para inicializar la aplicación:")

    # Controles de carga de archivos
    uploaded_file_mapa = st.sidebar.file_uploader(
        "Archivo de Metadatos de Estaciones (.csv)", type=['csv'], key="upload_metadata_file")
    uploaded_file_precip = st.sidebar.file_uploader(
        "Archivo de Precipitaciones/ENSO (.csv)", type=['csv'], key="upload_precip_file")
    uploaded_zip_shapefile = st.sidebar.file_uploader(
        "Shapefile de Municipios (.zip)", type=['zip'], key="upload_shapefile")

    st.sidebar.markdown("---")
    
    # --------------------------------------------------------------------------------
    # LÓGICA DE PROCESAMIENTO
    # --------------------------------------------------------------------------------
    
    if st.sidebar.button("Procesar y Almacenar Datos", key='process_data_button'):
        if all([uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile]):
            
            # 💥 CRÍTICO: Limpiar cachés de assets (incluye GIFs/archivos grandes) 💥
            st.cache_resource.clear() 
            
            with st.spinner("Procesando archivos y cargando datos... Esto puede tardar unos segundos."):
                
                # Llamada a la función de procesamiento (definida en data_processor.py)
                gdf_stations, gdf_municipios, df_long, df_enso = load_and_process_all_data(
                    uploaded_file_mapa, uploaded_file_precip, uploaded_zip_shapefile)
                    
                if gdf_stations is not None and df_long is not None and gdf_municipios is not None:
                    # Almacenar en session_state tras procesamiento exitoso
                    st.session_state.gdf_stations = gdf_stations
                    st.session_state.gdf_municipios = gdf_municipios
                    st.session_state.df_long = df_long
                    st.session_state.df_enso = df_enso
                    
                    st.session_state.data_loaded = True
                    st.sidebar.success("¡Datos cargados y listos!")
                    
                    # Forzar un rerun para actualizar la interfaz principal
                    st.rerun() 
                else:
                    st.sidebar.error("Hubo un error al procesar los archivos. Verifique el formato de los datos.")
        else:
            st.sidebar.warning("Debe cargar los 3 archivos requeridos antes de procesar.")
    
    st.sidebar.markdown("---")
    
    # --------------------------------------------------------------------------------
    # CONTROLES DE FILTRADO (Solo si los datos ya están cargados)
    # --------------------------------------------------------------------------------
    if st.session_state.data_loaded:
        st.sidebar.header("2. Filtros de Análisis")
        
        df = st.session_state.df_long
        
        # Filtro de Años
        min_year = int(df[Config.YEAR_COL].min())
        max_year = int(df[Config.YEAR_COL].max())
        
        selected_year_range = st.sidebar.slider(
            "Rango de Años:",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year),
            step=1
        )
        
        # Filtro de Meses
        all_months = list(Config.MONTH_MAP.keys())
        selected_months = st.sidebar.multiselect(
            "Meses:",
            options=all_months,
            default=all_months
        )

        # Filtro de Estaciones (por nombre o ID)
        all_stations = st.session_state.gdf_stations[Config.STATION_NAME_COL].unique()
        selected_stations = st.sidebar.multiselect(
            "Estaciones Seleccionadas:",
            options=all_stations,
            default=all_stations
        )
        
        # Almacenar filtros para que el main pueda acceder a ellos
        st.session_state.selected_year_range = selected_year_range
        st.session_state.selected_months = selected_months
        st.session_state.selected_stations = selected_stations


def main():
    """Función principal de la aplicación Streamlit."""
    
    initialize_session_state()
    sidebar_controls()
    
    st.title("Sistema de Información Hidroclimática - SIHCLIM V4")
    
    # --- LÓGICA DE CONTROL DEL FLUJO PRINCIPAL (CRÍTICA) ---
    
    # La aplicación principal solo se ejecuta si los datos están cargados
    if st.session_state.data_loaded and st.session_state.df_long is not None:
        
        # 1. Recuperar los datos base y los filtros
        df_long_base = st.session_state.df_long
        gdf_stations_base = st.session_state.gdf_stations
        gdf_municipios_base = st.session_state.gdf_municipios
        df_enso_base = st.session_state.df_enso
        
        # Recuperar filtros del sidebar
        selected_year_range = st.session_state.selected_year_range
        selected_months = st.session_state.selected_months
        selected_stations = st.session_state.selected_stations

        # 2. Aplicar Filtros (CRÍTICO: Esto debe hacerse sin fallos)
        
        # Convertir meses seleccionados a números (1-12)
        selected_month_nums = [Config.MONTH_MAP[m] for m in selected_months]
        
        # Aplicar filtro de tiempo y estación al DataFrame largo
        df_filtered = df_long_base[
            (df_long_base[Config.YEAR_COL] >= selected_year_range[0]) &
            (df_long_base[Config.YEAR_COL] <= selected_year_range[1]) &
            (df_long_base[Config.MONTH_COL].isin(selected_month_nums)) &
            (df_long_base[Config.STATION_NAME_COL].isin(selected_stations))
        ].copy()
        
        # Filtrar GeoDataFrame de estaciones
        gdf_filtered = gdf_stations_base[
            gdf_stations_base[Config.STATION_NAME_COL].isin(selected_stations)
        ].copy()
        
        # 3. Preparar DataFrames Agregados
        
        # Dataframe Anual Derivado (para Series de Tiempo)
        df_anual_melted = df_filtered.groupby([Config.YEAR_COL, Config.STATION_NAME_COL])[Config.PRECIP_COL].sum().reset_index()
        df_anual_melted.rename(columns={Config.PRECIP_COL: 'PPAM'}, inplace=True)
        
        # Dataframe Mensual (para Climatología/Anomalías)
        df_monthly_filtered = df_filtered.groupby([Config.YEAR_COL, Config.MONTH_COL])[Config.PRECIP_COL].mean().reset_index()

        # Nombres de las estaciones seleccionadas para el análisis
        stations_for_analysis = gdf_filtered[Config.STATION_NAME_COL].tolist()
        
        # --- 4. DESPLIEGUE DE PESTAÑAS (Aquí es donde se rompía el flujo) ---

        tab_names = [
            "1. Mapa de Análisis", "2. Series de Tiempo", "3. Análisis ENSO", 
            "4. Correlación y Pronóstico", "5. Pronóstico", "6. Estadísticas", 
            "7. Mapas Avanzados (GIF/IDW)", "8. Acerca De"
        ]
        
        # 💥 CRÍTICO: Crear las pestañas. El código debe llegar a este punto. 💥
        tabs = st.tabs(tab_names)
        
        # Verificar que se seleccionó al menos una estación antes de pasar a las visualizaciones
        if df_filtered.empty or gdf_filtered.empty:
            st.warning("No hay datos disponibles para los filtros aplicados. Ajuste la selección de años, meses o estaciones.")
            display_welcome_tab()
            return

        # 5. Lógica de Visualización (Llamada a las funciones de modules/visualizer.py)
        
        with tabs[0]:
            display_map_analysis_tab(gdf_filtered, gdf_municipios_base, df_filtered)

        with tabs[1]:
            display_temporal_analysis_tab(df_filtered, df_anual_melted, gdf_filtered)

        with tabs[2]:
            display_enso_tab(df_filtered, df_enso_base, gdf_filtered)

        with tabs[3]:
            display_correlation_tab(df_filtered)

        with tabs[4]:
            # El pronóstico requiere el DataFrame de series de tiempo
            display_forecast_tab(df_anual_melted)

        with tabs[5]:
            display_stats_tab(df_long_base, df_anual_melted, df_monthly_filtered, stations_for_analysis, gdf_filtered)

        with tabs[6]:
            # Mapas avanzados (GIF, IDW)
            display_advanced_maps_tab(gdf_filtered, df_filtered)
        
        with tabs[7]:
            display_about_tab()

    else:
        # Si los datos no se han cargado, mostrar la pantalla de bienvenida
        display_welcome_tab()
        st.info("Para comenzar, por favor cargue los 3 archivos requeridos en la barra lateral izquierda y haga clic en 'Procesar y Almacenar Datos'.")


if __name__ == "__main__":
    main()
