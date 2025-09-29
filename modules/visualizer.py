# modules/visualizer.py

#--- Importaciones Estándar y de Terceros
import streamlit as st
import pandas as pd
import geopandas as gpd
import altair as alt
import folium
from folium.plugins import MarkerCluster, MiniMap
from folium.raster_layers import WmsTileLayer
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import base64
import branca.colormap as cm
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from scipy import stats
from scipy.interpolate import Rbf
import pymannkendall as mk
from prophet.plot import plot_plotly
import io

#--- Importaciones de Módulos Propios
from modules.analysis import calculate_spi, calculate_spei, calculate_monthly_anomalies, \
    calculate_percentiles_and_extremes
from modules.config import Config
from modules.utils import add_folium_download_button
from modules.interpolation import create_interpolation_surface
from modules.forecasting import (
    generate_sarima_forecast, generate_prophet_forecast,
    get_decomposition_results, create_acf_chart, create_pacf_chart
)

#--- DISPLAY UTILS ---
def display_filter_summary(total_stations_count, selected_stations_count, year_range,
                         selected_months_count):
    if isinstance(year_range, tuple) and len(year_range) == 2:
        year_text = f"{year_range[0]}-{year_range[1]}"
    else:
        year_text = "N/A"
    summary_text = (
        f"**Estaciones Seleccionadas:** {selected_stations_count} de {total_stations_count} | "
        f"**Período:** {year_text} | "
        f"**Meses:** {selected_months_count} de 12"
    )
    st.info(summary_text)

def get_map_options():
    return {
        "CartoDB Positron (Predeterminado)": {"tiles": "cartodbpositron", "attr": '&copy; <a href="https://carto.com/attributions">CartoDB</a>', "overlay": False},
        "OpenStreetMap": {"tiles": "OpenStreetMap", "attr": '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors', "overlay": False},
        "Topografía (Open TopoMap)": {"tiles": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", "attr": 'Map data: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, <a href="http://viewfinderpanoramas.org">SRTM</a> | Map style: &copy; <a href="https://opentopomap.org">Open Topo Map</a> (<a href="https://creativecommons.org/licenses/by-sa/3.0/">CC-BY-SA</a>)', "overlay": False},
    }

def display_map_controls(container_object, key_prefix):
    map_options = get_map_options()
    base_maps = {k: v for k, v in map_options.items() if not v.get("overlay")}
    selected_base_map_name = container_object.selectbox("Seleccionar Mapa Base",
                                                      list(base_maps.keys()), key=f"{key_prefix}_base_map")
    return base_maps[selected_base_map_name], []

def generate_station_popup_html(row, df_anual_melted, include_chart=False, df_monthly_filtered=None):
    # This function seems to be from an older version in the PDF, but we'll keep its structure.
    # It might need further review if popups on maps are also an issue.
    station_name = row.get(Config.STATION_NAME_COL, 'N/A')
    popup_html = f"<h4>{station_name}</h4>"
    popup_html += f"<p><b>Municipio:</b> {row.get(Config.MUNICIPALITY_COL, 'N/A')}</p>"
    popup_html += f"<p><b>Altitud:</b> {row.get(Config.ALTITUDE_COL, 'N/A')} m</p>"
    return folium.Popup(popup_html, max_width=300)

def create_folium_map(location, zoom, base_map_config, overlays_config, fit_bounds_data=None):
    m = folium.Map(location=location, zoom_start=zoom, tiles=base_map_config.get("tiles", "OpenStreetMap"),
                   attr=base_map_config.get("attr", None))
    if fit_bounds_data is not None and not fit_bounds_data.empty:
        if len(fit_bounds_data) > 1:
            bounds = fit_bounds_data.total_bounds
            if np.all(np.isfinite(bounds)):
                m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        elif len(fit_bounds_data) == 1:
            point = fit_bounds_data.iloc[0].geometry
            m.location = [point.y, point.x]
            m.zoom_start = 12
    return m


#--- MAIN TAB DISPLAY FUNCTIONS
def display_welcome_tab():
    st.header("Bienvenido al Sistema de Información de Lluvias y Clima")
    st.markdown(Config.WELCOME_TEXT, unsafe_allow_html=True)
    if os.path.exists(Config.LOGO_PATH):
        try:
            with open(Config.LOGO_PATH, "rb") as f:
                logo_bytes = f.read()
            st.image(logo_bytes, width=250, caption="Corporación Cuenca Verde")
        except Exception:
            st.warning("No se pudo cargar el logo de bienvenida.")

def display_spatial_distribution_tab(gdf_filtered, stations_for_analysis, df_anual_melted,
                                     df_monthly_filtered):
    st.header("Distribución espacial de las Estaciones de Lluvia")
    display_filter_summary(
        total_stations_count=len(st.session_state.gdf_stations),
        selected_stations_count=len(stations_for_analysis),
        year_range=st.session_state.year_range,
        selected_months_count=len(st.session_state.meses_numeros)
    )
    # This section can be simplified or expanded as needed.
    st.map(gdf_filtered)

def display_graphs_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis, gdf_filtered):
    st.header("Visualizaciones de Precipitación")
    display_filter_summary(
        total_stations_count=len(st.session_state.gdf_stations),
        selected_stations_count=len(stations_for_analysis),
        year_range=st.session_state.year_range,
        selected_months_count=len(st.session_state.meses_numeros)
    )

    if not stations_for_analysis:
        st.warning("Por favor, seleccione al menos una estación para ver esta sección.")
        return

    # --- ENRIQUECIMIENTO DE DATAFRAMES ---
    # With the fix in data_processor, df_monthly_filtered should already have metadata.
    # However, df_anual_melted loses it during aggregation, so we must re-merge it.
    df_monthly_rich = df_monthly_filtered.copy()
    df_anual_rich = df_anual_melted.copy()
    
    metadata_cols = [col for col in [Config.STATION_NAME_COL, Config.MUNICIPALITY_COL, Config.ALTITUDE_COL] if col in gdf_filtered.columns]
    if len(metadata_cols) > 1:
        station_metadata = gdf_filtered[metadata_cols].drop_duplicates(subset=[Config.STATION_NAME_COL])
        df_anual_rich = pd.merge(df_anual_rich, station_metadata, on=Config.STATION_NAME_COL, how='left')

    sub_tab_anual, sub_tab_mensual, sub_tab_comparacion, sub_tab_distribucion, \
    sub_tab_acumulada, sub_tab_altitud, sub_tab_regional = \
        st.tabs(["Análisis Anual", "Análisis Mensual", "Comparación Rápida", "Distribución",
                 "Acumulada", "Relación Altitud", "Serie Regional"])

    with sub_tab_anual:
        if not df_anual_rich.empty:
            st.subheader("Precipitación Anual por Estación")
            chart_anual = alt.Chart(df_anual_rich.dropna(subset=[Config.PRECIPITATION_COL])).mark_line(point=True).encode(
                x=alt.X(f'{Config.YEAR_COL}:O', title='Año'),
                y=alt.Y(f'{Config.PRECIPITATION_COL}:Q', title='Precipitación (mm)'),
                color=alt.Color(f'{Config.STATION_NAME_COL}:N', title='Estaciones'),
                tooltip=[
                    alt.Tooltip(f'{Config.STATION_NAME_COL}:N', title='Estación'),
                    alt.Tooltip(f'{Config.YEAR_COL}:O', title='Año'),
                    alt.Tooltip(f'{Config.PRECIPITATION_COL}:Q', format='.0f', title='Ppt. Anual (mm)'),
                    alt.Tooltip(f'{Config.MUNICIPALITY_COL}:N', title='Municipio'),
                    alt.Tooltip(f'{Config.ALTITUDE_COL}:Q', format='.0f', title='Altitud (m)')
                ]
            ).interactive()
            st.altair_chart(chart_anual, use_container_width=True)
        else:
            st.warning("No hay datos anuales para mostrar.")

    with sub_tab_mensual:
        anual_graf_tab, anual_analisis_tab, tabla_datos_tab = st.tabs(["Gráfico de Serie Mensual", "Análisis ENSO en el Período", "Tabla de Datos"])
        with anual_graf_tab:
            if not df_monthly_rich.empty:
                st.subheader("Serie de Precipitación Mensual")
                # --- CORRECCIÓN FINAL DEL TOOLTIP ---
                base_chart = alt.Chart(df_monthly_rich).encode(
                    x=alt.X(f'{Config.DATE_COL}:T', title='Fecha'),
                    y=alt.Y(f'{Config.PRECIPITATION_COL}:Q', title='Precipitación (mm)'),
                    color=alt.Color(f'{Config.STATION_NAME_COL}:N', title='Estaciones'),
                    tooltip=[
                        alt.Tooltip(f'{Config.DATE_COL}:T', format='%Y-%m', title='Fecha'),
                        alt.Tooltip(f'{Config.PRECIPITATION_COL}:Q', format='.0f', title='Ppt. Mensual (mm)'),
                        alt.Tooltip(f'{Config.STATION_NAME_COL}:N', title='Estación'),
                        # Se elimina la línea de 'Origen'
                        alt.Tooltip(f'{Config.MONTH_COL}:O', title="Mes"),
                        alt.Tooltip(f'{Config.MUNICIPALITY_COL}:N', title='Municipio'),
                        alt.Tooltip(f'{Config.ALTITUDE_COL}:Q', format='.0f', title='Altitud (m)')
                    ]
                )
                final_chart = base_chart.mark_line(opacity=0.3) + base_chart.mark_point(filled=True, size=60)
                st.altair_chart(final_chart.interactive(), use_container_width=True)
            else:
                st.warning("No hay datos mensuales para mostrar.")
    # Add other tabs as needed...

def display_advanced_maps_tab(gdf_filtered, stations_for_analysis, df_anual_melted, df_monthly_filtered):
    st.header("Mapas Avanzados")
    gif_tab, temporal_tab, race_tab, anim_tab, compare_tab, kriging_tab = st.tabs([
        "Animación GIF (Antioquia)", "Visualización Temporal", "Gráfico de Carrera", 
        "Mapa Animado", "Comparación de Mapas", "Interpolación Comparativa"
    ])
    with gif_tab:
        st.subheader("Distribución Espacio-Temporal de la Lluvia en Antioquia")
        gif_path = Config.GIF_PATH
        if os.path.exists(gif_path):
            st.image(gif_path, caption="Animación PPAM", width=600)
        else:
            st.error(f"No se pudo encontrar el archivo GIF en la ruta: {gif_path}")
            
# Dummy functions for other tabs to make the file complete
def display_anomalies_tab(df_long, df_monthly_filtered, stations_for_analysis):
    st.header("Análisis de Anomalías")
    st.info("Contenido no implementado en esta versión.")

def display_drought_analysis_tab(df_monthly_filtered, gdf_filtered, stations_for_analysis):
    st.header("Análisis de Extremos Hidrológicos")
    st.info("Contenido no implementado en esta versión.")
    
def display_stats_tab(df_long, df_anual_melted, df_monthly_filtered, stations_for_analysis, gdf_filtered):
    st.header("Estadísticas")
    st.info("Contenido no implementado en esta versión.")

def display_correlation_tab(df_monthly_filtered, stations_for_analysis):
    st.header("Análisis de Correlación")
    st.info("Contenido no implementado en esta versión.")

def display_enso_tab(df_monthly_filtered, df_enso, gdf_filtered, stations_for_analysis):
    st.header("Análisis ENSO")
    st.info("Contenido no implementado en esta versión.")

def display_trends_and_forecast_tab(df_anual_melted, df_monthly_to_process, stations_for_analysis):
    st.header("Tendencias y Pronósticos")
    st.info("Contenido no implementado en esta versión.")

def display_downloads_tab(df_anual_melted, df_monthly_filtered, stations_for_analysis):
    st.header("Descargas")
    st.info("Contenido no implementado en esta versión.")

def display_station_table_tab(gdf_filtered, df_anual_melted, stations_for_analysis):
    st.header("Tabla de Estaciones")
    st.info("Contenido no implementado en esta versión.")
