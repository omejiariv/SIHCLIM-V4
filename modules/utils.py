# modules/utils.py

import streamlit as st
import base64
import pandas as pd

def standardize_numeric_column(series):
    """
    Convierte una serie de pandas a tipo numérico de forma robusta,
    manejando comas como separadores decimales y otros errores.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series
    # Reemplazar comas por puntos y convertir a numérico, forzando errores a NaN
    return pd.to_numeric(series.astype(str).str.replace(',', '.', regex=False), errors='coerce')


def add_folium_download_button(folium_map, filename="mapa.html"):
    """Añade un botón de descarga para un mapa de Folium."""
    map_html = folium_map.get_root().render()
    st.download_button(
        label=f"📥 Descargar Mapa ({filename})",
        data=map_html,
        file_name=filename,
        mime="text/html"
    )
