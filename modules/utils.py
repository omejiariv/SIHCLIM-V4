# modules/utils.py

import streamlit as st
import base64
import pandas as pd
import io
from fpdf import FPDF
from datetime import datetime

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

# Clase para definir el formato del PDF con encabezado y pie de página
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Reporte - Sistema de Información de Lluvias y Clima', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        page_num = self.page_no()
        self.cell(0, 10, f'Página {page_num}', 0, 0, 'C')

def generate_pdf_report():
    """
    Genera un reporte en PDF resumiendo el análisis actual.
    """
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', '', 12)

    # 1. Título y Fecha
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Resumen del Análisis de Precipitación', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 8, f"Generado el: {datetime.now().strftime('%Y%m%d_%H%M%S')}", 0, 1, 'L')
    pdf.ln(5)

    # 2. Resumen de Filtros Activos
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Filtros Aplicados', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    # Extraer filtros del session_state
    year_range_val = st.session_state.get('year_range', ('N/A', 'N/A'))
    if isinstance(year_range_val[0], int): # Modo normal
        pdf.multi_cell(0, 5, f"- Período: {year_range_val[0]} - {year_range_val[1]}")
    
    # --- INICIO DE LA CORRECCIÓN ---
    # En lugar de imprimir la lista completa, solo mostramos la cantidad.
    num_selected_stations = len(st.session_state.get('station_multiselect', []))
    pdf.multi_cell(0, 5, f"- Número de estaciones seleccionadas: {num_selected_stations}")
    # --- FIN DE LA CORRECCIÓN ---
    
    pdf.ln(10)

    # 3. Añadir Gráficos (si existen)
    if 'report_fig_anual_avg' in st.session_state:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Gráfico: Precipitación Media Multianual', 0, 1, 'L')
        
        fig = st.session_state['report_fig_anual_avg']
        img_bytes = io.BytesIO()
        fig.write_image(img_bytes, format='png', width=800, height=400, scale=2)
        img_bytes.seek(0)
        
        pdf.image(img_bytes, x=10, w=190)
        pdf.ln(10)

    # 4. Añadir Tablas (si existen)
    if 'report_df_stats_summary' in st.session_state:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Tabla: Resumen de Estadísticas Mensuales', 0, 1, 'L')
        pdf.set_font('Courier', '', 8)

        df_summary = st.session_state['report_df_stats_summary']
        
        header = "{:<25} | {:>10} | {:>10} | {:>10}".format(
            "Estación", "Máx (mm)", "Prom (mm)", "Mín (mm)"
        )
        pdf.cell(0, 5, header, 0, 1)
        pdf.cell(0, 2, "-"*65, 0, 1)
        
        for index, row in df_summary.iterrows():
            line = "{:<25.25} | {:>10.0f} | {:>10.0f} | {:>10.0f}".format(
                row['Estación'],
                row['Ppt. Máxima Mensual (mm)'],
                row['Promedio Mensual (mm)'],
                row['Ppt. Mínima Mensual (mm)']
            )
            pdf.cell(0, 5, line, 0, 1)
        pdf.ln(10)

    return pdf.output(dest='S').encode('latin-1')
