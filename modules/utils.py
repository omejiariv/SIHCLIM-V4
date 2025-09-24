# En modules/utils.py

# ... (tus otras importaciones y funciones existentes) ...
import io
from fpdf import FPDF
import streamlit as st
from datetime import datetime

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
    pdf.cell(0, 8, f"Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'L')
    pdf.ln(5)

    # 2. Resumen de Filtros Activos
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Filtros Aplicados', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    # Extraer filtros del session_state
    year_range_val = st.session_state.get('year_range', ('N/A', 'N/A'))
    if isinstance(year_range_val[0], int): # Modo normal
        pdf.multi_cell(0, 5, f"- Período: {year_range_val[0]} - {year_range_val[1]}")
    
    selected_stations = st.session_state.get('station_multiselect', [])
    pdf.multi_cell(0, 5, f"- Estaciones seleccionadas: {len(selected_stations)}")
    # (Puedes añadir más filtros aquí si lo deseas)
    pdf.ln(10)

    # 3. Añadir Gráficos (si existen)
    if 'report_fig_anual_avg' in st.session_state:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Gráfico: Precipitación Media Multianual', 0, 1, 'L')
        
        fig = st.session_state['report_fig_anual_avg']
        # Guardar la figura en un buffer de memoria como imagen PNG
        img_bytes = io.BytesIO()
        fig.write_image(img_bytes, format='png', width=800, height=400, scale=2)
        img_bytes.seek(0)
        
        # Insertar la imagen en el PDF (w=190mm para que ocupe casi todo el ancho)
        pdf.image(img_bytes, x=10, w=190)
        pdf.ln(10)

    # 4. Añadir Tablas (si existen)
    if 'report_df_stats_summary' in st.session_state:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Tabla: Resumen de Estadísticas Mensuales', 0, 1, 'L')
        pdf.set_font('Courier', '', 8) # Usar fuente monoespaciada para tablas

        df_summary = st.session_state['report_df_stats_summary']
        
        # Convertir DataFrame a un texto simple formateado para el PDF
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

    # Retornar el PDF como bytes
    return pdf.output(dest='S').encode('latin-1')
