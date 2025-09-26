# En modules/utils.py

# ... (tus otras funciones)

def display_filter_summary(total_stations_count, selected_stations_count, year_range, selected_months_count):
    """
    Muestra una caja de resumen de filtros estilizada y con un logo.
    """
    # Convertir el logo a base64 para insertarlo en el HTML
    logo_base64 = ""
    if os.path.exists(Config.LOGO_PATH):
        with open(Config.LOGO_PATH, "rb") as image_file:
            logo_base64 = base64.b64encode(image_file.read()).decode()

    # Formatear el rango de años
    if isinstance(year_range, tuple) and len(year_range) == 2:
        year_text = f"{year_range[0]} – {year_range[1]}"
    else:
        year_text = "N/A"

    # Crear el HTML para la caja de resumen
    summary_html = f"""
    <div style="
        border: 1px solid #99c2ff;
        border-radius: 5px;
        padding: 10px;
        background-color: #e6f0ff;
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    ">
        <img src="data:image/png;base64,{logo_base64}" width="40" style="margin-right: 15px;">
        <div>
            <b>Resumen de Filtros:</b> 
            Estaciones Seleccionadas: <b>{selected_stations_count} de {total_stations_count}</b> | 
            Período: <b>{year_text}</b> | 
            Meses: <b>{selected_months_count} de 12</b>
        </div>
    </div>
    """
    
    st.markdown(summary_html, unsafe_allow_html=True)
