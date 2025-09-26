# modules/interpolation.py

import pandas as pd
import numpy as np
import gstools as gs
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from modules.config import Config

def interpolate_idw(lons, lats, vals, grid_lon, grid_lat, power=2):
    nx, ny = len(grid_lon), len(grid_lat)
    grid_z = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            x, y = grid_lon[i], grid_lat[j]
            distances = np.sqrt((lons - x)**2 + (lats - y)**2)
            if np.any(distances < 1e-10):
                grid_z[j, i] = vals[np.argmin(distances)]
                continue
            weights = 1.0 / (distances**power)
            weighted_sum = np.sum(weights * vals)
            total_weight = np.sum(weights)
            if total_weight > 0:
                grid_z[j, i] = weighted_sum / total_weight
            else:
                grid_z[j, i] = np.nan
    return grid_z.T

def create_interpolation_surface(year, method, variogram_model, gdf_filtered_map, df_anual_non_na):
    data_year_with_geom = pd.merge(
        df_anual_non_na[df_anual_non_na[Config.YEAR_COL] == year],
        gdf_filtered_map.drop_duplicates(subset=[Config.STATION_NAME_COL]),
        on=Config.STATION_NAME_COL
    )

    clean_cols = [Config.LONGITUDE_COL, Config.LATITUDE_COL, Config.PRECIPITATION_COL]
    if method == "Kriging con Deriva Externa (KED)":
        clean_cols.append(Config.ELEVATION_COL)

    df_clean = data_year_with_geom.dropna(subset=clean_cols).copy()
    df_clean = df_clean[np.isfinite(df_clean[clean_cols]).all(axis=1)]
    df_clean = df_clean.drop_duplicates(subset=[Config.LONGITUDE_COL, Config.LATITUDE_COL])

    if len(df_clean) < 4:
        error_msg = f"No hay suficientes datos válidos para el año {year} después de la limpieza."
        fig = go.Figure().update_layout(title=error_msg, xaxis_visible=False, yaxis_visible=False)
        return fig, None, error_msg

    lons = df_clean[Config.LONGITUDE_COL].values
    lats = df_clean[Config.LATITUDE_COL].values
    vals = df_clean[Config.PRECIPITATION_COL].values
    
    bounds = gdf_filtered_map.total_bounds
    grid_lon = np.linspace(bounds[0] - 0.1, bounds[2] + 0.1, 100)
    grid_lat = np.linspace(bounds[1] - 0.1, bounds[3] + 0.1, 100)
    z_grid, fig_variogram, error_message = None, None, None

    try:
        if method in ["Kriging Ordinario", "Kriging con Deriva Externa (KED)"]:
            model_map = {
                'gaussian': gs.Gaussian(dim=2), 'exponential': gs.Exponential(dim=2),
                'spherical': gs.Spherical(dim=2), 'linear': gs.Linear(dim=2)
            }
            model = model_map.get(variogram_model, gs.Spherical(dim=2))

            bin_center, gamma = gs.vario_estimate((lons, lats), vals)
            model.fit_variogram(bin_center, gamma, nugget=True)
            
            fig_variogram, ax = plt.subplots()
            ax.plot(bin_center, gamma, 'o', label='Experimental')
            model.plot(ax=ax, label='Modelo Ajustado')
            ax.set_xlabel('Distancia (grados)'); ax.set_ylabel('Semivarianza')
            ax.set_title(f'Variograma para {year}'); ax.legend()

            if method == "Kriging Ordinario":
                krig = gs.krige.Ordinary(model, (lons, lats), vals)
            else:
                drift_vals = df_clean[Config.ELEVATION_COL].values
                krig = gs.krige.ExtDrift(model, (lons, lats), vals, drift_src=drift_vals)
            
            z_grid, _ = krig.structured([grid_lon, grid_lat])

        elif method == "IDW":
            z_grid = interpolate_idw(lons, lats, vals, grid_lon, grid_lat)
        elif method == "Spline (Thin Plate)":
            rbf = Rbf(lons, lats, vals, function='thin_plate')
            grid_x, grid_y = np.meshgrid(grid_lon, grid_lat)
            z_grid = rbf(grid_x, grid_y)

    except Exception as e:
        error_message = f"Error al calcular {method} para el año {year}: {e}"
        fig = go.Figure().update_layout(title=error_message, xaxis_visible=False, yaxis_visible=False)
        return fig, None, error_message

    if z_grid is not None:
        fig = go.Figure(data=go.Contour(z=z_grid.T, x=grid_lon, y=grid_lat,
                                        colorscale=px.colors.sequential.YlGnBu,
                                        contours=dict(showlabels=True,
                                                      labelfont=dict(size=10, color='white'),
                                                      labelformat=".0f")))
        
        # --- INICIO DE LA CORRECCIÓN ---
        # Se formatea el texto del hover para incluir más información
        hover_text = [
            f"<b>{row[Config.STATION_NAME_COL]}</b><br>"
            f"Municipio: {row[Config.MUNICIPALITY_COL]}<br>"
            f"Precipitación: {row[Config.PRECIPITATION_COL]:.0f} mm"
            for _, row in df_clean.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=lons, y=lats, mode='markers', 
            marker=dict(color='red', size=5), 
            name='Estaciones',
            text=hover_text,
            hoverinfo='text'
        ))
        fig.update_layout(title=f"Precipitación en {year} ({method})", height=600)
        return fig, fig_variogram, None

    return go.Figure().update_layout(title="Error: Método no implementado"), None, "Error: Método no implementado"
