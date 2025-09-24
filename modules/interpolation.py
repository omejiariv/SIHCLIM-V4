# modules/interpolation.py

import numpy as np
from scipy.interpolate import Rbf

def interpolate_idw(lons, lats, vals, grid_lon, grid_lat, power=2):
    """Realiza una interpolación por Distancia Inversa Ponderada (IDW)."""
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
    """
    Genera la superficie de interpolación.
    Versión simplificada sin Kriging.
    """
    # ... (Esta función ahora solo necesita la lógica para IDW y Spline)
    # ... por ahora, la dejamos deshabilitada para asegurar el despliegue.
    # Se puede reactivar solo con IDW y Spline después.
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.update_layout(title=f"Función de interpolación deshabilitada temporalmente.",
                      xaxis_visible=False, yaxis_visible=False)
    return fig, None, "Función deshabilitada."
