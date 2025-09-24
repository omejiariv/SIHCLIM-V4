# modules/analysis.py

import pandas as pd
import numpy as np
import spei
from scipy.stats import gamma, norm, loglaplace # <-- 1. IMPORTAR LA DISTRIBUCIÓN
from modules.config import Config

# --- Funciones de Análisis ---

def calculate_spi(series, window):
    """
    Calcula el Índice Estandarizado de Precipitación (SPI) para una serie de tiempo.
    Requiere que la serie esté en formato Series de Pandas con índice de tiempo.
    """
    # 1. Calcula la suma móvil de la precipitación
    rolling_sum = series.sort_index().rolling(window, min_periods=window).sum()

    # 2. Ajusta una distribución Gamma a los datos de la suma móvil
    params = gamma.fit(rolling_sum.dropna(), floc=0)
    shape, loc, scale = params

    # 3. Calcula la probabilidad acumulada (CDF) con la distribución Gamma
    cdf = gamma.cdf(rolling_sum, shape, loc=loc, scale=scale)

    # 4. Transforma la probabilidad acumulada a una distribución normal estándar (Z-score)
    spi = norm.ppf(cdf)

    # 5. Manejo de valores infinitos
    spi = np.where(np.isinf(spi), np.nan, spi)

    return pd.Series(spi, index=rolling_sum.index)


def calculate_monthly_anomalies(df_monthly_filtered, df_long):
    """
    Calcula la anomalía de la precipitación mensual respecto a la climatología
    histórica del DataFrame base (df_long).
    """
    # 1. Determinar la climatología del período histórico completo (df_long)
    df_climatology = df_long[
        df_long[Config.STATION_NAME_COL].isin(df_monthly_filtered[Config.STATION_NAME_COL].unique())
    ].groupby([Config.STATION_NAME_COL, Config.MONTH_COL])[Config.PRECIPITATION_COL].mean() \
     .reset_index().rename(columns={Config.PRECIPITATION_COL: 'precip_promedio_mes'})

    # 2. Fusionar la climatología con los datos filtrados (df_monthly_filtered)
    df_anomalias = pd.merge(
        df_monthly_filtered,
        df_climatology,
        on=[Config.STATION_NAME_COL, Config.MONTH_COL],
        how='left'
    )

    # 3. Calcular la anomalía
    df_anomalias['anomalia'] = df_anomalias[Config.PRECIPITATION_COL] - df_anomalias['precip_promedio_mes']

    return df_anomalias.copy()


def calculate_percentiles_and_extremes(df_long, station_name, p_lower=10, p_upper=90):
    """
    Calcula los percentiles mensuales y clasifica los meses de la serie filtrada
    como secos o húmedos extremos para una estación específica.
    """
    df_station_full = df_long[df_long[Config.STATION_NAME_COL] == station_name].copy()

    # 1. Calcular umbrales de percentil (climatología) para cada mes.
    df_thresholds = df_station_full.groupby(Config.MONTH_COL)[Config.PRECIPITATION_COL].agg(
        p_lower=lambda x: np.nanpercentile(x.dropna(), p_lower),
        p_upper=lambda x: np.nanpercentile(x.dropna(), p_upper),
        mean_monthly='mean'
    ).reset_index()

    # 2. Fusionar los umbrales con los datos originales de la estación
    df_station_extremes = pd.merge(
        df_station_full,
        df_thresholds,
        on=Config.MONTH_COL,
        how='left'
    )

    # 3. Clasificar los eventos
    df_station_extremes['event_type'] = 'Normal'

    # Sequía extrema (Ppt < P_lower)
    is_dry = (df_station_extremes[Config.PRECIPITATION_COL] < df_station_extremes['p_lower'])
    df_station_extremes.loc[is_dry, 'event_type'] = f'Sequía Extrema (< P{p_lower}%)'

    # Húmedo extremo (Ppt > P_upper)
    is_wet = (df_station_extremes[Config.PRECIPITATION_COL] > df_station_extremes['p_upper'])
    df_station_extremes.loc[is_wet, 'event_type'] = f'Húmedo Extremo (> P{p_upper}%)'

    return df_station_extremes.dropna(subset=[Config.PRECIPITATION_COL]), df_thresholds


def calculate_spei(precip_series, et_series, scale):
    """
    Calcula el SPEI usando una serie de evapotranspiración pre-calculada.

    Args:
        precip_series (pd.Series): Serie de tiempo de precipitación mensual.
        et_series (pd.Series): Serie de tiempo de evapotranspiración mensual (ET).
        scale (int): Escala de tiempo en meses para el cálculo (e.g., 3, 6, 12).

    Returns:
        pd.Series: Serie de tiempo con los valores del SPEI.
    """
    scale = int(scale)

    data = pd.DataFrame({'precip': precip_series, 'et': et_series}).dropna()

    if data.empty:
        return pd.Series(dtype=float)

    water_balance = data['precip'] - data['et']

    # --- LÍNEA CORREGIDA ---
    # 2. Se pasa el objeto de distribución 'loglaplace' directamente.
    spei_values = spei.spei(water_balance, loglaplace, scale)

    return spei_values
