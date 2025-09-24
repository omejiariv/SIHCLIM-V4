# modules/analysis.py

import pandas as pd
import numpy as np
from scipy.stats import gamma, norm, loglaplace
from modules.config import Config

# --- Funciones de Análisis ---

def calculate_spi(series, window):
    """
    Calcula el Índice Estandarizado de Precipitación (SPI) para una serie de tiempo
    usando una implementación directa con SciPy. Es más robusto que usar librerías externas.
    """
    # 1. Calcula la suma móvil de la precipitación
    rolling_sum = series.sort_index().rolling(window, min_periods=window).sum()
    
    # 2. Ajusta una distribución Gamma a los datos de la suma móvil
    # Se usa .dropna() para evitar errores con valores nulos
    params = gamma.fit(rolling_sum.dropna(), floc=0)
    shape, loc, scale = params
    
    # 3. Calcula la probabilidad acumulada (CDF) con la distribución Gamma
    cdf = gamma.cdf(rolling_sum, shape, loc=loc, scale=scale)
    
    # 4. Transforma la probabilidad acumulada a una distribución normal estándar (Z-score)
    spi = norm.ppf(cdf)
    
    # 5. Manejo de valores infinitos que pueden surgir de norm.ppf(0) o norm.ppf(1)
    spi = np.where(np.isinf(spi), np.nan, spi)
    
    return pd.Series(spi, index=rolling_sum.index)


def calculate_monthly_anomalies(df_monthly_filtered, df_long):
    """
    Calcula la anomalía de la precipitación mensual respecto a la climatología
    histórica del DataFrame base (df_long).
    """
    df_climatology = df_long[
        df_long[Config.STATION_NAME_COL].isin(df_monthly_filtered[Config.STATION_NAME_COL].unique())
    ].groupby([Config.STATION_NAME_COL, Config.MONTH_COL])[Config.PRECIPITATION_COL].mean() \
     .reset_index().rename(columns={Config.PRECIPITATION_COL: 'precip_promedio_mes'})

    df_anomalias = pd.merge(
        df_monthly_filtered,
        df_climatology,
        on=[Config.STATION_NAME_COL, Config.MONTH_COL],
        how='left'
    )
    df_anomalias['anomalia'] = df_anomalias[Config.PRECIPITATION_COL] - df_anomalias['precip_promedio_mes']
    return df_anomalias.copy()


def calculate_percentiles_and_extremes(df_long, station_name, p_lower=10, p_upper=90):
    """
    Calcula los percentiles mensuales y clasifica los meses de la serie filtrada
    como secos o húmedos extremos para una estación específica.
    """
    df_station_full = df_long[df_long[Config.STATION_NAME_COL] == station_name].copy()

    df_thresholds = df_station_full.groupby(Config.MONTH_COL)[Config.PRECIPITATION_COL].agg(
        p_lower=lambda x: np.nanpercentile(x.dropna(), p_lower),
        p_upper=lambda x: np.nanpercentile(x.dropna(), p_upper),
        mean_monthly='mean'
    ).reset_index()

    df_station_extremes = pd.merge(
        df_station_full,
        df_thresholds,
        on=Config.MONTH_COL,
        how='left'
    )

    df_station_extremes['event_type'] = 'Normal'
    is_dry = (df_station_extremes[Config.PRECIPITATION_COL] < df_station_extremes['p_lower'])
    df_station_extremes.loc[is_dry, 'event_type'] = f'Sequía Extrema (< P{p_lower}%)'
    is_wet = (df_station_extremes[Config.PRECIPITATION_COL] > df_station_extremes['p_upper'])
    df_station_extremes.loc[is_wet, 'event_type'] = f'Húmedo Extremo (> P{p_upper}%)'

    return df_station_extremes.dropna(subset=[Config.PRECIPITATION_COL]), df_thresholds


def calculate_spei(precip_series, et_series, scale):
    """
    Calcula el SPEI usando una implementación directa con SciPy.
    """
    scale = int(scale)
    
    df = pd.DataFrame({'precip': precip_series, 'et': et_series})
    df = df.sort_index().asfreq('MS')
    df.dropna(inplace=True)
    
    if len(df) < scale * 2:
        return pd.Series(dtype=float)

    # 1. Calcular el balance hídrico (Precipitación - Evapotranspiración)
    water_balance = df['precip'] - df['et']
    
    # 2. Acumular el balance hídrico en la escala de tiempo deseada
    rolling_balance = water_balance.rolling(window=scale, min_periods=scale).sum()

    # 3. Ajustar una distribución Log-Laplace (equivalente a Log-Logística)
    params = loglaplace.fit(rolling_balance.dropna())
    
    # 4. Calcular la probabilidad acumulada (CDF)
    cdf = loglaplace.cdf(rolling_balance, *params)
    
    # 5. Transformar a Z-score de la distribución normal
    spei = norm.ppf(cdf)
    
    # 6. Manejar valores infinitos
    spei = np.where(np.isinf(spei), np.nan, spei)
    
    return pd.Series(spei, index=rolling_balance.index)
