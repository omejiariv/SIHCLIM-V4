# modules/analysis.py

import pandas as pd
import numpy as np
from climate_indices import indices
from climate_indices.compute import Periodicity, Distribution # <-- 1. IMPORTACIÓN CORREGIDA
from scipy.stats import gamma, norm, loglaplace
from modules.config import Config

# --- Funciones de Análisis ---

def calculate_spi(series, window):
    """
    Calcula el Índice Estandarizado de Precipitación (SPI) para una serie de tiempo.
    """
    series = series.sort_index().asfreq('MS')
    data = series.dropna().to_numpy()
    
    if len(data) < window * 2:
        return pd.Series(dtype=float)

    # --- INICIO DE LA CORRECCIÓN ---
    # 2. Se usan los objetos enumeradores correctos importados anteriormente
    spi_values = indices.spi(
        values=data,
        scale=window,
        distribution=Distribution.gamma,
        periodicity=Periodicity.monthly,
        data_start_year=series.index.min().year,
        calibration_year_initial=series.index.min().year,
        calibration_year_final=series.index.max().year
    )
    # --- FIN DE LA CORRECCIÓN ---
    
    return pd.Series(spi_values, index=series.index[-len(spi_values):])


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
    Calcula el SPEI usando una serie de evapotranspiración pre-calculada.
    """
    scale = int(scale)
    
    df = pd.DataFrame({'precip': precip_series, 'et': et_series})
    df = df.sort_index().asfreq('MS')
    df.dropna(inplace=True)
    
    if len(df) < scale * 2:
        return pd.Series(dtype=float)

    # --- INICIO DE LA CORRECCIÓN ---
    # 2. Se usan los objetos enumeradores correctos importados anteriormente
    spei_values = indices.spei(
        precips_mm=df['precip'].to_numpy(),
        pet_mm=df['et'].to_numpy(),
        scale=scale,
        distribution=Distribution.log_logistic,
        periodicity=Periodicity.monthly,
        data_start_year=df.index.min().year,
        calibration_year_initial=df.index.min().year,
        calibration_year_final=df.index.max().year
    )
    # --- FIN DE LA CORRECCIÓN ---
    
    return pd.Series(spei_values, index=df.index[-len(spei_values):])
