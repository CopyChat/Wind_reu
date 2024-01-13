"""
to assess wind resource in Reunion
"""

__version__ = f'Version 2.0  \nTime-stamp: <2021-05-15>'
__author__ = "ChaoTANG@univ-reunion.fr"

import sys
import hydra
import numpy as np
import pandas as pd
from importlib import reload
from omegaconf import DictConfig
import GEO_PLOT


def jk():
    print(f'reloading GEO_PLOT....')
    reload(GEO_PLOT)

# ----------------------------- functions -----------------------------

@hydra.main(version_base='1.3', config_path="configs", config_name="wind_reu_config")
def wind_resource(cfg: DictConfig) -> None:
    """
    to find the physical link between the SSR classification and the large-scale variability
    over la reunion island
    """

    # reading data:
    mf: pd.DataFrame = GEO_PLOT.read_csv_into_df_with_header(f'{cfg.data.mf_all:s}')
    station: pd.DataFrame = pd.read_csv(f'{cfg.data.mf_station:s}')

    if cfg.job.voronoi:
        # load station data
        coords = np.array(station[['LON', 'LAT']])
        station_names = list(station['NOM'])

        # voronoi plot:
        GEO_PLOT.plot_voronoi_diagram_reu(
            points=coords[:], point_names=station_names, fill_color=None, out_fig=cfg.figure.reunion_voronoi_mf)

        # voronoi with color in alt
        alt = np.array(station['ALT'])
        GEO_PLOT.plot_voronoi_diagram_reu(
            points=coords[:], point_names=None, fill_color=alt, fill_color_name='altitude (m)', out_fig=cfg.figure.reunion_voronoi_mf_alt)

        # get mean of all hourly data
        hourly_mean = mf.groupby('NOM').mean()

        # voronoi with color in mean wind speed
        speed = hourly_mean['FF']
        GEO_PLOT.plot_voronoi_diagram_reu(
            points=coords[:], point_names=None, fill_color=speed, fill_color_name='mean wind speed (m/s)', out_fig=cfg.figure.reunion_voronoi_mf_speed)

    if cfg.job.missing_MF:
        print('working on MF missing data')

        for sta in station['NOM']:
            print(f'{sta:s}')
            sta1 = pd.DataFrame(mf[mf['NOM']==sta]['FF'])
            GEO_PLOT.check_missing_da_df(
                start='2000-01-01 00:00', end='2020-12-31 23:00',freq='H',
                data=sta1, plot=True, relative=True, output_plot_tag=f'MF station_{sta:s}')

    if cfg.job.climatology:
        print('working on climatology')

        for sta in station['NOM']:
            print(f'{sta:s}')
            sta1 = pd.DataFrame(mf[mf['NOM']==sta]['FF'])
            print(len(sta1))
            GEO_PLOT.plot_annual_diurnal_cycle_columns_in_df(
                df=sta1, columns=['FF',], title=f'{sta:s} wind speed 2000-2022', tag_on_plot=f'{sta:s}',
                linestyles=None, output_tag=f'{sta:s}', ylabel=f'hourly wind speed', with_marker=True,
                plot_errorbar=True, vmin=0, vmax=10)





    print(f'work done')
if __name__ == "__main__":
    sys.exit(wind_resource())
