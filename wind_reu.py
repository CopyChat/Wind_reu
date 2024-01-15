"""
to assess wind resource in Reunion
"""

__version__ = f'Version 2.0  \nTime-stamp: <2021-05-15>'
__author__ = "ChaoTANG@univ-reunion.fr"

import sys
import hydra
import matplotlib.pyplot as plt
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
            points=coords[:], point_names=station_names, fill_color=alt, fill_color_name='altitude (m)',
            cmap='Blues_r', out_fig=cfg.figure.reunion_voronoi_mf_alt)

        # get mean of all hourly data
        hourly_mean = mf.groupby('NOM').mean()

        # voronoi with color in mean wind speed
        speed = hourly_mean['FF']
        GEO_PLOT.plot_voronoi_diagram_reu(
            points=coords[:], point_names=station_names, fill_color=speed, fill_color_name='10m mean hourly wind speed (m/s)',
            cmap='Greens_r', out_fig=cfg.figure.reunion_voronoi_mf_speed_10m)

    if cfg.job.missing_MF:
        print('working on MF missing data')

        for sta in station['NOM']:
            print(f'{sta:s}')
            sta1 = pd.DataFrame(mf[mf['NOM']==sta]['FF'])
            GEO_PLOT.check_missing_da_df(
                start='2000-01-01 00:00', end='2020-12-31 23:00',freq='H',
                data=sta1, plot=True, relative=True, output_plot_tag=f'MF station_{sta:s}')

    if cfg.job.multi_year_mean:

        speed_mean = pd.DataFrame(mf.groupby('NOM').mean()[['FF', 'ALT', 'LON', 'LAT']])
        GEO_PLOT.plot_station_value_reu(lon=speed_mean.LON, lat=speed_mean.LAT, vmin=1, vmax=6,
                                        station_name=station['NOM'], output_fig=cfg.figure.station_mean_ff_10m,
                                        value=speed_mean.FF, cbar_label='mean hourly wind speed (m/s) 10m',
                                        fig_title=f'2000-2020 hourly mean at MF stations @10m')

        std_ff = pd.DataFrame(mf.groupby('NOM').std()[['FF', 'ALT', 'LON', 'LAT']])
        GEO_PLOT.plot_station_value_reu(lon=speed_mean.LON, lat=speed_mean.LAT, vmin=1, vmax=3,
                                        station_name=station['NOM'], output_fig=cfg.figure.station_std_ff_10m,
                                        value=std_ff.FF, cbar_label='std of hourly wind speed 10m',
                                        fig_title=f'2000-2020 hourly mean std of wind speed at MF stations 10m')

    if cfg.job.climatology:
        print('working on climatology')

        # climatology vs data completeness:
        for sta in station['NOM']:
            print(f'{sta:s}')
            sta1 = pd.DataFrame(mf[mf['NOM']==sta]['FF'])
            print(len(sta1))
            GEO_PLOT.plot_annual_diurnal_cycle_columns_in_df(
                df=sta1, columns=['FF',], title=f'{sta:s} wind speed 2000-2022 10m', tag_on_plot=f'{sta:s}',
                linestyles=None, output_tag=f'{sta:s}', ylabel=f'hourly wind speed 10m', with_marker=True,
                plot_errorbar=True, vmin=0, vmax=10)

        # climatology between stations:
        mf_pivot = mf.pivot(columns='NOM', values='FF')
        GEO_PLOT.plot_annual_diurnal_cycle_columns_in_df(
            df=mf_pivot, columns=station['NOM'], title='climatology between stations 10m', tag_on_plot='',
            linestyles=None, output_tag='all stations', ylabel=f'hourly wind speed 10m', with_marker=True,
            plot_errorbar=True, vmin=0, vmax=10)

    if cfg.job.MF_station_clustering:
        print('working on MF station clustering')

        mf_pivot = mf.pivot(columns='NOM', values='FF')
        # DBSCAN clustering on diurnal cycle + seasonal cycle
        from sklearn.cluster import DBSCAN
        import matplotlib.pyplot as plt

        diurnal = mf_pivot.groupby(mf_pivot.index.hour).mean()
        annual = mf_pivot.groupby(mf_pivot.index.month).mean()

        # use annual and diurnal cycles for clustering:
        climatology = np.array(pd.concat([annual, diurnal])).transpose()

        cluster_labels, distances = GEO_PLOT.clustering_station_climatology_reu(
            lon=station.LON, lat=station.LAT,station_name=station.NOM, climatology=climatology,
            eps=4, min_samples=2, title='DBSCAN_clustering of hourly wind speed (m/s) \n annual + diurnal cycle 2000-2020 @ 10m',
            out_fig=cfg.figure.DBSCAN_cluster_climatology_MF_ff_10m, show_params=True)



    print(f'work done')
if __name__ == "__main__":
    sys.exit(wind_resource())
