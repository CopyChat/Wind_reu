"""
to assess wind resource in Reunion
"""

__version__ = f'Version 2.0  \nTime-stamp: <2021-05-15>'
__author__ = "ChaoTANG@univ-reunion.fr"

import sys
import hydra
import matplotlib
import numpy as np
import pickle
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from importlib import reload
from omegaconf import DictConfig
# import geopandas as gpd
import GEO_PLOT
import DATA


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

    # calculate necessary data and save it for analysis:

    if any(GEO_PLOT.get_values_multilevel_dict(dict(cfg.job))):

        # get mean of all hourly data
        mf: pd.DataFrame = GEO_PLOT.read_csv_into_df_with_header(f'{cfg.data.mf_all:s}')

        if cfg.job.voronoi:
            # read station data
            station = pd.read_csv(f'{cfg.data.mf_station:s}')

            coords = np.array(station[['LON', 'LAT']])

            alt = np.array(station['ALT'])
            GEO_PLOT.plot_voronoi_diagram_reu(points=coords[:], fill_color=alt,
                                              out_fig=cfg.figure.reunion_voronoi_mf_alt)

            # get mean of all hourly data
            hourly_mean = mf.groupby('NOM').mean()

            speed = hourly_mean['FF']
            GEO_PLOT.plot_voronoi_diagram_reu(points=coords[:], fill_color=speed,
                                              out_fig=cfg.figure.reunion_voronoi_mf_speed)

            
        if cfg.job.missing_MF:
            print('working on MF missing data')

            for sta in station['NOM']:
                print(f'{sta:s}')
                sta1 = mf[mf['NOM']==sta]
                print(len(sta1))
                GEO_PLOT.check_missing_df_da_interval(
                    sta1, vmin=None, vmax=None, output_tag=sta, freq='H', columns=['FF',])

                GEO_PLOT.plot_missing_data(sta1, out_fig=f'{cfg.figure.reunion_missing_mf:s}_{sta:s}.png')



    print(f'work done')
if __name__ == "__main__":
    sys.exit(wind_resource())
