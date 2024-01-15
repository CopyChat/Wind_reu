"""
functions used for process and plot geo data
hard linked with that in CODE/
"""

__version__ = f'Version 1.0  \nTime-stamp: <2019-02-21>'
__author__ = "ChaoTANG@univ-reunion.fr"

import os
import scipy
import sys
from pathlib import Path
from typing import List
import warnings
import hydra
import seaborn as sns
from omegaconf import DictConfig
import cftime
import glob
import pandas as pd
import calendar
import numpy as np
from dateutil import tz
import xarray as xr
import cartopy.crs as ccrs

# to have the right backend for the font.
import matplotlib.pyplot as plt

import cartopy.feature as cfeature
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from statsmodels.stats.multitest import fdrcorrection as fdr_cor


def fig_add_headers(
        fig,  # input fig or GridSpec or GridSpecFromSubplotSpec
        *,
        row_headers=None,
        col_headers=None,
        row_pad=1,
        col_pad=5,
        rotate_row_headers=True,
        **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1), xycoords="axes fraction",
                xytext=(0, col_pad), textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )
        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                # default:
                # xytext=(-ax.yaxis.labelpad - row_pad, 0.5),
                # textcoords="offset points",
                xytext=(-0.25, 0.5),
                textcoords="axes fraction",
                xycoords=ax.yaxis.label,
                ha="center",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )


xytext = (-0.2, 0.5),


def rules_data_format():
    """

    :return:
    :rtype:
    """

    print(f'classification file in pd.DataFrame with DateTimeIndex')
    print(f'geo-field in DataArray with good name and units')
    print(f'see the function read_to_standard_da for the standard dim names')


# ----------------------------- definition -----------------------------

def get_possible_standard_coords_dims(name_for: str = 'coords', ndim: int = 1):
    """
    that the definition of all project, used by read nc to standard da
    :return:
    :rtype:
    """

    standard_coords = ['time', 'lev', 'lat', 'lon', 'number']
    standard_dims_1d = ['time', 'lev', 'y', 'x', 'num']
    standard_dims_2d = ['time', 'lev', ['y', 'x'], ['y', 'x'], 'num']
    # this definition is used for all projects, that's the best order do not change this
    # key word: order, dims, coords,

    if name_for == 'coords':
        return standard_coords
    if name_for == 'dims':

        if ndim == 1:
            return standard_dims_1d
        if ndim == 2:
            return standard_dims_2d
        else:
            return 0


# -----------------------------
class ValidationGrid:

    def __init__(self, vld_name: str, vld: xr.DataArray,
                 ref_name: str, ref: xr.DataArray):
        self.vld_name = vld_name
        self.ref_name = ref_name
        self.vld = vld
        self.ref = ref
        self.vld_var = vld.name
        self.ref_var = ref.name

    @property  # 用这个装饰器后这个方法调用就可以不加括号，即将其转化为属性
    def stats(self) -> None:
        print('name', 'dimension', 'shape', 'max', 'min', 'std')
        print(self.vld_name, self.vld.dims, self.vld.shape,
              self.vld.max().values, self.vld.min().values, self.vld.std())
        print(self.ref_name, self.ref.dims, self.ref.shape,
              self.ref.max().values, self.ref.min().values, self.ref.std())
        print(f'-----------------------')

        return None

    @property
    def plot_vis_a_vis(self):
        a = self.vld.values.ravel()
        b = self.ref.values.ravel()

        plt.title(f'{self.vld_name:s} vs {self.ref_name}')

        # lims = [
        #     np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        #     np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        # ]

        vmin = np.min([self.vld.min().values, self.ref.min().values])
        vmax = np.max([self.vld.max().values, self.ref.max().values])

        lims = [vmin, vmax]

        fig = plt.figure(dpi=220)
        ax = fig.add_subplot(1, 1, 1)

        plt.scatter(a, b)
        plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f'{self.ref_name:s} ({self.ref.assign_attrs().units:s})')
        ax.set_ylabel(f'{self.vld_name:s} ({self.vld.assign_attrs().units:s})')
        # fig.savefig('./plot/dd.png', dpi=300)

        plt.show()

        print(value_lonlatbox_from_area('d01'))

        return fig

    @property
    def plot_validation_matrix(self):
        """
        here plot maps of
        Returns
        -------
        """

        return 1


class CmipVarDir:

    def __init__(self, path: str):
        self.path = path

    @property  # 用这个装饰器后这个方法调用就可以不加括号，即将其转化为属性
    def nc_file(self):
        return glob.glob(f'{self.path:s}/*.nc')

    @property  # 用这个装饰器后这个方法调用就可以不加括号，即将其转化为属性
    def gcm(self):
        files = self.nc_file
        gcm = list(set([s.split('_')[2] for s in files]))
        gcm.sort()
        # todo: when produce new file in the dir, the results would be different/wrong
        return gcm

    @property  # 用这个装饰器后这个方法调用就可以不加括号，即将其转化为属性
    def ssp(self):
        files = self.nc_file
        ssp = list(set([s.split('_')[3] for s in files]))
        return ssp

    @property  # 用这个装饰器后这个方法调用就可以不加括号，即将其转化为属性
    def var(self):
        files = self.nc_file
        file_names = [s.split('/')[-1] for s in files]
        var = list(set([s.split('_')[0] for s in file_names]))
        return var

    @property
    def freq(self):
        files = self.nc_file
        freq = list(set([s.split('_')[1] for s in files]))
        return freq


def convert_multi_da_to_ds(list_da: list, list_var_name: list) -> xr.Dataset:
    ds = list_da[0].to_dataset(name=list_var_name[0])

    if len(list_da) > 1:
        # Save one DataArray as dataset
        for i in range(1, len(list_da)):
            # Add second DataArray to existing dataset (ds)
            ds[list_var_name[i]] = list_da[i]

    return ds


def read_binary_file(bf: str):
    """
    to read binary file
    :param bf:
    :type bf:
    :return:
    :rtype:
    """

    # example: f'./local_data/MSG+0415.3km.lon'

    data = np.fromfile(bf, '<f4')  # little-endian float32
    # data = np.fromfile(bf, '>f4')  # big-endian float32

    return data


def convert_ds_to_da(ds: xr.Dataset, varname: str = 'varname') -> xr.DataArray:
    """
    get all the value from ds and put them into a da, with the coordinates from ds

    Parameters
    ----------
    ds : with each variable < 2D
    varname: usually there are several var names in the ds, as numbers of ensemble, for example.

    Returns
    -------
    da: with dims=('var', 'latitude', 'longitude')
    """
    list_var = list(ds.keys())
    first_da = ds[list_var[0]]

    all_values = np.stack([ds[x].values for x in list_var], axis=0)

    da = 0

    if len(ds.coord_names) == 3:
        # there's another coord beside 'longitude' 'latitude'
        time_coord_name = [x for x in ds.coord_names if x not in ['longitude', 'latitude']][0]
        time_coord = ds.coords[time_coord_name]

        units = ds.attrs['units']
        da = xr.DataArray(data=all_values, coords={'var': list_var, time_coord_name: time_coord,
                                                   'latitude': first_da.latitude, 'longitude': first_da.longitude},
                          dims=('var', time_coord_name, 'latitude', 'longitude'), name=varname,
                          attrs={'units': units})

    if len(ds.coord_names) == 2:
        # only 'longitude' and 'latitude'

        da = xr.DataArray(data=all_values, coords={'var': list_var,
                                                   'latitude': first_da.latitude, 'longitude': first_da.longitude},
                          dims=('var', 'latitude', 'longitude'), name=varname)

    return da


def convert_da_to_360day_monthly(da: xr.DataArray) -> xr.DataArray:
    """
    Takes a DataArray. Change the
    calendar to 360_day and precision to monthly.
    @param da: input with time
    @return:
    @rtype:
    """
    val = da.copy()
    time1 = da.time.copy()
    i_time: int
    for i_time in range(val.sizes['time']):
        year = time1[i_time].dt.year.__int__()
        mon = time1[i_time].dt.month.__int__()
        day = time1[i_time].dt.day.__int__()
        # bb = val.time.values[i_time].timetuple()
        time1.values[i_time] = cftime.Datetime360Day(year, mon, day)

    val = val.assign_coords({'time': time1})

    return val


def find_two_bounds(vmin: float, vmax: float, n: int):
    """
    find the vmin and vmax in 'n' interval
    :param vmin:
    :param vmax:
    :param n:
    :return:
    """
    left = round(vmin / n, 0) * n
    right = round(vmax / n, 0) * n

    return left, right


def query_data(mysql_query: str, remove_missing_data=True):
    """
    select data from DataBase
    :return: DataFrame
    """

    from sqlalchemy import create_engine
    import pymysql
    pymysql.install_as_MySQLdb()

    db_connection_str = 'mysql+pymysql://pepapig:123456@localhost/SWIO'
    db_connection = create_engine(db_connection_str)

    df: pd.DataFrame = pd.read_sql(sql=mysql_query, con=db_connection)

    # ----------------------------- remove two stations with many missing data -----------------------------
    if remove_missing_data:
        df.drop(df[df['station_id'] == 97419380].index, inplace=True)
        df.drop(df[df['station_id'] == 97412384].index, inplace=True)

    # ----------------------------- remove two stations with many missing data -----------------------------
    df['Datetime'] = pd.to_datetime(df['DateTime'])
    df = df.set_index('Datetime')
    df = df.drop(['DateTime'], axis=1)

    return df


@hydra.main(version_base='1.3', config_path="configs", config_name="config.ctang")
def query_influxdb(cfg: DictConfig, query: str):
    """
    select data from DataBase
    :return: DataFrame
    """
    from influxdb import DataFrameClient

    host = cfg.SWIOdb.host
    user = cfg.SWIOdb.user
    password = cfg.SWIOdb.password
    # port = 8088
    dbname = cfg.SWIOdb.dbname

    client = DataFrameClient(host, user, password, dbname)

    df = client.query(query)

    df = df.set_index('DateTime')

    return df


def plot_wrf_domain(num_dom: int, domain_dir: str):
    """
    to plot the domain setting from the output of geogrid.exe
    Parameters
    ----------
    num_dom :
    domain_dir :

    Returns
    -------
    """

    # ----------------------------- Definition:
    Num_Domain = num_dom
    Title = 'Domain setting for WRF simulation'

    # ----------------------------- plot a empty map -----------------------------
    import matplotlib.patches as patches
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    fig = plt.figure(figsize=(10, 8), dpi=300)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=1.5, top=0.95, wspace=0.05)

    ax = plt.subplot(111, projection=ccrs.PlateCarree())
    # set map
    swio = value_lonlatbox_from_area('detect')
    ax.set_extent(swio, crs=ccrs.PlateCarree())
    # lon_left, lon_right, lat_north, lat_north

    ax.coastlines('50m')
    ax.add_feature(cfeature.LAND.with_scale('10m'))

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(list(range(0, 100, 2)))
    gl.ylocator = mticker.FixedLocator(list(range(-90, 90, 2)))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 12, 'color': 'gray'}
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    # ----------------------------- end of plot a empty map -----------------------------

    # swio:
    # resolution = [27, 9, 3, 1]
    # Poovanum:
    # resolution = [10, 2, 0.4, 0.4]
    # Chandra:
    # resolution = [27, 9, 3, 1]
    # detect
    resolution = [27, 9, 3, 1]

    colors = ['black', 'blue', 'green', 'red']

    resolution = resolution[4 - Num_Domain:]
    colors = colors[4 - Num_Domain:]

    for d in range(Num_Domain):
        # ----------------------------- read domains -----------------------------
        # swio:
        nc_file = f'{domain_dir:s}/geo_em.d0{d + 1:g}.nc'

        # Poovanum:
        # nc_file = f'{DATA_DIR:s}/Poovanum/geo_em.d0{d + 1:g}.nc'
        # Chandra:
        # nc_file = f'{DATA_DIR:s}/Chandra/domain/geo_em.d0{d + 1:g}.nc'

        lat = xr.open_dataset(nc_file).CLAT
        lon = xr.open_dataset(nc_file).CLONG
        # map_factor = xr.open_dataset(nc_file).MAPFAC_M
        height = lat.max() - lat.min()
        width = lon.max() - lon.min()
        # --------------------- plot the domains ------------------------------

        p = patches.Rectangle((lon.min(), lat.min()), width, height,
                              fill=False, alpha=0.9, lw=2, color=colors[d])
        ax.add_patch(p)

        # domain names and resolution:
        plt.text(lon.max() - 0.05 * width, lat.max() - 0.05 * height,
                 f'd0{d + 1:g}: {resolution[d]:g}km',
                 fontsize=8, fontweight='bold', ha='right', va='top', color=colors[d])
        # grid points:

        plt.text(lon.min() + 0.5 * width, lat.min() + 0.01 * height,
                 f'({lon.shape[2]:g}x{lat.shape[1]})',
                 fontsize=8, fontweight='bold', ha='center', va='bottom', color=colors[d])

    aladin_domain = 0

    if aladin_domain:
        # add ALADIN domain:
        #  [34°E-74°E ; 2°S-28°S]:
        p1 = patches.Rectangle((34, -28), 40, 26, fill=False, alpha=0.9, lw=2, color='black')
        ax.add_patch(p1)

        plt.text(64, -29, f'ALADIN domain @12km', fontsize=8, fontweight='bold', ha='center', va='bottom',
                 color='black')

    plt.grid(True)
    plt.title(Title, fontsize=12, fontweight='bold')
    plt.show()

    return fig


def plot_MF_station_names(lon: list, lat: list, station_name: list):
    for i in range(len(station_name)):
        plt.text(lon[i], lat[i] + 0.01, station_name[i], horizontalalignment = 'center',
                 verticalalignment = 'bottom', color = 'green', fontsize = 8)


def clustering_station_climatology_reu(lon, lat, station_name,
                                       climatology: np.array, eps=1, min_samples=2,
                                       title='Clustering_station_climatology',
                                       show_params = False,
                                       out_fig='./plot/DBSCAN_clustering.png'):
    """

    eps = 3  # The maximum dist. between two samples for one to be considered as in the neighborhood of the other
    min_samples = 2  # number of samples (or total weight) in a neighborhood for a point

    """

    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.cluster import DBSCAN

    X = climatology

    # Calculate pairwise distances
    distances = pairwise_distances(X)

    data_info = f'mean_dist.={distances.mean():2.1f}\n' \
                f'median_dist.={np.median(distances):2.1f}\n' \
                f'max_dist.={distances.max():2.1f}'


    # Use DBSCAN for clustering
    eps = 3  # The maximum dist. between two samples for one to be considered as in the neighborhood of the other
    min_samples = 2  # number of samples (or total weight) in a neighborhood for a point
    # to be considered as a core point
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(X)

    print(cluster_labels)

    # Visualize the clusters
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    plt.scatter(lon, lat, c=cluster_labels, cmap='viridis', s=500)

    if title is not None:
        plt.title(title, fontsize=12, fontweight='bold')
    else:
        plt.title('DBSCAN Clustering, MF annual+diurnal climatology, 2000-2020')

    plt.xlabel('LON')
    plt.ylabel('LAT')

    plot_coastline_reu()

    plot_MF_station_names(lon=lon, lat=lat, station_name=station_name)

    if show_params:
        plt.text(0.98, 0.98, f'eps={eps:2.1f}\n min_sample={min_samples:g}\n\n'
                             f'{data_info:s}',fontsize=8, color='red',
                 horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    plt.savefig(out_fig, dpi=300)
    plt.show()

    return cluster_labels, distances


def plot_station_value_reu(lon: pd.DataFrame, lat: pd.DataFrame,
                           station_name: list,
                           vmin: float, vmax: float, value: np.array, cbar_label: str,
                           fig_title: str = None, output_fig: str = None,bias=False):
    """
    plot station locations and their values
    :param bias:
    :type bias:
    :param fig_title:
    :param cbar_label: label of color bar
    :param lon:
    :param lat:
    :param value:
    :return: map show
    """
    import matplotlib as mpl


    if bias:
        cmap = plt.cm.coolwarm
        vmin = max(np.abs(vmin), np.abs(vmax)) * (- 1)
        vmax = max(np.abs(vmin), np.abs(vmax))
    else:
        cmap = plt.cm.YlOrRd

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    # ----------------------------- stations -----------------------------
    sc = plt.scatter(lon, lat, c=value, edgecolor='black',
                     zorder=2, vmin=vmin, vmax=vmax, s=1500, cmap=cmap)

    # ----------------------------- color bar -----------------------------
    cb = plt.colorbar(sc, orientation='vertical', shrink=0.7, pad=0.05, label=cbar_label)
    cb.ax.tick_params(labelsize=10)

    # add coastline:
    coastline = load_reunion_coastline()
    plt.scatter(coastline.longitude, coastline.latitude, marker='o', s=1, c='gray', edgecolor='gray', alpha=0.4)

    # add station names:
    if station_name is not None:
        for i in range(len(station_name)):
            ax.text(lon[i], lat[i] + 0.01, station_name[i],
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    color='green',
                    fontsize=8)

    ax.set_xlim(55.15, 55.9)
    ax.set_ylim(-21.5, -20.8)
    ax.set_aspect('equal', adjustable='box')

    if fig_title is not None:
        plt.title(fig_title, fontsize=12, fontweight='bold')

    if output_fig is not None:
        plt.savefig(output_fig, dpi=300,)

    plt.show()
    print(f'got plot')

# noinspection PyUnresolvedReferences
def plot_station_value(lon: pd.DataFrame, lat: pd.DataFrame, value: np.array, cbar_label: str,
                       fig_title: str, bias=False):
    """
    plot station locations and their values
    :param bias:
    :type bias:
    :param fig_title:
    :param cbar_label: label of color bar
    :param lon:
    :param lat:
    :param value:
    :return: map show
    """
    import matplotlib as mpl

    fig = plt.figure(dpi=220)
    fig.suptitle(fig_title)

    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([55, 56, -21.5, -20.8], crs=ccrs.PlateCarree())
    # lon_left, lon_right, lat_north, lat_north

    ax.add_feature(cfeature.LAND.with_scale('10m'))
    # ax.add_feature(cfeature.OCEAN.with_scale('10m'))
    # ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
    # ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
    # ax.add_feature(cfeature.LAKES.with_scale('10m'), alpha=0.5)
    # ax.add_feature(cfeature.RIVERS.with_scale('10m'))

    # ax.coastlines()

    # ----------------------------- stations -----------------------------

    if np.max(value) - np.min(value) < 10:
        round_number = 2
    else:
        round_number = 0

    n_cbar = 10
    vmin = round(np.min(value) / n_cbar, round_number) * n_cbar
    vmax = round(np.max(value) / n_cbar, round_number) * n_cbar

    if bias:
        cmap = plt.cm.coolwarm
        vmin = max(np.abs(vmin), np.abs(vmax)) * (- 1)
        vmax = max(np.abs(vmin), np.abs(vmax))
    else:
        cmap = plt.cm.YlOrRd

    bounds = np.linspace(vmin, vmax, n_cbar + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    # ----------------------------------------------------------
    sc = plt.scatter(lon, lat, c=value, edgecolor='black',
                     # transform=ccrs.PlateCarree(),
                     zorder=2, norm=norm, s=50, cmap=cmap)

    # ----------------------------- color bar -----------------------------
    cb = plt.colorbar(sc, orientation='horizontal', shrink=0.7, pad=0.05, label=cbar_label)
    cb.ax.tick_params(labelsize=10)

    ax.gridlines(draw_labels=True)
    plt.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.6)

    plt.show()
    print(f'got plot')


def get_zenith_angle(df: pd.DataFrame, datetime_col: str, utc: bool,
                     lon: np.ndarray, lat: np.ndarray, column_zenith: str):
    """
    get solar zenith angle at (lon, lat) according to df with DateTimeUTC
    :param column_zenith:
    :param utc:
    :param datetime_col:
    :param df:
    :param lon:
    :param lat:
    :return:
    """

    import pytz
    from pysolar.solar import get_altitude

    if utc:
        df['DateTimeUTC_2'] = df[datetime_col]
    else:
        # add timezone info, which is needed by pysolar
        df['DateTimeUTC_2'] = [df.index.to_pydatetime()[i].astimezone(pytz.timezone("UTC"))
                               for i in range(len(df.index))]

    print(f'starting calculation solar zenith angle')
    zenith = [90 - get_altitude(lat[i], lon[i], df['DateTimeUTC_2'][i]) for i in range(len(df))]
    # prime meridian in Greenwich, England

    # df_new = df.copy()
    # df_new['utc'] = df['DateTimeUTC_2']
    # df_new[column_zenith] = zenith
    df_new = pd.DataFrame(columns=[column_zenith], index=df.index, data=zenith)

    output_file = r'/Users/ctang/Microsoft_OneDrive/OneDrive/CODE/Prediction_PV/zenith.csv'
    df_new.to_csv(output_file)

    # ----------------------------- for test -----------------------------
    # import datetime
    # date = datetime.datetime(2004, 11, 1, 00, 00, 00, tzinfo=datetime.timezone.utc)
    #
    # for lat in range(100):
    #     lat2 = lat/100 - 21
    #     a = 90 - get_altitude(55.5, lat2, date)
    #     print(lat2, a)
    # ----------------------------- for test -----------------------------

    return df_new


def zenith_angle_reunion(df, ):
    """
    to get zenith angle @ la reunion
    input: df['DateTime']
    """
    from pysolar.solar import get_altitude

    lat = -22  # positive in the northern hemisphere
    lon = 55  # negative reckoning west from
    # prime meridian in Greenwich, England

    return [90 - get_altitude(lat, lon, df[i])
            for i in range(len(df))]


def get_color(n):
    """define some (8) colors to use for plotting ... """

    # return [plt.cm.Spectral(each)
    #         for each in np.linspace(0, 6, 8)]

    # return ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    if n <= 6:
        colors = ['red', 'darkorange',
                  'green', 'cyan', 'blue', 'darkviolet', ]
    else:
        colors = ['lightgrey', 'gray', 'lightcoral', 'firebrick', 'red', 'darkorange', 'gold', 'yellowgreen',
                  'green', 'cyan', 'deepskyblue', 'blue', 'darkviolet', 'magenta', 'pink']
    markers = ['o', 'v', '^', '<', '1', 's', 'p', 'P', '*', '+', 'x', 'd', 'D']

    return colors


def convert_ttr_era5_2_olr(ttr: xr.DataArray, is_reanalysis: bool):
    """
    as the name of function
    :param is_reanalysis:
    :param ttr:
    :return:
    """

    # using reanalyse of era5:
    # The thermal (also known as terrestrial or longwave) radiation emitted to space at the top of the atmosphere
    # is commonly known as the Outgoing Longwave Radiation (OLR). The top net thermal radiation (this parameter)
    # is equal to the negative of OLR. This parameter is accumulated over a particular time period which depends on
    # the data extracted. For the reanalysis, the accumulation period is over the 1 hour up to the validity date
    # and time. For the ensemble members, ensemble mean and ensemble spread, the accumulation period is over the
    # 3 hours up to the validity date and time. The units are joules per square metre (J m-2). To convert to
    # watts per square metre (W m-2), the accumulated values should be divided by the accumulation period
    # expressed in seconds. The ECMWF convention for vertical fluxes is positive downwards.

    if is_reanalysis:
        factor = -3600
    else:
        factor = -10800

    if isinstance(ttr, xr.DataArray):
        # change the variable name and units
        olr = xr.DataArray(ttr.values / factor,
                           coords=ttr.coords,
                           dims=ttr.dims,
                           name='OLR', attrs={'units': 'W m**-2', 'long_name': 'OLR'})

    return olr


def plot_class_occurrence_and_anomaly_time_series(classif: pd.DataFrame, anomaly: xr.DataArray):
    """
    as the title,
    project: MIALHE_2020 (ttt class occurrence series with spatial mean seasonal anomaly)
    :param classif:
    :type classif:
    :param anomaly:
    :type anomaly:
    :return:
    :rtype:
    """

    fig = plt.figure(figsize=(12, 8), constrained_layout=False)
    grid = fig.add_gridspec(2, 1, wspace=0, hspace=0)
    ax = fig.add_subplot(grid[0, 0])

    print(f'anomaly series')

    start = anomaly.index.year.min()
    end = anomaly.index.year.max()

    anomaly = anomaly.squeeze()

    ax.plot(range(start, end + 1), anomaly, marker='o')
    ax.set_ylim(-20, 20)

    ax.set_xticklabels(classif['year'])

    class_name = list(set(classif['class']))
    for i in range(len(class_name)):
        event = classif[classif['class'] == class_name[i]]['occurrence']
        cor = np.corrcoef(anomaly, event)[1, 0]

        ax.text(0.5, 0.95 - i * 0.08,
                f'cor with #{class_name[i]:g} = {cor:4.2f}', fontsize=12,
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    print(f'bar plot')
    fig.add_subplot(grid[-1, 0])
    sns.set(style="whitegrid")
    sns.barplot(x='year', y="occurrence", hue=f'class', data=classif)

    fig.suptitle(f'reu spatial mean ssr anomaly and olr regimes number each season')

    plt.savefig('./plot/anomaly_seris_olr_regimes.png', dpi=300)
    plt.show()


def multi_year_daily_mean(var: xr.DataArray):
    """
    calculate multi_year_daily mean of variable: var.
    :param var:
    :return:
    """
    ydaymean = var[var.time.dt.year == 2000].copy()  # get structure
    ydaymean[:] = var.groupby(var.time.dt.dayofyear).mean(axis=0, dim=xr.ALL_DIMS).values

    return ydaymean


def get_month_name_from_num(num):
    """
    applied project: Sky_clearness
    :param num:
    :return:
    """

    import calendar
    # print('Month Number:', num)

    # get month name
    # print('Month full name is:', calendar.month_name[num])
    # print('Month short name is:', calendar.month_abbr[num])

    return calendar.month_abbr[num]


def check_missing_da_df(start: str, end: str, freq: str, data: xr.DataArray, plot: bool = True,
                        output_plot_tag=None, show_fig=True, relative=False):
    """
    applied project Sky_clearness_2023:
    to find the missing data number in months and in hours
    :param start:
    :param end:
    :param freq:
    :param relative: plot/calculate in percentage
    :param data:
    :param plot:
    :return:
    """

    # Alias Description
    # B business day frequency
    # C custom business day frequency
    # D calendar day frequency
    # W weekly frequency
    # M month end frequency
    # SM semi - month end frequency(15 th and end of month)
    # BM business month end frequency
    # MS month start frequency
    # SMS semi - month start frequency(1 st and 15 th)
    # Q quarter end frequency
    # QS quarter start frequency
    # A, Y year-end frequency
    # H hourly frequency
    # T, min minutely frequency
    # S secondly frequency
    # L, ms milliseconds
    # U, us microseconds
    # N nanoseconds

    data = drop_nan_infinity(data)

    print(f'start working ...')

    total_time_steps = pd.date_range(start, end, freq=freq)
    total_number = len(total_time_steps)

    missing_num = total_number - len(data)

    print(f'there are {missing_num:g} missing values..')

    # number to calculate
    matrix_mon_hour = np.zeros((12, 24))

    monthly_missing_num = np.zeros((12,))
    monthly_total_num = np.zeros((12,))

    hourly_missing_num = np.zeros((24,))
    hourly_total_num = np.zeros((24,))

    if missing_num:
        # find the missing values:
        A = total_time_steps.strftime('%Y-%m-%d %H:%M')
        if isinstance(data, xr.DataArray):
            B = data.time.dt.strftime('%Y-%m-%d %H:%M')
        if isinstance(data, pd.DataFrame):
            B = data.index.strftime('%Y-%m-%d %H:%M')
        C = [i for i in A if i not in B]
        # Note: here, if error raised as B referenced before definition, then
        # the input is either pd.DataFrame or xr.DataArray.

        missing_datetime = pd.to_datetime(C)
        all_datetime = pd.to_datetime(A)

        for i in range(1, 13):

            miss_monthly = missing_datetime[missing_datetime.month == i]
            all_monthly = all_datetime[all_datetime.month == i]

            monthly_missing_num[i-1] = len(miss_monthly)
            monthly_total_num[i-1] = len(all_monthly)

            for h in range(0, 24):
                missing_hours = list(miss_monthly.groupby(miss_monthly.hour).keys())

                if h in missing_hours:
                    if relative:
                        matrix_mon_hour[i - 1, h] = miss_monthly.groupby(miss_monthly.hour)[h].size *100 \
                                                    / all_monthly.groupby(all_monthly.hour)[h].size
                    else:
                        matrix_mon_hour[i - 1, h] = miss_monthly.groupby(miss_monthly.hour)[h].size

        # to calculate the missing hours in all months/years:
        for i in range(24):
            miss_hourly = missing_datetime[missing_datetime.hour == i]
            all_hourly = all_datetime[all_datetime.hour == i]

            hourly_missing_num[i] = len(miss_hourly)
            hourly_total_num[i] = len(all_hourly)

    if plot:
        fig = plt.figure(figsize=(10, 6), dpi=300)
        im = plt.imshow(matrix_mon_hour, cmap="OrRd")

        if relative:
            cb_label = 'percentage of missing hours (%)'
            x_label = f'% of missing hours (total = {hourly_total_num.mean():g})'
        else:
            cb_label = 'number of missing hours'
            x_label = f'number of missing hours (total = {hourly_total_num.mean():g})'

        cb = plt.colorbar(im, orientation='horizontal', shrink=0.5, pad=0.1, label=cb_label)

        plt.ylabel('Month')
        plt.xlabel(x_label, labelpad=19) # labelpad: distance to plot

        plt.title(f'missing data @{freq:s}'
                  f' ({start:s} - {end:s}) '
                  f'{output_plot_tag:s}\n'
                  f'Hour')

        ax = plt.gca()
        # ----------- set the top left labels
        # put the major ticks at the middle of each cell
        x_ticks = np.arange(24)
        y_ticks = np.arange(12)

        x_ticks_label = np.arange(24)
        y_ticks_label = np.arange(12) + 1
        y_ticks_label = [get_month_name_from_num(x) for x in y_ticks_label]

        ax.xaxis.tick_top()
        # Major ticks
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        ax.set_xticklabels(x_ticks_label)
        ax.set_yticklabels(y_ticks_label)

        # Minor ticks
        ax.set_xticks(np.arange(-.5, 24, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 12, 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

        # ----------- set the right bottom labels
        # put the monthly/hourly sum on the right
        if relative:
            monthly_percent = monthly_missing_num * 100 / monthly_total_num
            mon_ticks_label = [f'{y_ticks_label[i]:s} {monthly_percent[i]:2.1f}% of {monthly_total_num[i]:g}' for i in range(12)]

            hourly_percent = hourly_missing_num * 100 / hourly_total_num
            hour_ticks_label = [f'{hourly_percent[i]:2.1f}' for i in range(24)]
        else:
            mon_ticks_label = [f'{y_ticks_label[i]:s} {monthly_missing_num[i]:g} ({monthly_total_num[i]:g})' for i in range(12)]
            hour_ticks_label = [ f'{hourly_missing_num[i]:g}' for i in range(24)]

        # Set the x-axis ticks on the bottom side
        ax_bottom = ax.secondary_xaxis("bottom")
        ax_bottom.set_xticks(x_ticks)
        ax_bottom.set_xticks(np.arange(-.5, 24, 1), minor=True)
        ax_bottom.set_xticklabels(hour_ticks_label, rotation=0, fontsize=8)

        # Set the y-axis ticks on the right side
        ax_right = ax.secondary_yaxis("right")
        ax_right.yaxis.tick_right()
        ax_right.set_yticks(y_ticks)
        ax_right.set_yticklabels(mon_ticks_label)
        ax_right.set_yticks(np.arange(-.5, 12, 1), minor=True)
        # -----------
        # adjust the y-axis ticks to the border of plot.
        plt.subplots_adjust(left=0.06)

        if output_plot_tag is not None:
            plt.savefig(f'./plot/missing.{output_plot_tag:s}.png', dpi=300)
        if show_fig:
            plt.show()

    print(total_time_steps)
    return matrix_mon_hour


def check_nan_inf_da_df(df):
    """
    check if there's nan or inf in the dataframe or dataarray

    Args:
        df (): 

    Returns:

    """

    if isinstance(df, xr.DataArray):
        df = df.to_dataframe()

    column_names = list(df.columns)

    for i in range(len(column_names)):
        num_nan = df[column_names[i]].isna().values.sum()
        index_nan = np.where(df[column_names[i]].isna().values)[0]
        num_inf = np.isinf(df[column_names[i]]).values.sum()
        index_inf = np.where(np.isinf(df[column_names[i]].isna()).values)[0]

        print(f'{column_names[i]:s} \t has {num_nan:g} NaN, '
              f'which is only {num_nan * 100 / df.shape[0]: 4.2f} %. Not Much. \t'
              f'| {column_names[i]:s} \t has {num_inf:g} INF, '
              f'which is only {num_inf * 100 / df.shape[0]: 4.2f} %. Not Much')

        if len(index_nan):
            print(f'------------- nan ------------')
            for j in range(len(index_nan)):
                index = index_nan[j]
                print(f'index = {index:g}', df.iloc[index])

        if len(index_inf):
            print(f'------------- inf ------------')
            for j in range(len(index_nan)):
                index = index_inf[j]
                print(f'index = {index:g}', df.iloc[index])


def plot_violin_boxen_df_1D(df: pd.DataFrame,
                            x: str = 'x', x_unit: str = None,
                            y: str = 'y', y_unit: str = None, ymin=0, ymax=100,
                            plot_type: str = 'violin',
                            scale='area',
                            x_label: str = None, y_label: str = None,
                            x_ticks_labels=None,
                            split=False,
                            add_number=False,
                            hue: str = 0,
                            suptitle_add_word: str = '',
                            ):
    """
    plot violin plot
    Args:
        x_unit:
        y_label ():
        x_label ():
        y_unit ():
        df ():
        x ():
        y ():
        hue ():
        suptitle_add_word ():

    Returns:

    """

    # definition:
    if x_label == 0:
        x_label = x
    if y_label == 0:
        y_label = y

    # set unit
    if y_unit != None:
        y_label = f'{y_label:s} ({y_unit:s})'
    if x_unit != None:
        x_label = f'{x_label:s} ({x_unit:s})'

    from seaborn import violinplot

    # calculate fig size:

    n_x = len(set(df[x].values))
    if hue:
        n_hue = len(set(df[hue].values))
    else:
        n_hue = 1

    max_df = np.max(df[y])

    fig_width = n_x * n_hue

    print(n_x, n_hue, fig_width)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    if plot_type == 'violin':
        if hue:
            ax = violinplot(x=x, y=y, hue=hue, scale=scale, data=df, split=split)
        else:
            ax = violinplot(x=x, y=y, scale=scale, data=df, split=split)

    if plot_type == 'boxen':
        if hue:
            ax = sns.boxenplot(x=x, y=y, hue=hue, data=df, scale=scale)
        else:
            ax = sns.boxenplot(x=x, y=y, data=df, scale=scale)

    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(x_label)

    if x_ticks_labels is not None:
        ax.set_xticklabels(x_ticks_labels, minor=False)

    plt.grid()

    plt.ylabel(f'{y_label:s}')
    title = f'{y:s} in {x:s}'

    if add_number:
        std = df.groupby([x])[y].std().values
        medians = df.groupby([x])[y].median().values

        y_pos = medians - std * 1.3
        nobs = df[x].value_counts().values
        nobs = [str(x) for x in nobs.tolist()]
        nobs = ["" + i for i in nobs]

        print(nobs, ax.get_xticklabels())
        # Add text to the figure
        pos = range(len(nobs))
        for tick, label in zip(pos, ax.get_xticklabels()):
            ax.text(tick, y_pos[tick] * 1.5, nobs[tick],
                    size='medium',
                    color='b',
                    weight='semibold',
                    horizontalalignment='center',
                    verticalalignment='center')

    if x_ticks_labels is not None:
        ax.set_xticklabels(x_ticks_labels)

    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    fig.suptitle(title)

    plt.savefig(f'./plot/{plot_type:s}.{title.replace(" ", "_"):s}.'
                f'.png', dpi=300)
    print(f'done')
    plt.show()


def convert_cf_to_octas(array_1D: np.ndarray):
    # intervals from Utrillas 2022 paper
    intervals = np.array([0.00001, 18.75, 31.25, 43.75, 56.25, 68.75, 81.25, 100, 110]) / 100
    # this is always increasing

    octas = (array_1D.reshape(-1, 1) < intervals).argmax(axis=1)

    return octas


def daily_mean_da(da: xr.DataArray):
    """
    calculate daily mean
    Args:
        da ():

    Returns:
        da
    """

    from datetime import datetime

    da = da.assign_coords(strtime=da.time.dt.strftime('%Y-%m-%d'))
    # 'strtime' could be any string

    da2 = da.groupby('strtime').mean('time', keep_attrs=True)

    da2 = da2.rename({'strtime': 'time'})

    datetime_obj = [datetime.strptime(a, '%Y-%m-%d') for a in da2.time.values]

    da2 = da2.assign_coords(time=datetime_obj)

    # key word: update coords

    return da2


def get_df_of_da_in_classif(da: xr.DataArray, classif: pd.DataFrame):
    """
    merge classif and nd dataArray, return df, with MultiIndex
    key word: boxplot for seaborn violin
    Args:
        da ():
        classif ():

    Returns:
        dataframe

    """

    b: xr.DataArray = get_data_in_classif(da=da, df=classif)

    list_class = list(set(classif['class'].to_list())) * np.prod(b.shape[:-1])

    c = b.to_dataframe()

    c['class'] = list_class

    c = c.dropna()

    return c


def monthly_mean_da(da: xr.DataArray):
    """
    return monthly mean, the time index will be the first day of the month, so easy to match two monthly das calculated
    by this function
    Args:
        da ():

    Returns:

    """

    print(f'monthly mean...')

    mean = da.groupby(da.time.dt.strftime('%Y-%m')).mean(keep_attrs=True).rename(strftime='time')

    from datetime import datetime

    dt = [datetime.strptime(x, '%Y-%m') for x in mean.time.values]
    # key word: converting string to datetime

    mean = mean.assign_coords(time=('time', dt))

    return mean


def anomaly_daily(da: xr.DataArray) -> xr.DataArray:
    """
    calculate daily anomaly, as the name, from the xr.DataArray
    :param da:
    :return:
    """

    print(f'daily anomaly ...')

    anomaly = da.groupby(da.time.dt.strftime('%m-%d')) - da.groupby(da.time.dt.strftime('%m-%d')).mean('time')

    return_anomaly = anomaly.assign_attrs({'units': da.assign_attrs().units, 'long_name': da.assign_attrs().long_name})

    return_anomaly.rename(da.name)

    return return_anomaly


def anomaly_monthly(da: xr.DataArray, percent: int = 0) -> xr.DataArray:
    """
    calculate monthly anomaly, as the name, from the xr.DataArray
    if the number of year is less than 30, better to smooth out the high frequency variance.
    :param da:
    :param percent: output in percentage
    :return: da with a coordinate named 'strftime' but do not have this dim
    dfd
    """

    if len(set(da.time.dt.year.values)) < 30:
        warnings.warn('CTANG: input less than 30 years ... better to smooth out the high frequency variance, '
                      'for more see project Mialhe_2020/src/anomaly.py')

    print(f'calculating monthly anomaly ... {da.name:s}')

    climatology = da.groupby(da.time.dt.month).mean('time')
    anomaly = da.groupby(da.time.dt.month) - climatology

    if percent:
        anomaly = anomaly.groupby(da.time.dt.month) / climatology

    return_anomaly = anomaly.assign_attrs({'units': da.assign_attrs().units, 'long_name': da.assign_attrs().long_name})

    return_anomaly.rename(da.name)

    return return_anomaly


def anomaly_hourly(da: xr.DataArray, percent: int = 0) -> xr.DataArray:
    """
    calculate hourly anomaly, as the name, from the xr.DataArray
    if the number of year is less than 30, better to smooth out the high frequency variance.
    :param da:
    :param percent: output in percentage
    :return: da with a coordinate named 'strftime' but do not have this dim
    dfd
    """

    if len(set(da.time.dt.year.values)) < 30:
        warnings.warn('CTANG: input less than 30 years ... better to smooth out the high frequency variance, '
                      'for more see project Mialhe_2020/src/anomaly.py')

    print(f'calculating hourly anomaly ... {da.name:s}')

    climatology = da.groupby(da.time.dt.strftime('%m-%d-%H')).mean('time')
    anomaly = da.groupby(da.time.dt.strftime('%m-%d-%H')) - climatology

    if percent:
        anomaly = anomaly.groupby(da.time.dt.strftime('%m-%d-%H')) / climatology

    return_anomaly = anomaly.assign_attrs({'units': da.assign_attrs().units, 'long_name': da.assign_attrs().long_name})

    return_anomaly.rename(da.name)

    return return_anomaly


def remove_duplicate_list(mylist: list) -> list:
    """
    Remove Duplicates From a Python list
    :param mylist:
    :return:
    """

    list_return = list(dict.fromkeys(mylist))

    return list_return


def plot_geo_subplot_map(geomap, vmin, vmax, bias, ax,
                         domain: str, tag: str,
                         plot_cbar: bool = True,
                         cmap=plt.cm.coolwarm,
                         plt_type: str = 'contourf',
                         statistics: bool = 1):
    """
    plot subplot
    Args:
        cmap:
        plt_type ():
        geomap ():
        vmin ():
        vmax ():
        bias ():
        ax ():
        domain ():
        tag ():
        plot_cbar ():
        statistics ():

    Returns:

    """

    plt.sca(ax)
    # active this subplot

    # set up map:
    set_basemap(ax, area=domain)

    # vmax = geomap.max()
    # vmin = geomap.min()
    cmap, norm = set_cbar(vmax=vmax, vmin=vmin, n_cbar=20, cmap=cmap, bias=bias)

    cf = 'wrong type'

    if plt_type == 'pcolormesh':
        cf = plt.pcolormesh(geomap.lon, geomap.lat, geomap,
                            cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    if plt_type == 'contourf':
        cf: object = ax.contourf(geomap.lon, geomap.lat, geomap, levels=norm.boundaries,
                                 cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='both')

    if plot_cbar:
        cbar_label = f'{geomap.name:s} ({geomap.assign_attrs().units:s})'
        plt.colorbar(cf, orientation='vertical', shrink=0.8, pad=0.05, label=cbar_label)

    ax.text(0.98, 0.96, f'{tag:s}', fontsize=11,
            horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    if statistics:
        mean_value = geomap.mean().values
        if np.abs(mean_value) > 0:
            ax.text(0.98, 0.2, f'{mean_value:4.2f}', fontsize=11,
                    horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    return cf


def get_data_in_classif(da: xr.DataArray, df: pd.DataFrame, significant: bool = 0, time_mean: bool = 0,
                        return_size: bool = False):
    """
    to get a new da with additional dim of classification df da and df may NOT in the same length of time.
    attention: the significant is calculated for all the available data,
        if the resolution of field is higher than classif, (hourly vs daily), all hours will get
        the same significant
    :param time_mean:
    :param significant: T test @0.05
    :param da: with DataTimeIndex,
    :param df: class with DataTimeIndex
    :return: in shape of (:,:,class)
    Note: this will produce a lot of nan if a multi time steps are not belonging to the same class.
    :rtype: da with additional dim named with class number
    """

    # get info:
    class_column_name = df.columns[0]
    class_names = np.sort(list(set(df[class_column_name])))

    print(f'get data in class...')

    dic = {}
    for i in range(len(class_names)):
        cls = class_names[i]
        date_class_one: pd.DatetimeIndex = df.loc[df[class_column_name] == cls].index
        if len(date_class_one) < 1:
            print(f'Sorry, I got 0 day in phase = {cls:g}')
            break
        class_1: xr.DataArray = \
            da.where(da.time.dt.strftime('%Y-%m-%d').isin(date_class_one.strftime('%Y-%m-%d')), drop=True)
        # key word: matching, selecting, match two DataArray by index,
        # note: works only on day, a day is a class, since the format is up to day

        dic[class_names[i]] = class_1.sizes['time']
        if significant:
            sig_map = value_significant_of_anomaly_2d_mask(field_3d=class_1)
            class_1 = filter_2d_by_mask(class_1, mask=sig_map)
            # class_1 is the only significant pixels (with nan) of all the time steps in one class.

        if i == 0:
            data_in_class = class_1
        else:
            data_in_class = xr.concat([data_in_class, class_1], dim='class')

        print(f'class = {cls:g}', data_in_class.shape, data_in_class.dims)

    # output:
    if time_mean:
        # each class hase a mean
        data_in_class = data_in_class.mean('time')

    output_da = data_in_class.assign_coords({'class': class_names}).rename(da.name).assign_attrs(
        {'units': da.attrs['units']}).transpose(..., 'class')

    # for i in range(len(class_names)):
    #     print(i, class_names[i], output_da[:, :, :, i].dropna(dim='time', how='all').sizes)

    if return_size:
        return output_da, dic
    else:
        return output_da


def convert_unit_era5_flux(flux: xr.DataArray, is_ensemble: bool = 0):
    """
    convert to W/m2
    :param is_ensemble:
    :type is_ensemble:
    :param flux:
    :type flux:
    :return:
    :rtype:
    """

    # ----------------------------- attention -----------------------------
    # For the reanalysis, the accumulation period is over the 1 hour
    # ending at the validity date and time. For the ensemble members,
    # ensemble mean and ensemble spread, the accumulation period is
    # over the 3 hours ending at the validity date and time. The units are
    # joules per square metre (J m-2 ). To convert to watts per square metre (W m-2 ),
    # the accumulated values should be divided by the accumulation period
    # expressed in seconds. The ECMWF convention for vertical fluxes is
    # positive downwards.

    print(f'convert flux unit to W/m**2 ...')
    if is_ensemble:
        factor = 3600 * 3
    else:
        factor = 3600 * 1

    da = flux / factor

    da = da.rename(flux.name).assign_attrs({'units': 'W/m**2',
                                            'long_name': flux.assign_attrs().long_name})

    return da


def plot_cyclone_in_classif(classif: pd.DataFrame,
                            radius: float = 5,
                            tag_subplot: str = 'Regime_',
                            cen_lon: float = 55.5,
                            cen_lat: float = -21.1,
                            suptitle_add_word: str = ''
                            ):
    """
    to plot classification vs cyclone

    note: there is a function to select_near_by_cyclone could be used in this function.
    Args:
        cen_lat ():
        cen_lon ():
        radius ():
        classif (): classification in df with DateTimeIndex, and only one column of 'class',
                    the column name could be any string
        suptitle_add_word ():

    Returns:
        maps with cyclone path

    """

    # read cyclone
    # cyclone_file = f'~/local_data/cyclones.2.csv'
    cyclone_file = f'./dataset/cyc_df.csv'
    cyc = pd.read_csv(cyclone_file)
    cyc['Datetime'] = pd.to_datetime(cyc['DateTime'])
    df_cyclone = cyc.set_index('Datetime')

    # ----------------------------- get definitions -----------------------------
    class_names = list(set(classif.values.ravel()))
    n_class = len(class_names)

    print(f'plot cyclone within {int(radius): g} degree ...')
    # ----------------------------- prepare fig -----------------------------
    fig, axs = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(12, 10), dpi=300,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.01, hspace=0.01)
    axs = axs.ravel()

    # plot in class
    for c in range(n_class):
        c_name = class_names[c]
        print(f'plot class = {str(c_name):s}')

        class_one: pd.DataFrame = classif[classif == c_name].dropna()

        # ----------------------------- plotting -----------------------------
        ax = axs[c]
        plt.sca(axs[c])  # active this subplot
        set_basemap(area='m_r_m', ax=ax)

        total = 0
        # loop of day from classification:
        for i in range(len(class_one)):
            all_cyc_1day = df_cyclone[df_cyclone.index.date == class_one.index.date[i]]
            # could be one or more cyclone, so length < 4 * n_cyc

            if len(all_cyc_1day) < 1:  # if no cycle that day:
                pass
            else:  # if with cyclone records, one or more

                name = all_cyc_1day['NOM_CYC']
                # name could be the length > 1, since the cyclone file is 6-hourly.

                # sometimes there are 2 cyclones at the same day:
                cyclone_name_1day = list(set(list(name.values)))
                num_cyclone_1day = len(cyclone_name_1day)
                # if num_cyclone_1day > 1:
                #     print(f'got more than one cyclones in one day:')
                #     print(f'-------------------------------------', class_one.index.date[i])
                #     print(cyclone_name_1day)

                # to see if these cyclones pass within the $radius
                # if @ reunion
                record_in_radius = 0
                for cyc in cyclone_name_1day:
                    # check every cyclone in this day, while num of day remains one even having two TCs.

                    cyc_day = all_cyc_1day[all_cyc_1day['NOM_CYC'] == cyc]

                    lat1 = cyc_day[cyc_day['NOM_CYC'] == cyc]['LAT']
                    lon1 = cyc_day[cyc_day['NOM_CYC'] == cyc]['LON']

                    for record in range(len(lat1)):

                        if distance_two_point_deg(
                                lon1=cen_lon,
                                lon2=lon1[record],
                                lat1=cen_lat,
                                lat2=lat1[record]
                        ) < radius:
                            record_in_radius += 1

                            # plot only the tc within radius
                            # plot path (up to 6 points) if one or more of these 6hourly records is within a radius
                            plt.plot(lon1, lat1, marker='.', label='')  # only path of the day
                            # plt.legend(loc='lower left', prop={'size': 8})

                if record_in_radius > 0:
                    # add this cyclone in today (do this in every day if satisfied)
                    total += 1

                    # full_path of this cyclone
                    # full_path_cyc = df_cyclone[df_cyclone['NOM_CYC'] == cyc]
                    # plt.plot(full_path_cyc['LON'], full_path_cyc['LAT'])

                    # output this cyclone:
                    print(i, total, record_in_radius, cyc_day)

        # ----------------------------- end of plot -----------------------------

        # plt.title(f'CL{c + 1:g}')

        ax.text(0.05, 0.95, f'{tag_subplot:s}{c + 1:g}',
                # ax.text(0.05, 0.95, f'Regimes_{c + 1:g}',
                fontsize=14, horizontalalignment='left', weight='bold', verticalalignment='top', transform=ax.transAxes)

        ax.text(0.96, 0.95,
                f'total day = {len(class_one):g}\n'
                f'TC day ={total:g}\n'
                f'{100 * total / len(class_one):4.1f}%',
                fontsize=14, horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        # ax.text(0.06, 0.01, f'plot only the path within a day',
        #         fontsize=12, horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

        # ----------------------------- end of plot -----------------------------
    title = f'cyclone within {radius:g} degree of Reunion'

    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    # fig.suptitle(title)
    plt.savefig(f'./plot/{title.replace(" ", "_"):s}.radius_{radius:g}.deg'
                f'.png', dpi=300)

    plt.show()
    print(f'got plot')


def distance_two_point_deg(lon1, lat1, lon2, lat2):
    return np.sqrt(np.square(lon2 - lon1) + np.square(lat2 - lat1))


def select_nearby_cyclone(cyc_df: pd.DataFrame,
                          lon_name: str = 'lon',
                          lat_name: str = 'lat',
                          radius: float = 5,
                          cen_lon: float = 55.5,
                          cen_lat: float = -21.1
                          ):
    """
    from cyclone record select nearby events
    Args:
        cyc_df ():
        lon_name ():
        lat_name ():
        radius ():
        cen_lon ():
        cen_lat ():

    Returns:
        df (): only nearby records

    key word: selecting, DataFrame, lonlat, radius, nearby, cyclone, df
    applied_project: Mialhe_2020
    """

    df = cyc_df.loc[
        distance_two_point_deg(
            lat1=cen_lat,
            lat2=cyc_df[lat_name],
            lon1=cen_lon,
            lon2=cyc_df[lon_name]) <= radius]

    return df


def select_pixel_da(da: xr.DataArray, lon, lat, n_pixel: int = 1,
                    plot: bool = True):
    """
    to select pixel from a da
    :param plot:
    :param da:
    :param lon:
    :param lat:
    :param n_pixel:
    :return:
    """

    # distance
    dis = (da.lon - lon) ** 2 + (da.lat - lat) ** 2
    index = dis.argmin(dim=['x', 'y'])
    index_x = index['x']
    index_y = index['y']

    print(f'---- index x and y')
    print(index_x, index_y)

    # Now I can use that index location to get the values at the x/y diminsion
    point_1 = da.sel(x=index_x, y=index_y)
    point_9 = da.sel(x=[index_x - 1, index_x, index_x + 1],
                     y=[index_y - 1, index_y, index_y + 1])

    if plot:
        # plot the nearest 9 points:
        point_9p = point_9[0]

        print(f' ctang: if there is a error of output of bounds, then check the input location')

        # plot
        plt.scatter(point_9p.lon.values.ravel(), point_9p.lat.values.ravel())

        # Plot requested lat/lon point blue
        plt.scatter(lon, lat, color='b')
        plt.text(lon, lat, 'requested', color='b')

        # Plot the nearest point in the array red
        plt.scatter(point_1.lon.values, point_1.lat.values, color='r')
        plt.text(point_1.lon.values, point_1.lat.values, 'nearest')

        plt.title('nearest point')
        plt.grid()
        plt.show()

    if n_pixel == 1:
        return point_1
    if n_pixel == 9:
        return point_9


def plot_diurnal_boxplot_in_classif(classif: pd.DataFrame, field: xr.DataArray,
                                    suptitle_add_word: str = '',
                                    anomaly: int = 0,
                                    relative_data: int = 0,
                                    ylimits: str = 'default',
                                    plot_big_data_test: int = 1):
    """

    Args:
        ylimits ():
        classif ():
        field (): dims = time, in this func by 'data_in_class', get da in 'time' & 'class'

        suptitle_add_word ():
        anomaly (int): 0
        relative_data (int): 0
        plot_big_data_test ():

    Returns:

    Applied_project:
        Mialhe_2020
    """
    # ----------------------------- data -----------------------------
    data_in_class = get_data_in_classif(da=field, df=classif, time_mean=False, significant=0)

    # to convert da to df: for the boxplot:
    print(f'convert DataArray to DataFrame ...')
    df = data_in_class.to_dataframe()
    df['Hour'] = df._get_label_or_level_values('time').hour
    df['Class'] = df._get_label_or_level_values('class')
    # key word: multilevel index, multi index, convert da to df, da2df

    # ----------------------------- get definitions -----------------------------
    class_names = list(set(classif.values.ravel()))
    n_class = len(class_names)

    # ----------------------------- plot -----------------------------
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), facecolor='w', edgecolor='k', dpi=300)

    sns.boxplot(x='Hour', y=data_in_class.name, hue='Class', data=df, ax=ax,
                showmeans=True, showfliers=False)
    # Seaborn's showmeans=True argument adds a mark for mean values in each box.
    # By default, mean values are marked in green color triangles.

    if anomaly:
        plt.axhline(y=0.0, color='r', linestyle='-', zorder=-5)

    ax.set_ylim(ylimits[0], ylimits[1])

    if relative_data:
        ax.set_ylabel(f'{data_in_class.name:s} (%)')
    else:
        ax.set_ylabel(f'{data_in_class.name:s} ({data_in_class.units})')

    title = f'{field.assign_attrs().long_name:s} percent={relative_data:g} anomaly={anomaly:g} in class'

    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    fig.suptitle(title)

    # ----------------------------- end of plot -----------------------------

    plt.savefig(f'./plot/diurnal.{title.replace(" ", "_"):s}.'
                f'test_{plot_big_data_test:g}.png', dpi=300)

    plt.show()
    print(f'got plot ')


def test_exact_mc_permutation(small, big, nmc, show: bool = True):
    """
    to test if two samples are same
    Args:
        small (): 
        big (): 
        nmc (): 

    Returns:

    """
    n, k = len(small), 0
    diff = np.abs(np.mean(small) - np.mean(big))
    zs = np.concatenate([small, big])

    list_diff = np.empty(nmc)
    for j in range(nmc):
        np.random.shuffle(zs)
        list_diff[j] = np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))

    p = np.float(k / nmc)
    # percentage that > than the diff = the probability to have a diff like diff of origional data

    if show:
        plt.close("all")

        f = plt.figure()
        ax = f.add_subplot(111)

        sns.set_palette("hls")  # 设置所有图的颜色，使用hls色彩空间
        sns.distplot(list_diff, color="g", label='distribution', bins=30, kde=True)  # kde=true，显示拟合曲线
        sns.ecdfplot(data=list_diff, color='r', label='ecdf')
        plt.axvline(x=diff, color='b', linestyle='-', linewidth=2, label='mean abs diff')
        plt.text(0.1, 0.95, f'permutation = {nmc:g}', ha='left', va='top', transform=ax.transAxes)
        plt.title('Permutation Test')
        plt.grid()
        plt.xlabel('difference')
        plt.ylabel('distribution')
        plt.legend()
        plt.show()

    return diff, list_diff, p


def get_confidence_interval(data, alpha: float = 0.95):
    """
    normal distribution
    Args:
        data ():
        alpha ():

    Returns:

    """
    if data.shape[0] < 30:
        warnings.warn('CTANG: num of data is too small, check and check it again')
    else:
        interval = stats.norm.interval(1 - alpha, np.mean(data), np.std(data))

    return interval


def get_values_multilevel_dict(dic: dict, level: int = 2):
    """
    as the function name, but max in level 2.
    used for the multilevel parameters.
    Args:
        dic ():
        level ():

    Returns: list

    """
    values_list = []
    for kk in dic.keys():
        # print(type(dic[kk]))
        if isinstance(dic[kk], int):
            values_list.append(dic[kk])
        else:
            in_dic = dict(dic[kk])
            for kkk in in_dic.keys():
                values_list.append(in_dic[kkk])

    return values_list


def cal_daily_total_energy(da: xr.DataArray, energy_unit: str = 'kWh'):
    """

    Args:
        energy_unit ():
        da (): with time, lon, lat, which will be interpolated into 24 hours per day

    Returns:
        da: DataArray, in unit of MJ/m2/day

    """

    print(f'interpolating into 24-hour/day ... to calculate the daily total energy density...')

    dates = list(set(da.time.dt.strftime('%Y-%m-%d').values))
    hours = [pd.date_range(start=x, periods=24, freq='H') for x in dates]
    hourly_da = da.interp(time=np.sort(np.array(hours).flatten()),
                          method='linear', kwargs={"fill_value": "extrapolate"})

    # add all hourly values to get a sum
    b = hourly_da.groupby(hourly_da.time.dt.strftime('%Y-%m-%d')).sum(dim='time')
    # ref: The daily irradiation in Wh/m2 will be obtained as the sum of all hourly values in W/m2.
    # For instance, if the irradiance is constant at 100 W/m2 during 10 hours,
    # the daily total irradiation is 1000 Wh/m2=1 kWh/m2.
    # The same principle applies to 10-min values, but then the total needs to be divided by 6.

    b = b.rename({"strftime": 'time'})
    b = b.assign_coords(time=('time', pd.DatetimeIndex(b.time)))

    # convert the units: works only we have 24 hours in a day:
    if energy_unit == 'kWh':
        b /= 1000
        b = b.assign_attrs({"units": "kWh/m2/day"})
    if energy_unit == 'MJ':
        b = b * 3600 / 10E6
        b = b.assign_attrs({"units": "MJ/m2/day"})

    # attrs will be gone after numerical calculation so do it at last
    b = b.assign_attrs({"long_name": "daily_total_energy_density"})

    return b


# def magnitude(a, b):
# ...     func = lambda x, y: np.sqrt(x ** 2 + y ** 2)
# ...     return xr.apply_ufunc(func, a, b)


def plot_climate_index(x: np.ndarray, y, index_name: str,
                       by_limit: bool = True,
                       by_percentage: bool = False,
                       limits=[-1, 1],
                       alpha=0.5,
                       x_label: str = 'time', y_label: str = 'index',
                       title_add_word: str = '',
                       scatter: bool = False,
                       ):
    """
    plot climate index, by percentage i.e. alpha or by limits i.e., limits
    Args:
        limits:
        by_percentage:
        by_limit:
        scatter ():
        title_add_word ():
        y_label ():
        x_label (): 
        index_name (): 
        x ():
        y ():
        alpha (): conf_interval: if 0.1 means

    Returns:

    """

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    plt.plot(x, y, label=index_name)

    if by_percentage:
        down, up = get_confidence_interval(y, alpha=alpha)

    if by_limit:
        down, up = limits

    if np.abs(down) != np.abs(up):
        plt.text(0.05, 0.05, 'up and bottom limits are different since the mean is not zero', color='r',
                 horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    plt.hlines(up, x[0], x[-1], colors='black', linewidth=1, linestyles='dashed')
    plt.hlines(down, x[0], x[-1], colors='black', linewidth=1, linestyles='dashed')

    if by_percentage:
        plt.fill_between(x, up, y, where=(y > up), interpolate=True, color='r', alpha=0.5,
                         label=str(f'{alpha * 100:4.2f} %'))
        plt.fill_between(x, down, y, where=(y < down), interpolate=True, color='r', alpha=0.5)
    if by_limit:
        plt.fill_between(x, up, y, where=(y > up), interpolate=True, color='r', alpha=0.5,
                         label=str(f'amplitude > {up:4.1f}'))
        plt.fill_between(x, down, y, where=(y < down), interpolate=True, color='r', alpha=0.5)

    if scatter:
        plt.scatter(x, y, marker='^', color='green')

    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)

    plt.legend(fontsize=18)

    title = index_name

    if title_add_word is not None:
        title = title + ' ' + title_add_word

    plt.title(title, fontsize=18)

    # ----------------------------- end of plot -----------------------------

    plt.savefig(f'./plot/index.{title.replace(" ", "_"):s}.'
                f'percentile_{alpha:4.2f}.png', dpi=300)
    plt.show()


def count_nan_2d_map(map: xr.DataArray):
    b = 0
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i, j]:
                pass
            else:
                b += 1
    return b


def plot_diurnal_curve_in_classif(classif: pd.DataFrame, field_1D: xr.DataArray,
                                  suptitle_add_word: str = '',
                                  anomaly: int = 0,
                                  percent: int = 0,
                                  ylimits='default',
                                  plot_big_data_test: int = 1):
    """

    Args:
        ylimits ():
        classif ():
        field_1D ():
        suptitle_add_word ():
        anomaly ():
        percent ():
        plot_big_data_test ():

    Returns:

    Applied_project:
     Mialhe_2020
    """
    # ----------------------------- data -----------------------------
    data_in_class = get_data_in_classif(da=field_1D, df=classif, time_mean=False, significant=0)

    # to convert da to df: for the boxplot:

    # ----------------------------- get definitions -----------------------------
    class_names = list(set(classif.values.ravel()))

    # ----------------------------- plot -----------------------------

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), facecolor='w', edgecolor='k', dpi=300)

    for i in range(len(class_names)):
        date_in_class = classif[classif['class'] == class_names[i]].index.date
        data_1class = field_1D.loc[field_1D.time.dt.date.isin(date_in_class)]

        y = data_1class.groupby(data_1class['time'].dt.hour).mean()
        x = y.hour

        plt.plot(x, y, label=str(class_names[i]))

    plt.legend()
    plt.grid(True)

    plt.xlabel('Hour')

    if percent:
        plt.ylabel(f'{data_in_class.name:s} (%)')
    else:
        plt.ylabel(f'{data_in_class.name:s} ({data_in_class.units})')

    if ylimits != 'default':
        ax.set_ylim(ylimits[0], ylimits[1])

    title = f'{field_1D.assign_attrs().long_name:s} percent={percent:g} anomaly={anomaly:g} in class'

    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    fig.suptitle(title)

    # ----------------------------- end of plot -----------------------------

    plt.savefig(f'./plot/diurnal.curve.{title.replace(" ", "_"):s}.'
                f'test_{plot_big_data_test:g}.png', dpi=300)
    plt.show()
    print(f'got plot ')


def plot_surface_circulation_diurnal_cycle_classif(axs, classif: pd.DataFrame, field: xr.DataArray,
                                                   moisture_flux: bool = False, anomaly: bool = False,
                                                   circulation_name: str = 'surface',
                                                   area: str = 'bigreu',
                                                   test: bool = False):
    """
    add circulation in subplot: 1) wind, 2) moisture flux by a) mean or b) anomaly
    Args:
        axs:
        circulation_name:
        area:
        test:


    Returns:
    :param circulation_name:
    :param moisture_flux:


    """
    print(f'plot surface circulation ...')
    # ----------------------------- get definitions -----------------------------
    class_names = list(set(classif.values.ravel()))
    n_class = len(class_names)

    hours = list(set(field.time.dt.hour.data))
    n_hour = len(hours)

    # class_column_name = classif.columns.to_list()[0]

    print(f'loading wind data ... ')
    warnings.warn('CTANG: load data from 1999-2016, make sure that the data period is correct,'
                  ' check and check it again')

    local_data = '/Users/ctang/local_data/era5'
    u_file = f'{local_data:s}/u10/u10.hourly.era5.1999-2016.{area:s}.local_daytime.nc'
    v_file = f'{local_data:s}/v10/v10.hourly.era5.1999-2016.{area:s}.local_daytime.nc'

    u = read_to_standard_da(u_file, 'u10')
    v = read_to_standard_da(v_file, 'v10')
    # classif OLR is from 1979 to 2018.

    if test:
        u = u.sel(time=slice('19990101', '20001201'))
        v = v.sel(time=slice('19990101', '20001201'))

    if moisture_flux:
        qs_file = f'/Users/ctang/local_data/era5/q_specific/sp.era5.hourly.1999-2016.bigreu.local_daytime.nc'
        qs = read_to_standard_da(qs_file, 'sp')
        u = u * qs
        u = u.assign_attrs({'units': 'g/kg m/s'})
        u = u.rename('moisture flux')

        v = v * qs
        v = v.assign_attrs({'units': 'g/kg m/s'})
        v = v.rename('moisture flux')

    if anomaly:
        print(f'plot surface wind anomaly ...')

        u_anomaly = anomaly_hourly(u)
        v_anomaly = anomaly_hourly(v)

        u = u_anomaly.assign_attrs({'units': u.units})
        v = v_anomaly.assign_attrs({'units': v.units})
        # after calculation the units are lost, get it back.

    u_in_class = get_data_in_classif(u, classif, significant=False, time_mean=False)
    v_in_class = get_data_in_classif(v, classif, significant=False, time_mean=False)

    for cls in range(n_class):
        print(f'plot surface flux in class = {cls + 1:g}')

        u_in_1class = u_in_class.where(u_in_class['class'] == class_names[cls], drop=True).squeeze()
        v_in_1class = v_in_class.where(v_in_class['class'] == class_names[cls], drop=True).squeeze()

        u_hourly_mean = u_in_1class.groupby(u_in_1class.time.dt.hour).mean()
        v_hourly_mean = v_in_1class.groupby(v_in_1class.time.dt.hour).mean()

        for hour in range(n_hour):
            plt.sca(axs[cls, hour])
            ax = axs[cls, hour]

            u_1hour = u_hourly_mean.sel(hour=hours[hour])
            v_1hour = v_hourly_mean.sel(hour=hours[hour])

            # parameter to define the quiver length and key:
            if anomaly:
                field_scale = u.std()
            else:
                field_scale = u.mean()

            plot_wind_subplot(area=area,
                              lon=u_1hour.lon, lat=v_1hour.lat,
                              u=u_1hour,
                              v=v_1hour,
                              bias=anomaly,
                              # key_input=1.5, # for wind anomaly bigreu
                              key_input=10,  # hourly wind bigreu
                              key_units=u.units,
                              plot_field_flux=True,
                              field_scale=field_scale,
                              ax=ax)


def plot_diurnal_cycle_field_in_classif(classif: pd.DataFrame,
                                        field: xr.DataArray,
                                        area: str, vmax, vmin,
                                        cmap='default',
                                        field_bias: bool = True,
                                        plot_circulation: bool = 0,
                                        circulation_anomaly: bool = 0,
                                        plot_moisture_flux: bool = 0,
                                        circulation_name: str = 'field flux',
                                        plt_type: str = 'pcolormesh',
                                        only_significant_points: bool = 0,
                                        str_class_names: list = None,
                                        row_headers: str = None, col_headers: str = None,
                                        suptitle_add_word: str = '',
                                        plot_num_record: bool = False,
                                        test_run: int = 1):
    """
        diurnal field in classification, data are processed before input to this question
    note:
        sfc circulation from era5 @27km, not anomaly, to show sea/land breeze during a day.
        the field, such as rsds, will be hourly anomaly, in one class, value at one hour - multiyear
        seasonal hourly mean, (%m-%d-%h), which is the difference between in-class value and all-day value.
    Args:
        plt_type:
        plot_moisture_flux:  plot surface field flux instead of only wind, or wind anomaly (controled by wind_anomaly)
        circulation_name:
        str_class_names:  class may have a name, otherwise use the value in column 'class' in the DataFrame
        circulation_anomaly:  if plot anomaly of surface wind
        classif (pandas.core.frame.DataFrame): DatetimeIndex with class name (1,2,3...)
        field (xarray.core.dataarray.DataArray): with time dim. field may in different days of classif,
            it will be selected by the available classif day by internal function.
        area (str):
        vmax (int):
        vmin (int):
        field_bias (bool):
        plot_circulation (int):
        only_significant_points (int):
        suptitle_add_word (str):
        test_run (int): if apply the data filter defined inside the function to boost the code,
            usually defined in the project level in the config file.

    Returns:

    """

    # ----------------------------- data -----------------------------
    if test_run:
        field = field.sel(time=slice('19990101', '20001201'))

    data_in_class, class_size = get_data_in_classif(da=field, df=classif, time_mean=False,
                                                    significant=0, return_size=True)
    # when the data freq is not the freq as cla
    print(f'good')
    # ----------------------------- get definitions -----------------------------
    class_names = list(set(classif.values.ravel()))
    n_class = len(class_names)

    hours = list(set(field.time.dt.hour.data))
    n_hour = len(hours)

    # class_column_name = classif.columns.to_list()[0]

    # ----------------------------- plotting -----------------------------
    fig, axs = plt.subplots(nrows=n_class, ncols=n_hour, sharex=True, sharey=True,
                            figsize=(20, 15), dpi=300,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95, wspace=0.02, hspace=0.02)

    # add headers:
    font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize=20)
    font_kwargs = dict(fontfamily="sans-serif", fontweight="regular", fontsize=20)
    fig_add_headers(fig, row_headers=row_headers, col_headers=col_headers, rotate_row_headers=True, **font_kwargs)

    for cls in range(n_class):
        print(f'plot class = {cls + 1:g}')

        in_class = data_in_class.where(data_in_class['class'] == class_names[cls], drop=True).squeeze()
        # nan comes from missing data and from non-significance
        in_class_dropna = in_class.dropna(dim='time')
        num_record = class_size[class_names[cls]]

        for hour in range(n_hour):

            plt.sca(axs[cls, hour])
            ax = axs[cls, hour]

            data_in_hour = in_class_dropna.where(in_class_dropna.time.dt.hour == hours[hour], drop=True)
            hourly_mean = data_in_hour.mean('time')

            if only_significant_points:
                sig_map = value_significant_of_anomaly_2d_mask(field_3d=data_in_hour,
                                                               fdr_correction=1)
                data_1h = filter_2d_by_mask(data_in_hour, mask=sig_map)
            else:
                data_1h = data_in_hour

            data_1h_mean = data_1h.mean('time')
            # hourly_mean = in_class.groupby(in_class.time.dt.hour).mean(keep_attrs=True)
            # here can NOT use in_class_dropna instead of in_class for groupby, if all values are nan,
            # got a value as Nan, otherwise (using in_class_dropna) the size of input of groupby is zero,
            # got an error.

            cf = plot_geo_subplot_map(geomap=data_1h_mean,
                                      vmax=vmax, vmin=vmin, bias=field_bias,
                                      plot_cbar=False,
                                      statistics=0,
                                      plt_type=plt_type,
                                      cmap=cmap,
                                      ax=ax, domain=area, tag='')

            mean_value = np.float(data_1h_mean.mean())

            if np.abs(mean_value) > 0:
                ax.text(0.98, 0.01, f'{mean_value:4.2f}', fontsize=16,
                        horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
            # num of record
            if plot_num_record:
                ax.text(0.01, 0.01, f'{num_record:g}', fontsize=16,
                        horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

            ax.set_ylabel(f'#_{str(class_names[cls]):s}', color='b')
            ax.set_xlabel('xxx')

    # ----------------------------- surface circulation-----------------------------

    plot_surface_circulation_diurnal_cycle_classif(axs=axs, classif=classif, field=field,
                                                   moisture_flux=plot_moisture_flux, anomaly=circulation_anomaly,
                                                   area=area, test=test_run)

    cbar_label = f'{field.name:s} ({field.assign_attrs().units:s})'
    cb_ax = fig.add_axes([0.15, 0.07, 0.7, 0.02])
    cb = plt.colorbar(cf, orientation='horizontal', shrink=0.8, pad=0.05, cax=cb_ax)
    cb.ax.tick_params(labelsize=24)
    cb.set_label(label=cbar_label, fontsize=24)

    # ----------------------------- title -----------------------------
    if field_bias:
        field_statistic = 'anomaly'
    else:
        field_statistic = 'mean'

    field_name = f'{field.assign_attrs().name:s}'

    if circulation_anomaly:
        circulation_statistic = 'anomaly'
    else:
        circulation_statistic = 'mean'

    title = f'{field_statistic:s}_{field_name}.' \
            f'{circulation_statistic:s}_{circulation_name}_flux.over_' \
            f'{area:s}.with_sig_{only_significant_points:g}.test_run_{test_run:g}'

    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    # fig.suptitle(title)

    # ----------------------------- end of plot -----------------------------

    plt.savefig(f'./plot/{title.replace(" ", "_"):s}.png', dpi=300)
    plt.show()
    print(f'got plot ')


def plot_wind_subplot(area: str, lon: xr.DataArray, lat: xr.DataArray,
                      u: xr.DataArray, v: xr.DataArray,
                      ax,
                      key_input=None,
                      key_units=r'$ {m}/{s}$',
                      plot_field_flux: bool = 0,
                      field_scale=1, bias: int = 0):
    """
    to plot circulation to a subplot
    Args:
        plot_field_flux: plot field flux or not
        field_scale: mean scales of field variable, n_scale and key will change accordingly.
        area ():
        lon ():
        lat ():
        u ():
        v ():
        ax (object):
        bias (int):

    Returns:

    """

    # speed = np.sqrt(u10 ** 2 + v10 ** 2)
    # speed = speed.rename('10m_wind_speed').assign_coords({'units': u10.attrs['units']})

    # Set up parameters for quiver plot. The slices below are used to subset the data (here
    # taking every 4th point in x and y). The quiver_kwargs are parameters to control the
    # appearance of the quiver so that they stay consistent between the calls.

    if area == 'bigreu':
        n_sclice = None
        headlength = 5
        headwidth = 8

        if bias == 0:
            n_scale = 20
            key = 5
            # set the n_scale to define vector length (higher value smaller vector), and key for sample vector in m/s
        if bias:
            n_scale = 3
            key = 0.5

    if area == 'SA_swio':
        headlength = 5
        headwidth = 3
        n_sclice = 8

        if bias == 0:
            n_scale = 3
            key = 10
        else:
            n_scale = 0.3
            key = 1

    if plot_field_flux:
        # the n_scale are really difficult to find, so use the default value
        n_scale = None

        if key == None:
            key = float(key * field_scale)
        else:
            key = key_input

    quiver_slices = slice(None, None, n_sclice)
    quiver_kwargs = {'headlength': headlength,
                     'headwidth': headwidth,
                     'scale_units': 'width',
                     'angles': 'uv',
                     'scale': 80}
    # a smaller scale parameter makes the arrow longer.

    circulation = ax.quiver(lon.values[quiver_slices],
                            lat.values[quiver_slices],
                            u.values[quiver_slices, quiver_slices],
                            v.values[quiver_slices, quiver_slices],
                            linewidth=1.01, edgecolors='k',
                            alpha=0.4,
                            # linewidths is only for controlling the outline thickness,
                            # when an outline of a different color is explicitly requested.
                            # it looks like you have to explicitly set the edgecolors kwarg
                            # to get what you want now
                            color='green', zorder=2, **quiver_kwargs)

    print(n_scale)
    ax.quiverkey(circulation, 0.18, -0.1, key, f'{key:g} ' + key_units,
                 labelpos='E',
                 color='k', labelcolor='k', coordinates='axes')
    # ----------------------------- end of plot wind -----------------------------


def plot_field_in_classif(field: xr.DataArray, classif: pd.DataFrame,
                          area: str, vmax='default', vmin='default',
                          bias: bool = 1,
                          plt_type: str = 'pcolormesh',
                          cmap=plt.cm.coolwarm,
                          plot_wind: bool = 0,
                          only_significant_points: bool = 0,
                          suptitle_add_word: str = ''):
    """
    to plot field in classif
    Args:
        cmap:
        field ():
        classif ():
        area ():
        vmax ():
        vmin ():
        bias ():
        plt_type ():
        plot_wind ():
        only_significant_points ():
        suptitle_add_word ():

    Returns:

    """

    # ----------------------------- get definitions -----------------------------
    class_names = list(set(classif.values.ravel()))
    n_class = len(class_names)

    total_num = len(classif)

    if any([vmax, vmin]) == 'default':
        vmax = field.max().values
        vmin = field.min().values

    # ----------------------------- data -----------------------------
    class_mean = get_data_in_classif(da=field, df=classif,
                                     time_mean=True,
                                     significant=only_significant_points)
    print(f'good')

    # ----------------------------- plot -----------------------------

    if n_class < 4:
        ncols = 1
        fig_size = (6, 9)  # to make text has the good size.
    else:
        ncols = 2
        fig_size = (10, 12)

    nrows = 1
    while nrows * ncols < n_class:
        nrows += 1
    # key word: find figure matrix setup automatic

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex='row', sharey='col',
                            figsize=fig_size, dpi=220, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.03, right=0.70, bottom=0.10, top=0.9, wspace=0.05, hspace=0.05)
    # axs = axs.flatten()
    axs = axs.ravel()

    for c in range(len(class_mean['class'])):
        cls = class_mean['class'].values[c]
        print(f'plot in class {cls:g} ...')

        num = len(classif[classif['class'] == cls])

        # use vertical alignment:
        if len(axs) < 4:
            # case of 1 column, see above lines
            subplot_num = c
        else:
            vertical_subplot_num = np.array([x * 2 for x in range(nrows)] +
                                            [x * 2 + 1 for x in range(nrows)])
            subplot_num = vertical_subplot_num[c]

        ax = set_active_axis(axs=axs, n=subplot_num)
        set_basemap(area=area, ax=ax)

        # plt.title('#' + str(int(cls)), fontsize=20, pad=3)

        # to use the cmap from input arguments:
        cmap = cmap

        cf = plot_geo_subplot_map(geomap=class_mean[:, :, c],
                                  vmax=vmax, vmin=vmin, bias=bias,
                                  domain=area,
                                  cmap=cmap,
                                  plot_cbar=False,
                                  statistics=0,
                                  plt_type=plt_type,
                                  ax=ax, tag='')

        fldmean = class_mean[:, :, c].mean().values
        # number of class
        ax.text(0.05, 0.95, '#' + str(int(cls)), fontsize=24,
                horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        ax.text(0.98, 0.95, f'{num * 100 / total_num: 4.0f}%', fontsize=24,
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        # ax.text(0.05, 0.95, f'{int(num):g} day,{num * 100 / total_num: 4.2f}%', fontsize=24,
        #         horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        ax.text(0.95, 0.000, f'mean: {fldmean:4.2f}', fontsize=26,
                horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    # ----------------------------- end of plot -----------------------------
    # horizontal cb position:
    # cb_ax = fig.add_axes([0.13, 0.1, 0.7, 0.015])
    # vertical cb position:
    cb_ax = fig.add_axes([0.72, 0.12, 0.03, 0.75])
    # horizontal starting point, vertical from bottom, width, and height

    cbar_label = f'{field.assign_attrs().long_name:s} ({field.assign_attrs().units:s})'
    # cb = plt.colorbar(cf, orientation='horizontal', shrink=0.7, pad=0.1,  cax=cb_ax)
    cb = plt.colorbar(cf, orientation='vertical', shrink=0.7, pad=0.1, cax=cb_ax)
    cb.ax.tick_params(labelsize=24)
    cb.set_label(label=cbar_label, fontsize=24)

    # ----------------------------- surface wind -----------------------------
    if plot_wind:
        print(f'plot surface wind ...')

        print(f'loading wind data ... ')

        local_data = '/Users/ctang/local_data/era5'
        u = read_to_standard_da(f'{local_data:s}/u10/u10.hourly.1999-2016.swio.day.nc', 'u10')
        v = read_to_standard_da(f'{local_data:s}/v10/v10.hourly.1999-2016.swio.day.nc', 'v10')

        # u = anomaly_daily(u)
        # v = anomaly_daily(v)

        u = get_data_in_classif(u, classif, significant=False, time_mean=True)
        v = get_data_in_classif(v, classif, significant=False, time_mean=True)

        # speed = np.sqrt(u10 ** 2 + v10 ** 2)
        # speed = speed.rename('10m_wind_speed').assign_coords({'units': u10.attrs['units']})

        # Set up parameters for quiver plot. The slices below are used to subset the data (here
        # taking every 4th point in x and y). The quiver_kwargs are parameters to control the
        # appearance of the quiver so that they stay consistent between the calls.

        # plot in subplot:
        for c in range(len(class_mean['class'])):
            cls = class_mean['class'].values[c]
            print(f'plot wind in class {cls:g} ...')

            ax = set_active_axis(axs=axs, n=c)

            u_1 = u[:, :, c]
            v_1 = v[:, :, c]

            plot_wind_subplot(area='bigreu',
                              lon=u_1.lon, lat=v_1.lat,
                              u=u_1, v=v_1,
                              ax=ax, bias=0)
            # by default plot mean circulation, not anomaly

        # ----------------------------- end of plot -----------------------------
        suptitle_add_word += ' (surface wind)'

    title = f'{field.assign_attrs().long_name:s} in class'

    if suptitle_add_word is not None:
        title = title + '\n ' + suptitle_add_word

    fig.suptitle(title, fontsize=20)
    plt.savefig(f'plot/{field.name:s}_{area:s}_sig{only_significant_points:g}_wind{plot_wind:g}_classif.'
                + title.replace(" ", "_").replace("\n", "") + '.png', dpi=220)
    plt.show()
    print(f'got plot')

def plot_voronoi_diagram_reu(points: np.ndarray, point_names: list, out_fig: str, fill_color=None,
                             fill_color_name:str='') -> object:

    from matplotlib.collections import PolyCollection
    from scipy.spatial import Voronoi, voronoi_plot_2d

    # add distant points to fill the infinite regions: do NOT works, distorted.
    # distant_point = 99999
    # points = np.append(points, [[distant_point, distant_point], [-distant_point, distant_point],
    #                             [distant_point, -distant_point], [-distant_point, -distant_point]], axis=0)

    vor = Voronoi(points, furthest_site=False)

    def voronoi_finite(vor):
        """
        Reconstruct infinite Voronoi regions in a finite space.

        Parameters:
        vor : scipy.spatial.Voronoi
            Voronoi diagram object.

        Returns:
        regions : list of list of tuple
            List of finite Voronoi regions.
        vertices : array
            Voronoi diagram vertices.
            :type vor: object
        """
        new_regions = []
        for region in vor.regions:
            if -1 not in region and len(region) > 0:
                new_regions.append([(vor.vertices[i, 0], vor.vertices[i, 1]) for i in region])
        return new_regions, vor.vertices

    # ---------------------------------------

    fig, ax = plt.subplots(figsize=(10, 8))

    title = 'Voronoi Diagram with MF stations'
    # Plot Voronoi diagram with filled color

    if fill_color is not None:

        regions, vertices = voronoi_finite(vor)
        polygons = PolyCollection(regions, edgecolor='black', cmap='viridis')
        polygons.set_array(fill_color)
        ax.add_collection(polygons)

        # Customize the colorbar
        cbar = plt.colorbar(polygons, ax=ax)
        cbar.set_label(fill_color_name)

        title = f'{title:s} {fill_color_name:s}'

    else:
        out_fig = f'{out_fig[:-4]:s}_no_fill.png'

    # plot lines and points:
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange',
                    line_width=2, line_alpha=0.8, point_size=10,
                    figsize=(10, 8), dpi=300)

    # add station names
    if  point_names is not None:
        for i in range(len(point_names)):
            ax.text(points[i, 0], points[i, 1]+0.01, point_names[i],
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    color='green',
                    fontsize=8)

    # add axis labels:
    plt.xlabel('Longitude ($^\circ$E)', fontsize=12)
    plt.ylabel('Latitude ($^\circ$N)', fontsize=12)

    # Customize the plot as needed
    ax.set_xlim(vor.min_bound[0] - 0.05, vor.max_bound[0] + 0.05)
    ax.set_ylim(vor.min_bound[1] - 0.05, vor.max_bound[1] + 0.05)

    ax.set_xlim(55.15, 55.9)
    ax.set_ylim(-21.5, -20.8)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(title)

    # add coastline:
    plot_coastline_reu()

    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def plot_coastline_reu():

    coastline = load_reunion_coastline()
    plt.scatter(coastline.longitude, coastline.latitude, marker='o', s=1, c='gray', edgecolor='gray', alpha=0.4)

    return 1


# =====
def plot_ttt_regimes(olr_regimes: pd.DataFrame, olr: xr.DataArray,
                     contour: bool = False,
                     area: str = 'SA_swio',
                     only_significant_points: int = 0,
                     paper_plot: int = False):
    """
    plot ttt phase by olr
    :param only_significant_points:
    :param olr_regimes:
    :param olr:
    :return:
    """
    # ----------- definition plot:

    if paper_plot:
        fontsize = 18
        plt_title = False
    else:
        plt_title = True
        fontsize = 24

    # ----------------------------- use the regime in sarah-e period -----------------------------

    year_min = olr.indexes['time'].year.min()
    year_max = olr.indexes['time'].year.max()
    olr_regimes = olr_regimes[np.logical_and(
        olr_regimes.index.year > year_min - 1,
        olr_regimes.index.year < year_max + 1)]

    month = 'NDJF'

    print(f'anomaly ...')

    olr_anomaly = olr.groupby(olr.time.dt.strftime('%m-%d')) - \
                  olr.groupby(olr.time.dt.strftime('%m-%d')).mean('time')

    olr_anomaly = olr_anomaly.assign_attrs(
        {'units': olr.assign_attrs().units, 'long_name': olr.assign_attrs().long_name})

    # the regime is defined by ERA5 ensemble data (B. P.),  but for 18h UTC
    # shift time by +1 day to match ensemble data timestamp
    olr_anomaly = convert_da_shifttime(olr_anomaly, second=-3600 * 18)
    olr = convert_da_shifttime(olr, second=-3600 * 18)

    # ----------------------------- fig config -----------------------------
    fig, axs = plt.subplots(nrows=4, ncols=2, sharex='row', sharey='col',
                            figsize=(8, 8), dpi=220, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.05, top=0.95, wspace=0.05, hspace=0.01)

    # to make it vertical, plot in column:
    axs = axs.T

    axs = axs.ravel()

    for regime in [1, 2, 3, 4, 5, 6, 7]:
        print(f'plot regime = {regime:g}')
        # ----------------------------- calculate mean in each phase -----------------------------
        date_phase_one: pd.DatetimeIndex = olr_regimes.loc[olr_regimes['class'] == regime].index
        if len(date_phase_one) < 1:
            print(f'Sorry, I got 0 day in phase = {regime:g}')
            print(olr_regimes)

            break

        anomaly_olr_1phase: xr.DataArray = olr_anomaly.sel(time=date_phase_one)  # filter
        # if there's an error: check
        # 1) if data_phase_one is empty
        olr_1phase = olr.where(
            olr.time.dt.strftime('%Y-%m-%d').isin(date_phase_one.strftime('%Y-%m-%d')), drop=True)

        # nday = anomaly_olr_1phase.shape[0]
        anomaly_mean: xr.DataArray = anomaly_olr_1phase.mean(axis=0)
        olr_mean = olr_1phase.mean(axis=0)

        # anomaly_ccub = sio.loadmat('./src/regime_maps.mat')['regimes_maps']
        # if not (anomaly_mean - anomaly_ccub[:, :, regime - 1]).max():
        #     print('the same')

        if only_significant_points:
            sig_map: xr.DataArray = value_olr_calssif_significant_map(
                phase=regime, grid=anomaly_mean, month=month, area=area)
            # olr_mean = filter_2d_by_mask(olr_mean, mask=sig_map)
            anomaly_mean = filter_2d_by_mask(anomaly_mean, mask=sig_map)

        ax = set_active_axis(axs=axs, n=regime - 1)
        set_basemap(area=area, ax=ax)

        # ----------------------------- start to plot -----------------------------
        # plt.title('#' + str(regime) + '/' + str(7), pad=3)

        vmax = 50
        vmin = -50

        lon, lat = np.meshgrid(anomaly_mean.lon, anomaly_mean.lat)
        level_anomaly = np.arange(vmin, vmax + 1, 5)
        cf1 = plt.contourf(lon, lat, anomaly_mean, level_anomaly, cmap='PuOr_r',
                           vmax=vmax, vmin=vmin, extend='both')

        if contour:
            level_olr = np.arange(140, 280, 20)
            cf2 = plt.contour(lon, lat, olr_mean, level_olr, cmap='magma_r', vmax=280, vmin=140)
            ax.clabel(cf2, level_olr, inline=True, fmt='%2d', fontsize='xx-small')

        ax.coastlines()

        # cf = plt.pcolormesh(lon, lat, consistency_percentage_map,
        #                     cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

        # ----------------------------- end of plot -----------------------------

        ax.text(0.98, 0.90, f'{month:s}', fontsize=fontsize,
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        ax.text(0.15, 0.98, f'#{regime:g}', fontsize=fontsize,
                horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

        ax.text(0.99, 0.01, f'{olr_anomaly.name:s}', fontsize=fontsize,
                horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

        test = 0
        if test:
            print(f'save netcdf file @regimes to check')
            file = f'./ttt.regime.{regime}.nc'
            anomaly_mean.to_netcdf(file)

    cbar_label = f'OLR ({olr_anomaly.assign_attrs().units:s})'
    cb = plt.colorbar(cf1, ticks=np.ndarray.tolist(level_anomaly), ax=axs, extend="max")
    cb.ax.tick_params(labelsize=18)
    cb.set_label(label=cbar_label, fontsize=18)

    title = f'olr regimes'
    if not paper_plot:
        plt.suptitle(title)

    # tag: specify the location of the cbar
    # cb_ax = fig.add_axes([0.13, 0.1, 0.7, 0.015])
    # cb = plt.colorbar(cf1, orientation='horizontal', shrink=0.7, pad=0.05, label=cbar_label, cax=cb_ax)

    plt.savefig(f'./plot/ttt_regimes_{area:s}_sig_{only_significant_points:g}.paper_plot_{paper_plot:g}.png', dpi=220)

    plt.show()
    print(f'got plot')


def plot_color_matrix(df: pd.DataFrame, ax, cbar_label: str, plot_number: bool = False, cmap='Blues'):
    """
    plot matrix by df, where x is column, y is index,
    :param plot_number:
    :type plot_number:
    :param cbar_label:
    :type cbar_label: str
    :param df:
    :type df:
    :param ax:
    :type ax:
    :return:
    :rtype: ax
    """

    import math

    c = ax.pcolor(df, cmap=plt.cm.get_cmap(cmap, df.max().max() + 1))

    x_ticks_label = df.columns
    y_ticks_label = df.index

    # put the major ticks at the middle of each cell
    x_ticks = np.arange(df.shape[1]) + 0.5
    y_ticks = np.arange(df.shape[0]) + 0.5
    ax.set_xticks(x_ticks, minor=False)
    ax.set_yticks(y_ticks, minor=False)

    ax.set_xticklabels(x_ticks_label, minor=False)
    ax.set_yticklabels(y_ticks_label, minor=False)

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False)

    # ax.set_ylabel('year')
    # ax.set_xlabel('month')

    vmin = int(df.values.min())
    vmax = int(df.values.max())

    # print(vmin, vmax)

    if vmin + vmax < vmax:
        c = ax.pcolor(df, cmap=plt.cm.get_cmap('coolwarm', df.max().max() + 1))
        cbar_ticks = [x for x in range(vmin, vmax + 1, math.ceil((vmax - vmin) / 10))]
    else:
        cbar_ticks = [x for x in range(vmin, vmax, math.ceil((vmax - vmin) / 10))]

    if plot_number:
        for i in range(df.shape[1]):  # x direction
            for j in range(df.shape[0]):  # y direction
                c = df.iloc[j, i]
                # notice to the order of
                ax.text(x_ticks[i], y_ticks[j], f'{c:2.0f}', va='center', ha='center')
        # put cbar label
        ax.yaxis.set_label_position("right")
    else:
        cb = plt.colorbar(c, ax=ax, label=cbar_label, ticks=cbar_ticks)
        loc = [x + 0.5 for x in cbar_ticks]
        cb.set_ticks(loc)
        cb.set_ticklabels(cbar_ticks)

    return ax


def value_cbar_ticks_from_vmax_vmin(vmax, vmin, num_bin):
    """ this is only an automatic cbar ticks """
    import math

    mi = math.floor(np.log10(vmax - vmin))
    list_bin = np.array([x * 10 ** (mi - 1) for x in range(1, 10)])
    interval = list_bin.flat[np.abs(list_bin - (vmax - vmin) / num_bin).argmin()]
    cbar_ticks = np.round([x for x in np.arange(vmin, vmax * (1 + 1 / 10 / num_bin), interval)], np.abs(mi - 1))
    print(cbar_ticks)

    #
    # from matplotlib.colors import TwoSlopeNorm
    # if bias:
    #     # to make uneven colorbar with zero in white
    #     norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    # else:
    #     norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmax - vmin) / 2 + vmin, vmax=vmax)
    return cbar_ticks


def find_symmetric_difference(list1, list2):
    """
    show difference
    :param list1:
    :type list1:
    :param list2:
    :type list2:
    :return:
    :rtype:
    """

    difference = set(list1).symmetric_difference(set(list2))
    list_difference = list(difference)

    return list_difference


def plot_join_heatmap_boxplot(da: xr.DataArray):
    # ----------------------------- prepare data -----------------------------
    # for heatmap:
    da_matrix = da.groupby(da.time.dt.strftime("%m-%H")).mean(keep_attrs=True)
    matrix = pd.DataFrame(da_matrix.to_numpy().reshape(24, -1, order='F')).transpose()

    y_sticks = value_str_month_name(range(1, 13))
    x_sticks = [f'{x:g}H' for x in range(24)]

    # y direction:
    name_y = 'month'
    # x direction:
    name_x = 'hour'

    fig = plt.figure()
    widths = [1, 3]
    heights = [1, 2]
    gridspec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)
    gridspec.update(wspace=0.1, hspace=0.2)  # set the spacing between axes.

    # heatmap:
    ax = fig.add_subplot(gridspec[1, 1])
    heat = sns.heatmap(matrix, ax=ax,
                       xticklabels=2, yticklabels=True)

    # boxplot for month:
    mon_mean = monthly_mean_da(da).to_dataframe()
    mon_mean['month'] = mon_mean.index.month

    ax_left = fig.add_subplot(gridspec[1, 0])
    sns.set_theme(style="ticks")
    mon_box_plot = sns.boxplot(x='month', y=da.name, data=mon_mean, ax=ax_left,
                               orient="h", palette="vlag", showmeans=True)

    # mon_box_plot = sns.boxplot(x='month', y=da.name, data=mon_mean, ax=ax_left,
    #                                orient="h", palette="vlag", showmeans=True)
    plt.show()


def drop_nan_infinity(df):
    df_filter = df.isin([np.nan, np.inf, -np.inf])
    # Mask df with the filter
    df = df[~df_filter]

    df.dropna(inplace=True)

    return df


def check_missing_df_da_interval(df, vmin=None, vmax=None, output_tag='', freq='H', columns=''):
    for i in range(len(columns)):
        col = columns[i]

        print(col)
        if vmax is not None:
            df = df[df <= vmax]

        if vmin is not None:
            df = df[df >= vmin]
            df = df.dropna()

        matrix = check_missing_da_df(start=str(df.index.date[0]), end=str(df.index.date[-1]),
                                     freq=freq, data=df, output_plot_tag=f'{output_tag:s}_{col:s}')
    return matrix


def plot_power_spectral_density_multi_columns_df(df: pd.DataFrame, columns: list = ['', ], title='',
                                                 linestyles=None,
                                                 vmax=None, vmin=None, check_missing=True,
                                                 xlabel='', output_tag: str = ''):
    """
    applied project Sky_clearness_2023:
    linestyles is a list of linestyle that could be applied by orders in the plot to group the lines.
    """
    from scipy import signal

    colors = get_color(len(columns))

    fig = plt.figure(figsize=(12, 8), dpi=300)

    for i in range(len(columns)):
        col = columns[i]

        print(col)

        if linestyles is not None:
            linestyle = linestyles[i]
        else:
            linestyle = '-'

        df1 = df[{col}]

        # when compare PDF in a value range:
        # such as Sky_Clearness 2023: for CI
        # if vmax is not None:
        #     df1 = df1[df1 <= vmax]
        #
        # if vmin is not None:
        #     df1 = df1[df1 >= vmin]

        # check missing data after the selecting above.

        # if vmax is not None:
        #     plt.xlim(vmin, vmax)

        # -----------------
        df1 = df1.fillna(0)
        signal_data = df1.values.ravel()
        fs = 1000.0  # 1 kHz sampling frequency
        (f, S) = signal.welch(signal_data, fs=3600, nperseg=3600)

        plt.semilogy(f, S, label=col, color=colors[i], linewidth=2, linestyle=linestyle, )

        plt.xlim([0, 100])

        # ax = sns.distplot(df1, hist=False, kde=True,
        #                   label=col, bins=30, color=colors[i], hist_kws={'edgecolor': 'black'},
        #                   kde_kws={'linewidth': 2, 'linestyle': linestyle})

    # plt.legend(loc='upper left', prop={'size': 12})
    plt.legend(prop={'size': 12})

    plt.xlabel(xlabel)
    plt.title(title)

    plt.savefig(f'./plot/psd.{output_tag:s}.png', dpi=300)
    plt.show()


def plot_pdf_multi_columns_df(df: pd.DataFrame, columns: list = ['', ], title='',
                              linestyles=None,
                              sel_v_max=None, sel_v_min=None, check_missing=True,
                              xlabel='', output_tag: str = ''):
    """
    applied project Sky_clearness_2023:
    linestyles is a list of linestyle that could be applied by orders in the plot to group the lines.
    """

    colors = get_color(len(columns))

    fig = plt.figure(figsize=(12, 8), dpi=300)

    for i in range(len(columns)):
        col = columns[i]

        print(col)

        if linestyles is not None:
            linestyle = linestyles[i]
        else:
            linestyle = '-'

        df1 = df[{col}]

        # when compare PDF in a value range:
        # such as Sky_Clearness 2023: for CI
        if sel_v_max is not None:
            df1 = df1[df1 <= sel_v_max]

        if sel_v_min is not None:
            df1 = df1[df1 >= sel_v_min]

        # check missing data after the selecting above.

        ax = sns.distplot(df1, hist=False, kde=True,
                          label=col, bins=30, color=colors[i], hist_kws={'edgecolor': 'black'},
                          kde_kws={'linewidth': 2, 'linestyle': linestyle})

    # plt.legend(loc='upper left', prop={'size': 12})
    plt.legend(prop={'size': 12})

    plt.xlabel(xlabel)
    plt.title(title)

    plt.savefig(f'./plot/kde.{output_tag:s}.png', dpi=300)
    plt.show()


def check_hourly_density_df(df: pd.DataFrame, columns=None, vmax=None, vmin=None, title='',
                            limit_line=False, limit_value=1,
                            output_tag: str = 'output_tag'):
    """
    applied project Sky_clearness_2023: to check if some SSR is larger than 1367 for example

    easy to change to other plot for multi column df:
    """

    if columns is not None:
        print(f'user specified columns')
    else:
        columns = df.columns

    print(f'columns used: ')
    print(columns)

    n_col = len(columns)

    if n_col > 4:
        n_raw = 4
    else:
        n_raw = 2

    fig_width = n_col * 2 + 1
    fig_height = n_raw * 2 + 4

    fig, axs = plt.subplots(nrows=n_raw, ncols=int(n_col / n_raw + 1),
                            sharex=False, sharey=False,
                            figsize=(fig_width, fig_height), dpi=300)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.10, top=0.9, wspace=0.1, hspace=0.1)
    axs = axs.ravel()

    print(f'fig created ')

    for i in range(n_col):
        column = columns[i]

        print(f'plot {column:s}')

        df0 = df[{column}]

        # when compare PDF in a value range:
        # such as Sky_Clearness 2023: for CI
        if vmax is not None:
            df0 = df0[df0 <= vmax]

        if vmin is not None:
            df0 = df0[df0 >= vmin]

        df0['hour'] = df0.index.hour
        ax = axs[i]

        sns.histplot(data=df0, x='hour', y=column,
                     bins=30, discrete=(True, False), log_scale=(False, False),
                     # kde=True,
                     # stat="density",
                     cbar=True, cbar_kws=dict(shrink=.75),
                     ax=ax,
                     )

        ax.set_xticks(np.arange(5, 20, ), minor=False)

        if limit_line:
            plt.sca(ax)
            plt.axhline(y=limit_value, color='r', linestyle='-', label=f'y={limit_value:2.1f}')

        if vmax is not None:
            ax.set_ylim(vmin, vmax)

        # plt.grid(zorder=-1)
        plt.legend()

    plt.suptitle(title + f' in [{vmin:2.1f}-{vmax:2.1f}]')

    plt.savefig(f'./plot/hourly.density.{output_tag:s}.png', dpi=300)
    plt.show()

    print(f'image saved')


def plot_density_df(df: pd.DataFrame,
                    title='',
                    output_tag: str = 'output_tag'):
    fig = plt.figure(figsize=(10, 16), dpi=300)

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.scatter(range(df.size), df.values)
    plt.grid()
    plt.title(title)

    ax2 = fig.add_subplot(2, 1, 2)
    df.hist(bins=50, alpha=0.8, ax=ax2)
    df.plot(kind='kde', secondary_y=True, ax=ax2)
    plt.grid(zorder=-1)

    plt.savefig(f'./plot/density.{output_tag:s}.png', dpi=300)
    plt.show()

    from scipy import stats
    print(stats.normaltest(df.values))


def plot_matrix_2d_df(
        df: pd.DataFrame = None,
        x_column: str = 'x', y_column: str = 'y',
        x_label: str = None, y_label: str = None,
        z_column: str = 'z', z_label: str = None,
        x_plt_limit: list = [-1, 1], y_plt_limit: list = [-1, 1],
        cut_off: bool = 0, cut_value: float = 1,
        z_plt_limit: list = [-1, 1],
        statistics: bool = 1,
        occurrence: bool = 1,
        suptitle_add_word: str = ""):
    # definition:
    class_names_x: list = list(set(df[x_column].sort_values().values))
    class_names_y: list = list(set(df[y_column].sort_values().values))

    n_class_x = len(class_names_x)
    n_class_y = len(class_names_y)

    if occurrence:
        # get cross matrix: occurrence
        cross = np.zeros((len(class_names_y), len(class_names_x)))
        for i in range(len(class_names_y)):
            class_one = df.loc[df[y_column] == class_names_y[i]]
            for j in range(len(class_names_x)):
                class_cross = class_one.loc[class_one[x_column] == class_names_x[j]]
                cross[i, j] = len(class_cross)
        cross_df = pd.DataFrame(data=cross, index=class_names_y, columns=class_names_x).astype(int)

    # ----------------------------- plot -----------------------------
    # fig size:
    fig_width = int(n_class_x * 1.5)
    fig_height = n_class_y + 1

    n_column = n_class_x + 1
    n_raw = n_class_y + 1

    fig, axs = plt.subplots(nrows=n_raw, ncols=n_column, sharex=True, sharey=True,
                            figsize=(fig_width, fig_height), dpi=300)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.10, top=0.9, wspace=0.05, hspace=0.05)

    # plot for matrix in [1 ]
    # mixing this two class first.

    for x in range(n_class_x):
        for y in range(n_class_y):
            # find days in this mixing class
            mix_class = df.where(
                (df[x_column] == class_names_x[x]) &
                (df[y_column] == class_names_y[y])).dropna()

            ax = axs[y + 1, x + 1]
            plt.sca(axs[y + 1, x + 1])
            # plot:
            sns.histplot(data=mix_class[z_column], kde=True, stat='density',
                         binwidth=0.5,
                         ax=ax)
            if y == n_class_y:
                ax.set(xlabel=f'Month {class_names_x[x]}')
            else:
                ax.set(xlabel=None)
            ax.set(ylabel=None)
            plt.xlim(z_plt_limit[0], z_plt_limit[1])
            # plt.legend(z_column)
            print(x, y)

            if statistics:
                count = len(mix_class)
                ratio = count * 100 / len(df)
                tag = f'{count:g},{ratio:4.0f}%'
                ax.text(0.9, 0.95, tag, fontsize=12, horizontalalignment='right', verticalalignment='top',
                        transform=ax.transAxes)

                if cut_off:
                    plt.axvline(x=cut_value, color='red', linestyle='--', linewidth=2, label='')
                    cut_count = len(mix_class[mix_class[z_column] > cut_value].dropna())
                    ax.text(0.9, 0.75,
                            f'{cut_count * 100 / count:4.1f}%', fontsize=10, color='red',
                            horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
                print(f'cut', cut_count, count, len(df))

    # plot x class: in [0, 1] to [0, -1]
    for x in range(n_class_x):

        plt.sca(axs[0, x + 1])
        ax = axs[0, x + 1]

        # data:
        mix_class = df.where(df[x_column] == class_names_x[x]).dropna()

        # plot:
        sns.histplot(data=mix_class[z_column], kde=True, stat='density',
                     binwidth=0.5,
                     ax=ax)
        ax.set(ylabel=None)
        plt.xlim(z_plt_limit[0], z_plt_limit[1])

        plt.title(f'{x_label:s} {class_names_x[x]}')

        if statistics:
            count = len(mix_class)
            ratio = count * 100 / len(df)
            tag = f'{count:g},{ratio:4.0f}%'
            ax.text(0.9, 0.95, tag, fontsize=12, horizontalalignment='right', verticalalignment='top',
                    transform=ax.transAxes)

            if cut_off:
                plt.axvline(x=cut_value, color='red', linestyle='--', linewidth=2, label='')
                cut_count = len(mix_class[mix_class[z_column] > cut_value].dropna())
                ax.text(0.9, 0.75,
                        f'{cut_count * 100 / count:4.1f}%', fontsize=10, color='red',
                        horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

                print(f'cut', cut_count, count, len(df))
        # plt.legend(z_column)

    # plot y class: in [y, 0] to [-1, 0]
    for y in range(n_class_y):

        plt.sca(axs[y + 1, 0])
        ax = axs[y + 1, 0]

        # data:
        mix_class = df.where(df[y_column] == class_names_y[y]).dropna()

        # plot:
        sns.histplot(data=mix_class[z_column], kde=True, stat='density',
                     binwidth=0.5,
                     ax=ax)

        ax.set(xlabel=None)
        plt.xlim(z_plt_limit[0], z_plt_limit[1])
        if statistics:
            count = len(mix_class)
            ratio = count * 100 / len(df)
            tag = f'{count:g},{ratio:4.0f}%'
            ax.text(0.9, 0.95, tag, fontsize=12, horizontalalignment='right', verticalalignment='top',
                    transform=ax.transAxes)

            if cut_off:
                plt.axvline(x=cut_value, color='red', linestyle='--', linewidth=2, label='')
                cut_count = len(mix_class[mix_class[z_column] > cut_value].dropna())
                ax.text(0.9, 0.75,
                        f'{cut_count * 100 / count:4.1f}%', fontsize=10, color='red',
                        horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
                print(f'cut', cut_count, count, len(df))

    title = f'{suptitle_add_word:s} {z_label:s} in ' \
            f'{x_column:s}_vs_{y_column:s} ' \
            f'cut_off_{cut_off:g} ' \
            f'cut_value_{cut_value:4.2f}'

    fig.suptitle(title)

    plt.savefig(f'./plot/{title.replace(" ", "_"):s}.'
                f'.png', dpi=300)
    plt.show()

    print(f'job done')


def plot_matrix_class_vs_class_field(class_x: pd.DataFrame,
                                     class_y: pd.DataFrame,
                                     field: xr.DataArray,
                                     plt_type: str = 'pcolormesh',
                                     vmax=30, vmin=-30, bias=1,
                                     only_significant_points=0,
                                     output_plot: str = 'class_vs_class_matrix_maps',
                                     occurrence: bool = 1,
                                     suptitle_add_word: str = ""):
    """
    plot the matrix of class vs class, with maps, of each single class and mixing class
    class_df: DataFrame of one columns of classifications with DateTimeIndex, columns' names will be used.
    $$: if plot occurrence, impact of class_x on class_y

    param class_y:
    :type class_y: pandas.core.frame.DataFrame
    :param class_x:
    :type class_x: pandas.core.frame.DataFrame
    :param output_plot:
    :type output_plot: str
    :param occurrence: if occurrence is True, will plot numbers in the matrix by default
    :type occurrence:
    :param suptitle_add_word:
    :type suptitle_add_word: str
    :return:
    :rtype: None

    Args:
        field:
    """

    # the input DataFrames may have different index, so merge two classes with DataTimeIndex:
    class_df = class_x.merge(class_y, left_index=True, right_index=True)

    # x direction:
    column_name_x = class_x.columns[0]
    # y direction:
    column_name_y = class_y.columns[0]

    class_name_x = sorted(set(class_x.values.ravel()))
    class_name_y = sorted(set(class_y.values.ravel()))

    # get cross matrix: occurrence
    cross = np.zeros((len(class_name_y), len(class_name_x)))
    for i in range(len(class_name_y)):
        class_one = class_df.loc[class_df[column_name_y] == class_name_y[i]]
        for j in range(len(class_name_x)):
            class_cross = class_one.loc[class_one[column_name_x] == class_name_x[j]]
            cross[i, j] = len(class_cross)
    cross_df = pd.DataFrame(data=cross, index=class_name_y, columns=class_name_x).astype(int)

    # contingency:
    sig, expected = value_sig_neu_test_2d(contingency=cross_df.values, alpha=0.05, output_expected=True)
    observed = cross_df.values

    output = {'sig': sig, 'observed': observed, 'expected': expected}

    n_class_x = len(set(class_x.values.ravel()))
    n_class_y = len(set(class_y.values.ravel()))

    # ----------------------------- plot -----------------------------
    fontsize = 14
    fig_width = n_class_x + 1
    fig_height = n_class_y + 1

    n_column = n_class_x + 1
    n_raw = n_class_y + 1

    # prepare data:
    field_in_class_x, class_size_x = get_data_in_classif(da=field, df=class_df[{column_name_x}],
                                                         significant=only_significant_points,
                                                         time_mean=True, return_size=True)
    field_in_class_y, class_size_y = get_data_in_classif(da=field, df=class_df[{column_name_y}],
                                                         significant=only_significant_points,
                                                         time_mean=True, return_size=True)

    # for plot mjo vs ttt change the fig size:
    fig_height = n_class_y

    fig, axs = plt.subplots(nrows=n_raw, ncols=n_column, sharex=True, sharey=True,
                            figsize=(fig_width + 1, fig_height), dpi=300, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.104, right=0.8, bottom=0.11, top=0.9, wspace=0.01, hspace=0.01)

    # plot for matrix in [1 ]
    # mixing this two class first.

    for x in range(n_class_x):
        for y in range(n_class_y):
            # find days in this mixing class
            mix_class = class_df.where(
                (class_df[column_name_x] == class_name_x[x]) &
                (class_df[column_name_y] == class_name_y[y])).dropna()

            field_mix_class = field.sel(time=mix_class.index.date)

            if only_significant_points:
                print(x, y)
                print(field_mix_class.sizes)
                sig_map: xr.DataArray = value_significant_of_anomaly_2d_mask(field_3d=field_mix_class, conf_level=0.05)
                print(f'only work for anomaly/change/trend/ !!! compare to ZERO!!')
                field_to_plot = filter_2d_by_mask(field_mix_class, mask=sig_map).mean(axis=0)
            else:
                field_to_plot = field_mix_class.mean(dim='time')

            # plot:
            ax = axs[y + 1, x + 1]

            cf = plot_geo_subplot_map(
                geomap=field_to_plot,
                bias=bias, domain='reu', tag=f'',
                vmax=vmax, vmin=vmin, plt_type=plt_type,
                plot_cbar=False,
                ax=ax)
            if sig[y, x]:
                ax.text(0.98, 0.97, f'{observed[y, x]:g}'
                # f'',
                                    f'({np.round(expected[y, x]):g})',
                        fontsize=10, weight='bold',
                        horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
            else:
                ax.text(0.98, 0.97, f'{observed[y, x]:g}'
                # f'',
                                    f' ({np.round(expected[y, x]):g})',
                        fontsize=10,
                        horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

            # ax.text(0.01, 0.98, f'{class_name_x[x]:g}', fontsize=12,
            #         horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

    # plot x class: in [0, 1] to [0, -1]
    for x in range(n_class_x):
        data = field_in_class_x.loc[{'class': class_name_x[x]}]
        ax = axs[0, x + 1]
        plot_geo_subplot_map(
            geomap=data,
            bias=bias, domain='reu', tag=f'{class_size_x[class_name_x[x]]:g}',
            vmax=vmax, vmin=vmin, plt_type=plt_type,
            plot_cbar=0,
            ax=ax)

        # ax.text(0.01, 0.98, f'#{class_name_x[x]:g}', fontsize=12,
        #         horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

    # plot y class: in [y, 0] to [-1, 0]
    for y in range(n_class_y):
        data = field_in_class_y.loc[{'class': class_name_y[y]}]
        ax = axs[y + 1, 0]
        plot_geo_subplot_map(
            geomap=data,
            bias=bias, domain='reu', tag=f'{class_size_y[class_name_y[y]]:g}',
            vmax=vmax, vmin=vmin, plt_type=plt_type,
            plot_cbar=0,
            ax=ax)

        # ax.text(0.01, 0.98, f'#{class_name_y[y]:g}', fontsize=12,
        #         horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    # ================================== end plot ==================================

    # xlable:
    for x in range(n_class_x):
        ax = axs[n_class_y, x + 1]
        ax.set_xlabel(f'Reg_{x + 1:g}')
    # ylable:
    for y in range(n_class_y):
        ax = axs[y + 1, 0]
        ax.set_ylabel(f'Phase_{y + 1:g}')

    title_x = ''
    for i in range(n_class_x):
        title_x += f'Reg_{class_name_x[i]:g}   '
    plt.figtext(0.21, 0.91, title_x, fontsize=fontsize)

    title_y = ''
    for i in range(n_class_y):
        title_y += f'Pha_{class_name_y[7 - i]:g}  '
    plt.figtext(0.075, 0.12, title_y, rotation='vertical', fontsize=fontsize)

    # vertical cb position:
    cb_ax = fig.add_axes([0.82, 0.12, 0.02, 0.75])
    # horizontal starting point, vertical from bottom, width, and height

    cbar_label = f'{field.assign_attrs().long_name:s} ({field.units:s})'

    # for Mialhe_2021:
    cbar_label = f'SSR daily anomaly (W m**-2)'

    # cb = plt.colorbar(cf, orientation='horizontal', shrink=0.7, pad=0.1,  cax=cb_ax)
    cb = plt.colorbar(cf, orientation='vertical', shrink=0.7, pad=0.1, cax=cb_ax)
    cb.ax.tick_params(labelsize=fontsize)
    cb.set_label(label=cbar_label, fontsize=fontsize)

    title = f'{suptitle_add_word:s} {column_name_x:s}_vs_{column_name_y}_in_{field.name:s}_' \
            f'{output_plot:s}'

    plt.savefig(f'./plot/{title.replace(" ", "_"):s}'
                f'significant_{only_significant_points:g}'
                f'.png', dpi=300)
    # for Mialhe_2021:

    plt.figtext(0.04, 0.42, 'MJO phase NDJF', rotation='vertical', fontsize=fontsize)
    plt.figtext(0.4, 0.95, 'OLR regime NDJF', fontsize=fontsize)
    plt.savefig(f'./plot/Figure.8.png', dpi=300)
    print(f'./plot/{title.replace(" ", "_"):s}.png')
    plt.show()
    print(title)

    return output

    # ------


def value_sig_neu_test_2d(contingency: np.ndarray,
                          output_expected: np.ndarray = None,
                          alpha: float = 0.05,
                          p_expected: bool = False):
    """
    if input is 1D, p_expected has to be given.
    Args:
        contingency: 
        output_expected: 
        alpha: 
        p_expected: 

    Returns:

    """
    # for Neu's test:
    n_row = contingency.shape[0]
    n_col = contingency.shape[1]
    k = (n_row - 1) * (n_col - 1)
    # for individual cell @(1-alpha/k) confidence level,
    # the upper tail z_value, i.e., the (1-alpha/k)/2 th percentile.
    z_score = stats.norm.interval(1 - alpha / k)[1]

    # n_dim = 2
    # z_score = stats.norm.interval(1 - alpha / (2 * n_dim), 0, 1)[1]

    # total number of all columns, OLR regimes for example
    count_cols = contingency.sum(axis=0)

    # total number of all row, MJO phases for example
    count_rows = contingency.sum(axis=1)
    print(count_cols, count_rows)

    # calculate expected P:
    # if only one colum input,
    if contingency.shape[1] < 2:
        # if no input P_expected, quit
        if output_expected == None:
            quit(0)
        else:
            p_expect = p_expected
    else:
        # p_expected, same for each column, each e.g., for each OLR regime
        p_expect = np.array(contingency.sum(axis=1) / np.sum(contingency))

    # p_observed, value in cell divided by column sum
    p_obs = np.array([contingency[:, i] / contingency[:, i].sum() for i in range(contingency.shape[1])]).T

    significant = []
    left_all = []
    right_all = []
    # loop for each column
    for x in range(contingency.shape[1]):
        # observed
        p_i = p_obs[:, x]

        left = p_i - z_score * np.sqrt(p_i * (1 - p_i) / count_cols[x])
        right = p_i + z_score * np.sqrt(p_i * (1 - p_i) / count_cols[x])

        sig = [np.logical_or(p_expect[y] < left[y], p_expect[y] > right[y]) for y in range(len(p_i))]

        significant.append(sig)
        right_all.append(right)
        left_all.append(left)

    # transpose:
    significant = np.array(significant).T
    left_all = np.array(left_all).T
    right_all = np.array(right_all).T

    # prepare outputs:

    if output_expected:
        # expected proportion:
        expected_num = np.outer(p_expect, count_cols)

        diff = contingency - expected_num

        for m in range(contingency.shape[1]):
            np.set_printoptions(precision=3)
            print(f'column =', m + 1),
            print(contingency[:, m])
            print(expected_num[:, m])
            print(diff[:, m])
            print('observed', p_obs[:, m])
            print('expected', p_expect)

            print(left_all[:, m])
            print(right_all[:, m])
            print(significant[:, m])

        print(f'input alpha= {alpha:4.2f}, '
              f'the upper probability tail area is {alpha / 2 / k:4.4f}, '
              f'individual confidence level is {1 - alpha / k: 4.4f}, '
              f'z_score={z_score:4.3f}, k={k:g}')
        return significant, expected_num
    else:
        print(z_score, f'k={k:g}', f'alpha={alpha:4.2f}')
        return significant


def test_neu_test():
    pa = np.array([0.214, 0.188, 0.256, 0.342])
    # values from Neu's paper 1974
    alpha = 0.1
    k = 4
    # z_score = stats.norm.interval(1 - alpha / 2 / n_dim, 0, 1)[1]
    # the upper tail z_value
    z_score = stats.norm.interval(1 - alpha / k, 0, 1)[1]

    n_all = 117
    left = pa - z_score * np.sqrt(pa * (1 - pa) / n_all)
    right = pa + z_score * np.sqrt(pa * (1 - pa) / n_all)

    print(left, right)

    print(f'input alpha= {alpha:4.2f}, '
          f'the upper probability tail area is {alpha / 2 / k:4.4f}, '
          f'individual confidence level is {1 - alpha / k: 4.4f}, '
          f'z_score={z_score:4.3f}, k={k:g}')


# def plot_monthly_diurnal_maps(field: xr.DataArray, ax):
def monthly_diurnal_boxplot_matrix_df(class_x: pd.DataFrame, class_y: pd.DataFrame, plot: bool = False,
                                      output_figure: str = 'contingency.png'):
    # the input DataFrames may have different index, so merge two classes with DataTimeIndex:
    class_df = class_y.merge(class_x, left_index=True, right_index=True)

    # y direction:
    name_y = class_df.columns[0]
    # x direction:
    name_x = class_df.columns[1]

    class_name_y = list(set(class_df.iloc[:, 0]))
    class_name_x = list(set(class_df.iloc[:, 1]))

    # get cross matrix
    cross = np.zeros((len(class_name_y), len(class_name_x)))
    for i in range(len(class_name_y)):
        class_one = class_df.loc[class_df[name_y] == class_name_y[i]]
        for j in range(len(class_name_x)):
            class_cross = class_one.loc[class_one[name_x] == class_name_x[j]]
            cross[i, j] = len(class_cross)
            print(f'x = {j + 1:g}, y = {i + 1:g}, cross_size = {cross[i, j]:g}')

    cross_df = pd.DataFrame(data=cross, index=class_name_y, columns=class_name_x).astype(int)

    sig, expected = value_sig_neu_test_2d(contingency=cross_df.values, alpha=0.05, output_expected=True)

    # output = {'sig': sig[::-1], 'expected': expected[::-1], 'observed': cross[::-1]}
    output = {'sig': sig, 'expected': expected, 'observed': cross}

    if plot:

        fontsize = 14
        # ----------------------------- plot -----------------------------
        fig = plt.figure(figsize=(12.5, 8), dpi=300)
        widths = [1, 3]
        heights = [1, 2]
        gridspec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)
        gridspec.update(wspace=0.08, hspace=0.15)  # set the spacing between axes.

        # matrix:
        ax = fig.add_subplot(gridspec[1, 1])
        cbar_label = 'count'

        plot_number = True  # it's better to plot number with occurrence
        cbar_label = 'observed - expected (N. of day)'

        # ======== the color matrix:

        import math
        df = cross_df

        x_ticks_label = df.columns
        y_ticks_label = [np.int(x) for x in class_name_y[::-1]]

        # put the major ticks at the middle of each cell
        x_ticks = np.arange(df.shape[1]) + 0.5
        y_ticks = np.arange(df.shape[0]) + 0.5

        # inverse y_ticks, since the plot will go from bottom to up when using ax.text
        y_ticks = y_ticks[::-1]  # up to down

        ax.set_xticks(x_ticks, minor=False)
        ax.set_yticks(y_ticks[::-1], minor=False)

        ax.set_xticklabels(x_ticks_label, minor=False, fontsize=fontsize)
        ax.set_yticklabels(y_ticks_label, minor=False, fontsize=fontsize)

        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False)

        diff = cross_df.values - expected

        vmin = -30
        vmax = 30
        vmin = np.int(diff.min())
        vmax = np.int(diff.max())

        from matplotlib.colors import TwoSlopeNorm
        if vmin * vmax < 0:
            vmax = np.max([vmin, vmax])

            vmax = 12
            vmin = vmax * -1

            # to make uneven colorbar with zero in white
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        cmap = plt.cm.get_cmap('RdYlGn', vmax + 1).reversed()
        cmap = plt.cm.get_cmap('PiYG', vmax + 1).reversed()

        # cmap = plt.cm.get_cmap('coolwarm', df.max().max() + 1)
        # plot the inverse color: since pcolor plot it from bottom to up:
        cf = ax.pcolor(diff[::-1], cmap=cmap, norm=norm)

        if vmin + vmax < vmax:
            cbar_ticks = [x for x in range(vmin, vmax + 1, math.ceil((vmax - vmin) / 10))]
        else:
            cbar_ticks = [x for x in range(vmin, vmax, math.ceil((vmax - vmin) / 10))]

        print(cbar_ticks)
        if plot_number:
            number = cross
            for i in range(number.shape[1]):  # x direction
                for j in range(number.shape[0]):  # y direction
                    c = number[j, i]
                    # notice to the order of
                    if sig[j, i]:
                        ax.text(x_ticks[i], y_ticks[j], f'{c:2.0f}', va='center', ha='center',
                                weight='bold', fontsize=fontsize)
                    else:
                        ax.text(x_ticks[i], y_ticks[j], f'{c:2.0f}', va='center', ha='center',
                                fontsize=fontsize)
            # put cbar label
            ax.yaxis.set_label_position("right")

        # add colorbar
        cb_ax = fig.add_axes([0.92, 0.11, 0.02, 0.48])
        cb = plt.colorbar(cf, orientation='vertical', shrink=0.8, pad=0.05, cax=cb_ax,
                          ax=ax, label=cbar_label, ticks=cbar_ticks)
        loc = [x for x in cbar_ticks]
        cb.set_ticks(loc)
        cb.set_ticklabels(cbar_ticks)
        cb.ax.tick_params(labelsize=fontsize)
        cb.set_label(label=cbar_label, fontsize=fontsize)

        # ======== the color matrix: end

        ax.set_xlabel('SSR class', fontsize=fontsize)

        # histogram in x direction:
        ax = fig.add_subplot(gridspec[0, 1])
        bars = class_name_x
        data = class_df[name_x]
        height = [len(data[data == x]) for x in class_name_x]

        colors = ['darkgray', 'lightskyblue', 'firebrick', 'indianred', 'darkgray',
                  'cornflowerblue', 'royalblue', 'darkgray', 'darkgray']

        y_pos = np.arange(len(bars))
        ax.bar(bars, height, align='center', color=colors)

        plt.grid(axis='y')
        ax.set_xlim(0.5, y_pos[-1] + 1.5)  # these limit is from test
        # x_ticks = np.arange(len(class_name_x)) + 0.5

        # ax.set_xticks([], minor=False)
        ax.set_xticklabels([], minor=False, fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.set_ylabel('day', fontsize=fontsize)
        ax.set_title('SSR class', fontsize=fontsize)

        # histogram in y direction:
        ax = fig.add_subplot(gridspec[1, 0])
        bars = class_name_y
        data = class_df[name_y]
        height = [len(data[data == x]) for x in class_name_y]

        colors_olr = ['darkgray', 'royalblue', 'indianred', 'darkgray', 'indianred', 'firebrick', 'darkgray']
        # colors_olr = ['darkgray', 'blueviolet', 'orange', 'darkgray', 'orange', 'darkorange', 'darkgray']

        y_pos = np.arange(len(bars))
        ax.barh(bars, height, align='center', color='orange')

        plt.grid(axis='x')

        ax.set_ylim(0.5, y_pos[-1] + 1.5)
        # these limit is from test
        ax.invert_xaxis()
        ax.invert_yaxis()

        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.yaxis.tick_right()
        ax.set_yticklabels([], minor=False, fontsize=fontsize)

        ax.set_xlabel('day', fontsize=fontsize)
        ax.set_ylabel(name_y, fontsize=fontsize)

        # end of plotting:
        # title = f'{name_x:s} vs {name_y:s}'
        # if suptitle_add_word is not None:
        #     title = title + ' ' + suptitle_add_word
        #
        # fig.suptitle(title)
        #
        # plt.savefig(output_plot, dpi=300)

        plt.savefig(f'./plot/{output_figure:s}', dpi=300)
        plt.show()

        print(f'job done')

    return output


def contingency_2df_table(class_x: pd.DataFrame, class_y: pd.DataFrame, plot: bool = False,
                          output_figure: str = 'contingency.png'):
    # the input DataFrames may have different index, so merge two classes with DataTimeIndex:
    class_df = class_y.merge(class_x, left_index=True, right_index=True)

    # y direction:
    name_y = class_df.columns[0]
    # x direction:
    name_x = class_df.columns[1]

    class_name_y = list(set(class_df.iloc[:, 0]))
    class_name_x = list(set(class_df.iloc[:, 1]))

    # get cross matrix
    cross = np.zeros((len(class_name_y), len(class_name_x)))
    for i in range(len(class_name_y)):
        class_one = class_df.loc[class_df[name_y] == class_name_y[i]]
        for j in range(len(class_name_x)):
            class_cross = class_one.loc[class_one[name_x] == class_name_x[j]]
            cross[i, j] = len(class_cross)
            print(f'x = {j + 1:g}, y = {i + 1:g}, cross_size = {cross[i, j]:g}')

    cross_df = pd.DataFrame(data=cross, index=class_name_y, columns=class_name_x).astype(int)

    sig, expected = value_sig_neu_test_2d(contingency=cross_df.values, alpha=0.05, output_expected=True)

    # output = {'sig': sig[::-1], 'expected': expected[::-1], 'observed': cross[::-1]}
    output = {'sig': sig, 'expected': expected, 'observed': cross}

    if plot:

        fontsize = 14
        # ----------------------------- plot -----------------------------
        fig = plt.figure(figsize=(12.5, 8), dpi=300)
        widths = [1, 3]
        heights = [1, 2]
        gridspec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)
        gridspec.update(wspace=0.08, hspace=0.15)  # set the spacing between axes.

        # matrix:
        ax = fig.add_subplot(gridspec[1, 1])
        cbar_label = 'count'

        plot_number = True  # it's better to plot number with occurrence
        cbar_label = 'observed - expected (N. of day)'

        # ======== the color matrix:

        import math
        df = cross_df

        x_ticks_label = df.columns
        y_ticks_label = [np.int(x) for x in class_name_y[::-1]]

        # put the major ticks at the middle of each cell
        x_ticks = np.arange(df.shape[1]) + 0.5
        y_ticks = np.arange(df.shape[0]) + 0.5

        # inverse y_ticks, since the plot will go from bottom to up when using ax.text
        y_ticks = y_ticks[::-1]  # up to down

        ax.set_xticks(x_ticks, minor=False)
        ax.set_yticks(y_ticks[::-1], minor=False)

        ax.set_xticklabels(x_ticks_label, minor=False, fontsize=fontsize)
        ax.set_yticklabels(y_ticks_label, minor=False, fontsize=fontsize)

        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False)

        diff = cross_df.values - expected

        vmin = -30
        vmax = 30
        vmin = np.int(diff.min())
        vmax = np.int(diff.max())

        from matplotlib.colors import TwoSlopeNorm
        if vmin * vmax < 0:
            vmax = np.max([vmin, vmax])

            vmax = 12
            vmin = vmax * -1

            # to make uneven colorbar with zero in white
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        cmap = plt.cm.get_cmap('RdYlGn', vmax + 1).reversed()
        cmap = plt.cm.get_cmap('PiYG', vmax + 1).reversed()

        # cmap = plt.cm.get_cmap('coolwarm', df.max().max() + 1)
        # plot the inverse color: since pcolor plot it from bottom to up:
        cf = ax.pcolor(diff[::-1], cmap=cmap, norm=norm)

        if vmin + vmax < vmax:
            cbar_ticks = [x for x in range(vmin, vmax + 1, math.ceil((vmax - vmin) / 10))]
        else:
            cbar_ticks = [x for x in range(vmin, vmax, math.ceil((vmax - vmin) / 10))]

        print(cbar_ticks)
        if plot_number:
            number = cross
            for i in range(number.shape[1]):  # x direction
                for j in range(number.shape[0]):  # y direction
                    c = number[j, i]

                    if c >= expected[j, i]:
                        weight = 'bold'
                    if c <= expected[j, i]:
                        weight = 'normal'

                    # notice to the order of
                    if sig[j, i]:
                        style = 'italic'
                    else:
                        style = 'normal'

                    # Boldface values are overrepresented and the others values are underrepresented.Italics denote 95 % significance
                    # according to Neu’s test.

                    ax.text(x_ticks[i], y_ticks[j], f'{c:2.0f}', va='center', ha='center',
                            weight=weight, style=style, fontsize=fontsize)
            # put cbar label
            ax.yaxis.set_label_position("right")

        # add colorbar
        cb_ax = fig.add_axes([0.92, 0.11, 0.02, 0.48])
        cb = plt.colorbar(cf, orientation='vertical', shrink=0.8, pad=0.05, cax=cb_ax,
                          ax=ax, label=cbar_label, ticks=cbar_ticks)
        loc = [x for x in cbar_ticks]
        cb.set_ticks(loc)
        cb.set_ticklabels(cbar_ticks)
        cb.ax.tick_params(labelsize=fontsize)
        cb.set_label(label=cbar_label, fontsize=fontsize)

        # ======== the color matrix: end

        ax.set_xlabel('SSR class', fontsize=fontsize)

        # histogram in x direction:
        ax = fig.add_subplot(gridspec[0, 1])
        bars = class_name_x
        data = class_df[name_x]
        height = [len(data[data == x]) for x in class_name_x]

        colors = ['darkgray', 'lightskyblue', 'firebrick', 'indianred', 'darkgray',
                  'cornflowerblue', 'royalblue', 'darkgray', 'darkgray']

        y_pos = np.arange(len(bars))
        ax.bar(bars, height, align='center', color=colors)

        plt.grid(axis='y')
        ax.set_xlim(0.5, y_pos[-1] + 1.5)  # these limit is from test
        # x_ticks = np.arange(len(class_name_x)) + 0.5

        # ax.set_xticks([], minor=False)
        ax.set_xticklabels([], minor=False, fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.set_ylabel('day', fontsize=fontsize)
        ax.set_title('SSR class', fontsize=fontsize)

        # histogram in y direction:
        ax = fig.add_subplot(gridspec[1, 0])
        bars = class_name_y
        data = class_df[name_y]
        height = [len(data[data == x]) for x in class_name_y]

        colors_olr = ['darkgray', 'royalblue', 'indianred', 'darkgray', 'indianred', 'firebrick', 'darkgray']
        patterns = [" ", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
        # patterns = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
        # colors_olr = ['darkgray', 'blueviolet', 'orange', 'darkgray', 'orange', 'darkorange', 'darkgray']

        y_pos = np.arange(len(bars))
        ax.barh(bars, height, align='center', color='orange')

        plt.grid(axis='x')

        ax.set_ylim(0.5, y_pos[-1] + 1.5)
        # these limit is from test
        ax.invert_xaxis()
        ax.invert_yaxis()

        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.yaxis.tick_right()
        ax.set_yticklabels([], minor=False, fontsize=fontsize)

        ax.set_xlabel('day', fontsize=fontsize)
        ax.set_ylabel(name_y, fontsize=fontsize)

        # end of plotting:
        # title = f'{name_x:s} vs {name_y:s}'
        # if suptitle_add_word is not None:
        #     title = title + ' ' + suptitle_add_word
        #
        # fig.suptitle(title)
        #
        # plt.savefig(output_plot, dpi=300)

        plt.savefig(f'./plot/{output_figure:s}', dpi=300)
        plt.show()

        print(f'job done')

    return output


def plot_matrix_class_vs_class(class_x: pd.DataFrame,
                               class_y: pd.DataFrame,
                               output_plot: str = 'class_vs_class_matrix',
                               occurrence: bool = 1,
                               significant: bool = True,
                               suptitle_add_word: str = ""):
    """
    plot the matrix of class vs class, color bar is number of points
    class_df: DataFrame of one columns of classifications with DateTimeIndex, columns' names will be used.
    $$: if plot occurrence, impact of class_x on class_y

    :param class_y:
    :type class_y: pandas.core.frame.DataFrame
    :param class_x:
    :type class_x: pandas.core.frame.DataFrame
    :param output_plot:
    :type output_plot: str
    :param occurrence: if occurrence is True, will plot numbers in the matrix by default
    :type occurrence:
    :param suptitle_add_word:
    :type suptitle_add_word: str
    :return:
    :rtype: None
    """

    # the input DataFrames may have different index, so merge two classes with DataTimeIndex:
    class_df = class_x.merge(class_y, left_index=True, right_index=True)

    # y direction:
    name_y = class_df.columns[0]
    # x direction:
    name_x = class_df.columns[1]

    class_name_y = list(set(class_df.iloc[:, 0]))
    class_name_x = list(set(class_df.iloc[:, 1]))

    # get cross matrix
    cross = np.zeros((len(class_name_y), len(class_name_x)))
    for i in range(len(class_name_y)):
        class_one = class_df.loc[class_df[name_y] == class_name_y[i]]
        for j in range(len(class_name_x)):
            class_cross = class_one.loc[class_one[name_x] == class_name_x[j]]
            cross[i, j] = len(class_cross)
    cross_df = pd.DataFrame(data=cross, index=class_name_y, columns=class_name_x).astype(int)

    # ----------------------------- plot -----------------------------
    fig = plt.figure(figsize=(8, 6), dpi=300)
    widths = [1, 3]
    heights = [1, 2]
    gridspec = fig.add_gridspec(ncols=2, nrows=2, width_ratios=widths, height_ratios=heights)
    gridspec.update(wspace=0.1, hspace=0.2)  # set the spacing between axes.

    # matrix:
    ax = fig.add_subplot(gridspec[1, 1])
    cbar_label = 'count'
    plot_number = False

    if occurrence:
        print(f'occurrence: {name_y:s} introduced changes of {name_x:s} occurrence')
        occ = cross_df
        for i in range(len(class_name_x)):
            ssr_class = class_name_x[i]
            avg_freq = len(class_df[class_df[name_x] == ssr_class]) / len(class_df)
            print(len(class_df[class_df[name_x] == ssr_class]), len(class_df), avg_freq)

            for j in range(len(class_name_y)):
                large_class = class_name_y[j]
                freq = cross[j, i] / len(class_df[class_df[name_y] == large_class])
                occ.iloc[j, i] = (freq - avg_freq) * 100
                print(i, j, cross_df.iloc[j, i], len(class_df[class_df[name_y] == large_class]), freq)

        plot_number = True  # it's better to plot number with occurrence

        cross_df = occ

        cbar_label = 'occurrence (%)'

    plot_color_matrix(df=cross_df, ax=ax, cbar_label=cbar_label, plot_number=plot_number)

    ax.set_xlabel(name_x)

    # histogram in x direction:
    ax = fig.add_subplot(gridspec[0, 1])
    bars = class_name_x
    data = class_df[name_x]
    height = [len(data[data == x]) for x in class_name_x]

    y_pos = np.arange(len(bars))
    ax.bar(bars, height, align='center', color='red')

    ax.set_xlim(0.5, y_pos[-1] + 1.5)  # these limit is from test
    # x_ticks = np.arange(len(class_name_x)) + 0.5

    ax.set_xticks([], minor=False)

    ax.set_xticklabels([], minor=False)
    ax.set_ylabel('n_day')
    # ax.set_xlabel(name_x)

    # histogram in y direction:
    ax = fig.add_subplot(gridspec[1, 0])
    bars = class_name_y
    data = class_df[name_y]
    height = [len(data[data == x]) for x in class_name_y]

    y_pos = np.arange(len(bars))
    ax.barh(bars, height, align='center', color='orange')

    ax.set_ylim(0.5, y_pos[-1] + 1.5)
    # these limit is from test
    ax.invert_xaxis()
    ax.set_xlabel('n_day')
    ax.set_ylabel(name_y)

    # end of plotting:
    title = f'{name_x:s} vs {name_y:s}'
    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    fig.suptitle(title)

    plt.savefig(output_plot, dpi=300)

    plt.show()

    print(f'job done')


def plot_matrix_classification_at_year_and_month(class_df: pd.DataFrame, output_plot: str):
    """
    calculate classification at different month
    class_df: DataFrame of one class with DateTimeIndex
    """

    # get info from input:
    n_class = int(class_df.max())

    year_start = class_df.index.year.min()
    year_end = class_df.index.year.max()
    n_year = year_end - year_start + 1

    month_list = list(set(class_df.index.month))

    for i in range(n_class):
        class_1 = class_df.loc[class_df.values == i + 1]
        cross = np.zeros((n_year, len(month_list)))

        for y in range(n_year):
            for im in range(len(month_list)):
                cross[y, im] = class_1.loc[
                    (class_1.index.year == y + year_start) &
                    (class_1.index.month == month_list[im])].__len__()

        print(f'# ----------------- {n_class:g} -> {i + 1:g} -----------------------------')

        df = pd.DataFrame(data=cross, index=range(year_start, year_start + n_year), columns=month_list)
        df = df.astype(int)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6), facecolor='w', edgecolor='k', dpi=300)

        plot_color_matrix(df=df, ax=ax, cbar_label='count')

        plt.suptitle(f'C{n_class:g} -> C{i + 1:g}')
        plt.savefig(f'./plot/{output_plot:s}', dpi=220)
        plt.show()

    print(f'job done')


def plot_12months_geo_map_significant(da: xr.DataArray, area: str, sig_dim: str, only_sig_point: bool):
    fig, axs = plt.subplots(nrows=4, ncols=3, sharex='row', sharey='col',
                            figsize=(14, 12), dpi=220, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.85, bottom=0.12, top=0.9, wspace=0.1, hspace=0.1)
    axs = axs.ravel()

    print(sig_dim)

    for imon in range(12):
        print(f'plot month = {imon + 1:g}')
        ax = set_active_axis(axs=axs, n=imon)

        # get data:
        month_ly = da.sel(month=imon + 1)

        if only_sig_point:
            sig_map: xr.DataArray = value_significant_of_anomaly_2d_mask(field_3d=month_ly, conf_level=0.05)
            print(f'only work for anomaly/change/trend/ !!! compare to ZERO!!')
            month_to_plot = filter_2d_by_mask(month_ly, mask=sig_map).mean(axis=0)
        else:
            month_to_plot = month_ly.mean(axis=0)

        # ----------------------------- plot -----------------------------
        set_basemap(area=area, ax=ax)

        lon, lat = np.meshgrid(month_to_plot.longitude, month_to_plot.latitude)

        vmax, vmin = value_cbar_max_min_of_da(month_to_plot)

        # TODO:
        # vmax = 20
        # vmin = -20
        level_anomaly = np.arange(vmin, vmax + 1, 2)

        cf1 = plt.contourf(lon, lat, month_to_plot, level_anomaly, cmap='PuOr_r', vmax=vmax, vmin=vmin)

        # ----------------------------- end of plot -----------------------------

        ax.text(0.9, 0.95, f'{calendar.month_abbr[imon + 1]:s}', fontsize=12,
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        ax.text(0.9, 0.1, f'{da.name:s}', fontsize=12,
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        cbar_label = f'({da.assign_attrs().units:s})'
        plt.colorbar(cf1, ticks=np.ndarray.tolist(level_anomaly), label=cbar_label, ax=ax)

    return fig, axs


def select_day_time_df_da(hour1, hour2, da=None, df=None):
    if da is not None:
        return da.where(np.logical_and(da.time.dt.hour >= hour1, da.time.dt.hour <= hour2), drop=True)

    if df is not None:
        return df[df.index.hour.isin(range(hour1, hour2 + 1))]


def select_area_from_str(da: xr.DataArray, area: str):
    lonlat = value_lonlatbox_from_area(area)
    da1 = da.where(np.logical_and(da.lon >= lonlat[0], da.lon < lonlat[1]), drop=True)
    da2 = da1.where(np.logical_and(da1.lat >= lonlat[2], da1.lat < lonlat[3]), drop=True)

    return da2


def get_month_name_from_num(num):
    import calendar
    # print('Month Number:', num)

    # get month name
    # print('Month full name is:', calendar.month_name[num])
    # print('Month short name is:', calendar.month_abbr[num])

    return calendar.month_abbr[num]


def plot_mjo_monthly_distribution(mjo: pd.DataFrame,
                                  instense: bool = False):
    """
    plot histogram of mjo
    :return:
    """

    mjo = mjo.assign(Month=lambda df: df.index.month)

    # ----------------------------- filters -----------------------------
    if instense:
        mjo = mjo[mjo.amplitude > 1]

    # no big difference when applying these filters,
    # seems weak MJO is random.
    # ----------------------------- filters -----------------------------

    count: pd.Series = mjo.groupby(['phase', 'Month']).phase.count()
    # coding: counting one column

    df = pd.DataFrame({'count': count}).reset_index()
    fig = plt.figure(figsize=(12, 6), dpi=300)
    ax = fig.add_subplot(111)

    sns.set(style="whitegrid")
    ax = sns.barplot(x='Month', y="count", hue=f'phase', data=df)

    title1 = f'mjo monthly distribution big_amplitude={instense:g}'
    plt.suptitle(title1)
    plt.savefig(f'./plot/barplot.{title1.replace(" ", "_"):s}.'
                f'.png', dpi=300)
    plt.show()

    # plot every phase:
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 14), dpi=220)
    axs = axs.flatten()

    for phase in range(8):
        phase_1: pd.Series = count.loc[phase + 1]
        # multi-index selection
        df_1 = pd.DataFrame({f'phase_{phase + 1:g}': phase_1})
        ax_1 = df_1.plot(kind='bar', ax=axs[phase], rot=0)
        # coding: direction of x label in barplot

    plt.suptitle(title1)
    plt.savefig(f'./plot/month_matrix.{title1.replace(" ", "_"):s}.'
                f'.png', dpi=300)
    plt.show()
    print(f'job done')


def plot_annual_diurnal_cycle_columns_in_df(df: pd.DataFrame, columns=None,
                                            title=' ',
                                            linestyles=None,
                                            output_tag='',
                                            tag_on_plot='',
                                            count_bar_plot=True,
                                            colors=None, markers=None,
                                            ylabel='',
                                            with_marker=True,
                                            plot_errorbar=False,
                                            vmin=None, vmax=None):
    """
    applied project LW_XGBoost_Cloud, Final figure
    :param df:
    :param columns:
    :return:
    """

    # note: to convert df to multiple columns based on the values in one column.
    # use this line:
    # mf_pivot = mf.pivot(columns='NOM', values='FF')
    # ----------------------------- set parameters -----------------------------

    if columns is not None:
        print(f'user specified columns')
    else:
        columns = df.columns

    print(f'columns used: \t')
    print(columns)

    # ----------------------------------- color and markers ------------------------------
    if colors is None:
        colors = ['lightgrey', 'gray', 'lightcoral', 'firebrick', 'red', 'darkorange', 'gold', 'yellowgreen',
                  'green', 'cyan', 'deepskyblue', 'blue', 'darkviolet', 'magenta', 'pink'] * 4

        if len(columns) < 5:
            colors = ['blue', 'red', 'green', 'black']

    if markers is None:
        markers = ['o', 'v', '^', '<', '1', 's', 'p', 'P', '*', '+', 'x', 'd', 'D'] * 4
        if len(columns) < 5:
            markers = ['o', 'v', 's', '*', '+', 'x', 'd', 'D']
    # ----------------------------- set fig -----------------------------
    alpha_bar_plot = 0.7
    fontsize = 12

    nrows = 2 + 2 * (int(count_bar_plot))

    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(8, 3 * nrows),
                             facecolor='w', edgecolor='k', dpi=300)  # figsize=(w,h)
    fig.subplots_adjust(left=0.15, right=0.9, bottom=0.1, top=0.95, wspace=0.05, hspace=0.3)

    axes = axes.flatten()

    # ----------------------------- plotting -----------------------------
    # diurnal cycle:
    num_plot = 0
    ax = set_active_axis(axs=axes, n=num_plot)
    for i in range(len(columns)):
        column = columns[i]

        df1 = df[{column}]

        # if vmax is not None:
        #     df1 = df1[df1 <= vmax]
        #
        # if vmin is not None:
        #     df1 = df1[df1 >= vmin]

        df0 = drop_nan_infinity(df1)

        mean = df0.groupby(df0.index.hour).mean()[{column}]
        std = df0.groupby(df0.index.hour).std()[{column}]

        x = mean.index.values
        y = mean[column].values
        y_err = std[column].values

        if plot_errorbar:
            capsize = 5
        else:
            capsize = 0

        if linestyles is not None:
            linestyle = linestyles[i]
        else:
            linestyle = '-'

        if with_marker:
            plt.errorbar(x, y, yerr=y_err, marker=markers[i], color=colors[i],
                         capsize=capsize, capthick=1,  # error bar format.
                         label=f'{column:s}', linestyle=linestyle)
        else:
            plt.errorbar(x, y, yerr=y_err, color=colors[i],
                         capsize=capsize, capthick=1,  # error bar format.
                         label=f'{column:s}', linestyle=linestyle)

        print(column, x, y)

        x_stick_label = [f'{i:g}' for i in x]
        plt.xticks(ticks=x, labels=x_stick_label, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # plt.legend(loc='upper right', fontsize=8)
        plt.legend(fontsize=fontsize)
        plt.xlabel(f'Hour', fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)

        # plot tag_on_plot:
        ax.text(0.1, 0.95, f'{tag_on_plot:s}', fontsize=14,
                horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

        if len(columns) == 1:
            plt.ylim(vmin, vmax)

    if count_bar_plot:
        num_plot += 1
        # count bar plot hourly
        ax = set_active_axis(axs=axes, n=num_plot)
        df1 = df[columns]
        count = df1.groupby(df.index.hour).count()
        count.plot(kind='bar', color=colors, ax=ax, grid=True, alpha=alpha_bar_plot, fontsize=fontsize)
        plt.xticks(ticks=[xx - np.min(x) for xx in x], labels=x_stick_label, rotation=0, fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.ylabel('data count', fontsize=fontsize)
        plt.xlabel('Hour', fontsize=fontsize)

    # change color if only one variable to plot
    if len(columns) == 1:
        colors = ['red']
    # annual cycle:
    num_plot += 1
    ax = set_active_axis(axs=axes, n=num_plot)
    for i in range(len(columns)):
        column = columns[i]
        df1 = df[{column}]
        df0 = drop_nan_infinity(df1)

        mean = df0.groupby(df0.index.month).mean()[{column}]
        std = df0.groupby(df0.index.month).std()[{column}]

        x = mean.index.values
        y = mean[column].values
        y_err = std[column].values

        if plot_errorbar:
            capsize = 5
        else:
            capsize = 0

        if linestyles is not None:
            linestyle = linestyles[i]
        else:
            linestyle = '-'

        if with_marker:
            plt.errorbar(x, y, yerr=y_err, marker=markers[i], color=colors[i],
                         capsize=capsize, capthick=1,  # error bar format.
                         label=f'{column:s}', linestyle=linestyle)
        else:
            plt.errorbar(x, y, yerr=y_err, color=colors[i],
                         capsize=capsize, capthick=1,  # error bar format.
                         label=f'{column:s}', linestyle=linestyle)

        print(column, x, y)

        if len(columns) == 1:
            plt.ylim(vmin, vmax)

        x_stick_label = [get_month_name_from_num(i) for i in x]
        plt.xticks(ticks=x, labels=x_stick_label, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # plt.legend(loc='upper right', fontsize=8)
        plt.legend(fontsize=fontsize)
        plt.xlabel(f'Month', fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)

        # plot tag_on_plot:
        ax.text(0.1, 0.95, f'{tag_on_plot:s}', fontsize=14,
                horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

    if count_bar_plot:
        num_plot += 1
        # count bar plot monthly
        ax = set_active_axis(axs=axes, n=num_plot)
        df1 = df[columns]
        count = df1.groupby(df.index.month).count()
        count.plot(kind='bar', color=colors, ax=ax, grid=True, alpha=alpha_bar_plot, fontsize=fontsize)
        x_stick_label = [get_month_name_from_num(i) for i in x]
        plt.xticks(ticks=range(0, 12), labels=x_stick_label, rotation=0, fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.ylabel('data count', fontsize=fontsize)
        plt.xlabel('Month', fontsize=fontsize)

    plt.savefig(f'./plot/annual_diurnal_cycle_with_countbar_{int(count_bar_plot):g}.{output_tag:s}.png', dpi=300)
    plt.show()
    print(f'got the plot')


def plot_diurnal_cycle_columns_in_df(df: pd.DataFrame, columns=None,
                                     title=' ', linestyles=None,
                                     output_tag='',
                                     ylabel='', with_marker=True,
                                     plot_errorbar=False,
                                     vmin=None, vmax=None):
    """
    applied project Sky_clearness_2023:
    :param months:
    :param suptitle:
    :param df:
    :param columns:
    :return:
    """

    if columns is not None:
        print(f'user specified columns')
    else:
        columns = df.columns

    print(f'columns used: \t')
    print(columns)

    # ----------------------------- set parameters -----------------------------
    # months = [11, 12, 1, 2, 3, 4]
    colors = ['lightgrey', 'gray', 'lightcoral', 'firebrick', 'red', 'darkorange', 'gold', 'yellowgreen',
              'green', 'cyan', 'deepskyblue', 'blue', 'darkviolet', 'magenta', 'pink']
    markers = ['o', 'v', '^', '<', '1', 's', 'p', 'P', '*', '+', 'x', 'd', 'D']

    if len(columns) < 5:
        colors = ['red', 'blue', 'green', 'black']
        markers = ['o', 'v', 's', '*', '+', 'x', 'd', 'D']

    fontsize = 12
    # ----------------------------- set fig -----------------------------
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6),
                           facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    # ----------------------------- plotting -----------------------------
    for i in range(len(columns)):
        column = columns[i]
        print(column)

        df1 = df[{column}]
        # when compare PDF in a value range:
        # such as Sky_Clearness 2023: for CI
        if vmax is not None:
            df1 = df1[df1 <= vmax]

        if vmin is not None:
            df1 = df1[df1 >= vmin]

        df0 = drop_nan_infinity(df1)

        mean = df0.groupby(df0.index.hour).mean()[{column}]
        std = df0.groupby(df0.index.hour).std()[{column}]

        x = mean.index.values
        y = mean[column].values
        y_err = std[column].values

        if plot_errorbar:
            capsize = 5
        else:
            capsize = 0

        if linestyles is not None:
            linestyle = linestyles[i]
        else:
            linestyle = '-'

        if with_marker:
            if plot_errorbar:
                plt.errorbar(x, y, yerr=y_err, marker=markers[i], color=colors[i],
                             capsize=capsize, capthick=1,  # error bar format.
                             label=f'{column:s}', linestyle=linestyle)
            else:
                # without errorbar:
                plt.plot(x, y, marker=markers[i], color=colors[i], label=f'{column:s}', linestyle=linestyle)
        else:
            plt.errorbar(x, y, yerr=y_err, color=colors[i],
                         capsize=capsize, capthick=1,  # error bar format.
                         label=f'{column:s}', linestyle=linestyle)

        plt.ylim(vmin, vmax)
        # plt.ylim(0., 0.8)
        # plt.ylim(400, 1000)
        # plt.ylim(200, 600)

        x_stick_label = [f'{i:g}' for i in x]

        plt.xticks(ticks=x, labels=x_stick_label)

        # plt.legend(loc='upper right', fontsize=8)
        plt.legend(fontsize=fontsize)
        plt.xlabel(f'Hour (local time)', fontsize=fontsize)
        plt.ylabel(ylabel, fontsize=fontsize)

    if plot_errorbar:
        ax.grid(axis='y')

    plt.title(title + ' (errorbar = std)', fontsize=fontsize)
    plt.savefig(f'./plot/diurnal_cycle_single_fig.{output_tag:s}.png', dpi=300)
    plt.show()
    print(f'got the plot')


def plot_annual_cycle_columns_in_df(df: pd.DataFrame, columns=None,
                                    title=' ', linestyles=None,
                                    output_tag='',
                                    ylabel='', with_marker=True,
                                    plot_errorbar=False,
                                    colors=None, markers=None,
                                    vmin=None, vmax=None):
    """
    applied project Sky_clearness_2023:
    :param df:
    :param columns:
    :return:
    """

    if columns is not None:
        print(f'user specified columns')
    else:
        columns = df.columns

    print(f'columns used: ')
    print(columns)

    # ----------------------------- set parameters -----------------------------
    # months = [11, 12, 1, 2, 3, 4]
    if colors is None:
        colors = ['lightgrey', 'gray', 'lightcoral', 'firebrick', 'red', 'darkorange', 'gold', 'yellowgreen',
                  'green', 'cyan', 'deepskyblue', 'blue', 'darkviolet', 'magenta', 'pink']

        if len(columns) < 5:
            colors = ['red', 'blue', 'green', 'black']

    if markers is None:
        markers = ['o', 'v', '^', '<', '1', 's', 'p', 'P', '*', '+', 'x', 'd', 'D']
        if len(columns) < 5:
            markers = ['o', 'v', 's', '*', '+', 'x', 'd', 'D']

    # ----------------------------- set fig -----------------------------
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6),
                           facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    # ----------------------------- plotting -----------------------------
    for i in range(len(columns)):
        column = columns[i]
        print(column)

        df1 = df[{column}]
        # when compare PDF in a value range:
        # such as Sky_Clearness 2023: for CI
        if vmax is not None:
            df1 = df1[df1 <= vmax]

        if vmin is not None:
            df1 = df1[df1 >= vmin]

        df0 = drop_nan_infinity(df1)

        mean = df0.groupby(df0.index.month).mean()[{column}]
        std = df0.groupby(df0.index.month).std()[{column}]

        x = mean.index.values
        y = mean[column].values
        y_err = std[column].values

        if plot_errorbar:
            capsize = 5
        else:
            capsize = 0

        if linestyles is not None:
            linestyle = linestyles[i]
        else:
            linestyle = '-'

        if with_marker:
            plt.errorbar(x, y, yerr=y_err, marker=markers[i], color=colors[i],
                         capsize=capsize, capthick=1,  # error bar format.
                         label=f'{column:s}', linestyle=linestyle)
        else:
            plt.errorbar(x, y, yerr=y_err, color=colors[i],
                         capsize=capsize, capthick=1,  # error bar format.
                         label=f'{column:s}', linestyle=linestyle)

        plt.ylim(vmin, vmax)
        # plt.ylim(0.3, 0.8)
        # plt.ylim(400, 1000)
        # plt.ylim(200, 600)

        x_stick_label = [get_month_name_from_num(i) for i in x]

        plt.xticks(ticks=x, labels=x_stick_label)

        # plt.legend(loc='upper right', fontsize=8)
        plt.legend(fontsize=8)
        plt.xlabel(f'Month')
        plt.ylabel(ylabel)

    plt.title(title + ' (errorbar = std)')
    plt.savefig(f'./plot/monthly_cycle_single_fig.{output_tag:s}.png', dpi=300)
    plt.show()
    print(f'got the plot')


def plot_monthly_diurnal_single_fig_df(df: pd.DataFrame, column=None, suptitle=' ',
                                       months=None, output_tag='',
                                       ylabel='', with_marker=False,
                                       plot_errorbar=False,
                                       vmin=None, vmax=None):
    """
    plot hourly curves by /month/ for the columns in list
    :param months:
    :param suptitle:
    :param df:
    :return:
    """

    if months is None:  # 👍
        months = [12, 1, 2, ]

    # ----------------------------- set parameters -----------------------------
    # months = [11, 12, 1, 2, 3, 4]
    colors = ['lightgrey', 'gray', 'lightcoral', 'firebrick', 'red', 'darkorange', 'gold', 'yellowgreen',
              'green', 'cyan', 'deepskyblue', 'blue', 'darkviolet', 'magenta', 'pink']
    markers = ['o', 'v', '^', '<', '1', 's', 'p', 'P', '*', '+', 'x', 'd', 'D']

    # ----------------------------- set fig -----------------------------
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6),
                           facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    # ----------------------------- plotting -----------------------------
    for mon in months:
        data_slice = df[df.index.month == mon]

        mean = data_slice.groupby(data_slice.index.hour).mean()[{column}]
        std = data_slice.groupby(data_slice.index.hour).std()[{column}]

        plt.plot(mean.index, mean)

        x = mean.index.values
        y = mean[column].values
        y_err = std[column].values

        if plot_errorbar:
            capsize = 5
        else:
            capsize = 0

        if with_marker:
            plt.errorbar(x, y, yerr=y_err, marker=markers[mon], color=colors[mon],
                         capsize=capsize, capthick=1,  # error bar format.
                         label=f'{calendar.month_abbr[mon]:s}')
        else:
            plt.errorbar(x, y, yerr=y_err, color=colors[mon],
                         capsize=capsize, capthick=1,  # error bar format.
                         label=f'{calendar.month_abbr[mon]:s}')

        plt.legend(loc='upper right', fontsize=8)
        plt.xlabel(f'Hour')
        plt.ylabel(ylabel)

    plt.suptitle(suptitle + ' (errorbar = std)')
    plt.savefig(f'./plot/monthly_diurnal_cycle.{output_tag:s}.png', dpi=300)
    plt.show()
    print(f'got the plot')


def plot_mjo_phase(mjo_phase: pd.DataFrame, olr: xr.DataArray, high_amplitude: bool, month: str,
                   only_significant_points: int = 0):
    """
    plot mjo phase by olr
    :param only_significant_points:
    :param month:
    :param mjo_phase:
    :param high_amplitude: if plot high amplitude > 1
    :param olr: has to input olr, anomaly and mean both needed.
    :return:
    """
    # ----------------------------- prepare the data -----------------------------
    olr_daily_anomaly = anomaly_daily(olr)
    anomaly_in_class = get_data_in_classif(da=olr_daily_anomaly, df=mjo_phase[{'phase'}],
                                           time_mean=True, significant=only_significant_points)
    # for contour:
    olr_in_class = get_data_in_classif(da=olr, df=mjo_phase[{'phase'}],
                                       time_mean=True, significant=0)
    # ----------------------------- some predefined values of cbar limits -----------------------------

    fig, axs = plt.subplots(nrows=4, ncols=2, sharex='row', sharey='col',
                            figsize=(8, 10), dpi=220, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    # axs = axs.flatten()
    axs = axs.ravel()

    for phase in [1, 2, 3, 4, 5, 6, 7, 8]:
        print(f'plot class = {phase:g}')
        # ----------------------------- calculate mean in each phase -----------------------------
        anomaly_mean = anomaly_in_class.where(anomaly_in_class['class'] == phase, drop=True).squeeze()
        olr_mean = olr_in_class.where(olr_in_class['class'] == phase, drop=True).squeeze()
        # nan comes from missing data and from non-significance

        ax = set_active_axis(axs=axs, n=phase - 1)
        set_basemap(area='swio', ax=ax)
        # set_basemap(area='SA_swio', ax=ax)

        # ----------------------------- start to plot -----------------------------
        plt.title('#' + str(phase), pad=3, fontsize=18)

        lon, lat = np.meshgrid(anomaly_mean.lon, anomaly_mean.lat)
        level_anomaly = np.arange(-30, 31, 5)
        cf1 = plt.contourf(lon, lat, anomaly_mean, level_anomaly, cmap='PuOr_r', vmax=30, vmin=-30, extend='both')
        level_olr = np.arange(140, 280, 20)
        cf2 = plt.contour(lon, lat, olr_mean, level_olr, cmap='magma_r', vmax=280, vmin=140)
        ax.clabel(cf2, level_olr, inline=True, fmt='%2d', fontsize='xx-small')
        ax.coastlines()

        # cf = plt.pcolormesh(lon, lat, consistency_percentage_map,
        #                     cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

        # ----------------------------- end of plot -----------------------------

        ax.text(0.9, 0.95, f'{month:s}', fontsize=18,
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        ax.text(0.95, 0.02, f'{olr.name:s}', fontsize=18,
                horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

    cbar_label = f'OLR ({olr.assign_attrs().units:s})'
    cb = plt.colorbar(cf1, ticks=np.ndarray.tolist(level_anomaly), label=cbar_label, ax=axs, extend='max')
    cb.ax.tick_params(labelsize=18)
    cb.set_label(label=cbar_label, fontsize=18)

    title = f'MJO phase in {month:s}'
    # plt.suptitle(title)

    # tag: specify the location of the cbar
    # cb_ax = fig.add_axes([0.13, 0.1, 0.7, 0.015])
    # cb = plt.colorbar(cf1, orientation='horizontal', shrink=0.7, pad=0.05, label=cbar_label, cax=cb_ax)

    plt.savefig(f'./plot/mjo_phases_sig_{only_significant_points:g}.'
                f'{title.replace(" ", "_"):s}.'
                f'high_amplitude_{high_amplitude:g}'
                f'.png', dpi=300)

    plt.show()
    print(f'got plot')


def plot_hourly_curve_by_month(df: pd.DataFrame, columns: list, suptitle=' ', months=None,
                               vmin=None, vmax=None):
    """
    plot hourly curves by /month/ for the columns in list
    :param months:
    :param suptitle:
    :param df:
    :param columns:
    :return:
    """

    if months is None:  # 👍
        months = [11, 12, 1, 2, 3, 4]

    # ----------------------------- set parameters -----------------------------
    # months = [11, 12, 1, 2, 3, 4]
    colors = ['black', 'green', 'orange', 'red']
    data_sources = columns

    # ----------------------------- set fig -----------------------------
    nrows = len(months)

    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 15),
                            facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.9, wspace=0.05)

    axs = axs.ravel()
    # ----------------------------- plotting -----------------------------

    for v in range(len(columns)):
        for i in range(nrows):
            plt.sca(axs[i])  # active this subplot

            month = months[i]
            data_slice = df[df.index.month == month]
            x = range(len(data_slice))

            label = data_sources[v]

            plt.plot(x, data_slice[[columns[v]]], color=colors[v], label=label)

            print(f'month = {months[i]:g}, var = {columns[v]:s}')

            # ----------------------------- format of fig -----------------------------
            # if input data is hourly, reform the fig axis
            nday = len(set(data_slice.index.day))

            # if len(data_slice) > 31:
            #     custom_ticks = range(11, len(data_slice), 24)
            #     custom_ticks_labels = range(1, nday + 1)
            # else:
            #     custom_ticks = x
            #     custom_ticks_labels = [y + 1 for y in custom_ticks]
            #
            # axs[i].set_xticks(custom_ticks)
            # axs[i].set_xticklabels(custom_ticks_labels)

            axs[i].set_xlim(0, len(data_slice) * 1.2)

            if vmin is not None:
                axs[i].set_ylim(vmin, vmax)

            # axs[i].xaxis.set_ticks_position('top')
            # axs[i].xaxis.set_ticks_position('bottom')

            plt.legend(loc='upper right', fontsize=8)
            plt.xlabel(f'num of data point')
            plt.ylabel(r'$SSR\ (W/m^2)$')
            axs[i].text(0.5, 0.95, data_slice.index[0].month_name(), fontsize=20,
                        horizontalalignment='right', verticalalignment='top', transform=axs[i].transAxes)

    plt.suptitle(suptitle)
    plt.savefig(f'./plot/hourly_curve_by_month.{columns[v]:s}.png', dpi=300)
    plt.show()
    print(f'got the plot')


def get_T_value(conf_level: float = 0.05, dof: int = 10):
    """
    get value of T
    two tail = 0.95:
    :param conf_level:
    :param dof:
    :return:
    """

    print(conf_level)

    T_value = [
        12.71, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
        2.040, 2.037, 2.035, 2.032, 2.030, 2.028, 2.026, 2.024, 2.023, 2.021,
        2.020, 2.018, 2.017, 2.015, 2.014, 2.013, 2.012, 2.011, 2.010, 2.009,
        2.008, 2.007, 2.006, 2.005, 2.004, 2.003, 2.002, 2.002, 2.001, 2.000,
        2.000, 1.999, 1.998, 1.998, 1.997, 1.997, 1.996, 1.995, 1.995, 1.994,
        1.994, 1.993, 1.993, 1.993, 1.992, 1.992, 1.991, 1.991, 1.990, 1.990,
        1.990, 1.989, 1.989, 1.989, 1.988, 1.988, 1.988, 1.987, 1.987, 1.987,
        1.986, 1.986, 1.986, 1.986, 1.985, 1.985, 1.985, 1.984, 1.984, 1.984]

    # infinity:
    if dof > 100:
        return 1.960
    else:
        return T_value[dof - 1]


# ===================================================
# one tail t test table:
# dof       0.90    0.95   0.975    0.99   0.995   0.999
# 1.       3.078   6.314  12.706  31.821  63.657 318.313
# 2.       1.886   2.920   4.303   6.965   9.925  22.327
# 3.       1.638   2.353   3.182   4.541   5.841  10.215
# 4.       1.533   2.132   2.776   3.747   4.604   7.173
# 5.       1.476   2.015   2.571   3.365   4.032   5.893
# 6.       1.440   1.943   2.447   3.143   3.707   5.208
# 7.       1.415   1.895   2.365   2.998   3.499   4.782
# 8.       1.397   1.860   2.306   2.896   3.355   4.499
# 9.       1.383   1.833   2.262   2.821   3.250   4.296
# 10.       1.372   1.812   2.228   2.764   3.169   4.143
# 11.       1.363   1.796   2.201   2.718   3.106   4.024
# 12.       1.356   1.782   2.179   2.681   3.055   3.929
# 13.       1.350   1.771   2.160   2.650   3.012   3.852
# 14.       1.345   1.761   2.145   2.624   2.977   3.787
# 15.       1.341   1.753   2.131   2.602   2.947   3.733
# 16.       1.337   1.746   2.120   2.583   2.921   3.686
# 17.       1.333   1.740   2.110   2.567   2.898   3.646
# 18.       1.330   1.734   2.101   2.552   2.878   3.610
# 19.       1.328   1.729   2.093   2.539   2.861   3.579
# 20.       1.325   1.725   2.086   2.528   2.845   3.552
# 21.       1.323   1.721   2.080   2.518   2.831   3.527
# 22.       1.321   1.717   2.074   2.508   2.819   3.505
# 23.       1.319   1.714   2.069   2.500   2.807   3.485
# 24.       1.318   1.711   2.064   2.492   2.797   3.467
# 25.       1.316   1.708   2.060   2.485   2.787   3.450
# 26.       1.315   1.706   2.056   2.479   2.779   3.435
# 27.       1.314   1.703   2.052   2.473   2.771   3.421
# 28.       1.313   1.701   2.048   2.467   2.763   3.408
# 29.       1.311   1.699   2.045   2.462   2.756   3.396
# 30.       1.310   1.697   2.042   2.457   2.750   3.385
# 31.       1.309   1.696   2.040   2.453   2.744   3.375
# 32.       1.309   1.694   2.037   2.449   2.738   3.365
# 33.       1.308   1.692   2.035   2.445   2.733   3.356
# 34.       1.307   1.691   2.032   2.441   2.728   3.348
# 35.       1.306   1.690   2.030   2.438   2.724   3.340
# 36.       1.306   1.688   2.028   2.434   2.719   3.333
# 37.       1.305   1.687   2.026   2.431   2.715   3.326
# 38.       1.304   1.686   2.024   2.429   2.712   3.319
# 39.       1.304   1.685   2.023   2.426   2.708   3.313
# 40.       1.303   1.684   2.021   2.423   2.704   3.307
# 41.       1.303   1.683   2.020   2.421   2.701   3.301
# 42.       1.302   1.682   2.018   2.418   2.698   3.296
# 43.       1.302   1.681   2.017   2.416   2.695   3.291
# 44.       1.301   1.680   2.015   2.414   2.692   3.286
# 45.       1.301   1.679   2.014   2.412   2.690   3.281
# 46.       1.300   1.679   2.013   2.410   2.687   3.277
# 47.       1.300   1.678   2.012   2.408   2.685   3.273
# 48.       1.299   1.677   2.011   2.407   2.682   3.269
# 49.       1.299   1.677   2.010   2.405   2.680   3.265
# 50.       1.299   1.676   2.009   2.403   2.678   3.261
# 51.       1.298   1.675   2.008   2.402   2.676   3.258
# 52.       1.298   1.675   2.007   2.400   2.674   3.255
# 53.       1.298   1.674   2.006   2.399   2.672   3.251
# 54.       1.297   1.674   2.005   2.397   2.670   3.248
# 55.       1.297   1.673   2.004   2.396   2.668   3.245
# 56.       1.297   1.673   2.003   2.395   2.667   3.242
# 57.       1.297   1.672   2.002   2.394   2.665   3.239
# 58.       1.296   1.672   2.002   2.392   2.663   3.237
# 59.       1.296   1.671   2.001   2.391   2.662   3.234
# 60.       1.296   1.671   2.000   2.390   2.660   3.232
# 61.       1.296   1.670   2.000   2.389   2.659   3.229
# 62.       1.295   1.670   1.999   2.388   2.657   3.227
# 63.       1.295   1.669   1.998   2.387   2.656   3.225
# 64.       1.295   1.669   1.998   2.386   2.655   3.223
# 65.       1.295   1.669   1.997   2.385   2.654   3.220
# 66.       1.295   1.668   1.997   2.384   2.652   3.218
# 67.       1.294   1.668   1.996   2.383   2.651   3.216
# 68.       1.294   1.668   1.995   2.382   2.650   3.214
# 69.       1.294   1.667   1.995   2.382   2.649   3.213
# 70.       1.294   1.667   1.994   2.381   2.648   3.211
# 71.       1.294   1.667   1.994   2.380   2.647   3.209
# 72.       1.293   1.666   1.993   2.379   2.646   3.207
# 73.       1.293   1.666   1.993   2.379   2.645   3.206
# 74.       1.293   1.666   1.993   2.378   2.644   3.204
# 75.       1.293   1.665   1.992   2.377   2.643   3.202
# 76.       1.293   1.665   1.992   2.376   2.642   3.201
# 77.       1.293   1.665   1.991   2.376   2.641   3.199
# 78.       1.292   1.665   1.991   2.375   2.640   3.198
# 79.       1.292   1.664   1.990   2.374   2.640   3.197
# 80.       1.292   1.664   1.990   2.374   2.639   3.195
# 81.       1.292   1.664   1.990   2.373   2.638   3.194
# 82.       1.292   1.664   1.989   2.373   2.637   3.193
# 83.       1.292   1.663   1.989   2.372   2.636   3.191
# 84.       1.292   1.663   1.989   2.372   2.636   3.190
# 85.       1.292   1.663   1.988   2.371   2.635   3.189
# 86.       1.291   1.663   1.988   2.370   2.634   3.188
# 87.       1.291   1.663   1.988   2.370   2.634   3.187
# 88.       1.291   1.662   1.987   2.369   2.633   3.185
# 89.       1.291   1.662   1.987   2.369   2.632   3.184
# 90.       1.291   1.662   1.987   2.368   2.632   3.183
# 91.       1.291   1.662   1.986   2.368   2.631   3.182
# 92.       1.291   1.662   1.986   2.368   2.630   3.181
# 93.       1.291   1.661   1.986   2.367   2.630   3.180
# 94.       1.291   1.661   1.986   2.367   2.629   3.179
# 95.       1.291   1.661   1.985   2.366   2.629   3.178
# 96.       1.290   1.661   1.985   2.366   2.628   3.177
# 97.       1.290   1.661   1.985   2.365   2.627   3.176
# 98.       1.290   1.661   1.984   2.365   2.627   3.175
# 99.       1.290   1.660   1.984   2.365   2.626   3.175
# 100.       1.290   1.660   1.984   2.364   2.626   3.174
# infinity   1.282   1.645   1.960   2.326   2.576   3.090


def value_transition_next_day_classif(df: pd.DataFrame):
    # classification transition:

    from datetime import timedelta

    classif = df
    column_name = classif.columns.values[0]

    # total num of days in each class
    class_count = classif.value_counts().sort_index()

    n_class = len(class_count)

    next_day_occurrence = []
    for ii in range(len(class_count)):
        class_name = class_count.index[ii]

        days = classif[classif[column_name] == class_name].index

        next_days = days + timedelta(seconds=3600 * 24)

        class_next_days = classif[classif.index.isin(next_days.values)]

        # the num of next day may be smaller than today, since sarah_e has missing data
        n_next_day = len(class_next_days)

        count_next_days = class_next_days.value_counts().sort_index()

        percentage = count_next_days * 100 / n_next_day
        # percentage = count_next_days * 100 / len(days)

        print(f'CL{class_name[0]:g}', percentage)

        next_day_occurrence += [list(percentage.values)]

    next_day_occurrence = np.array(next_day_occurrence)

    # print output: a table
    for i in range(n_class):
        print(f'CL{class_count.index[i][0]:g}', class_count.iloc[i],
              class_count.iloc[i] * 100 / class_count.sum(), *next_day_occurrence[i, :], sep=' ')

    print([f'CL{class_count.index[i][0]:g}' for i in range(n_class)])
    print([class_count.iloc[i] * 100 / class_count.sum() for i in range(n_class)])

    return next_day_occurrence


def value_olr_calssif_significant_map(phase: int, grid: xr.DataArray = 0, month: str = 0,
                                      area: str = 'bigreu') -> xr.DataArray:
    """
    calculate significant map of olr classification e.g., mjo phase or ttt regimes,
     depend on the input olr data from era5 analysis
    ONLY in the swio area
    :param month: like JJA and DJF, etc
    :param grid: output sig_map remapped to gird. if grid = 0 , no interp
    :param phase:
    :return:
    """
    mjo_phase: pd.DataFrame = read_mjo()

    # ----------------------------- read necessary data: era5 ttr reanalysis data
    # ttr_swio = xr.open_dataset(f'~/local_data/era5/ttr.era5.1999-2016.day.swio.nc')['ttr']
    if area == 'SA_swio':
        ttr_era5 = read_to_standard_da(f'~/local_data/era5/ttr/ttr.era5.1999-2016.day.swio.nc', var='ttr')
    if area == 'bigreu':
        ttr_era5 = read_to_standard_da(f'~/local_data/era5/ttr/ttr.era5.1999-2016.day.reu.nc', var='ttr')

    if isinstance(month, str):
        ttr_era5 = filter_xr_by_month(ttr_era5, month=month)
        mjo_phase: pd.DataFrame = filter_df_by_month(mjo_phase, month=month)

    # ----------------------------- anomaly OLR -----------------------------
    olr_swio = convert_ttr_era5_2_olr(ttr=ttr_era5, is_reanalysis=True)
    olr_swio_anomaly = anomaly_daily(olr_swio)

    # select phase:
    date_index: pd.DatetimeIndex = mjo_phase.loc[mjo_phase['phase'] == phase].index
    olr_swio_anomaly_1phase: xr.DataArray = olr_swio_anomaly.sel(time=date_index)  # tag: filter

    # ----------------------------- calculate sig_map -----------------------------
    print(f'calculating significant map, dims={str(olr_swio_anomaly_1phase.shape):s}, waiting ... ')
    sig_map_olr: xr.DataArray = value_significant_of_anomaly_2d_mask(field_3d=olr_swio_anomaly_1phase, conf_level=0.05)

    # to see if remap is necessary:
    if grid == 0:
        sig = sig_map_olr.copy()
        # no remap

    else:

        new_sig_map = np.zeros(grid.shape)
        old_lon = olr_swio.lon
        old_lat = olr_swio.lat

        new_lon = grid.lon.values
        new_lat = grid.lat.values

        # get closest lon:
        for lon in range(grid.lon.size):
            new_lon[lon] = old_lon[np.abs(old_lon - new_lon[lon]).argmin()]
        # get closest lat:
        for lat in range(grid.lat.size):
            new_lat[lat] = old_lat[np.abs(old_lat - new_lat[lat]).argmin()]

        for lat in range(grid.lat.size):
            for lon in range(grid.lon.size):
                new_sig_map[lat, lon] = sig_map_olr.where((sig_map_olr.lat == new_lat[lat]) &
                                                          (sig_map_olr.lon == new_lon[lon]), drop=True).values

        sig = xr.DataArray(new_sig_map.astype(bool), coords=[grid.lat, grid.lon], dims=grid.dims)

    return sig


def value_significant_of_anomaly_2d_mask(field_3d: xr.DataArray, conf_level: float = 0.05,
                                         show: bool = False,
                                         fdr_correction: bool = True,
                                         check_nan_every_grid: bool = False) -> xr.DataArray:
    """
    calculate 2d map of significant of values in true false
    :param conf_level: default = 0.05
    :param field_3d: have to be in (time, lat, lon)
    :return: 2d array of true false xr.DataArray

    Args:
        fdr_correction (): if I do a false discoveries rate correction
        check_nan_every_grid (): to check all the pixel, since they could have nan at different times
        show ():
    """

    print(f'significant: in put file size =')
    print(field_3d.sizes)

    # change the order of dims
    transpose_dims = ['y', 'x']

    # there's another coord beside 'y' and 'x', which is the dim of significant !!!
    sig_coord_name = [x for x in field_3d.dims if x not in transpose_dims][0]

    # tag, note: change order of dim:
    new_dims = [sig_coord_name] + transpose_dims
    field_3d = field_3d.transpose(*new_dims)

    p_map = np.zeros((field_3d.shape[1], field_3d.shape[2]))

    print(f'to get significant map...')
    if check_nan_every_grid:
        for lat in range(field_3d.shape[1]):
            print(f'significant ----- {lat * 100 / len(field_3d.lat): 4.2f} % ...')
            for lon in range(field_3d.shape[2]):
                grid = field_3d[:, lat, lon]

                # select value not nan, removing the nan value
                grid_nonnan = grid[np.logical_not(np.isnan(grid))]

                # check if there's a bad grid point with all values = nan
                # when pass a 3D values to t test with nan, it gives only nan, so better to check each pixel
                if len(grid_nonnan) < 1:
                    print("----------------- bad point")
                    p_map[lat, lon] = 0
                else:
                    # t-test
                    t_statistic, p_value_2side = stats.ttest_1samp(grid_nonnan, 0)
                    p_map[lat, lon] = p_value_2side

                # print(sig_2d.shape, lat, lon)
    else:
        # to make simple and faster:
        # check if the operation of dropna reduce seriously the size of data
        field = field_3d.dropna(sig_coord_name)
        if len(field) / len(field_3d) < 0.30:
            # sometime we have to allow many nan since there are maybe a map with only land
            sys.exit('too much nan, > 70% in data check significant function')
        t_map, p_map = stats.ttest_1samp(field, 0)

    if fdr_correction:
        rejected, pvalue_corrected = \
            fdr_cor(p_map.ravel(), alpha=conf_level, method='indep', is_sorted=False)

        # method{‘i’, ‘indep’, ‘p’, ‘poscorr’, ‘n’, ‘negcorr’}, optional
        # Which method to use for FDR correction.
        # {'i', 'indep', 'p', 'poscorr'} all refer to fdr_bh (Benjamini/Hochberg
        # for independent or positively correlated tests).
        # {'n', 'negcorr'} both refer to fdr_by (Benjamini/Yekutieli for
        # general or negatively correlated tests). Defaults to 'indep'.

        rejected = rejected.reshape(p_map.shape)
        pvalue_corrected = pvalue_corrected.reshape(p_map.shape)
        print(f'correction of FDR is made')

        # update p_map with fdr corrections
        p_map = pvalue_corrected

    # option 2:
    # 根据定义，p值大小指原假设H0为真的情况下样本数据出现的概率。
    # 在实际应用中，如果p值小于0.05，表示H0为真的情况下样本数据出现的概率小于5 %，
    # 根据小概率原理，这样的小概率事件不可能发生，因此我们拒绝H0为真的假设
    sig_map = p_map < conf_level

    # return sig map in 2D xr.DataArray:
    sig_map_da = field_3d.mean(sig_coord_name)
    sig = xr.DataArray(sig_map.astype(bool), coords=sig_map_da.coords, dims=sig_map_da.dims)

    if show:
        plt.close("all")
        sig.plot.pcolormesh(vmin=0, vmax=1)
        plt.show()

    return sig


def welch_test(a: xr.DataArray, b: xr.DataArray, conf_level: float = 0.95, equal_var: bool = False,
               show: bool = True, title: str = 'default'):
    """
    two samples test
    Args:
        a ():
        b ():
        conf_level ():
        equal_var ():
        show ():
        title ():

    Returns:

    """

    # plot welch's test:
    t_sta, p_2side = scipy.stats.ttest_ind(a, b, equal_var=False)

    da1 = xr.zeros_like(a[-1])
    da1[:] = p_2side

    sig = da1 < conf_level
    sig = sig.assign_attrs(dict(units='significance'))

    if show:
        plt.close("all")
        sig.plot.pcolormesh(vmin=0, vmax=1)
        plt.title(title)
        plt.savefig(f'./plot/welch_test.{title.replace(" ", "_"):s}.'
                    f'.png', dpi=300)

        plt.show()

    return sig


def value_remap_a_to_b(a: xr.DataArray, b: xr.DataArray):
    """
    remap a to b, by method='cubic'
    :param a:
    :param b:
    :return:
    """

    # remap cmsaf to mf:
    # if remap by only spatial dimensions:
    lon_name = get_time_lon_lat_name_from_da(b)['lon']
    lat_name = get_time_lon_lat_name_from_da(b)['lat']

    interpolated = a.interp(lon=b[lon_name], lat=b[lat_name])
    # @you, please make sure in the you give the right dimension names (such as lon and lat).
    # cmsaf and mf are both xr.DataArray

    # if remap by all the dimensions:
    # interpolated = a.interp_like(b)
    # that's the same as line below
    # interpolated = a.interp(lon=b.lon, lat=b.lat, time=b.time)
    # we are not usually do the remap in time dimension,
    # if you want to do this, make sure that there is no duplicate index

    return interpolated


def value_season_mean_ds(ds, time_calendar='standard'):
    """

    :param ds:
    :type ds:
    :param time_calendar:
    :type time_calendar:
    :return:
    :rtype:
    """

    print(f'calendar is {time_calendar:s}')
    # Make a DataArray with the number of days in each month, size = len(time)
    month_length = ds.time.dt.days_in_month

    # Calculate the weights by grouping by 'time.season'
    weights = month_length.groupby('time.season') / month_length.groupby('time.season').sum()

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby('time.season').sum().values, np.ones(4))

    # Calculate the weighted average
    return (ds * weights).groupby('time.season').sum(dim='time')


def value_map_corner_values_from_coverage(coverage: str):
    """
    get location of lon/lat box:
    :param coverage:
    :return:
    """

    lon_low = lon_up = lat_low = lat_up = -999

    if coverage == 'reunion':
        lon_low, lon_up, lat_low, lat_up = 54.75, 56.25, -21.75, -20.25
    if coverage == 'swio':
        lon_low, lon_up, lat_low, lat_up = 0, 120, -50, 10
    if coverage == 'SouthernAfrica':
        lon_low, lon_up, lat_low, lat_up = 0, 60, -40, 0

    return lon_low, lon_up, lat_low, lat_up


def value_replace_in_xr(data, dim_name: str, new_values: np.ndarray):
    """
    replace values in xr.DataArray or xr.DataSet, by array
    :param dim_name:
    :type dim_name:
    :param data:
    :param new_values:
    :return:
    """

    data[dim_name] = new_values

    return data


def convert_utc2local_da(test: bool, da):
    """
    convert utc time to local time
    :param test:
    :type test:
    :param da:
    :return:
    """

    time_local = []
    for i in range(da.time.size):
        utc = da.indexes['time'][i].replace(tzinfo=tz.tzutc())
        local_time = utc.astimezone(tz.tzlocal())
        local_time = local_time.replace(tzinfo=None)
        time_local.append(local_time)

        if test:
            print(utc, local_time)

    value_replace_in_xr(data=da, dim_name='time', new_values=np.array(time_local))

    return da


def cal_persistence(classif: pd.DataFrame):
    classif['duration'] = np.full(len(classif), fill_value=1)
    # calculate duration:

    today = 0
    while today < len(classif) - 1:
        # today:
        cl_today = classif.iloc[today].values[0]
        d = 1

        coming_day = today

        print('starting, today =', today)
        while classif.iloc[coming_day + 1].values[0] == cl_today:
            d += 1
            coming_day += 1
            print('coming day + 1 =', coming_day)
            # save this number:
            classif.iloc[today, -1] = int(d)
            classif.iloc[today + 1:today + d, -1] = np.full(d - 1, fill_value=np.nan)
            # to stop checking coming day t
            if coming_day == len(classif) - 1:
                today = len(classif)
                break
        else:
            print('final d = ', d)
            today += d
            print('final today = ', today)

    return classif


def value_consistency_sign_with_mean_in_percentage_2d(field_3d: xr.DataArray):
    """
    get a map of percentage: days of the same signs with statistics.
    :param field_3d:
    :return: percentage_data_array in format ndarray in (lat,lon)
    """
    # note: change order of dim:
    field_3d = field_3d.transpose("time", "latitude", "longitude")
    # note: mean of one specific dim:
    time_mean_field = field_3d.mean(dim='time')

    num_time = field_3d.time.shape[0]
    num_lon = time_mean_field.longitude.shape[0]
    num_lat = time_mean_field.latitude.shape[0]

    compare_field = np.zeros(field_3d.shape)

    percentage_2d_map = np.zeros((num_lat, num_lon))

    for t in range(num_time):
        compare_field[t, :, :] = time_mean_field * field_3d[t]

    for lat in range(num_lat):
        for lon in range(num_lon):
            time_series = compare_field[:, lat, lon]
            positive_count = len(list(filter(lambda x: (x >= 0), time_series)))

            percentage_2d_map[lat, lon] = positive_count / num_time * 100

    return percentage_2d_map


def set_basemap(ax: plt.Axes, area: str):
    """
    set basemap
    :param ax:
    :type ax: cartopy.mpl.geoaxes.GeoAxesSubplot
    :param area:
    :return:
    """
    area_name = area

    area = value_lonlatbox_from_area(area_name)
    ax.set_extent(area, crs=ccrs.PlateCarree())
    # lon_left, lon_right, lat_north, lat_north

    ax.coastlines('50m')
    ax.add_feature(cfeature.LAND.with_scale('10m'))


def set_active_axis(axs: np.ndarray, n: int):
    """
    active this axis of plot or subplot
    :param axs: nd array of subplots' axis
    :param n:
    :return:
    """

    ax = axs[n]
    plt.sca(axs[n])
    # active this subplot

    return ax


# noinspection PyUnresolvedReferences
def set_cbar(vmax, vmin, n_cbar, bias, cmap: str = 'default'):
    """
    set for color bar
    :param n_cbar:
    :param vmin:
    :param vmax:
    :param bias:
    :return: cmap, norm

    Args:
        cmap: if a specific cmap is requared
    """
    import matplotlib as mpl

    if cmap == 'default':
        if bias == 1:
            cmap = plt.cm.coolwarm
            vmin = max(np.abs(vmin), np.abs(vmax)) * (- 1)
            vmax = max(np.abs(vmin), np.abs(vmax))
        if bias == 0:
            cmap = plt.cm.YlOrRd

        # using the input of min and max, but make (max-min/2) in the middle
        if bias == 3:
            cmap = plt.cm.coolwarm
    else:
        print(f'use cbar specific')

    from matplotlib.colors import TwoSlopeNorm
    if bias:
        # to make uneven colorbar with zero in white
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=(vmax - vmin) / 2 + vmin, vmax=vmax)

    return cmap, norm


def value_season_name_from_str(string: str):
    """
    get monthly tag, such as JJA/DJF from string
    :param string: str
    :return:
    """

    tag_dict = dict(

        summer='JJA',
        winter='DJF',
        austral_summer='DJF',
        austral_winter='JJA',

    )

    return tag_dict[string]


def print_data(data, dim: int = 1):
    """
    print all data
    :param dim:
    :type dim:
    :param data:
    :return:
    """

    if dim == 1:
        for i in range(data.shape[0]):
            print(data[i])

    if dim == 2:

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                print(data[i, j])


def read_mjo(match_ssr_avail_day: bool = 0):
    """
    to read mjo phase
    :return: pd.DataFrame
    """
    # mjo_sel = f'SELECT ' \
    #           f'ADDTIME(CONVERT(left(dt, 10), DATETIME), MAKETIME(HOUR(dt), ' \
    #           f'floor(MINUTE(dt) / 10)*10,0)) as DateTime, ' \
    #           f'AVG(GHI_020_Avg) as ssr, ' \
    #           f'MONTH(dt) as Month, HOUR(dt) as Hour, ' \
    #           f'floor(MINUTE(dt) / 10)*10 as Minute ' \
    #           f'from GYSOMATE.GHI_T_P_Rain_RH_Moufia_1min ' \
    #           f'where GHI_020_Avg>=0 and ' \
    #           f'(dt>="2018-03-01" and dt<="2019-05-01") ' \
    #           f'group by date(dt), hour(dt), floor(minute(dt) / 10);'
    #
    # print(mjo_sel)

    mjo_query = f'SELECT dt as DateTime, rmm1, rmm2, phase, amplitude ' \
                f'from SWIO.MJO_index ' \
                f'where year(dt)>=1999 and year(dt)<=2016;'
    # NOTE: amplitude is the square root of rmm1^2 + rmm2^2

    df = query_data(mysql_query=mjo_query, remove_missing_data=False)

    if match_ssr_avail_day:
        return df
    else:
        return df


def sellonlatbox(da: xr.DataArray, lonlatbox: list):
    """
    used on dataarray, as that of cdo command
    Parameters
    ----------
    da : input data
    lonlatbox : [lon1, lon2, lat1, lat2], south and west are negative

    Returns
    -------
    DataArray
    """
    # TODO: consider the definition of negative values of lon/lat

    coords = da.coords.variables.mapping

    if 'lon' in coords.keys():
        lon_name = 'lon'
    if 'longitude' in coords.keys():
        lon_name = 'longitude'

    if 'lat' in coords.keys():
        lat_name = 'lat'
    if 'latitude' in coords.keys():
        lat_name = 'latitude'

    da1 = da.where(np.logical_and(da[lon_name] > min(lonlatbox[0], lonlatbox[1]),
                                  da[lon_name] < max(lonlatbox[0], lonlatbox[1])), drop=True)
    da2 = da1.where(np.logical_and(da1[lat_name] > min(lonlatbox[2], lonlatbox[3]),
                                   da1[lat_name] < max(lonlatbox[2], lonlatbox[3])), drop=True)

    return da2


def filter_remove_b_from_a_daily_df(a: pd.DataFrame, b: pd.DataFrame):
    mask = [a.index.strftime("%Y-%m-%d")[i] in b.index.strftime("%Y-%m-%d") for i in range(len(a))]

    mask_inverse = [not (i) for i in mask]

    return a[mask_inverse]


def filter_2d_by_mask(data: xr.DataArray, mask: xr.DataArray):
    """
    filtering 2d data by mask
    :param data:
    :param mask:
    :return: do not change the data format
    """

    # check if the dims of data and mask is the same:
    check_lat = data.lat == mask.lat
    check_lon = data.lon == mask.lon

    if np.logical_or(False in check_lat, False in check_lon):
        print(f'maks and data in different lonlat coords...check ...')
        breakpoint()

    # if isinstance(data, np.ndarray):
    #     data_to_return: np.ndarray = data[mask]
    #     # Attention: if the mask is not square/rectangle of Trues, got 1d array;

    if isinstance(data, xr.DataArray):
        # build up a xr.DataArray as mask, only type of DataArray works.
        lookup = xr.DataArray(mask, dims=('y', 'x'))
        # use the standard dim names 'time', 'y', 'x'

        # lookup = xr.DataArray(mask, dims=data.dims) # possible works for 3D mask

        data_to_return = data.where(lookup)

    return data_to_return


def select_land_only_reunion_by_altitude(da: xr.DataArray):
    """
    select a area for only reunion
    Args:
        da ():

    Returns:
        da with nan
    """

    altitude = value_altitude_from_lonlat_reunion(lon=da.lon.values, lat=da.lat.values)
    lookup = altitude > 0

    new_da = da.where(lookup).dropna(dim='y', how='all').dropna(dim='x', how='all')

    return new_da


def value_month_from_str(month: str):
    """
    get month as array from string such as 'JJA'
    :rtype: int
    :param month:
    :return: int
    """

    # TODO: from first letter to number

    if month == 'JJA':
        mon = (6, 7, 8)
    if month == 'DJF':
        mon = (12, 1, 2)
    if month == 'NDJF':
        mon = (11, 12, 1, 2)

    return mon


def value_str_month_name(months: list):
    """
    from list of month to get month names
    Args:
        months ():

    Returns:

    """

    all_months = list(range(1, 13))
    all_names = ['Jan', 'Feb', 'Mar', 'Apr',
                 'May', 'Jun', 'Jul', 'Aug',
                 'Sep', 'Oct', 'Nov', 'Dec']

    names = [all_names[months[x] - 1] for x in range(len(months)) if months[x] in all_months]

    return names


def filter_df_by_month(data: pd.DataFrame, month: str) -> pd.DataFrame:
    """
    filtering pd.DataFrame by input string of season
    :param data:
    :param month: such as 'JJA'
    :return:
    """

    if isinstance(data, pd.DataFrame):
        # TODO: to be updated:
        season_index = ((data.index.month % 12 + 3) // 3).map({1: 'DJF', 2: 'MAM', 3: 'JJA', 4: 'SON'})
        return_data = data[season_index == month]

    return return_data


def filter_xr_by_month(data: xr.DataArray, month: str) -> xr.DataArray:
    """
    filtering xr.DataArray by input string of season
    :param data:
    :param month: such as 'JJA', 'DJF', et 'NDJF'
    :return:
    """

    if isinstance(data, xr.DataArray):
        month = value_month_from_str(month)

        mask = [True if x in month else False for x in data.time.dt.month]
        lookup = xr.DataArray(mask, dims=data.dims[0])

        data_to_return = data.where(lookup, drop=True)

    if isinstance(data, xr.Dataset):
        # TODO: to be updated:
        print(f'function to update')

    return data_to_return


def filter_by_season_name(data: xr.DataArray, season_name: str):
    """
    filtering data by input string of season
    :param data:
    :param season_name: summer, winter, austral_summer, austral_winter, etc..
    :return:
    """

    # get tag such as 'JJA'
    season_tag = value_season_name_from_str(season_name)

    if isinstance(data, xr.DataArray):
        return_data = data[data.time.dt.season == season_tag]

    if isinstance(data, pd.DataFrame):
        season_index = ((data.index.month % 12 + 3) // 3).map({1: 'DJF', 2: 'MAM', 3: 'JJA', 4: 'SON'})
        return_data = data[season_index == season_tag]

    return return_data


def data_filter_by_key_limit_value(data, key: str, how: str, value: float):
    """
    filtering data by a value of keyword
    :param data:
    :param key: which column
    :param how: lt, gt, eq, lteq
    :param value: float
    :return: data after filtered
    """
    #
    # how_dict = dict(
    #     lt='<',
    #     gt='>',
    #     eq='==',
    #     lteq='<=',
    #     gteq='<=')

    # filter_str = f'{key:s} {how_dict[how]} {value:.2f}'

    if isinstance(data, pd.DataFrame):
        if how == 'gt':
            return_data = data[data[key] > value]

    if isinstance(data, xr.DataArray):
        return_data = data[data.time.dt.season == how]

    return return_data


def reduce_ndim_coord(coord: xr.DataArray, dim_name: str, random: bool = True,
                      max_check_len: int = 500, check_ratio: float = 0.1):
    """
    to check if the input da, usually a coord or dim of a geo map, is static or not,
    if it's static try to reduce the num of dim of this coord

    :param max_check_len:
    :type max_check_len:
    :param dim_name: to check if this dim is change as a function of others
    :type dim_name: such as 'south_north' if the input coord is lon
    :param check_ratio: percentage of the total size, for check if random is True
    :type check_ratio:
    :param coord: input coord such as lon or lat
    :type coord:
    :param random: to use random data, 10 percent of the total data
    :type random:
    :return: ndim-reduced coord as da
    :rtype:
    """

    original_coord = coord
    check_coord = coord

    other_dim = list(original_coord.dims)
    other_dim.remove(dim_name)

    check_coord = check_coord.stack(new_dim=other_dim)

    # random check of 10 percent is not enough if the length is smaller than say 500
    # example: icare data, SAF_NWF reunion selection in project iCare_Cloud, lat=24 lon=28,
    # we have to check all of these 24 or 28, since only few of them are different (dim dependent).
    if check_coord.shape[0] < 500:
        random = False

    if random:
        from random import randint
        check_len = int(min(check_coord.shape[0] * check_ratio, max_check_len))

        check_index = [randint(0, check_len - 1) for x in range(check_len)]

        # select some sample to boost
        check_coord = check_coord.isel(new_dim=check_index)

    # starting to check if dim of dim_name is changing as a function of other_dims
    check_coord = check_coord.transpose(..., dim_name)

    diff = 0
    for i in range(check_coord.shape[0]):
        if np.array_equal(check_coord[i], check_coord[0]):
            pass
        else:
            diff += 1
            # print('not same: ', i)
            # print(f' random number is ', check_index)

    if diff:
        return original_coord
    else:
        # every lon, such as lon[i, :] is the same
        return check_coord[0]


def get_time_lon_lat_from_da(da: xr.DataArray):
    """

    Parameters
    ----------
    da :

    Returns
    -------
    dict :     return {'time': time, 'lon': lon, 'lat': lat, 'number': number}
    """
    coords_names: dict = get_time_lon_lat_name_from_da(da, name_from='coords')
    # attention: here the coords names maybe the dim names, if coords is missing in the *.nc file.

    coords = dict()

    for lonlat in ['lon', 'lat']:
        if lonlat in coords_names:
            lon_or_lat: xr.DataArray = da[coords_names[lonlat]]

            if lon_or_lat.ndim == 1:
                lon_or_lat = lon_or_lat.values

            if lon_or_lat.ndim > 1:
                dim_names = get_time_lon_lat_name_from_da(lon_or_lat, name_from='dims')
                lon_or_lat = reduce_ndim_coord(coord=lon_or_lat, dim_name=dim_names[lonlat], random=True).values

            if lonlat == 'lon':
                if lon_or_lat.ndim == 1:
                    lon_or_lat = np.array([x - 360 if x > 180 else x for x in lon_or_lat])

            coords[lonlat] = lon_or_lat

    if 'time' in coords_names:
        time = da[coords_names['time']].values
        coords.update(time=time)

    if 'lev' in coords_names:
        lev = da[coords_names['lev']].values
        coords.update(lev=lev)

    if 'number' in coords_names:
        number = da[coords_names['number']].values
        coords.update(number=number)

    return coords


def get_time_lon_lat_name_from_da(da: xr.DataArray,
                                  name_from: str = 'coords'):
    """

    Parameters
    ----------
    da ():
    name_from (): get name from coords by default, possible get name from dims.

    Returns
    -------
    dict :     return {'time': time, 'lon': lon, 'lat': lat, 'number': number}
    """
    # definitions:
    possible_coords_names = {
        'time': ['time', 'datetime', 'XTIME', 'Time', 'WEDCEN2'],
        'lon': ['lon', 'west_east', 'rlon', 'longitude', 'nx', 'x', 'XLONG', 'XLONG_U', 'XLONG_V'],
        'lat': ['lat', 'south_north', 'rlat', 'latitude', 'ny', 'y', 'XLAT', 'XLAT_U', 'XLAT_V'],
        'lev': ['height', 'bottom_top', 'lev', 'level', 'xlevel', 'lev_2'],
        'number': ['number', 'num', 'model']
        # the default name is 'number'
    }
    # attention: these keys will be used as the standard names of DataArray

    # ----------------------------- important:
    # coords = list(da.dims)
    # ATTENTION: num of dims is sometimes larger than coords: WRF has bottom_top but not such coords
    # ATTENTION: according to the doc of Xarray: len(arr.dims) <= len(arr.coords) in general.
    # ATTENTION: dims names are not the same as coords, WRF dim='south_north', coords=XLAT
    # so save to use the coords names.
    # -----------------------------
    if name_from == 'coords':
        da_names = list(dict(da.coords).keys())
        # CTANG: coords should be a list not a string.
        # since: 't' is in 'time', 'level' is in 'xlevel'
        # and: ['t'] is not in ['time']; 't' is not in ['time']

        dims = list(da.dims)

    if name_from == 'dims':
        da_names = list(da.dims)
        dims = list(da.dims)

    # construct output
    output_names = {}

    for key, possible_list in possible_coords_names.items():
        coord_name = [x for x in possible_list if x in da_names]
        if len(coord_name) == 1:
            output_names.update({key: coord_name[0]})
        else:
            # check if this coords is missing in dims
            dim_name = [x for x in possible_list if x in dims]
            if len(dim_name) == 1:
                output_names.update({key: dim_name[0]})

                print(f'coords {key:s} not found, using dimension name: {dim_name[0]}')
                warnings.warn('coords missing')

    return output_names


def value_cbar_max_min_of_da(da: xr.DataArray):
    max_value = np.float(max(np.abs(da.min()), np.abs(da.max())))

    return max_value, max_value * (-1)


def value_max_min_of_var(var: str, how: str):
    # data:
    # ----------------------------- do not change the following lines:
    var_name = ['sst', 'v10', 'msl', 'q', 'ttr', 'OLR', 'sp', 'SIS', 'ssrd', 'SWDOWN']
    max_mean = [30.00, 7.600, 103100, 0.0200, -190.00, -190.00, 400.0, 170.0, 170.00, 170.00]
    min_mean = [14.00, -6.60, 99900., 0.0000, -310.00, -310.00, 100.0, 0.000, 0.0000, 0.0000]
    max_anom = [0.500, 0.020, 900.00, 0.0020, 50.0000, 50.0000, 5.000, 30.00, 20.000, 20.000]
    min_anom = [-0.50, -0.02, -900.0, -0.002, -50.000, -50.000, -5.00, -30.0, -20.00, -20.00]

    # ----------------------------- do not change above lines.
    if how == 'time_mean':
        vmax = max_mean[var_name.index(var)]
        vmin = min_mean[var_name.index(var)]
    if how == 'anomaly_mean':
        vmax = max_anom[var_name.index(var)]
        vmin = min_anom[var_name.index(var)]

    return vmax, vmin


def get_min_max_ds(ds: xr.Dataset):
    """
    get min and max of ds
    Parameters
    ----------
    ds :
    -------
    """
    list_var = list(ds.keys())
    vmax = np.max([ds[v].max().values for v in list_var])
    vmin = np.min([ds[v].min().values for v in list_var])

    return vmin, vmax


def plot_hourly_boxplot_ds_by(list_da: list, list_var_name: list, by: str = 'Month', comment='no comment'):
    """
    plot hourly box plot by "Month" or "Season"
    :param comment:
    :type comment:
    :param list_da: the input da should be in the same coords
    :param list_var_name:
    :param by: 'Month' or 'season'
    :return:
    """

    ds = convert_multi_da_to_ds(list_da=list_da, list_var_name=list_var_name)

    # define the variable to use:
    monthly = seasonal = None
    months = list(range(1, 13))
    seasons = ['DJF', 'MAM', 'JJA', 'SON']

    vmin, vmax = get_min_max_ds(ds)
    # vmax = 500
    # vmin = -500

    if by in ['Month', 'month', 'months']:
        monthly = True
        nrow = 6
        ncol = 2
        tags = months
    if by in ['season', 'Season', 'seasons']:
        print(f'plot seasonal plots')
        seasonal = True
        nrow = 2
        ncol = 2
        tags = seasons

    if by is None:
        nrow = 1
        ncol = 1
        tags = None

    fig, axs = plt.subplots(nrows=nrow, ncols=ncol,
                            figsize=(ncol * 9, nrow * 3), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.1, hspace=0.9, top=0.95, wspace=0.2)

    if by is not None:
        axs = axs.ravel()

    # 1) prepare data:

    for i in range(len(tags)):
        if by is None:
            ax = axs
        else:
            # plt.sca(axs[i])  # active this subplot
            ax = axs[i]

        if monthly:
            data_slice: xr.Dataset = ds.where(ds.time.dt.month == tags[i], drop=True)
        if seasonal:
            data_slice: xr.Dataset = ds.where(ds.time.dt.season == tags[i], drop=True)

        if by is None:
            data_slice = ds.copy()

        # to convert da to df: for the boxplot:
        print(f'convert DataArray to DataFrame ...')
        all_var = pd.DataFrame()
        for col in range(len(list_var_name)):
            var = pd.DataFrame()
            da_slice = data_slice[list_var_name[col]]
            var['target'] = da_slice.values.ravel()
            var['Hour'] = da_slice.to_dataframe().index.get_level_values(0).hour
            var['var'] = [list_var_name[col] for _ in range(len(da_slice.values.ravel()))]
            # var['var'] = [list_var_name[col] for x in range(len(da_slice.values.ravel()))]
            all_var = all_var.append(var)

        sns.boxplot(x='Hour', y='target', hue='var', data=all_var, ax=ax, showmeans=True)
        # Seaborn's showmeans=True argument adds a mark for mean values in each box.
        # By default, mean values are marked in green color triangles.

        ax.set_xlim(4, 20)
        ax.set_ylim(vmin, vmax)
        if by is not None:
            ax.set_title(f'{by:s} = {str(tags[i]):s}', fontsize=18)

        if comment != 'no comment':
            ax.text(0.02, 0.95, f'{comment:s}', fontsize=14,
                    horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

        # Get the handles and labels. For this example it'll be 2 tuples
        # of length 4 each.
        handles, labels = ax.get_legend_handles_labels()

        # When creating the legend, only use the first two elements
        # to effectively remove the last two.
        plt.legend(handles[0:4], labels[0:4], bbox_to_anchor=(0.95, 0.95), borderaxespad=0.,
                   loc="upper right", fontsize=18)

        ax.set_xlabel(f'Hour', fontsize=18)
        ax.set_ylabel(f'SSR ($W/m^2$)', fontsize=18)
        ax.tick_params(labelsize=16)

        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')

    print(f'save/show the plot ...')
    plt.savefig(f'./plot/{"-".join(list_var_name):s}.hourly_boxplot_by_{by:s}.png', dpi=200)

    plt.show()

    print(f'got this plot')

    return fig


def plot_hourly_boxplot_by(df: pd.DataFrame, columns: list, by: str,
                           vmin=None, vmax=None, title='', ylabel='',
                           output_tag: str = None):
    """
    applied project Sky_clearness_2023:
    plot hourly box plot by "Month" or "Season"
    :param df:
    :param columns:
    :param by:
    :return:
    """

    if by == 'Month':
        nrow = 4
        ncol = 3
    if by is None:
        nrow = 1
        ncol = 1

    # n_plot = nrow * ncol

    fig, axs = plt.subplots(nrows=nrow, ncols=ncol,
                            figsize=(18, 13), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05, hspace=0.4, top=0.90, wspace=0.2)

    if by == 'Month':
        axs = axs.ravel()

    # months = [11, 12, 1, 2, 3, 4, 4]
    months = list(range(1, 13))
    for i in range(len(months)):
        if by == 'Month':
            # plt.sca(axs[i])  # active this subplot
            ax = axs[i]
        if by is None:
            ax = axs

        if by == 'Month':
            data_slice = df[df.index.month == months[i]]
        if by is None:
            data_slice = df.copy()

        all_var = pd.DataFrame()
        for col in range(len(columns)):
            # calculate normalised value:
            var = pd.DataFrame()
            var['target'] = data_slice[columns[col]]
            var['Hour'] = data_slice.index.hour
            var['var'] = [columns[col] for _ in range(len(data_slice))]
            all_var = all_var.append(var)

        ax.set_xlim(4, 20)
        sns.boxplot(x='Hour', y='target', hue='var', data=all_var, ax=ax, showmeans=True)

        if vmax is not None:
            ax.set_ylim(vmin, vmax)

        if by is not None:
            ax.set_title(f'{by:s} = {months[i]:g}')

        # plt.legend()

        # Get the handles and labels. For this example it'll be 2 tuples
        # of length 4 each.
        handles, labels = ax.get_legend_handles_labels()

        # When creating the legend, only use the first two elements
        # to effectively remove the last two.
        plt.legend(handles[0:4], labels[0:4], bbox_to_anchor=(0.95, 0.95), borderaxespad=0.,
                   loc="upper right", fontsize=18)

        plt.ylabel(f'distribution')
        plt.suptitle(title, fontsize=18)
        ax.set_xlabel(f'Hour', fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(labelsize=16)

        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')

    print(f'save/show the plot ...')

    plt.savefig(f'./plot/hourly_boxplot_by{by:s}.{output_tag:s}.png')

    plt.show()

    print(f'got this plot')


def plot_scatter_color_by(x: pd.DataFrame, y: pd.DataFrame, label_x: str, label_y: str,
                          color_by_column: str, size: float = 8):
    """

    :param size:
    :param x:
    :param y:
    :param label_x:
    :param label_y:
    :param color_by_column:
    :return:
    """

    # default is color_by = month

    months = [11, 12, 1, 2, 3, 4]
    month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

    colors = ['pink', 'darkviolet', 'blue', 'forestgreen', 'darkorange', 'red',
              'deeppink', 'blueviolet', 'royalblue', 'lightseagreen', 'limegreen', 'yellowgreen', 'tomato',
              'silver', 'gray', 'black']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 6), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, hspace=0.4, top=0.8, wspace=0.05)

    if color_by_column is None:
        xx = x
        yy = y
        plt.scatter(xx, yy, c=colors[1], s=size, edgecolors=colors[1], alpha=0.8)

    if color_by_column == 'Month':
        for i in range(len(months)):
            xx = x[x.index.month == months[i]]
            yy = y[y.index.month == months[i]]

            # plt.plot(xx, yy, label=month_names[i], color=colors[i])
            plt.scatter(xx, yy, c=colors[i], label=month_names[i],
                        s=size, edgecolors=colors[i], alpha=0.8)

        plt.legend(loc="upper right", markerscale=6, fontsize=16)

    ax.set_xlabel(label_x, fontsize=18)
    ax.set_ylabel(label_y, fontsize=18)
    ax.tick_params(labelsize=16)

    plt.grid(True)

    return fig, ax


# ==================================
def get_random_color(num_color: int):
    """
    return color as a list
    :param num_color:
    :return:
    """
    import random

    number_of_colors = num_color

    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
             for _ in range(number_of_colors)]

    return color


# ==================================
def station_data_missing_map_hourly_by_month(df: pd.DataFrame, station_id: str):
    """
    plot hourly missing data map by month
    :param station_id:
    :param df:
    :return:
    """

    # ----------------------------- set parameters -----------------------------
    # TODO: read month directly
    months = [11, 12, 1, 2, 3, 4]
    station_id = list(set(df[station_id]))
    # ----------------------------- set fig -----------------------------
    nrows = len(months)

    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 30),
                            facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.05, hspace=0.4, top=0.96, wspace=0.05)

    axs = axs.ravel()
    # ----------------------------- plotting -----------------------------

    # plot data in each month:
    for i in range(len(months)):
        month = months[i]

        plt.sca(axs[i])  # active this subplot

        month_data = df[df.index.month == month].sort_index()

        # get all time steps to check missing value
        first_timestep = month_data.index[0]
        last_timestep = month_data.index[-1]
        time_range = pd.date_range(first_timestep, last_timestep, freq='60min')
        daytime_range = [x for x in time_range if (8 <= x.hour <= 17)]
        all_daytime_index = pd.Index(daytime_range)

        # for v in range(2):
        for v in range(len(station_id)):

            data_slice = month_data[month_data['station_id'] == station_id[v]]
            nday = len(set(data_slice.index.day))

            print(f'month = {month:g}, station_id = {v:g}, day = {nday:g}')

            # find missing time steps:
            diff = all_daytime_index.difference(data_slice.index)

            if len(diff) == 0:
                print(f'all complete ...')
                plt.hlines(v, 0, 320, colors='blue', linewidth=0.2, linestyles='dashed', label='')
            else:
                print(f'there is missing data ...')

                plt.hlines(v, 0, 320, colors='red', linewidth=0.4, linestyles='dashed', label='')
                for k in range(len(all_daytime_index)):
                    if all_daytime_index[k] in diff:
                        plt.scatter(k, v, edgecolor='black', zorder=2, s=50)

        # ----------------------------- format of fig -----------------------------

        # ----------------------------- x axis -----------------------------
        # put the ticks in the middle of the day, means 12h00
        custom_ticks = range(4, len(data_slice), 10)

        custom_ticks_labels = range(1, nday + 1)
        axs[i].set_xticks(custom_ticks)
        axs[i].set_xticklabels(custom_ticks_labels)
        axs[i].set_xlim(0, 320)

        # axs[i].xaxis.set_ticks_position('top')
        # axs[i].xaxis.set_ticks_position('bottom')

        # ----------------------------- y axis -----------------------------
        custom_ticks = range(len(station_id))

        custom_ticks_labels = station_id

        axs[i].set_yticks(custom_ticks)
        axs[i].set_yticklabels(custom_ticks_labels)

        axs[i].set_ylim(-1, len(station_id) + 1)

        # plt.legend(loc='upper right', fontsize=8)
        plt.xlabel(f'day')
        plt.ylabel(f'station_id (blue (red) means (not) complete in this month)')
        plt.title(data_slice.index[0].month_name())

    suptitle = f'MeteoFrance missing data at each station during daytime (8h - 17h)'
    plt.suptitle(suptitle)

    # plt.show()
    print(f'got the plot')
    plt.savefig('./meteofrance_missing_map.png', dpi=200)


# noinspection PyUnresolvedReferences
def plot_station_value_by_month(lon: pd.DataFrame, lat: pd.DataFrame, value: pd.DataFrame,
                                cbar_label: str, fig_title: str, bias=False):
    """
    plot station locations and their values
    :param bias:
    :param fig_title:
    :param cbar_label: label of color bar
    :param lon:
    :param lat:
    :param value:
    :return: map show
    """

    print(fig_title)
    import matplotlib as mpl

    months = [11, 12, 1, 2, 3, 4]
    month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

    plt.figure(figsize=(5, 24), dpi=200)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=1.5, top=0.95, wspace=0.05)

    # plot in each month:
    for m in range(len(months)):

        # data:
        monthly_data = value[value.index.month == months[m]]

        station_group = monthly_data.groupby('station_id')
        station_mean_bias = station_group[['bias']].mean().values[:, 0]

        # set map
        ax = plt.subplot(len(months), 1, m + 1, projection=ccrs.PlateCarree())
        ax.set_extent([55, 56, -21.5, -20.8], crs=ccrs.PlateCarree())
        # ax.set_extent([20, 110, -51, 9], crs=ccrs.PlateCarree())
        # lon_left, lon_right, lat_north, lat_north

        ax.coastlines('50m')
        ax.add_feature(cfeature.LAND.with_scale('10m'))
        # ax.add_feature(cfeature.OCEAN.with_scale('10m'))
        # ax.add_feature(cfeature.COASTLINE.with_scale('10m'))
        # ax.add_feature(cfeature.BORDERS.with_scale('10m'), linestyle=':')
        # ax.add_feature(cfeature.LAKES.with_scale('10m'), alpha=0.5)
        # ax.add_feature(cfeature.RIVERS.with_scale('10m'))
        # ax.coastlines()

        # ----------------------------- cbar -----------------------------
        if np.max(station_mean_bias) - np.min(station_mean_bias) < 10:
            round_number = 2
        else:
            round_number = 0

        n_cbar = 10
        vmin = round(np.min(station_mean_bias) / n_cbar, round_number) * n_cbar
        vmax = round(np.max(station_mean_bias) / n_cbar, round_number) * n_cbar

        if bias:
            cmap = plt.cm.coolwarm
            vmin = max(np.abs(vmin), np.abs(vmax)) * (- 1)
            vmax = max(np.abs(vmin), np.abs(vmax))
            print(vmax)
        else:
            cmap = plt.cm.YlOrRd

        vmin = -340
        vmax = 340

        bounds = np.linspace(vmin, vmax, n_cbar + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # ----------------------------------------------------------
        # plot:
        # ax.quiver(x, y, u, v, transform=vector_crs)
        sc = plt.scatter(lon, lat, c=station_mean_bias, edgecolor='black',
                         # transform=ccrs.PlateCarree(),
                         zorder=2, norm=norm, vmin=vmin, vmax=vmax, s=50, cmap=cmap)

        # ----------------------------- color bar -----------------------------
        cb = plt.colorbar(sc, orientation='horizontal', shrink=0.7, pad=0.1, label=cbar_label)
        cb.ax.tick_params(labelsize=10)

        # ax.xaxis.set_ticks_position('top')

        ax.gridlines(draw_labels=False)

        ax.text(0.93, 0.95, month_names[m],
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    plt.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.6)
    plt.suptitle(f'monthly daytime (8h 17h) \n mean bias at MeteoFrance stations')

    plt.show()
    print(f'got plot')


def monthly_circulation(lon: xr.DataArray, lat: xr.DataArray,
                        u: xr.DataArray, v: xr.DataArray, p: xr.DataArray, domain: str,
                        cbar_label: str, fig_title: str, bias=False):
    """
    to plot monthly circulation, u, v winds, and mean sea level pressure (p)
    :param domain: one of ['swio', 'reu-mau', 'reu']
    :param p:
    :param v:
    :param u:
    :param bias:
    :param fig_title:
    :param cbar_label: label of color bar
    :param lon:
    :param lat:
    :return: map show
    """

    print(cbar_label, fig_title, bias)

    months = [11, 12, 1, 2, 3, 4]
    dates = ['2004-11-01', '2004-12-01', '2005-01-01', '2005-02-01', '2005-03-01', '2005-04-01']
    month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    # nrows = len(months)

    plt.figure(figsize=(5, 24), dpi=300)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=1.5, top=0.95, wspace=0.05)

    # plot in each month:
    for m in range(len(months)):
        ax = plt.subplot(len(months), 1, m + 1, projection=ccrs.PlateCarree())

        # ax.gridlines(draw_labels=False)

        print(f'plot month = {month_names[m]:s}')
        # ----------------------------- plot u and v winds -----------------------------
        # data:
        x = lon.longitude.values
        y = lat.latitude.values
        monthly_u = u.sel(time=dates[m]).values
        monthly_v = v.sel(time=dates[m]).values
        monthly_p = p.sel(time=dates[m]).values

        # set map
        area_name = domain

        if area_name == 'swio':
            n_slice = 1
            n_scale = 2
        if area_name == 'reu_mau':
            n_slice = 1
            n_scale = 10

        if area_name == 'reu':
            n_slice = 2
            n_scale = 10

        area = value_lonlatbox_from_area(area_name)
        ax.set_extent(area, crs=ccrs.PlateCarree())
        # lon_left, lon_right, lat_north, lat_north

        ax.coastlines('50m')
        ax.add_feature(cfeature.LAND.with_scale('10m'))

        # ----------------------------- mean sea level pressure -----------------------------
        # Contour the heights every 10 m
        contours = np.arange(98947, 102427, 300)

        c = ax.contour(x, y, monthly_p, levels=contours, colors='green', linewidths=1)
        ax.clabel(c, fontsize=10, inline=1, inline_spacing=3, fmt='%i')

        # ----------------------------- wind -----------------------------
        # Set up parameters for quiver plot. The slices below are used to subset the data (here
        # taking every 4th point in x and y). The quiver_kwargs are parameters to control the
        # appearance of the quiver so that they stay consistent between the calls.
        quiver_slices = slice(None, None, n_slice)
        quiver_kwargs = {'headlength': 5, 'headwidth': 3, 'angles': 'uv', 'scale_units': 'xy', 'scale': n_scale}

        # Plot the wind vectors
        ax.quiver(x[quiver_slices], y[quiver_slices],
                  monthly_u[quiver_slices, quiver_slices], monthly_v[quiver_slices, quiver_slices],
                  color='blue', zorder=2, **quiver_kwargs)

        ax.text(0.93, 0.95, month_names[m],
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

        # ax.gridlines(draw_labels=False)
        # # ----------------------------------------------------------
        # # plot:

        clevs = np.arange(-10, 12, 2)
        # clevs = np.arange(200, 370, 15)
        cf = ax.contourf(x, y, monthly_u, clevs, cmap=plt.cm.coolwarm,
                         norm=plt.Normalize(-10, 10), transform=ccrs.PlateCarree())

        # ----------------------------- color bar -----------------------------
        cb = plt.colorbar(cf, orientation='horizontal', shrink=0.7, pad=0.05,
                          # label='ssr')
                          label='east <-- 850hPa zonal wind --> west')

        cb.ax.tick_params(labelsize=10)

        # # ax.xaxis.set_ticks_position('top')

        # ax.text(0.53, 0.95, month_names[m] + 'from ERA5 2004-2005',
        #         horizontalalignment='right', verticalalignment='top',
        #         transform=ax.transAxes)

        plt.title(month_names[m] + ' (ERA5 2004-2005)')

    # plt.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.6)
    plt.suptitle(f'circulation wind at 850 hPa')

    plt.show()
    print(f'got plot')

    plt.savefig(f'./monthly_circulation.png', dpi=220)


def convert_multi_da_by_new_dim(list_da: list, new_dim: dict):
    """
    Args:
        list_da ():
        new_dim ():

    Returns: a new da

    """

    # merge data from all models:

    ensemble = xr.concat(list_da, list(new_dim.values())[0]).rename({'concat_dim': list(new_dim.keys())[0]})
    # change order of dims so that cdo could read correctly
    # import itertools
    # ensemble_da = ensemble.transpose(itertools.permutations(list(list_da[0].dims) + list(new_dim.keys())))
    # TODO: change dims order

    return ensemble


def get_gcm_list_in_dir(var: str, path: str):
    files: list = glob.glob(f'{path:s}/{var:s}*nc')

    gcm = list(set([s.split('_')[2] for s in files]))

    gcm.sort()

    return gcm


def nc_mergetime(list_file: list, var: str, output_tag: str = 'mergetime', save: bool = True):
    """
    # since CDO mergetime function will lose the lon/lat by unknown reason,
    # I make a function in Python.
    :param output_tag: monthly, daily, e.g.
           use it to avoid over writing with this function is called in loop, usually use the month
    :param list_file:
    :param var:
    :param save:
    :return: a nc file merged, named as the 1st file in the list, with "mergetime.nc" in the end.
    """

    # read the first da
    da = read_to_standard_da(list_file[0], var)

    for i in range(len(list_file)):
        if i > 0:
            print(f'merging {i:g} of {len(list_file):g} ...')

            da1 = read_to_standard_da(list_file[i], var)

            da = xr.concat([da, da1], dim='time')

    da = da.sortby(da.time)

    if save:
        # save it to NetCDF file with the lon and lat (2D).
        output_name = f'{Path(list_file[0]).stem:s}.{output_tag:s}.nc'
        # output files may have the timestamp in the file name, that's the starting time of
        # the merged file
        input_dir = os.path.split(list_file[0])[0]

        # output to the same dir:
        da.to_netcdf(f'{input_dir:s}/{output_name:s}')
        print(f'saved to {input_dir:s}/{output_name:s}')

    return da


def ctang_convention():
    """
    print out my convention used in the analysis:

    Returns:
    """
    print(f'use units for attrs of DataArray... ref: ERA5 reanalysis')

    print(f'try to have name and units for any xr.DataArray')

    return 111


def convert_cmip6_ensemble_2_standard_da(
        var: str,
        ssp: str,
        freq: str,
        output_nc: str,
        year_start: int,
        year_end: int,
        raw_data_dir: str,
        output_dir: str,
        raw_file_tag: str,
        set_same_cal: bool = False,
        remapping: bool = False):
    """
    to read the raw cmip6 files, before mergetime, and do processes to save it to single netcdf file
    with the same lon lat, same calender.

    rules to follow: attention 1) make input and output separately in different dirs 2) temp file do not have .nc
    in the end.

    Args:
        output_nc ():
        set_same_cal ():
        remapping (bool):
        raw_file_tag ():
        year_end ():
        year_start ():
        var ():
        ssp ():
        raw_data_dir ():
        output_dir ():
        freq ():

    Returns:
        path of output file absolute

    """

    # definition:

    ensemble = 'r1i1p1f1'
    grid = 'gn'
    # ----------------------------- 1st, merge the data to 1970-2099.nc -----------------------------

    # attention this part is done with the code in ./src/
    # all model are merged with the same period, num of year for example

    # ----------------------------- 2nd, prepare the standard data xr.DataArray-----------------------------
    # with the same dims names

    # example of file: rsds_Amon_MPI-ESM1-2-LR_ssp585_r1i1p1f1_gn_2015-2099.year.global_mean.nc
    wildcard_raw_file = f'{raw_data_dir:s}/{var:s}*{freq:s}*{ssp:s}_{ensemble:s}_{grid:s}_' \
                        f'{year_start:g}-{year_end:g}*{raw_file_tag:s}.nc'

    raw_files: list = glob.glob(wildcard_raw_file)

    # gcm list
    gcm_list = list(set([a.split("_")[2] for a in raw_files]))
    gcm_list.sort()

    for ff in range(len(raw_files)):
        # attention: when using matching, careful that CESM2 ~ CESM2-XXX
        gcm_name = [model for model in gcm_list if raw_files[ff].find(f'_{model:s}_') > 0][0]
        # gcm_name = [model for model in gcm_list if model in raw_files[ff].split('_')][0]

        # find

        da = read_to_standard_da(raw_files[ff], var=var)
        da = da.assign_attrs(model_name='gcm_name')
        # here the name is saved to the final ensemble da
        # each name of gcm is in the coords of model_name (new dim)

        if not ff:
            lon = get_time_lon_lat_from_da(da)['lon']
            time = get_time_lon_lat_from_da(da)['time']
            lat = get_time_lon_lat_from_da(da)['lat']

        # create a new DataArray:
        new_da = xr.DataArray(data=da.values.astype(np.float32),
                              dims=('time', 'lat', 'lon'),
                              coords={'time': time, 'lat': lat, 'lon': lon},
                              name=var)
        new_da = new_da.assign_attrs({'model_name': gcm_name,
                                      'units': da.attrs['units'],
                                      'missing_value': np.NAN})

        # new_da = new_da.dropna(dim='latitude')

        if set_same_cal:
            # convert time to the same calendar:
            new_da = convert_da_to_360day_monthly(new_da)

        if remapping:
            # since the lon/lat are different in nc files, remap them to the same/smallest domain:
            # select smaller domain as "GERICS-REMO2015_v1"
            if ff == 0:
                ref_da = new_da.mean(axis=0)
            else:
                print(f'ref:', ref_da.shape, f'remap:', new_da.shape)
                new_da = value_remap_a_to_b(a=new_da, b=ref_da)
                print(new_da.shape)

        # temp data, in the raw_data dir

        temp_data = f'{raw_files[ff]:s}.temp'
        new_da.to_netcdf(temp_data)
        print(f'save to {temp_data:s}')

    # merge data from all models:
    da_list: List[xr.DataArray] = [xr.open_dataarray(file + '.temp') for file in raw_files]

    new_dim = [f'{aa.assign_attrs().model_name:s}' for aa in da_list]

    # noinspection PyTypeChecker
    ensemble_da = xr.concat(da_list, new_dim).rename({'concat_dim': 'model'})
    ensemble_da = ensemble_da.squeeze(drop=True)

    # rename attribute: model_name:
    ensemble_da = ensemble_da.assign_attrs(model_name='ensemble')

    # TODO: cdo sinfo do not works

    # make time the 1st dim
    # change order of dims so that cdo could read correctly
    # dims = list(ensemble_da.dims)
    # dims.remove('time')
    # new_order = ['time'] + dims

    ensemble_da = ensemble_da.transpose('time', 'model')

    # output file name, to save in data dir
    ensemble_da.to_netcdf(f'{output_dir:s}/{output_nc:s}')
    # ----------------------------- ok, clean -----------------------------
    # OK, remove the temp data:
    os.system(f'rm -rf *nc.temp')

    print(f'all done, the data is saved in {output_dir:s} as {output_nc:s}')

    return f'{output_dir:s}/{output_nc:s}'


def convert_cordex_ensemble_2_standard_da(
        var: str,
        domain: str,
        gcm: list,
        rcm: list,
        rcp: str,
        raw_data_dir: str,
        output_dir: str,
        output_tag: str,
        statistic: str,
        test: bool):
    """
    to read the original netcdf files, before mergetime, and do processes to save it to single netcdf file
    with the same lon lat, same calender.
    :param var:
    :type var:
    :param domain:
    :type domain:
    :param gcm:
    :type gcm:
    :param rcm:
    :type rcm:
    :param rcp:
    :type rcp:
    :param raw_data_dir:
    :type raw_data_dir:
    :param output_dir:
    :type output_dir:
    :param output_tag:
    :type output_tag:
    :param statistic:
    :type statistic:
    :param test:
    :type test:
    :return:
    :rtype:
    """

    # ----------------------------- 1st, merge the data to 1970-2099.nc -----------------------------
    # clean:
    # os.system(f'./local_data/{VAR:s}/merge.sfcWind.codex.hist.rcp85.sh -r')
    # merge:
    # os.system(f'./local_data/{VAR:s}/merge.sfcWind.codex.hist.rcp85.sh')
    # already done on CCuR
    # ----------------------------- 2nd, prepare the standard data xr.DataSet -----------------------------

    for window in ['1970-1999', '2036-2065', '2070-2099', '1970-2099']:

        # output file name, to save in data dir
        ensemble_file_output = f'{output_dir:s}/{var:s}/' \
                               f'{var:s}.{statistic:s}.{domain:s}.{rcp:s}.ensemble.{output_tag:s}.{window:s}.nc'

        raw_files: list = glob.glob(f'{raw_data_dir:s}/*{var:s}*{rcp:s}*{window:s}*.{output_tag:s}.{statistic:s}.nc')

        if len(raw_files) == 0:
            continue

        for ff in range(len(raw_files)):
            gcm_name = [model for model in gcm if raw_files[ff].find(model) > 0][0]
            rcm_name = [model for model in rcm if raw_files[ff].find(model) > 0][0]

            da = xr.open_dataset(raw_files[ff])
            da = da[var]
            da = da.assign_attrs(units='hour_per_month')

            if not ff:
                lon = get_time_lon_lat_from_da(da)['lon']
                time = get_time_lon_lat_from_da(da)['time']
                lat = get_time_lon_lat_from_da(da)['lat']

            # create a new DataArray:
            new_da = xr.DataArray(data=da.values.astype(np.float32), dims=('time', 'latitude', 'longitude'),
                                  coords={'time': time, 'latitude': lat, 'longitude': lon},
                                  name=var)
            new_da = new_da.assign_attrs({'model_id': rcm_name, 'driving_model_id': gcm_name,
                                          'units': da.attrs['units'],
                                          'missing_value': np.NAN})

            # new_da = new_da.dropna(dim='latitude')

            if test:
                print(ff, gcm_name, rcm_name)

            set_same_cal = 0
            remapping = 0

            if set_same_cal:
                # convert time to the same calendar:
                new_da = convert_da_to_360day_monthly(new_da)

            if remapping:
                # since the lon/lat are different in nc files, remap them to the same/smallest domain:
                # select smaller domain as "GERICS-REMO2015_v1"
                if ff == 0:
                    ref_da = new_da.mean(axis=0)
                else:
                    print(f'ref:', ref_da.shape, f'remap:', new_da.shape)
                    new_da = value_remap_a_to_b(a=new_da, b=ref_da)
                    print(new_da.shape)
                    print(new_da.shape)

            # temp data, in the raw_data dir
            output_file = f'{raw_data_dir:s}/' \
                          f'{var:s}.{statistic:s}.{domain:s}.{rcp:s}.{gcm_name:s}-{rcm_name:s}.{window:s}.nc.temp'
            new_da.to_netcdf(output_file)

            # ds = xr.merge([ds, new_da.to_dataset()])

        # merge data from all models:
        files_to_merge = f'{raw_data_dir:s}/{var:s}.{statistic:s}.{domain:s}.{rcp:s}.*.{window:s}.nc.temp'
        files = glob.glob(files_to_merge)
        da_list: List[xr.DataArray] = [xr.open_dataarray(file) for file in files]
        new_dim = [f'{aa.assign_attrs().driving_model_id:s}->{aa.assign_attrs().model_id:s}'
                   for aa in da_list]

        # noinspection PyTypeChecker
        ensemble_da = xr.concat(da_list, new_dim).rename({'concat_dim': 'model'})
        # change order of dims so that cdo could read correctly
        ensemble_da = ensemble_da.transpose('time', 'latitude', 'longitude', 'model')

        # drop nan dimension, created by cdo remap:
        ensemble_da = ensemble_da[:, 1:-1, 1:-1, :]
        ensemble_da = ensemble_da.dropna(dim='latitude')
        ensemble_da.to_netcdf(ensemble_file_output)

        # TODO: cdo sinfo do not works

    # ----------------------------- ok, clean -----------------------------
    # OK, remove the temp data:
    os.system(f'rm -rf *nc.temp')

    print(f'all done, the data {window:s} is in ./data/{var:s}/')


def plot_geo_map(data_map: xr.DataArray,
                 bias: int = 0,
                 grid: bool = 1,
                 grid_num: int = 10,
                 plt_limits: str = 'default',
                 cb_limits: str = 'default',
                 plt_type: str = 'pcolormesh',
                 suptitle_add_word: str = None):
    """
    
    Args:
        grid ():
        grid_num ():
        data_map ():
        bias (): 
        plt_limits (): 
        cb_limits (): 
        plt_type (): 
        suptitle_add_word (): 

    Returns:

    """

    fig = plt.figure(figsize=(8, 8), dpi=220)
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())

    if plt_limits == 'default':
        lon = data_map.lon.values
        lat = data_map.lat.values
        area = [
            np.min(lon), np.max(lon),
            np.min(lat), np.max(lat)]
    else:
        # limits = [x1, x2, y1, y2]
        area = plt_limits

    ax.set_extent(area, crs=ccrs.PlateCarree())
    # lon_left, lon_right, lat_north, lat_north

    ax.coastlines('50m')
    ax.add_feature(cfeature.LAND.with_scale('10m'))

    if bias == 1:
        how = 'anomaly_mean'
    else:
        how = 'time_mean'

    # if max and min is default, then use the value in my table:
    if cb_limits == 'default':
        vmax, vmin = np.max(data_map), np.min(data_map)
    else:
        if cb_limits == 'defined':
            vmax, vmin = value_max_min_of_var(var=str(data_map.name), how=how)
        else:
            vmin = cb_limits[0]
            vmax = cb_limits[1]

    cmap, norm = set_cbar(vmax=vmax, vmin=vmin, n_cbar=10, bias=bias)

    lon = get_time_lon_lat_from_da(data_map)['lon']
    lat = get_time_lon_lat_from_da(data_map)['lat']

    if plt_type == 'pcolormesh':
        cf = plt.pcolormesh(lon, lat, data_map, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    if plt_type == 'contourf':
        cf: object = ax.contourf(lon, lat, data_map, levels=norm.boundaries,
                                 cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='both')

    cbar_label = f'{data_map.name:s} ({data_map.assign_attrs().units:s})'
    # cb_ax = ax.add_axes([0.87, 0.2, 0.01, 0.7])
    # cb = plt.colorbar(cf, orientation='vertical', shrink=0.8, pad=0.05, label=cbar_label)
    cb = plt.colorbar(cf, orientation='horizontal', shrink=0.8, pad=0.1, label=cbar_label)

    if grid:
        ax.grid()

    # set lon and lat ----------------------------- set ticks
    reso = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2, 2.5, 3])
    reso_x = (area[1] - area[0]) / grid_num
    reso_x_plot = reso[(np.abs(reso - reso_x)).argmin()]

    reso_y = (area[3] - area[2]) / grid_num
    reso_y_plot = reso[(np.abs(reso - reso_y)).argmin()]

    ax.set_xticks(np.arange(
        (area[0] // reso_x_plot) * reso_x_plot,
        area[1] + reso_x_plot,
        reso_x_plot), crs=ccrs.PlateCarree())

    ax.set_yticks(np.arange(
        (area[2] // reso_y_plot) * reso_y_plot,
        area[-1] + reso_y_plot,
        reso_y_plot), crs=ccrs.PlateCarree())

    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    # ----------------------------- set ticks

    title = data_map.assign_attrs().long_name.replace(" ", "_")

    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    plt.suptitle(title)
    # tag: additional word added to suptitle

    plt.savefig(f'./plot/{data_map.name:s}.{title.replace(" ", "_"):s}.png', dpi=220)
    plt.show()
    print(f'got plot ')


def test_pixel_position_python():
    """
    to test if the plt.pcolormesh will plot the pixel with lon and lat in it's central point
    Returns:

    """

    # get era5 data:
    file_path = f'~/local_data/era5/u10/u10.hourly.era5.1999-2016.bigreu.local_daytime.nc'
    u = read_to_standard_da(file_path, 'u10')

    map_2d = u[5, 8:10, 5:7]
    plot_geo_map(data_map=map_2d, plt_limits=[55, 56, -21.5, -20.5],
                 cb_limits=[-1.2, 1.2])
    print(map_2d.lon, map_2d.lat)

    # test also pcolormesh in dataarray:
    map_2d.plot.pcolormesh()
    plt.show()

    print(f'conclusion: matplotlib will plot the lon and lat in the center of pixels')


def if_same_coords(map1: xr.DataArray, map2: xr.DataArray, coords_to_check=None):
    """
    return yes or not if 2 maps in the same coordinates
    :param coords_to_check:
    :type coords_to_check:
    :param map2:
    :param map1:
    :return:

    Parameters
    ----------
    coords_to_check : list of the dims to check
    """

    if coords_to_check is None:
        coords_to_check = ['lat', 'lon']

    coords1 = get_time_lon_lat_from_da(map1)
    coords2 = get_time_lon_lat_from_da(map2)

    possible_coords = coords_to_check

    for k in possible_coords:
        if k in coords1.keys() & k in coords2.keys():
            one = coords1[k]
            two = coords2[k]
            if_same: bool = (one == two).all()
        else:
            if_same = False

        same = False and if_same

    return same


def add_timezone_da(da: xr.DataArray, timezone_str: str):
    """
    this function is not yet tested
    Args:
        da ():
        timezone_str ():

    Returns:

    """
    times = pd.DatetimeIndex(da.time.values).tz_localize(tz=timezone_str).to_pydatetime()

    da['time'] = times
    print(f'time zone is {timezone_str:s}')

    return da


def read_to_standard_da(file_path: str, var: str, timezone_str: str = 'default'):
    """
    read da and change the dim names/order to time, lon, lat, lev, number
    note: the coords may have several dims, which will be reduced to one, if the coord is not changing
            according to other dims, by the function reduce_ndim_coord
    attention: if the lon and lat could not be reduce to 2D, or if the time could not be reduce to 1D,
               this function will be crash.
    note: order/name of output da are defined in function get_possible_standard_coords_dims

    attention: standard dim is 'time', 'y', 'x'. if da has no coords that's the default dims names
        - better to make it diff if the da has coords or not, so da.lon will always be coords not dims.
        - use the function convert_da_to_std_dim_coords_names

    Parameters
    ----------
    file_path :
    var :

    Returns
    -------

    Args:
        timezone_str (): not tested yet

    """

    ds = xr.open_dataset(file_path)

    da = ds[var]

    # change the order of dims: necessary
    da = convert_da_standard_dims_order(da)

    new_da = convert_da_to_std_dim_coords_names(da)

    if timezone_str != 'default':
        new_da = add_timezone_da(new_da, timezone_str)

    # ----------------------------- the following code is replace by the function convert_da_to_std_dim_coords_names
    # coords = get_time_lon_lat_from_da(da)
    #
    # possible_coords = get_possible_standard_coords_dims()
    # # ['time', 'lev', 'lat', 'lon', 'number']
    #
    # # prepare new coords with the order and standard names of function: get_possible_standard_coords_dims
    # new_coords = dict()
    # max_coords_ndim = 1
    # for d in possible_coords:
    #     if d in coords:
    #         new_coords[d] = coords[d]
    #         max_coords_ndim = max(max_coords_ndim, coords[d].ndim)
    #
    # # to prepare dims names for data and for coords
    # possible_dims_coords = get_possible_standard_coords_dims(name_for='dims', ndim=max_coords_ndim)  # 1d or 2d
    # possible_dims_data_1d = get_possible_standard_coords_dims(name_for='dims', ndim=1)
    #
    # # get new dim, for data (time, x, y) and for coords lon: [x,y] separately
    # new_dims_data = dict()
    # new_dims_coords = dict()
    # for d in possible_coords:
    #     if d in coords:
    #         new_dims_coords[d] = possible_dims_coords[possible_coords.index(d)]
    #         new_dims_data[d] = possible_dims_data_1d[possible_coords.index(d)]
    #
    # # prepare coords parameters for da:
    # coords_param = dict()
    # for cod, dim in zip(new_coords.keys(), new_dims_coords.values()):
    #     coords_param[cod] = (dim, new_coords[cod])
    #
    #     # example of coords_param:
    #     # kkk={"time": (new_coords['time']),
    #     # "lat": (['y', 'x'], new_coords['lat']),
    #     # "lon": (['y', 'x'], new_coords['lon'])}
    #
    # # create new da:
    # new_da = xr.DataArray(da.values,
    #                       dims=list(new_dims_data.values()),
    #                       coords=coords_param,
    #                       name=var, attrs=da.attrs)

    # if max_coords_ndim == 2:
    #     new_da = xr.DataArray(da.values,
    #                           dims=list(new_dims.values()),
    #                           coords={
    #                               "time": (new_coords['time']),
    #                               "lat": (['y', 'x'], new_coords['lat']),
    #                               "lon": (['y', 'x'], new_coords['lon']),
    #                           },
    #                           name=var, attrs=da.attrs)
    # --- the above code is replace by the function convert_da_to_std_dim_coords_names

    return new_da


def convert_da_to_std_dim_coords_names(da):
    """
    note: two dimensional coords not tested yet.

    Args:
        da ():

    Returns:

    """
    coords = get_time_lon_lat_from_da(da)

    possible_coords = get_possible_standard_coords_dims()
    # ['time', 'lev', 'lat', 'lon', 'number']

    # prepare new coords with the order and standard names of function: get_possible_standard_coords_dims
    new_coords = dict()
    max_coords_ndim = 1
    for d in possible_coords:
        if d in coords:
            new_coords[d] = coords[d]
            max_coords_ndim = max(max_coords_ndim, coords[d].ndim)

    # to prepare dims names for data and for coords
    possible_dims_coords = get_possible_standard_coords_dims(name_for='dims', ndim=max_coords_ndim)  # 1d or 2d
    possible_dims_data_1d = get_possible_standard_coords_dims(name_for='dims', ndim=1)

    # get new dim, for data (time, x, y) and for coords lon: [x,y] separately
    new_dims_data = dict()
    new_dims_coords = dict()
    for d in possible_coords:
        if d in coords:
            new_dims_coords[d] = possible_dims_coords[possible_coords.index(d)]
            new_dims_data[d] = possible_dims_data_1d[possible_coords.index(d)]

    # prepare coords parameters for da:
    coords_param = dict()
    for cod, dim in zip(new_coords.keys(), new_dims_coords.values()):
        coords_param[cod] = (dim, new_coords[cod])

        # example of coords_param:
        # kkk={"time": (new_coords['time']),
        # "lat": (['y', 'x'], new_coords['lat']),
        # "lon": (['y', 'x'], new_coords['lon'])}

    # create new da:
    new_da = xr.DataArray(da.values,
                          dims=list(new_dims_data.values()),
                          coords=coords_param,
                          name=da.name, attrs=da.attrs)

    # new_da = xr.DataArray(da.values.reshape(-1, 1, 1),
    #                       dims=list(new_dims_data.values()),
    #                       coords=coords_param,
    #                       name=da.name, attrs=da.attrs)

    # if max_coords_ndim == 2:
    #     new_da = xr.DataArray(da.values,
    #                           dims=list(new_dims.values()),
    #                           coords={
    #                               "time": (new_coords['time']),
    #                               "lat": (['y', 'x'], new_coords['lat']),
    #                               "lon": (['y', 'x'], new_coords['lon']),
    #                           },
    #                           name=var, attrs=da.attrs)

    return new_da


def read_csv_into_df_with_header(csv: str):
    """
    and also set datatimeindex csv has to have a column as 'DateTime'
    Parameters
    ----------
    csv :

    Returns
    -------
    pd.DataFrame

    """

    df = pd.read_csv(csv, na_values=['-9999'])

    df['DateTimeIndex'] = pd.to_datetime(df['DateTime'])

    df = df.set_index('DateTimeIndex')
    del df['DateTime']

    # TODO: test todo
    return df


def match_station_to_grid_data(df: pd.DataFrame, column: str, da: xr.DataArray):
    """
    matching in situ data, type csv with header, to gridded data in DataArray
    merge in space and time

    keyword: station, pixel, gridded, select, match, create
    note: the codes to get station values are lost. not found so far.


    Parameters
    ----------
    df : dataframe with DateTimeIndex
    column : column name of the values to match
    da :  make sure they're in the same timezone

    Returns
    -------
    xr.DataArray
    the output da will be in the same lon-lat grid,
    while the length of time dimension is the same as the in-situ data
    """
    # ----------------------------- read -----------------------------
    # loop in time
    dt_index_mf = df.index.drop_duplicates().dropna().sort_values()
    dt_index_mf = dt_index_mf.tz_localize(None)

    # initialize and make all the values as nan:
    matched_da = da.reindex(time=dt_index_mf)

    # do not touch the unit of the df
    matched_da = matched_da.assign_attrs(units='as original in df')

    matched_da = matched_da.where(matched_da.lon > 360)

    # rename the array with column name
    matched_da = matched_da.rename(column)

    for dt in range(len(dt_index_mf)):
        df_1 = df.loc[dt_index_mf[dt]]
        da_1 = da.sel(time=dt_index_mf[dt])

        for i in range(len(df_1)):
            lat = df_1.latitude[i]
            lon = df_1.longitude[i]
            da_sta = da_1.sel(lat=lat, lon=lon, method='nearest')
            df_sta = df_1.loc[(df_1['longitude'] == lon) & (df_1['latitude'] == lat)]

            nearest_lat = da_sta.lat
            nearest_lon = da_sta.lon

            # update value:
            matched_da.loc[dict(lon=nearest_lon, lat=nearest_lat, time=matched_da.time[dt])] \
                = float(df_sta[column])

            print(dt_index_mf[dt], f'lon={lon:4.2f}, lat={lat:4.2f}, {df_1.station_name[i]:s}')

    print(f'good')

    return matched_da


def plot_nothing(ax):
    # Hide axis
    # plt.setp(ax.get_xaxis().set_visible(False))
    # plt.setp(ax.get_yaxis().set_visible(False))

    # plt.setp(ax.get_xaxis().set_ticks([]))
    # plt.setp(ax.get_yaxis().set_ticks([]))

    # plt.setp(ax.get_yticklabels(),visible=False)
    # plt.setp(ax.get_xticklabels(),visible=False)

    # plt.tick_params(axis="x", which="both", bottom=False, top=False)
    # plt.tick_params(axis="y", which="both", left=False, right=False)
    # ax.tick_params(axis='both', which='both', length=0)

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    # Hide the right and top spines
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def plot_diurnal_cycle_maps_dataset(list_da: list, bias: int, var_list, hour_list, title: str,
                                    lonlatbox=None, comment='no comment'):
    """
    plot diurnal cycle from dataset, the dataArray may have different coords, vmax and vmin, just a function to show
    all the diurnal cycle
    :param comment:
    :type comment:
    :param bias:
    :type bias:
    :param list_da:
    :type list_da:
    :param title:
    :param lonlatbox: default area: SWIO_1km domain
    :param hour_list:
    :param var_list:
    :return:
    """

    if lonlatbox is None:
        lonlatbox = [54.8, 58.1, -21.9, -19.5]

    # ----------------------------- filtering data by season -----------------------------
    fig, axs = plt.subplots(ncols=len(hour_list), nrows=len(var_list), sharex='row', sharey='col',
                            figsize=(len(hour_list) * 2, len(var_list) * 4), dpi=220,
                            subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.85, bottom=0.02, top=0.8, wspace=0.1, hspace=0.1)

    ds = convert_multi_da_to_ds(list_da, var_list)

    vmin, vmax = get_min_max_ds(ds)
    # vmin = -600
    # vmax = 600

    for var in range(len(var_list)):
        print(f'plot var = {var + 1:g}')

        da = ds[var_list[var]]

        coords = get_time_lon_lat_from_da(da)

        # comment these two lines when using different cbar over subplots
        # vmax = float(da.max(skipna=True))
        # vmin = float(da.min(skipna=True))
        # vmax, vmin = GEO_PLOT.value_max_min_of_var(var=field.name, how=statistic)

        for h in range(len(hour_list)):
            hour = hour_list[h]
            print(f'hour = {hour:g}')

            hourly = da[da.indexes['time'].hour == hour][0]

            # ----------------------------- plotting -----------------------------
            if len(axs.shape) == 1:
                # if the input dataset has only one DataArray
                ax = axs[h]
            else:
                ax = axs[var, h]
            # set map
            ax.set_extent(lonlatbox, crs=ccrs.PlateCarree())
            # lon_left, lon_right, lat_north, lat_north
            ax.coastlines('50m')
            ax.add_feature(cfeature.LAND.with_scale('10m'))

            cmap, norm = set_cbar(vmax=vmax, vmin=vmin, n_cbar=20, bias=bias)
            cf = ax.contourf(coords['lon'], coords['lat'], hourly, levels=norm.boundaries,
                             cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='both')

            ax.text(0.02, 0.95, f'{var_list[var]:s}', fontsize=8,
                    horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
            if comment != 'no comment':
                ax.text(0.92, 0.95, f'{comment:s}', fontsize=8,
                        horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

            if var == 0:
                ax.set_title(f'{hour:g}H00')
            # ----------------------------- end of plot -----------------------------
            if h + 1 == len(hour_list):
                cax = inset_axes(ax,
                                 width="5%",  # width = 5% of parent_bbox width
                                 height="100%",  # height : 50%
                                 loc='lower left',
                                 bbox_to_anchor=(1.05, 0., 1, 1),
                                 bbox_transform=ax.transAxes,
                                 borderpad=0,
                                 )

                cbar_label = f'{var_list[var]:s} ({da.assign_attrs().units:s})'
                # ax.text(0.9, 0.9, cbar_label, ha='center', va='center', transform=ax.transAxes)

                cb = plt.colorbar(cf, orientation='vertical', cax=cax, shrink=0.8, pad=0.05, label=cbar_label)
                cb.ax.tick_params(labelsize=10)

    plt.suptitle(title)
    plt.savefig(f'./plot/{"-".join(var_list):s}.hourly_maps.png', dpi=200)
    plt.show()
    print(f'got plot ')


def plot_compare_2geo_maps(map1: xr.DataArray, map2: xr.DataArray, tag1: str = 'A', tag2: str = 'B',
                           suptitle_add_word: str = None):
    """
    to compare 2 geo-maps,
    :param suptitle_add_word:
    :type suptitle_add_word:
    :param tag2:
    :param tag1:
    :param map1: model, to be remapped if necessary
    :param map2: ref,
    :return:

    Parameters
    ----------
    suptitle_add_word :  str
    suptitle_add_word :  add word to the plot sup title

    """

    # to check the if remap is necessary:

    if not if_same_coords(map1, map2, coords_to_check=['lat', 'lon']):
        print(f'coords not same, have to perform remapping to compare...')
        map1 = value_remap_a_to_b(a=map1, b=map2)
        tag1 += f'_remap'

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex='row', sharey='col',
                             figsize=(10, 10), dpi=220, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.8, wspace=0.05)

    axes = axes.flatten()

    # ----------------------------- map 1 -----------------------------
    plot_geo_map(data_map=map1, bias=0,
                 # ax=axes[0],
                 vmax=max(map1.max(), map2.max()), vmin=min(map1.min(), map2.min()))
    axes[0].text(0.93, 0.95, tag1, fontsize=12,
                 horizontalalignment='right', verticalalignment='top', transform=axes[0].transAxes)

    axes[0].text(0.93, 0.05, f'mean: {map1.mean().values:4.2f}', fontsize=14,
                 horizontalalignment='right', verticalalignment='bottom', transform=axes[0].transAxes)

    # ----------------------------- map 2 -----------------------------
    plot_geo_map(data_map=map2, bias=0,
                 # ax=axes[1],
                 vmax=max(map1.max(), map2.max()), vmin=min(map1.min(), map2.min()))
    axes[1].text(0.93, 0.95, tag2, fontsize=12,
                 horizontalalignment='right', verticalalignment='top', transform=axes[1].transAxes)

    axes[1].text(0.93, 0.05, f'mean: {map2.mean().values:4.2f}', fontsize=14,
                 horizontalalignment='right', verticalalignment='bottom', transform=axes[1].transAxes)

    # ----------------------------- plot bias -----------------------------
    bias_map = xr.DataArray(map1.values - map2.values, coords=[map1.lat, map1.lon], dims=map1.dims,
                            name=map1.name, attrs={'units': map1.assign_attrs().units})
    plot_geo_map(data_map=bias_map, bias=1,
                 # ax=axes[2],
                 vmax=max(np.abs(bias_map.max()), np.abs(bias_map.min())),
                 vmin=min(-np.abs(bias_map.max()), -np.abs(bias_map.min())))
    axes[2].text(0.93, 0.95, f'{tag1:s}-{tag2:s}', fontsize=14,
                 horizontalalignment='right', verticalalignment='top', transform=axes[2].transAxes)

    axes[2].text(0.93, 0.05, f'mean: {bias_map.mean().values:4.2f}', fontsize=14,
                 horizontalalignment='right', verticalalignment='bottom', transform=axes[2].transAxes)

    # ----------------------------- plot bias in % -----------------------------
    bias_in_percent = xr.DataArray((map1.values - map2.values) / map2.values * 100,
                                   coords=[map1.lat, map1.lon],
                                   dims=map1.dims, name=map1.name, attrs={'units': f'%'})

    vmax = max(np.abs(bias_in_percent.max()), np.abs(bias_in_percent.min()))
    vmin = min(-np.abs(bias_in_percent.max()), -np.abs(bias_in_percent.min()))

    # set the max of %
    if vmax > 1000 or vmin < -1000:
        vmax = 1000
        vmin = -1000

    plot_geo_map(data_map=bias_in_percent, bias=1,
                 # ax=axes[3],
                 vmax=vmax, vmin=vmin)
    axes[3].text(0.93, 0.95, f'({tag1:s}-{tag2:s})/{tag2:s} %', fontsize=14,
                 horizontalalignment='right', verticalalignment='top', transform=axes[3].transAxes)

    axes[3].text(0.93, 0.05, f'mean: {bias_in_percent.mean().values:4.2f} %', fontsize=14,
                 horizontalalignment='right', verticalalignment='bottom', transform=axes[3].transAxes)

    date = convert_time_coords_to_datetime(map1).date()
    hour = convert_time_coords_to_datetime(map1).hour

    timestamp = str(date) + 'T' + str(hour)

    title = f'{tag1:s} vs {tag2:s}' + f' ({timestamp:s})'

    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    plt.suptitle(title)
    # tag: additional word added to suptitle

    plt.savefig(f'./plot/{map1.name:s}.{title.replace(" ", "_"):s}.{timestamp:s}.png', dpi=220)
    plt.show()
    print(f'got plot ')


def convert_df_shifttime(df: pd.DataFrame, second: int):
    """
    shift time by second
    Parameters
    ----------
    df :
    second :
    Returns
    -------
    """

    from datetime import timedelta

    time_shifted = df.index + timedelta(seconds=second)

    new_df = pd.DataFrame(data=df.values, columns=df.columns, index=time_shifted)

    return new_df


def convert_da_standard_dims_order(da: xr.DataArray):
    """
    read da and change the dim order to time, lon, lat, lev, number
    however, the names are not changed

    note: this function may take time when access values of da by da.values

    :param da:
    :type da:
    :return:
    :rtype:
    """

    dims_names = get_time_lon_lat_name_from_da(da, name_from='dims')

    dims_order = []

    possible_coords = get_possible_standard_coords_dims()
    # ['time', 'lev', 'lat', 'lon', 'number']

    possible_name = [x for x in possible_coords if x in dims_names.keys()]
    for i in range(len(dims_names)):
        dims_order.append(dims_names[possible_name[i]])

    new_da = da.transpose(*dims_order)

    return new_da


def convert_da_shifttime(da: xr.DataArray, second: int):
    """
    shift time by second
    Parameters
    ----------
    da :
    second :
    Returns
    -------
    """

    from datetime import timedelta

    coords = get_time_lon_lat_from_da(da)

    time_shifted = da.time.get_index('time') + timedelta(seconds=second)

    new_coords = dict(time=time_shifted)
    # possible_coords = ['lev', 'lat', 'lon']  # do not change the order
    # for i in range(len(possible_coords)):
    #     c = possible_coords[i]
    #     if c in coords:
    #         # my_dict['name'] = 'Nick'
    #         new_coords[c] = coords[c]

    # do not change the input dims and coords:

    if da.coords['lat'].ndim == 1:
        new_da = xr.DataArray(da.values, dims=da.dims, name=da.name, attrs=da.attrs)
        new_da = new_da.assign_coords(time=("time", time_shifted),
                                      lat=("y", da.lat.data), lon=("x", da.lon.data))

    if da.coords['lat'].ndim == 2:
        new_da = xr.DataArray(da.values,
                              dims=da.dims, name=da.name, attrs=da.attrs,
                              coords={
                                  "time": time_shifted,
                                  "lat": (['y', 'x'], da['lat'].data),
                                  "lon": (['y', 'x'], da['lon'].data)})

    return new_da


def convert_time_coords_to_datetime(da: xr.DataArray):
    """
    convert time coordinates in dataArray to datetime object
    :param da:
    :return:
    """

    dt_object = pd.Timestamp(da.time.values).to_pydatetime()

    return dt_object


def vis_a_vis_plot(x, y, xlabel: str, ylabel: str, title: str):
    """
    plot scatter plot
    :param title:
    :type title:
    :param xlabel:
    :param ylabel:
    :param x:
    :param y:
    :return:
    """

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), facecolor='w', edgecolor='k', dpi=200)  # figsize=(w,h)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.4, top=0.8, wspace=0.05)

    plt.scatter(x, y, marker='^', c='b', s=50, edgecolors='blue', alpha=0.8, label=ylabel)

    plt.title(title)

    # plt.legend(loc="upper right", markerscale=1, fontsize=16)

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(labelsize=16)

    plt.grid(True)

    plt.show()


# noinspection PyUnresolvedReferences
def plot_altitude_bias_by_month(df: pd.DataFrame, model_column: str, obs_column: str,
                                cbar_label: str,
                                bias=False):
    """
    plot station locations and their values
    :param obs_column:
    :type obs_column:
    :param df:
    :type df:
    :param cbar_label:
    :type cbar_label:
    :param model_column:
    :type model_column:
    :param bias:
    :return: map show
    """
    import matplotlib as mpl

    # data:
    df['bias'] = df[model_column] - df[obs_column]

    months = [11, 12, 1, 2, 3, 4]
    month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']
    # nrows = len(months)

    plt.figure(figsize=(5, 24), dpi=200)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=1.5, top=0.95, wspace=0.05)

    # plot in each month:
    for m in range(len(months)):
        ax = plt.subplot(len(months), 1, m + 1)

        # data:
        monthly_data = df[df.index.month == months[m]]
        station_group = monthly_data.groupby('station_id')
        station_mean_bias = station_group[['bias']].mean().values[:, 0]
        # station_mean_height = station_group[['altitude']].mean().values[:, 0]

        lon = df['longitude']
        lat = df['latitude']

        # ----------------------------- cbar -----------------------------
        if np.max(station_mean_bias) - np.min(station_mean_bias) < 10:
            round_number = 2
        else:
            round_number = 0

        n_cbar = 10
        vmin = round(np.min(station_mean_bias) / n_cbar, round_number) * n_cbar
        vmax = round(np.max(station_mean_bias) / n_cbar, round_number) * n_cbar

        if bias:
            cmap = plt.cm.coolwarm
            vmin = max(np.abs(vmin), np.abs(vmax)) * (- 1)
            vmax = max(np.abs(vmin), np.abs(vmax))
        else:
            cmap = plt.cm.YlOrRd

        # human chosen values
        # vmin = -340
        # vmax = 340

        bounds = np.linspace(vmin, vmax, n_cbar + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        sc = plt.scatter(lon, lat, c=station_mean_bias, edgecolor='black',
                         # transform=ccrs.PlateCarree(),
                         zorder=2, norm=norm, vmin=vmin, vmax=vmax, s=50, cmap=cmap)

        # ----------------------------- color bar -----------------------------
        cb = plt.colorbar(sc, orientation='horizontal', shrink=0.7, pad=0.1, label=cbar_label)
        cb.ax.tick_params(labelsize=10)

        ax.gridlines(draw_labels=False)

        ax.text(0.93, 0.95, month_names[m],
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    plt.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.6)
    plt.suptitle(f'monthly daytime (8h 17h) \n mean bias at MeteoFrance stations')

    plt.show()
    print(f'got plot')


def plot_scatter_contourf(lon: xr.DataArray, lat: xr.DataArray, cloud: xr.DataArray, cbar_label: str,
                          lon_mf: np.ndarray, lat_mf: np.ndarray, value: pd.DataFrame, cbar_mf: str,
                          bias_mf=True):
    """
    to plot meteofrance stationary value and a color filled map.

    :param value:
    :type value:
    :param bias_mf:
    :type bias_mf:
    :param cbar_mf:
    :param lat_mf:
    :param lon_mf:
    :param cloud:
    :param cbar_label: label of color bar
    :param lon:
    :param lat:
    :return: map show
    """

    import matplotlib as mpl
    import datetime as dt

    hours = [x for x in range(8, 18, 1)]
    # dates = ['2004-11-01', '2004-12-01', '2005-01-01', '2005-02-01', '2005-03-01', '2005-04-01']
    # month_names = ['Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr']

    plt.figure(figsize=(10, 20), dpi=300)
    # fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=1.5, top=0.95, wspace=0.05)

    # plot in each hour:
    for h in range(len(hours)):

        # noinspection PyTypeChecker
        ax = plt.subplot(len(hours) / 2, 2, h + 1, projection=ccrs.PlateCarree())

        print(f'plot hour = {hours[h]:g}')

        # ----------------------------- mean cloud fraction -----------------------------
        # data:

        hourly_cloud = cloud.sel(time=dt.time(hours[h])).mean(axis=0)

        # set map
        reu = value_lonlatbox_from_area('reu')
        ax.set_extent(reu, crs=ccrs.PlateCarree())
        # lon_left, lon_right, lat_north, lat_north

        ax.coastlines('50m')
        ax.add_feature(cfeature.LAND.with_scale('10m'))

        # Plot Color fill of hourly mean cloud fraction
        # normalize color to not have too dark of green at the top end
        clevs = np.arange(60, 102, 2)
        cf = ax.contourf(lon, lat, hourly_cloud, clevs, cmap=plt.cm.Greens,
                         norm=plt.Normalize(60, 102), transform=ccrs.PlateCarree())

        # cb = plt.colorbar(cf, orientation='horizontal', pad=0.1, aspect=50)
        plt.colorbar(cf, orientation='horizontal', shrink=0.7, pad=0.05, label=cbar_label)
        # cb.set_label(cbar_label)

        # ----------------------------- hourly mean bias wrf4.1 - mf -----------------------------

        # data:

        hourly_bias = value[value.index.hour == hours[h]]
        hourly_bias = hourly_bias.groupby('station_id').mean().values.reshape((37,))

        vmax = 240
        vmin = vmax * -1

        if bias_mf:
            cmap = plt.cm.coolwarm
            vmin = max(np.abs(vmin), np.abs(vmax)) * (- 1)
            vmax = max(np.abs(vmin), np.abs(vmax))
        else:
            cmap = plt.cm.YlOrRd

        n_cbar = 20

        bounds = np.linspace(vmin, vmax, n_cbar + 1)
        # noinspection PyUnresolvedReferences
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # ----------------------------------------------------------
        sc = plt.scatter(lon_mf, lat_mf, c=hourly_bias, edgecolor='black',
                         # transform=ccrs.PlateCarree(),
                         zorder=2, norm=norm, vmin=vmin, vmax=vmax, s=50, cmap=cmap)

        # ----------------------------- color bar -----------------------------
        cb = plt.colorbar(sc, orientation='vertical', shrink=0.7, pad=0.05, label=cbar_mf)
        cb.ax.tick_params(labelsize=10)

        # ----------------------------- end of plot -----------------------------
        ax.xaxis.set_ticks_position('top')

        ax.gridlines(draw_labels=False)

        ax.text(0.98, 0.95, f'{hours[h]:g}h00\nDJF mean\n2004-2005', horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes)

        ax.text(0.01, 0.16, f'0.05x0.05 degree\nMVIRI/SEVIRI on METEOSAT',
                horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    # plt.tight_layout(pad=0.1, w_pad=0.3, h_pad=0.6)
    plt.suptitle(f'cloud fraction at daytime hours \n as mean of DJF during 2004 - 2005')

    plt.show()
    print(f'got plot')


def calculate_climate_rolling_da(da: xr.DataArray):
    """
    calculate 30 year rolling mean
    Parameters
    ----------
    da :

    Returns
    -------
    mean
    """
    year = da.groupby(da.time.dt.year).mean(dim='time').rename({'year': 'time'})

    mean: xr.DataArray = year.rolling(time=30, center=True).mean().dropna("time")

    return mean


def get_lon_lat_from_area(area: str):
    """
    get the lon and lat of this position
    Args:
        area ():

    Returns:
        lon, lat

    """

    if area == 'reu':
        lon = 55.5
        lat = -21.1

    return lon, lat


def value_elevation_from_lonlat(lon, lat):
    import geocoder
    g = geocoder.elevation([lat, lon])
    print(g.meters)

    return g.meters


def value_lon_lat_from_address(location: str = 'saint denis, reunion'):
    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="re")

    location = geolocator.geocode("st denis, reunion")

    return location.longitude, location.latitude


def value_humidity_specific_era5_Bolton(surface_pressure: xr.DataArray,
                                        dew_point_2m_temp: xr.DataArray,
                                        test: bool = 0):
    # Compute the Specific Humidity(Bolton 1980):
    # https://archive.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html
    # e = 6.112 * exp((17.67 * Td) / (Td + 243.5));
    # q = (0.622 * e) / (p - (0.378 * e));
    # where:
    # e = vapor pressure in mb;
    # Td = dew point in deg C;
    # p = surface pressure in mb;
    # q = specific humidity in kg / kg.

    # using ERA5 single level data.

    # check input units:

    if test:
        # use a slice of data
        dew_point_2m_temp = dew_point_2m_temp[0:100]
        surface_pressure = surface_pressure[0:100]

    # change units
    dew_point_2m_temp = dew_point_2m_temp - 273.5
    dew_point_2m_temp = dew_point_2m_temp.assign_attrs({'unit': 'C'})

    surface_pressure = surface_pressure / 100
    surface_pressure = surface_pressure.assign_attrs({'unit': 'mb'})

    # e = 6.112 * exp((17.67 * Td) / (Td + 243.5));
    e = 6.112 * np.exp((17.67 * dew_point_2m_temp) / (dew_point_2m_temp + 243.5))
    # q = (0.622 * e) / (p - (0.378 * e));
    q = (0.622 * e) / (surface_pressure - (0.378 * e))

    q = q.rename('specific_humidity')
    q = q.assign_attrs({
        'units': 'kg/kg',
        'long_name': 'specific humidity',
        'name': 'specific_humidity',
        'data': 'era5 single level'
    })

    return q


def value_reso_from_da(grid: xr.DataArray):
    # get resolution from grid
    lat = grid.lat.values
    lon = grid.lon.values

    from decimal import Decimal
    reso_lat = lat[1:] - lat[:-1]
    reso_lat_list = [Decimal(x).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP") for x in reso_lat]
    reso_lat = np.float16(list(set(reso_lat))[0])

    reso_lon = lon[1:] - lon[:-1]
    reso_lon_list = [Decimal(x).quantize(Decimal("0.01"), rounding="ROUND_HALF_UP") for x in reso_lon]
    reso_lon = np.float16(list(set(reso_lon))[0])

    if reso_lat == reso_lon:
        return reso_lat
    else:
        return {'lat': reso_lat, 'lon': reso_lon}


def plot_topo_mauritius_high_reso(plot: bool = True, grid: xr.DataArray = None,
                                  plot_max: bool = True,
                                  add_point: list = None,
                                  vmax=100, output_tag: str = ''):
    # The map is based on the ASTER Global Digital Elevation Model
    # from NASA Jet Propulsion Laboratory

    file1 = f'~/local_data/topo/ASTGTMV003_S20E057_dem.nc'
    file2 = f'~/local_data/topo/ASTGTMV003_S21E057_dem.nc'

    ref1 = read_to_standard_da(file1, 'ASTER_GDEM_DEM')
    ref2 = read_to_standard_da(file2, 'ASTER_GDEM_DEM')

    ref = xr.concat([ref1, ref2[1:, :]], dim='y')
    ref = ref.rename({'x': 'lon', 'y': 'lat'})

    land = ref.where(ref != 0)

    geomap = land

    # -------------------------------------------------------------------
    # plot:

    cmap = plt.cm.terrain

    fig = plt.figure(figsize=(8, 6), facecolor='w', edgecolor='k', dpi=300)
    ax = plt.subplot(111, projection=ccrs.PlateCarree())

    west = 57.2
    east = 57.9
    north = -19.9
    south = -20.6

    ax.set_extent([west, east, north, south], crs=ccrs.PlateCarree())

    cmap, norm = set_cbar(vmax=vmax, vmin=0, n_cbar=20, cmap=cmap, bias=0)

    print(f'max = {geomap.max().values:4.2f}')
    cf = plt.pcolormesh(geomap.lon, geomap.lat, geomap,
                        cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    plt.xticks(np.arange(west, east, 0.1))
    plt.yticks(np.arange(south, north, 0.1))

    cb = plt.colorbar(cf, orientation='vertical', shrink=0.9, pad=0.05)
    cb.ax.tick_params(labelsize=14)

    cbar_label = f'elevation (meter)'
    cb.set_label(label=cbar_label, size=14)

    if plot_max:
        marker = '^'
        top = geomap.where(geomap == geomap.max(), drop=True)
        lat_max = top.coords['lat'].values[0]
        lon_max = top.coords['lon'].values[0]
        plt.scatter(lon_max, lat_max, marker=marker, s=80, c='r', edgecolor='k')

    if add_point:
        plt.scatter(add_point[0], add_point[1], marker=add_point[2], s=50, c='green', edgecolor='white')

    if grid is not None:
        # plot grid:
        reso = value_reso_from_da(grid)

        lon = np.round(grid.lon.values, decimals=2)
        lat = np.round(grid.lat.values, decimals=2)

        lon_grid = list(lon - reso * 0.5)
        lon_grid.append(lon[-1] + 0.5 * reso)

        lat_grid = list(lat - reso * 0.5)
        lat_grid.append(lat[-1] + 0.5 * reso)

        # plot grid lines:
        plt.hlines(lat_grid, xmin=0, xmax=100, linestyle='--', color='gray', linewidth=0.5)
        plt.vlines(lon_grid, ymin=-30, ymax=0, linestyle='--', color='gray', linewidth=0.5)

    plt.savefig(f'./plot/reu.topo.{output_tag:s}.png', dpi=300)
    plt.show()

    print(f'done')


def plot_topo_mauritius_high_reso(plot: bool = True, grid: xr.DataArray = None,
                                  plot_max: bool = True,
                                  add_point: list = None,
                                  vmax=100, output_tag: str = ''):
    # The map is based on the ASTER Global Digital Elevation Model
    # from NASA Jet Propulsion Laboratory
    # attention: if dpi=300, it takes 1 hour to plot. use dpi=220, then 1 minute

    file1 = f'~/local_data/topo/ASTGTMV003_S20E057_dem.nc'
    file2 = f'~/local_data/topo/ASTGTMV003_S21E057_dem.nc'

    ref1 = read_to_standard_da(file1, 'ASTER_GDEM_DEM')
    ref2 = read_to_standard_da(file2, 'ASTER_GDEM_DEM')

    ref = xr.concat([ref1, ref2[1:, :]], dim='y')
    ref = ref.rename({'x': 'lon', 'y': 'lat'})

    land = ref.where(ref != 0)

    geomap = land

    fontsize = 12
    # -------------------------------------------------------------------
    # plot:

    cmap = plt.cm.terrain

    fig = plt.figure(figsize=(8, 6), facecolor='w', edgecolor='k', dpi=300)
    ax = plt.subplot(111, projection=ccrs.PlateCarree())

    west = 57.2
    east = 57.9
    north = -19.9
    south = -20.6

    ax.set_extent([west, east, north, south], crs=ccrs.PlateCarree())

    cmap, norm = set_cbar(vmax=vmax, vmin=0, n_cbar=20, cmap=cmap, bias=0)

    print(f'max = {geomap.max().values:4.2f}')
    print(f'waiting ...')

    cf = plt.pcolormesh(geomap.lon, geomap.lat, geomap,
                        cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    plt.xticks(np.arange(west, east, 0.1))
    plt.yticks(np.arange(south, north, 0.1))

    cb = plt.colorbar(cf, orientation='vertical', shrink=0.9, pad=0.05)
    cb.ax.tick_params(labelsize=14)

    cbar_label = f'elevation (meter)'
    cb.set_label(label=cbar_label, size=14)

    plt.xlabel('Longitude', fontsize=fontsize)
    plt.ylabel('Latitude', fontsize=fontsize)

    if plot_max:
        marker = '^'
        max_value = geomap.max()
        top = geomap.where(geomap == max_value, drop=True)
        plt.scatter(top.lon, top.lat, marker=marker, s=80, c='r', edgecolor='k',
                    label=f'summit @{int(max_value):g}m')

    if add_point:
        plt.scatter(add_point[0], add_point[1], marker=add_point[2], s=80,
                    c='green', edgecolor='white', label=add_point[3])

    plt.legend(fontsize=fontsize)

    if grid is not None:
        # plot grid:
        reso = value_reso_from_da(grid)

        lon = np.round(grid.lon.values, decimals=2)
        lat = np.round(grid.lat.values, decimals=2)

        lon_grid = list(lon - reso * 0.5)
        lon_grid.append(lon[-1] + 0.5 * reso)

        lat_grid = list(lat - reso * 0.5)
        lat_grid.append(lat[-1] + 0.5 * reso)

        # plot grid lines:
        plt.hlines(lat_grid, xmin=0, xmax=100, linestyle='--', color='gray', linewidth=0.5)
        plt.vlines(lon_grid, ymin=-30, ymax=0, linestyle='--', color='gray', linewidth=0.5)

    plt.savefig(f'./plot/reu.topo.{output_tag:s}.png', dpi=300)
    plt.show()

    print(f'done')


def load_reunion_coastline():
    csv = '~/local_data/topo/reu/coastline_reu.csv'
    return pd.read_csv(csv)

def get_coastline_from_topo_reu(plot: bool = True, csv:str ='~/local_data/topo/reu/coastline_reu.csv'):
    from scipy.ndimage import generic_filter

    # get topo into DataArray:

    file1 = f'~/local_data/topo/reu/ASTGTMV003_S21E055_dem.nc'
    file2 = f'~/local_data/topo/reu/ASTGTMV003_S22E055_dem.nc'

    ref1 = read_to_standard_da(file1, 'ASTER_GDEM_DEM')
    ref2 = read_to_standard_da(file2, 'ASTER_GDEM_DEM')

    ref = xr.concat([ref1, ref2[1:, :]], dim='y')
    ref = ref.rename({'x': 'lon', 'y': 'lat'})

    land = ref.where(ref != 0)

    geomap = land

    # a func of filter
    def find_pixels_with_nan_and_positive_neighbours(data_array):
        def condition(arr):
            return np.isnan(arr).any() and (arr > 0).any()

        # Apply the condition to each pixel using a 3x3 neighborhood
        mask = generic_filter(data_array, condition, size=(3, 3), mode='constant', cval=np.nan)

        return xr.DataArray(mask, coords=data_array.coords, dims=data_array.dims)

    coastline_pixels = find_pixels_with_nan_and_positive_neighbours(geomap)

    def get_lon_lat_pairs_with_value(data_array, value=1):
        # Find indices where the pixel value is equal to the specified value
        indices = np.argwhere(data_array.values == value)

        # Extract lon and lat coordinates based on the indices
        lon_values = data_array.lon.values[indices[:, 1]]
        lat_values = data_array.lat.values[indices[:, 0]]

        # Create a list of (lon, lat) pairs
        lon_lat_pairs = list(zip(lon_values, lat_values))

        return np.array(lon_lat_pairs)
        # ----

    coastline = get_lon_lat_pairs_with_value(coastline_pixels, value=1)

    df = pd.DataFrame(coastline)  # A is a numpy 2d array
    C = ['longitude', 'latitude']
    df.to_csv(csv, header=C, index=False)  # C is

    if plot:
        fig = plt.figure(figsize=(8, 8), dpi=220)
        plt.scatter(coastline[:, 0], coastline[:, 1], marker='o', s=1, c='k', edgecolor='k')
        plt.show()

    return df


def plot_topo_reunion_high_reso(plot: bool = True, grid: xr.DataArray = None,
                                plot_max: bool = True,
                                add_point: list = None,
                                dpi: int = 100,
                                plot_wind: bool = False,
                                vmax=100, output_tag: str = ''):
    # The map is based on the ASTER Global Digital Elevation Model
    # from NASA Jet Propulsion Laboratory
    # attention: if dpi=300, it takes 1 hour to plot. use dpi=220, then 1 minute

    file1 = f'~/local_data/topo/reu/ASTGTMV003_S21E055_dem.nc'
    file2 = f'~/local_data/topo/reu/ASTGTMV003_S22E055_dem.nc'

    ref1 = read_to_standard_da(file1, 'ASTER_GDEM_DEM')
    ref2 = read_to_standard_da(file2, 'ASTER_GDEM_DEM')

    ref = xr.concat([ref1, ref2[1:, :]], dim='y')
    ref = ref.rename({'x': 'lon', 'y': 'lat'})

    land = ref.where(ref != 0)

    geomap = land

    fontsize = 12
    # -------------------------------------------------------------------
    # plot:

    cmap = plt.cm.terrain

    fig = plt.figure(figsize=(8, 6), facecolor='w', edgecolor='k', dpi=dpi)
    ax = plt.subplot(111, projection=ccrs.PlateCarree())

    west = 55.05
    east = 56
    north = -20.7
    south = -21.55

    west = 55.
    east = 56.3
    north = -20.3
    south = -22.

    ax.set_extent([west, east, north, south], crs=ccrs.PlateCarree())

    cmap, norm = set_cbar(vmax=vmax, vmin=0, n_cbar=20, cmap=cmap, bias=0)

    print(f'max = {geomap.max().values:4.2f}')
    print(f'waiting ...')

    # cf = plt.pcolormesh(geomap.lon, geomap.lat, geomap,
    #                     cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    plt.xticks(np.arange(west, east, 0.1))
    plt.yticks(np.arange(south, north, 0.1))

    # cb = plt.colorbar(cf, orientation='vertical', shrink=0.9, pad=0.05)
    # cb.ax.tick_params(labelsize=fontsize)

    # cbar_label = f'elevation (meter)'
    # cb.set_label(label=cbar_label, size=fontsize)

    plt.xlabel('Longitude ($^\circ$E)', fontsize=fontsize)
    plt.ylabel('Latitude ($^\circ$N)', fontsize=fontsize)

    if plot_max:
        marker = '^'
        max_value = geomap.max()
        top = geomap.where(geomap == max_value, drop=True)
        plt.scatter(top.lon, top.lat, marker=marker, s=80, c='r', edgecolor='k',
                    label=f'summit @{int(max_value):g}m')

    if add_point:
        plt.scatter(add_point[0], add_point[1], marker=add_point[2], s=80,
                    c='green', edgecolor='white', label=add_point[3])

    plt.legend(fontsize=fontsize)

    if grid is not None:
        # plot grid:
        reso = value_reso_from_da(grid)

        lon = np.round(grid.lon.values, decimals=2)
        lat = np.round(grid.lat.values, decimals=2)

        lon_grid = list(lon - reso * 0.5)
        lon_grid.append(lon[-1] + 0.5 * reso)

        lat_grid = list(lat - reso * 0.5)
        lat_grid.append(lat[-1] + 0.5 * reso)

        # plot grid lines:
        plt.hlines(lat_grid, xmin=0, xmax=100, linestyle='--', color='gray', linewidth=0.5)
        plt.vlines(lon_grid, ymin=-30, ymax=0, linestyle='--', color='gray', linewidth=0.5)

    # ----------------------------- surface wind -----------------------------
    if plot_wind:
        print(f'plot surface wind ...')

        print(f'loading wind data ... ')

        # local_data = '/Users/ctang/local_data/era5'
        local_data = '~/Microsoft_OneDrive/OneDrive/CODE/Mialhe_2021/local_data'
        file = f'{local_data:s}/u10.v10.big_reunion.1979-2022.monmean.timmean.nc'
        u = read_to_standard_da(file, 'u10')
        v = read_to_standard_da(file, 'v10')

        # u = anomaly_daily(u)
        # v = anomaly_daily(v)

        # speed = np.sqrt(u10 ** 2 + v10 ** 2)
        # speed = speed.rename('10m_wind_speed').assign_coords({'units': u10.attrs['units']})

        # Set up parameters for quiver plot. The slices below are used to subset the data (here
        # taking every 4th point in x and y). The quiver_kwargs are parameters to control the
        # appearance of the quiver so that they stay consistent between the calls.

        u_1 = u[0, :, :]
        v_1 = v[0, :, :]

        plot_wind_subplot(area='bigreu',
                          lon=u_1.lon, lat=v_1.lat,
                          u=u_1, v=v_1,
                          ax=ax, bias=0)
        # by default plot mean circulation, not anomaly

        wind_tag = f'surface_wind'
    else:
        wind_tag = 'no_wind'

    # ----------------------------- end of plot -----------------------------

    print(f'print output ...')

    plt.savefig(f'./plot/reu.topo.{output_tag:s}.{wind_tag:s}.png', dpi=dpi)
    plt.show()

    print(f'done')


def value_altitude_from_lonlat_reunion(lon: np.ndarray, lat: np.ndarray,
                                       method: str = 'linear',
                                       show: bool = True):
    """
    interpolate the altitude maps from ASTER GDEM V3,
    note: works only for Reunion

    Args:
        method ():
        lon ():
        lat ():
        show ():

    Returns:
        da map

    """
    # read ref
    # The map is based on the ASTER Global Digital Elevation Model
    # from NASA Jet Propulsion Laboratory

    file1 = f'~/local_data/topo/ASTGTMV003_S21E055_dem.nc'
    file2 = f'~/local_data/topo/ASTGTMV003_S22E055_dem.nc'

    ref1 = read_to_standard_da(file1, 'ASTER_GDEM_DEM')
    ref2 = read_to_standard_da(file2, 'ASTER_GDEM_DEM')

    ref = xr.concat([ref1, ref2[1:, :]], dim='y')

    ref = ref.rename({'x': 'lon', 'y': 'lat'})

    interpolated = ref.interp(lon=lon, lat=lat, method=method, kwargs={"fill_value": "extrapolate"})

    new_da = xr.DataArray(data=interpolated.values,
                          dims=('lat', 'lon'),
                          coords={'lat': lat, 'lon': lon},
                          attrs={
                              'units': 'meters',
                              'grid_mapping': 'crs',
                              'standard_name': 'altitude'},
                          name='altitude')
    if show:
        new_da.plot()
        plt.show()

    # always keep the std dim and coords names:
    da = convert_da_to_std_dim_coords_names(new_da)

    return da


def value_aod_reunion(times: pd.DatetimeIndex, wavelength: float = 700):
    """

    Args:
        wavelength ():
        times ():

    Returns:

    """

    aod_file = f'~/local_data/AERONET/aod.aeronet.reunion.csv'

    aod = read_csv_into_df_with_header(aod_file)

    # not finished yet

    return 1


def value_clearsky_radiation(
        times: pd.DatetimeIndex,
        lon: np.ndarray,
        lat: np.ndarray,
        model: str = 'climatology',
        show: bool = 1):
    import pvlib
    from pvlib.location import Location

    # ----------------------------- definition -----------------------------
    if model == 'climatology':
        clearsky_model = 'ineichen'  # ineichen with climatology table by default
    if model == 'local_atmosphere':
        clearsky_model = 'simplified_solis'

        # prepare AOD
        # aod_df = value_aod_reunion(times)

        # prepare total column water vapour
        # wv_da = value_total_column_water_vapour(times, lon=lon, lat=lat)

    # prepare time
    import timezonefinder
    tf = timezonefinder.TimezoneFinder()
    timezone_str = tf.closest_timezone_at(lat=lat.mean(), lng=lon.mean())
    times_tz = pd.DatetimeIndex(times).tz_localize(tz=timezone_str)
    # tz is required for clear_sky calculation.
    # while it's better to output ds/da without tz, since is not natively supported by pandas and/so xarray

    # prepare altitude
    altitude_da = value_altitude_from_lonlat_reunion(lon=lon, lat=lat)

    nd = np.zeros((len(times), 3, len(lat), len(lon)))

    for i in range(len(lon)):
        for j in range(len(lat)):
            print(f'calculate clearsky radiation at each pixel: lon={lon[i]:4.2f}, lat={lat[j]:4.2f}, '
                  f'altitude={altitude_da.values[j, i]:4.2f}')
            # pixel = Location(lon[i], lat[j], timezone_str, name=name)
            pixel = Location(longitude=lon[i], latitude=lat[j], altitude=altitude_da[j, i].values, tz=timezone_str)
            cs: pd.DataFrame = pixel.get_clearsky(times_tz, model=clearsky_model)
            nd[:, :, j, i] = cs.values

    ds = xr.Dataset(
        data_vars=dict(
            ghi=(["time", "y", "x"], nd[:, 0, :, :]),
            ndi=(["time", "y", "x"], nd[:, 1, :, :]),
            dhi=(["time", "y", "x"], nd[:, 2, :, :]),
        ),
        coords=dict(
            time=times,
            lat=(["y"], lat),
            lon=(["x"], lon),
            # keep the good order
        ),
        attrs=dict(description="clearsky radiation from pvlib",
                   units="W/m2"),
    )

    if show:
        print(f'plot the last point and last 72 timestep...')
        cs[0:24].plot()
        plt.ylabel('Irradiance $W/m^2$')
        plt.grid()
        plt.show()

    return ds


def value_lonlatbox_from_area(area: str):
    """
    get lonlat box of an area by names
    :param area:
    :return: list
    """

    if area == 'southern_Africa':
        box = [0, 59, -40, 1]

    if area == 'AFR-22':
        box = [-24, 59, -47.5, 43.8]

    if area == 'SA_swio':
        box = [0, 90, -50, 10]

    if area == 'cyc_swio':
        box = [25, 90, -40, -5]

    if area == 'cyc_swio_big':
        box = [25, 90, -40, 0]

    if area == 'reu':
        box = [55.05, 56., -21.55, -20.7]

    if area == 'bigreu':
        box = [54, 57, -22, -20]

    if area == 'small_reu':
        box = [55.2, 55.9, -21.4, -20.8]

    if area == 'swio':
        box = [20, 110, -50, 9]

    if area == 'reu_mau':
        box = [52, 60, -17.9, -23]

    if area == 'swio-domain':
        box = [32, 76, -34, 4]

    if area == 'd01':
        box = [41, 70, -33, -6]

    if area == 'd02':
        box = [53.1, 60, -22.9, -18.1]

    if area == 'reu-mau':
        box = [52.1, 59.9, -22.9, -18.33]

    if area == 'd_1km':
        box = [54.8, 58.1, -21.9, -19.5]

    if area == 'detect':
        box = [44, 64, -28, -12]

    if area == 'm_r_m':
        box = [40, 64, -30, -10]

    return box


def cluster_mean_gaussian_mixture(var_history, n_components, max_iter, cov_type):
    """
    input days with similar temp profile, return a dataframe with values = most common cluster mean.

    :param var_history: pd.DateFrame
    :param n_components:
    :param max_iter:
    :param cov_type:
    :return: pd.DateFrame of DateTimeIndex
    """

    from sklearn.mixture import GaussianMixture

    # clustering by Gaussian Mixture
    gm = GaussianMixture(n_components=n_components, max_iter=max_iter, covariance_type=cov_type)

    var_clusters = gm.fit(var_history)

    cluster_mean = var_clusters.means_

    labels = gm.predict(var_history)

    return cluster_mean, labels


def plot_daily_cluster_mean(mean, locations, labels, ylabel, title):
    fig = plt.figure(figsize=(10, 6), dpi=220)
    # fig.suptitle(fig_title)

    ax = fig.add_subplot(1, 2, 1)
    ax.set_aspect(aspect=0.015)
    # ----------------------------- plotting -----------------------------

    colors = ['blue', 'red', 'orange']
    markers = ['o', '^', 's']
    group_names = ['group 1', 'group 2', 'group 3']

    # get x in hours, even when only have sunny hours:
    x = range(8, 18)

    for c in range(mean.shape[0]):
        plt.plot(x, mean[c, :], color=colors[c], marker=markers[c], label=group_names[c])

    plt.hlines(0, 8, 17, colors='black')

    # plt.text(0.98, 0.95, 'text',
    #          horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

    plt.legend(loc='upper right', prop={'size': 8})
    plt.title(title)
    # ----------------------------- location of group members -----------------------------

    ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    ax.set_extent([55, 56, -21.5, -20.8], crs=ccrs.PlateCarree())
    ax.coastlines('50m')

    # ----------------------------- plot cloud fraction from CM SAF -----------------------------

    # cloud fraction cover
    cfc_cmsaf = f'~/local_data/obs/CFC.cmsaf.hour.reu.DJF.nc'
    cloud = xr.open_dataset(cfc_cmsaf).CFC

    mean_cloud = cloud.mean(dim='time')

    clevs = np.arange(60, 82, 2)
    cf = ax.contourf(cloud.lon, cloud.lat, mean_cloud, clevs, cmap=plt.cm.Greens,
                     norm=plt.Normalize(60, 82), transform=ccrs.PlateCarree(), zorder=1)

    # cb = plt.colorbar(cf, orientation='horizontal', pad=0.1, aspect=50)
    plt.colorbar(cf, orientation='horizontal', shrink=0.7, pad=0.05, label='daily mean cloud fraction CM SAF')

    # ----------------------------- locations -----------------------------
    # plot location of stations
    for i in range(len(locations)):
        label = labels[i]

        lon = locations['longitude'].values[i]
        lat = locations['latitude'].values[i]

        plt.scatter(lon, lat, color=colors[label],
                    edgecolor='black', zorder=2, s=50, label=group_names[label] if i == 0 else "")

    # sc = plt.scatter(locations['longitude'], locations['latitude'], c=labels,
    #                  edgecolor='black', zorder=2, s=50)

    ax.gridlines(draw_labels=True)

    plt.xlabel(u'$hour$')
    plt.ylabel(ylabel)
    plt.legend(loc='upper right', prop={'size': 8})


def plot_cordex_ensemble_monthly_changes_map(past: xr.DataArray, future: xr.DataArray,
                                             vmax: float, vmin: float,
                                             significance, big_title: str):
    """
    to plot climate changes based on ensemble of model outputs
    1. the input model outputs are in the same shape

    Parameters
    ----------
    big_title :
    past : past, windows defined before in the cfg file
    future :
    Returns
    -------
    :param big_title:
    :type big_title:
    :param significance:
    :type significance:
    :param past:
    :type past:
    :param future:
    :type future:
    :param vmin:
    :type vmin:
    :param vmax:
    :type vmax:

    """

    # windows = {
    #     'past': f'{past.time.dt.year.values[0]:g}-{past.time.dt.year.values[-1]:g}',
    #     'future': f'{future.time.dt.year.values[0]:g}-{future.time.dt.year.values[-1]:g}',
    # }

    fig, axs = plt.subplots(nrows=4, ncols=3, sharex='row', sharey='col',
                            figsize=(15, 10), dpi=220, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.12, top=0.9, wspace=0.1, hspace=0.1)
    axs = axs.flatten()
    axs = axs.ravel()

    # ----------------------------- subplot data:
    changes = (future - past.assign_coords(time=future.time))

    if significance:
        print(f'calculate significant area ... ')
        ens_mean = []
        for i in range(len(changes)):
            sig_mask: xr.DataArray = value_significant_of_anomaly_2d_mask(field_3d=changes[i], conf_level=0.05)
            masked_map = filter_2d_by_mask(changes[i], mask=sig_mask).mean(axis=2)
            masked_map = masked_map.assign_attrs(units=future.assign_attrs().units)
            ens_mean.append(masked_map)
    else:
        ens_mean = changes.mean(dim=['model'], keep_attrs=True)
        ens_mean = ens_mean.assign_attrs(units=future.assign_attrs().units)

    # vmax = np.max(changes)
    # vmin = np.min(changes)

    print(vmin, vmax)

    for i in range(12):
        print(i)
        plot_geo_subplot_map(geomap=ens_mean[i],
                             vmin=vmin, vmax=vmax,
                             bias=1, ax=axs[i], domain='reu-mau', tag=f'month={i + 1:g}',
                             statistics=True,
                             )

    # ----------------------------- plot 4: ensemble std

    plt.suptitle(big_title)

    plt.savefig(f'./plot/{big_title.replace(" ", "_"):s}.png', dpi=220)
    # add the line blow to the command parameter, to disable the output_dir default by hydra:
    # "hydra.run.dir=. hydra.output_subdir=null"
    # or:
    # add hydra.run.dir = .
    # in the default.yaml.

    plt.show()

    print(f'done')


def plot_multi_scenario_ensemble_time_series(da: xr.DataArray,
                                             plot_every_model: int = 0,
                                             suptitle_add_word: str = '',
                                             highlight_model_list=None,
                                             ):
    """
    plot time series, the input
    Args:
        da (): only have 3 dims: time and number and SSP, with model names in coords
        suptitle_add_word ():
        highlight_model_list (): if plot highlight model, so every_model is off.
        plot_every_model ():

    Returns:
        plot
    """
    if highlight_model_list is None:
        highlight_model_list = []
    if len(highlight_model_list):
        plot_every_model = False

    plt.subplots(figsize=(9, 6), dpi=220)

    scenario = list(da.SSP.data)

    colors = ['blue', 'darkorange', 'green', 'red']

    x = da.time.dt.year

    for s in range(len(scenario)):

        data = da.sel(SSP=scenario[s]).dropna('number')
        num = len(data.number)
        scenario_mean = data.mean('number')

        if plot_every_model:
            for i in range(len(da.number)):
                model_name = list(da.number.data)[i]

                print(f'{model_name:s}, {str(i + 1):s}/{len(da.number):g} model')
                model = str(da.number[i].data)
                data_one_model = da.sel(number=model, SSP=scenario[s])

                plt.plot(x, data_one_model, linestyle='-', linewidth=1.0,
                         alpha=0.2, color=colors[s], zorder=1)

        else:
            # plot range of std
            scenario_std = data.std('number')

            # 95% spread
            low_limit = np.subtract(scenario_mean, 1.96 * scenario_std)
            up_limit = np.add(scenario_mean, 1.96 * scenario_std)

            plt.plot(x, low_limit, '-', color=colors[s], linewidth=0.1, zorder=1)
            plt.plot(x, up_limit, '-', color=colors[s], linewidth=0.1, zorder=1)
            plt.fill_between(x, low_limit, up_limit, color=colors[s], alpha=0.2, zorder=1)

        if len(highlight_model_list):
            j = 0
            for i in range(len(data.number)):
                model_name = list(data.number.data)[i]

                if model_name in highlight_model_list:
                    print(f'highlight this model: {model_name:s}')
                    j += 1

                    data_one_model = da.sel(number=model_name, SSP=scenario[s])

                    plt.plot(x, data_one_model, linestyle=get_linestyle_list()[j][1], linewidth=2.0,
                             alpha=0.8, label=model_name, color=colors[s], zorder=1)

        plt.plot(x, scenario_mean, label=f'{scenario[s]:s} ({num:g} GCMs)', linestyle='-', linewidth=2.0,
                 alpha=1, color=colors[s], zorder=2)

    plt.legend(loc='upper left', prop={'size': 14})

    plt.ylim(-10, 10)

    plt.ylabel(f'{da.name:s} ({da.units:s})')
    plt.xlabel('year')
    # plt.pause(0.05)
    # for interactive plot model, do not works for remote interpreter, do not work in not scientific mode.

    title = f'projected changes, 95% multi model spread'

    if suptitle_add_word is not None:
        title = title + ' ' + suptitle_add_word

    plt.suptitle(title)

    plt.savefig(f'{title.replace(" ", "_"):s}.every_model_{plot_every_model:g}.png', dpi=300)

    plt.show()


def get_linestyle_list():
    """
    to use like this linestyle=get_linestyle_list()[i][1]
    Returns:

    """
    linestyles = [
        ('solid', 'solid'),  # Same as (0, ()) or '-'
        ('dotted', 'dotted'),  # Same as (0, (1, 1)) or '.'
        ('dashed', 'dashed'),  # Same as '--'
        ('dashdot', 'dashdot'),  # Same as '-.
        ('loosely dotted', (0, (1, 10))),
        ('dotted', (0, (1, 1))),
        ('densely dotted', (0, (1, 1))),

        ('loosely dashed', (0, (5, 10))),
        ('dashed', (0, (5, 5))),
        ('densely dashed', (0, (5, 1))),

        ('loosely dashdotted', (0, (3, 10, 1, 10))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),

        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

    return linestyles


def plot_cordex_ensemble_changes_map(past: xr.DataArray, mid: xr.DataArray, end: xr.DataArray, big_title: str):
    """
    to plot climate changes based on ensemble of model outputs
    1. the input model outputs are in the same shape

    Parameters
    ----------
    big_title :
    past : past, windows defined before in the cfg file
    mid :
    end :
    Returns
    -------

    """

    # windows = {
    #     'past': f'{past.time.dt.year.values[0]:g}-{past.time.dt.year.values[-1]:g}',
    #     'mid': f'{mid.time.dt.year.values[0]:g}-{mid.time.dt.year.values[-1]:g}',
    #     'end': f'{end.time.dt.year.values[0]:g}-{end.time.dt.year.values[-1]:g}',
    # }

    fig, axs = plt.subplots(nrows=4, ncols=3, sharex='row', sharey='col',
                            figsize=(15, 10), dpi=220, subplot_kw={'projection': ccrs.PlateCarree()})
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.12, top=0.9, wspace=0.1, hspace=0.1)
    axs = axs.flatten()
    axs = axs.ravel()

    # ----------------------------- subplot data:
    # 1, 2, 3
    ensemble_yearmean = [a.mean(dim=['time', 'model'], keep_attrs=True) for a in [past, mid, end]]
    # 4
    past_ensemble_std = past.mean(dim='time', keep_attrs=True).std(dim='model', keep_attrs=True)
    # 5
    mid_change = (mid - past.assign_coords(time=mid.time)).mean(dim=['time', 'model'], keep_attrs=True)
    mid_change = mid_change.assign_attrs(units=mid.assign_attrs().units)
    # 6
    end_change = (end - past.assign_coords(time=end.time)).mean(dim=['time', 'model'], keep_attrs=True)
    end_change = end_change.assign_attrs(units=mid.assign_attrs().units)
    # 7
    # emergence = value_time_of_emergence(std=past_ensemble_std, time_series=end_change)
    # TODO: running mean needed, to rewrite this function
    # emergence = past_ensemble_std
    # emergence = emergence.assign_attrs(units='time of emergence')

    # 8,
    past_mean = past.mean(dim=['time', 'model'], keep_attrs=True)
    mid_change_p = mid_change * 100 / (past_mean + 0.001)
    mid_change_p = mid_change_p.assign_attrs(units='%')

    # 9
    end_change_p = end_change * 100 / (past_mean + 0.001)
    end_change_p = end_change_p.assign_attrs(units='%')

    # 10: nothing
    nothing = past_ensemble_std
    # 11
    mid_ensemble_std = mid.mean(dim='time', keep_attrs=True).std(dim='model', keep_attrs=True)
    end_ensemble_std = end.mean(dim='time', keep_attrs=True).std(dim='model', keep_attrs=True)

    # running mean needed.
    emergence = past_ensemble_std

    maps = ensemble_yearmean + [
        past_ensemble_std, mid_change, end_change,
        emergence, mid_change_p, end_change_p,
        nothing, mid_ensemble_std, end_ensemble_std,
    ]
    # -----------------------------
    # for wind changes the max and min :
    # vmin = [np.min(ensemble_yearmean)] * 3 + [0.2, ] + [-0.3, ] * 2 + [2000, -0.15, -0.25, 0, 0.3, 0.3]
    # vmax = [np.max(ensemble_yearmean)] * 3 + [1.2, ] + [+0.3, ] * 2 + [2100, 0.15, 0.25, 1, 1.2, 1.2]
    vmin = [3.8, ] * 3 + [0.2, ] + [-0.3, ] * 2 + [2000, -0.15, -0.25, 0, 0.3, 0.3]
    vmax = [9.5, ] * 3 + [1.2, ] + [+0.3, ] * 2 + [2100, 0.15, 0.25, 1, 1.2, 1.2]
    # for wind changes the max and min :
    # -----------------------------

    # for duration changes, the max and min:
    # vmin = [500, ] * 3 + [5, ] + [-15, ] * 2 + [2000, -25, -25, 0, ] + [5, ] * 2
    # vmax = [800, ] * 3 + [120, ] + [15, ] * 2 + [2100, 25, 25, 10000] + [120, ] * 2

    # -----------------------------

    tags = ['past', 'mid', 'end', 'ensemble_std_past'] + ['mid-change', 'end-change', ] + \
           ['time of emergence', 'mid-change percentage', 'end-change percentage',
            'nothing', 'mid_ensemble_std', 'end_ensemble_std']
    bias = [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0]

    # vmax = np.max(ensemble_yearmean)
    # vmin = np.min(ensemble_yearmean)

    for i in range(12):
        subplot_kwargs = {
            'geomap': maps[i],
            'vmin': vmin[i],
            'vmax': vmax[i],
            'ax': axs[i],
            'domain': 'reu-mau',
            'tag': tags[i],
            'bias': bias[i]}

        print(i)
        plot_geo_subplot_map(**subplot_kwargs)

    # ----------------------------- plot 4: ensemble std

    plt.suptitle(big_title)
    plt.savefig(f'./plot/{big_title.replace(" ", "_"):s}.png', dpi=200)
    plt.show()
    print(f'done')


# change from Mialhe


def py_note(keyword):
    if keyword in ['da', 'select', 'replace']:
        print(
            f' clearsky_index = clearsky_index.where(np.logical_not(np.isinf(clearsky_index)), 0)'
        )
