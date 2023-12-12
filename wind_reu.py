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
import GEO_PLOT
import MIALHE_2021
import DATA

import Final_Figure


def jk():
    reload(GEO_PLOT)


# ----------------------------- functions -----------------------------

@hydra.main(config_path="configs", config_name="scale_interaction")
def interaction(cfg: DictConfig) -> None:
    """
    to find the physical link between the SSR classification and the large-scale variability
    over la reunion island
    """

    # calculate necessary data and save it for analysis:
    # ==================================================================== data:

    if any(GEO_PLOT.get_values_multilevel_dict(dict(cfg.job.data))):

        if cfg.job.data.prepare_tc_data:
            # add dist_reu to the raw data from 1982 - 2016-04
            tc_6h = DATA.prepare_tc_6h_data(raw=cfg.input.tc_raw, output=cfg.input.tc_6h)

        if cfg.job.data.prepare_anomaly_pauline:
            import os
            import glob
            path = './dataset/nc_f28_2'
            files: list = glob.glob(f'{path:s}/SISan??????????0000.nc')
            files.sort()
            for i in range(len(files)):
                print(i, len(files), i / len(files))
                # get variable name:
                dir, name = os.path.split(files[i])
                year = name[5:9]
                month = name[9:11]
                day = name[11:13]
                hour = name[13:15]
                t_stamp = pd.to_datetime(f'{year:s}-{month:s}-{day:s}-{hour:s}')
                v_name = f'X{year:s}.{month:s}.{day:s}.{hour:s}.00.00'
                ds = xr.open_dataset(files[i])
                da = xr.open_dataset(files[i])[v_name]

                lon = da.longitude.data
                lat = da.latitude.data

                data = da.data.reshape(1, da.shape[0], -1)

                # inverse the lat
                lat2 = np.sort(lat)
                data = data[:, ::-1, :]

                # create a new DataArray:
                new_da = xr.DataArray(data=data,
                                      dims=('time', 'lat', 'lon'),
                                      coords={'time': np.array([t_stamp]), 'lat': lat2, 'lon': lon},
                                      name='SIS')

                new_da = new_da.assign_attrs({'units': 'W/m2', 'statistics': 'anomaly',
                                              'timezone': 'UTC',
                                              'author': 'Pauline'})

                if i == 0:
                    n_da = new_da
                else:
                    n_da = xr.merge([n_da, new_da])

                print(n_da.sizes)

            # save it as UTC 00. which is better for clearsky radiation calculation
            n_da.to_netcdf(cfg.input.ssr_anomaly_pauline)
            print(dir, cfg.input.ssr_anomaly_pauline)

            # ====-======================
            # over land only: overland
            coords_df = pd.read_csv(f'./local_data/Pauline_data/GEO/coordinates.csv')
            da = GEO_PLOT.read_to_standard_da(cfg.input.ssr_anomaly_pauline, 'SIS')
            land_mask = np.zeros((len(da.lat), len(da.lon)))

            lats = da.lat.values
            lons = da.lon.values

            land = coords_df['onlandarea']
            for i in range(len(coords_df)):

                nearest_lat_index = np.abs(lats - coords_df['y'][i]).argmin()
                nearest_lon_index = np.abs(lons - coords_df['x'][i]).argmin()

                print(i, f'lat', coords_df['y'][i], lats[nearest_lat_index])

                # if the coords are not in large diff:
                if np.abs(lats[nearest_lat_index] - coords_df['y'][i]) < 0.05 * 0.5:
                    if np.abs(lons[nearest_lon_index] - coords_df['x'][i]) < 0.05 * 0.5:
                        land_mask[nearest_lat_index, nearest_lon_index] = bool(land[i])
                    else:
                        print(f'find bad point')
                        break
                else:
                    print(f'find bad point')
                    break

            # save this land mask to disk
            np.save(cfg.input.land_mask_pauline, land_mask)
            # read it back from disk, 'npy' is auto added by line above
            # land = np.load(f'{cfg.input.land_mask_pauline:s}.npy')

            # ====-====================== CAL
            cal_raw = GEO_PLOT.read_to_standard_da('~/local_data/cmsaf/CAL/CAL.1999-2016.hour.nc', 'CAL')

            # select min:
            cal_1h = cal_raw.where(cal_raw.time.dt.minute == 0, drop=True)

            # shifttime:
            cal_local = GEO_PLOT.convert_da_shifttime(cal_1h, second=3600 * 4)

            # select hour:
            cal_day = cal_local.where(cal_local.time.dt.hour.isin([8, 9, 10, 11, 12, 13, 14, 15, 16, 17]), drop=True)

            # cut into reu:
            cal_reu = cal_day.where(np.logical_and(cal_day.lon >= 55.05, cal_day.lon <= 56), drop=True)
            cal_reu = cal_reu.where(np.logical_and(cal_reu.lat <= -20.688, cal_reu.lat >= -21.55), drop=True)

            # save to disk
            # select year:
            cal = cal_reu.where(np.logical_and(cal_reu.time.dt.year <= 2016, cal_reu.time.dt.year >= 1999), drop=True)

            cal.to_netcdf(cfg.input.cal_sarah_2_pauline)
            # ====-====================== clear sky era5
            anomaly_pauline = GEO_PLOT.read_to_standard_da(cfg.input.ssr_anomaly_pauline, 'SIS')
            local = GEO_PLOT.convert_da_shifttime(anomaly_pauline, 3600 * 4)

            # ====-====================== clear sky era5 24 hours:
            vars = ['ssrdc', 'ssrd', 'fdir', 'cdir']
            raw_file = f'/Users/ctang/local_data/era5/learn/rad.era5.hour.reu.nc'

            for i in range(4):
                var = vars[i]
                da = GEO_PLOT.read_to_standard_da(raw_file, var)

                da_local = GEO_PLOT.convert_da_shifttime(da, 3600 * 4)

                # convert units:
                da_local = GEO_PLOT.convert_unit_era5_flux(da_local, is_ensemble=0)

                da_local.to_netcdf(f'./dataset/{var:s}.era5.2005-2020.local_24h.reu.nc')
                print(i)

            # ====-====================== clear sky era5 day time
            ssrdc = GEO_PLOT.read_to_standard_da(f'/Users/ctang/local_data/era5/rsdscs/'
                                                 f'ssrdc.ssrd.era5.199-2016.daytime.reu.nc', 'ssrdc')

            ssrdc_local = GEO_PLOT.convert_da_shifttime(ssrdc, 3600 * 4)

            # convert units:
            ssrdc_local = GEO_PLOT.convert_unit_era5_flux(ssrdc_local, is_ensemble=0)

            # save:
            ssrdc_local.to_netcdf(f'./dataset/ssrdc.era5.1999-2016.local_daytime.reu.nc')

            # remap:
            a = ssrdc_local.rename({'x': 'lon', 'y': 'lat'})
            b = anomaly_pauline.rename({'x': 'lon', 'y': 'lat'})
            ssrdc_local_remap = a.interp(lon=b.lon.values, lat=b.lat.values)

            ssrdc_local_remap.to_netcdf(f'./dataset/ssrdc.era5.1999-2016.local_daytime.reu.remap.nc')

            # clearsky calculation
            clearsky = GEO_PLOT.value_clearsky_radiation(
                times=local.time,
                lon=n_da.lon.values,
                lat=n_da.lat.values
            )
            clearsky = clearsky.assign_attrs({'units': 'W/m2'})
            # save to file:
            clearsky.to_netcdf(cfg.input.ssr_clearsky_pauline)

            # ====-======================
            # save raw ssr in pauline format:
            ssr_hour = DATA.reading_sarah_e_hour_reu(
                select_only_land=False,
                select_day_time_hours=True, select_only_NDJF=False,
                for_test=False, clearsky=False)
            # in local time

            # cut into reu:
            ssr_hour_reu = ssr_hour.where(np.logical_and(ssr_hour.lon >= 55.05, ssr_hour.lon <= 56), drop=True)
            ssr_hour_reu = ssr_hour_reu.where(np.logical_and(ssr_hour_reu.lat <= -20.688, ssr_hour_reu.lat >= -21.55),
                                              drop=True)
            # lon and lat are slightly changed, so get Pauline's coords:
            ssr_hour_reu = ssr_hour_reu.assign_coords({"lon": ("x", clearsky.lon.values)})
            ssr_hour_reu = ssr_hour_reu.assign_coords({"lat": ("y", clearsky.lat.values)})

            ssr_hour_reu.to_netcdf(cfg.input.ssr_raw_pauline)
            # ssr_hour_reu.to_netcdf('./dataset/ssr_hour_raw_format_pauline.nc')

            # ====-======================
            # for the CFC from CM_SAF:
            cfc = GEO_PLOT.read_to_standard_da(cfg.input.cfc_cmsaf, 'CFC')
            # cut into reu:
            cfc = cfc.where(np.logical_and(cfc.lon >= 55.05, cfc.lon <= 56), drop=True)
            cfc = cfc.where(np.logical_and(cfc.lat <= -20.688, cfc.lat >= -21.55), drop=True)

            cfc.to_netcdf(cfg.input.cfc_cmsaf_pauline)

            # ====-======================
            # save daily anomaly as sum of all 10 hours format pauline:
            ssr_hour_anomaly_pauline = GEO_PLOT.read_to_standard_da(cfg.input.ssr_anomaly_pauline, 'SIS')
            ssr_day_anomaly = ssr_hour_anomaly_pauline.groupby(
                ssr_hour_anomaly_pauline.time.dt.strftime("%Y-%m-%d")).mean(dim='time')

            new_da = xr.DataArray(ssr_day_anomaly.values, dims=('time', 'y', 'x'), name='SIS',
                                  attrs=ssr_hour_anomaly_pauline.attrs)
            new_da = new_da.assign_coords(time=("time", pd.to_datetime(ssr_day_anomaly.strftime)),
                                          lat=("y", ssr_day_anomaly.lat.data), lon=("x", ssr_day_anomaly.lon.data))
            new_da = new_da.assign_attrs({'long_name': 'Sarah-e anomaly produced by Pauline'})

            new_da.to_netcdf(cfg.input.ssr_daily_anomaly_pauline)

        if cfg.job.data.save_siod_2_nc:
            siod_month: pd.DataFrame = GEO_PLOT.read_csv_into_df_with_header(f'{cfg.dir.data:s}/{cfg.input.siod_raw:s}')
            # ----------------------------- save to nc -----------------------------
            new_da = xr.DataArray(data=siod_month.values.ravel(), dims=('time',),
                                  coords={'time': siod_month.index.values}, name='index')
            new_da = new_da.assign_attrs({'units': 'degC', 'long_name': 'SIOD'})
            new_da.to_netcdf(f'{cfg.dir.data:s}/{cfg.input.siod:s}')

        if cfg.job.data.save_enso_2_nc:
            df = pd.read_csv(f'{cfg.dir.data:s}/{cfg.input.enso_raw:s}', na_values=['-9999'],
                             skiprows=[1, ],
                             nrows=42, header=0, delimiter=r"\s+")
            # key word: read space text dataframe, skip, number of row
            # headers are added as the first row: YEAR DJ JF FM MA AM MJ JJ JA AS SO ON ND

            # select the good year:
            df = df[(df.YEAR >= cfg.param.sarah_e_start_year) & (df.YEAR <= cfg.param.sarah_e_end_year)]
            df = df.set_index('YEAR')

            # interpolate to daily
            dates = pd.date_range(f'{cfg.param.sarah_e_start_year:g}-01',
                                  f'{cfg.param.sarah_e_end_year:g}-12',
                                  freq='MS')
            # convert to da
            new_da = xr.DataArray(data=df.values.ravel(order='C'), dims=('time',),
                                  coords={'time': dates},
                                  name='index')
            new_da = new_da.assign_attrs({'units': 'index', 'long_name': 'Multivariate ENSO Index'})

            # note that the raw data are still in bimonthly.

            new_da.to_netcdf(f'{cfg.dir.data:s}/{cfg.input.enso:s}')

        if cfg.job.data.cal_cyclone_class:
            cyclone_class = DATA.cal_cyclone_class_in_ssr_period(
                ssr_cluster=DATA.load_ssr_cluster(cfg.input.ssr_clusters, check_missing=True),
                raw_cyclone_data=cfg.input.tc_6h,
                **cfg.param.cal_cyclone_class_in_ssr_period)

            print(cyclone_class.columns)

        if cfg.job.data.cal_clearsky_flux:
            ssr_sarah_e = DATA.reading_sarah_e_hour_reu(
                for_test=False,
                clearsky=False,
                select_day_time_hours=False,
                select_only_land=False,
                select_only_NDJF=False)

            clearsky: xr.Dataset = GEO_PLOT.value_clearsky_radiation(
                times=ssr_sarah_e.time.values,
                lon=ssr_sarah_e.lon.values,
                lat=ssr_sarah_e.lat.values,
                model='climatology')

            clearsky['ghi'].to_netcdf(cfg.output.ssr_clearsky_flux_hour_reu)
            print(f'clearsky data are in local time')

        if cfg.job.data.save_ssr_files:
            # save them to temp dir so that simple for analysis:

            ssr_hour = DATA.reading_sarah_e_hour_reu(
                select_only_land=False,
                select_day_time_hours=True, select_only_NDJF=False,
                for_test=False, clearsky=False)

            ssr_day = GEO_PLOT.daily_mean_da(ssr_hour)

            ssr_hour_anomaly = GEO_PLOT.anomaly_hourly(ssr_hour)
            ssr_day_anomaly = GEO_PLOT.anomaly_daily(ssr_day)

            # save it:
            print(f'starting to save')
            ssr_hour.to_netcdf(cfg.output.ssr_total_sky_hour_reu)
            ssr_hour_anomaly.to_netcdf(cfg.output.ssr_total_sky_hour_reu_anomaly)
            ssr_day.to_netcdf(cfg.output.ssr_total_sky_day_reu)
            ssr_day_anomaly.to_netcdf(cfg.output.ssr_total_sky_day_reu_anomaly)

        if cfg.job.data.save_olr_files:
            ttr_swio = GEO_PLOT.read_to_standard_da(f'~/local_data/era5/ttr/ttr.era5.1999-2016.day.swio.nc',
                                                    var='ttr')
            olr_swio = GEO_PLOT.convert_ttr_era5_2_olr(ttr=ttr_swio, is_reanalysis=True)
            olr_swio.to_netcdf(cfg.input.olr_day_swio)

        if cfg.job.data.save_olr_ens_files:
            ttr_swio = GEO_PLOT.read_to_standard_da(f'{cfg.input.ttr_ens_file:s}', 'ttr')
            # In ERA - 5, the variable used to compute the clusters is called Top Net Thermal Radiation.It is given in J
            # / m2, so it needs to be multiplied by 10800, i.e.the nb of seconds between two analyses (3 hours),
            # to obtain W / m2.

            ttr = DATA.prepare_ttr_data(ttr_swio)
            # pre-process of ensemble data: change unit, etc
            olr = GEO_PLOT.convert_ttr_era5_2_olr(ttr=ttr.squeeze(drop=True), is_reanalysis=False)

            # save it:
            olr.to_netcdf(cfg.input.olr_ens_swio)

        if cfg.job.data.ssr_cluster_seasonal_series:
            ssr_cluster = DATA.load_ssr_cluster(data=cfg.input.ssr_clusters, check_missing=True)
            print()

            # starting to output a seasonal series
            # note that can not select [11,12,1,2], since that will result in groups of different seasons.
            c9 = ssr_cluster[{'C9'}]
            class_names = [f'CL_{x + 1:g}' for x in range(9)]
            class_names_dict = dict(zip(np.arange(1, 10), class_names))

            months = 'JJAS'
            months = 'NDJF'

            df_ssr_c9_count = pd.DataFrame()
            season_count = []
            for year in np.arange(1998, 2017):

                # select NDJF
                if months == 'NDJF':
                    season_1 = c9[(c9.index >= f'{year:g}-11-01') &
                                  (c9.index < f'{year + 1:g}-03-01')]
                    ind = [f'{year:g}-{year + 1:g}']
                if months == 'JJAS':
                    season_1 = c9[(c9.index >= f'{year:g}-06-01') &
                                  (c9.index < f'{year:g}-10-01')]
                    ind = [f'{year:g}']

                count = season_1.value_counts().sort_index()
                count_index = [list(x)[0] for x in count.index]
                count_class = count.values.reshape(1, -1)

                columns = [class_names_dict[i] for i in count_index]

                df_count = pd.DataFrame(data=count_class,
                                        columns=columns,
                                        index=ind)

                df_ssr_c9_count = df_ssr_c9_count.append(df_count)
                season_count.append(len(season_1))

            df_ssr_c9_count['total_days'] = season_count
            df_ssr_c9_count = df_ssr_c9_count.replace({np.nan: 0}).astype(int)
            df_ssr_c9_count['season'] = [months] * len(df_ssr_c9_count)
            df_ssr_c9_count = df_ssr_c9_count[df_ssr_c9_count.index != '1998']

            # change the order of columns:
            new_order = class_names + ['total_days', 'season']
            df_ssr_c9_count = df_ssr_c9_count[new_order]

            print(df_ssr_c9_count)
            # to save it
            output = f'./dataset/ssr_class_{months:s}_series'
            df_ssr_c9_count.to_csv(f'{output:s}.csv')

            df_ssr_c9_count.to_pickle(output)
            # to read it
            df = pd.read_pickle(output)

    print(f'data prepared, done')

    # ==================================================================== loading
    # definition:

    # ==================================== loading data and definition
    nearby_radius = 5

    central_lat = -21.1
    central_lon = 55.5

    ssr_cluster = DATA.load_ssr_cluster(data=cfg.input.ssr_clusters, check_missing=True)

    # ssr hour anomaly
    ssr_hour_anomaly_pauline = GEO_PLOT.read_to_standard_da(cfg.input.ssr_anomaly_pauline, 'SIS')
    ssr_hour_anomaly_pauline = GEO_PLOT.convert_da_shifttime(ssr_hour_anomaly_pauline, 3600 * 4)
    # 154 days with missing records are removed

    # uncomment this lines to check missing date of SSR
    # total_time_steps = pd.date_range(start='1999-01-01', end='2016-12-31', freq='D')
    # daily = GEO_PLOT.daily_mean_da(ssr_hour_anomaly_pauline)[:, 3, 3]
    # missing_date = [a for a in total_time_steps if a not in daily.time.dt.date]
    # # this missing_date has 5 days in 29 Feb. used to calculate climatology by filtering, so missing values are removed
    # # after this calculation.
    #
    # missing_date2 = [a for a in total_time_steps if a.date() not in ssr_cluster.index.date]
    # # this missing date2 have 154 days, including 5 days of 29th Feb and 149 days with missing hours.

    ssr_raw_pauline = GEO_PLOT.read_to_standard_da(cfg.input.ssr_raw_pauline, var='SIS')
    # select ssr in the core of cyclone season, DJFM, the whole cyclone season: DJFM

    ssr_DJFM = ssr_raw_pauline.where(ssr_raw_pauline.time.dt.month.isin([1, 2, 3, 12]), drop=True)

    # the 89 pixels pauline used for classification
    # land_mask = np.load(f'{cfg.input.land_mask_pauline:s}.npy')
    #
    # clearsky = GEO_PLOT.read_to_standard_da(cfg.input.ssr_clearsky_pauline, 'ghi')
    # clearsky = clearsky.assign_attrs({'units': 'W/m2'})

    # ==================================================================== loading cyclone

    cyclone_class, cyclone_num, all_nearby_cyc_99_16 = DATA.cal_cyclone_class_in_ssr_period(
        ssr_cluster=ssr_cluster[{'9Cl'}],
        raw_cyclone_data=cfg.input.tc_6h,
        nearby_radius=nearby_radius,
        find_day_with_more_than_one_cyclone=1)

    # attention, the len of cyclone_class is not the total number of tc, there're two days with 2 cyclones
    # ssr_cluster is changed by this function
    # cyc in all month
    # cyclone_class have all sarah_e available date 6421

    # only 2 day in SSR class 2 have two nearby tc in a day: 2008-01-31 and 2008-02-01,
    # both these days are belongs to SSR class 2, CL2. and belongs to olr_regime 2.

    # so, totally 165 tc days in 99-16, 3 days are missing in sarah_e, so only 162 tc days with ssr.

    cyc_day = cyclone_class[cyclone_class > 0].dropna()

    print(len(cyc_day))
    for m in range(1, 13):
        print(f'mon={m:g} = {len(cyc_day[cyc_day.index.month == m].dropna())}')

    # all cyc day 71, from 1999-2016, all months:
    cyc_day = cyclone_class[cyclone_class > 0].dropna().index
    non_cyc_day = cyclone_class[cyclone_class['class'] == 0].index

    # ==================================================================== olr regime
    olr_regime = DATA.load_olr_regimes(
        mat_file=cfg.input.ttt_Ben_class_file, classif='dominant')
    # days(NDJF, 29 Feb suppressed for leap yrs) x 39 seasons (1979-80 to 2017-18) x 10 members.
    b = olr_regime[olr_regime.index.year >= 1999]
    olr_regime = b[b.index.year <= 2016]

    # cyclone_class_olr = DATA.cal_cyclone_class_in_olr_period(
    #     ssr_cluster=olr_regime,
    #     raw_cyclone_data=cfg.input.cyclone_all,
    #     nearby_radius=nearby_radius,
    #     find_day_with_more_than_one_cyclone=1)

    # ======================== analysis ========================== ssr:

    if any(GEO_PLOT.get_values_multilevel_dict(dict(cfg.job.ssr))):

        Final_Figure.figure2_monthly_num_ssr_class2(df=ssr_cluster[{'9Cl'}])

        if cfg.job.ssr.plot_topo_reu_in_sarah_e_grid:
            GEO_PLOT.plot_topo_reunion_high_reso(plot=True,
                                                 grid=ssr_hour_anomaly_pauline[0, :, :],
                                                 dpi=100,
                                                 plot_wind=1,
                                                 output_tag='sarah_e_grid')

        if cfg.job.ssr.statistics:
            # persistence in big groups:
            persistence_big_group = Final_Figure.persistence_big_group(ssr_cluster[{'9Cl'}].copy())

            # persistence:
            persistence, persistence_all_class = Final_Figure.persistence(ssr_cluster[{'9Cl'}])

            # transition: the next day:
            transition = GEO_PLOT.value_transition_next_day_classif(df=ssr_cluster[{'9Cl'}])

        if cfg.job.ssr.classification.plot_classification:
            # monthly - diurnal maps:

            Final_Figure.ssr_cluster_in_month_hour(field=ssr_hour_anomaly_pauline, classif=ssr_cluster[{'9Cl'}],
                                                   only_sig=0,
                                                   vmin=-800, vmax=400, bias=1,
                                                   cmap=plt.cm.seismic,
                                                   cbar_label=f'($^\_$N)',
                                                   output='Fig.xx.ssr.cluster.monthly.png')

            # anomaly
            Final_Figure.figure_1_ssr_classification(field=ssr_hour_anomaly_pauline,
                                                     classif=ssr_cluster[{'9Cl'}],
                                                     vmin=-800, vmax=400,
                                                     output='Fig.1.png',
                                                     cbar_label=f'SSR (W m**-2)',
                                                     bias=1,
                                                     cmap=plt.cm.seismic,
                                                     only_sig=False)

            Final_Figure.figure2_monthly_num_ssr_class(df=ssr_cluster[{'9Cl'}])

            # clearsky calculated
            Final_Figure.figure_1_ssr_classification(field=clearsky,
                                                     classif=ssr_cluster[{'9Cl'}],
                                                     vmin=300, vmax=1400,
                                                     output='Fig.A1.png',
                                                     bias=0,
                                                     cbar_label=f'clear sky estimated SSR (W m**-2)',
                                                     cmap=plt.cm.YlOrRd,
                                                     only_sig=False)
            # clearsky era5 raw:
            clearsky_era5_raw = GEO_PLOT.read_to_standard_da(cfg.input.ssr_clearsky_era5_raw, 'ssrdc')
            Final_Figure.figure_3_ssr_classification_clearsky(field=clearsky_era5_raw,
                                                              classif=ssr_cluster[{'9Cl'}],
                                                              vmin=300, vmax=1400,
                                                              output='Fig.clearsky_era5_raw.png',
                                                              bias=0,
                                                              add_triangle=0,
                                                              cbar_label=f'clear sky SSR (W m**-2)',
                                                              cmap=plt.cm.YlOrRd,
                                                              only_sig=False)

            # test ratio of total / clear sky SSR by ERA5 raw data:

            ssrd = GEO_PLOT.read_to_standard_da(f'./dataset/ssrd.era5.2005-2020.local_24h.reu.nc', 'ssrd')
            ssrdc = GEO_PLOT.read_to_standard_da(f'./dataset/ssrdc.era5.2005-2020.local_24h.reu.nc', 'ssrdc')

            ratio_era5 = ssrd / ssrdc
            ratio_era5 = ratio_era5.assign_attrs({'units': 'W/m2'})
            ratio_era5 = ratio_era5.rename('SIS')

            Final_Figure.figure_3_ssr_classification_clearsky_era5_data_only(
                field=ratio_era5, classif=ssr_cluster[{'9Cl'}], vmin=0.1, vmax=1.5,
                output='Fig.clearsky_ratio_era5_data_only.png', cbar_label=f'all_sky/clear_sky',
                bias=0, cmap=plt.cm.Blues, only_sig=False)
            # ----------------------------------------------------------
            clearsky_era5_remap = GEO_PLOT.read_to_standard_da(cfg.input.ssr_clearsky_era5_pauline, 'ssrdc')
            Final_Figure.figure_1_ssr_classification(field=clearsky_era5_remap,
                                                     classif=ssr_cluster[{'9Cl'}],
                                                     vmin=300, vmax=1400,
                                                     output='Fig.dif_clearsky_era5.remap.png',
                                                     bias=0,
                                                     cbar_label=f'clear sky SSR (W m**-2)',
                                                     cmap=plt.cm.YlOrRd,
                                                     only_sig=False)

            dif_clear = clearsky - clearsky_era5_remap
            dif_clear = dif_clear.assign_attrs({'units': 'W m**-2'})
            dif_clear = dif_clear.rename('dif_clear')

            Final_Figure.figure_3_ssr_classification_clearsky(
                field=dif_clear, classif=ssr_cluster[{'9Cl'}], vmin=-300, vmax=300,
                output='Fig.dif_clearsky_cal-era5.png', bias=0, cbar_label=f'clear sky SSR (W m**-2)',
                cmap=plt.cm.YlOrRd, only_sig=False)

            # CAL from SARAH_2
            cal = GEO_PLOT.read_to_standard_da(cfg.input.cal_sarah_2_pauline, 'CAL')
            Final_Figure.figure_1_ssr_classification(
                field=cal, classif=ssr_cluster[{'9Cl'}], vmin=0, vmax=1, output='Fig.CAL.png', bias=0,
                cbar_label=f'CAL', cmap=plt.cm.YlOrRd, only_sig=False)

            ssr_hour_reu = GEO_PLOT.read_to_standard_da('./dataset/ssr_hour_raw_format_pauline.nc', 'SIS')

            Final_Figure.figure_1_ssr_classification(
                field=ssr_hour_reu, classif=ssr_cluster[{'9Cl'}], vmin=300, vmax=1400, output='Fig.test.png',
                cbar_label=f'all_sky', bias=0, cmap=plt.cm.Reds, only_sig=False)

            # ratio of all sky to clearksy:
            ratio = ssr_hour_reu / clearsky_era5_remap

            ratio = ratio.assign_attrs({'units': 'W/m2'})
            ratio = ratio.rename('SIS')

            Final_Figure.figure_3_ssr_classification_clearsky(
                field=ratio, classif=ssr_cluster[{'9Cl'}], vmin=0.1, vmax=1.5, output='Fig.clearsky_ratio_era5.png',
                cbar_label=f'all_sky/clear_sky', bias=0, cmap=plt.cm.Blues, only_sig=False)

            # clear sky estimated by cal and ssr raw
            clear_esti = ssr_hour_reu / (1 - cal)
            clear_esti = clear_esti.assign_coords(lon=('x', ssr_hour_reu.lon.values),
                                                  lat=('y', ssr_hour_reu.lat.values))

            clear_esti = clear_esti.assign_attrs({'units': 'W/m2'})
            clear_esti = clear_esti.rename('SIS')

            Final_Figure.figure_1_ssr_classification(
                field=clear_esti, classif=ssr_cluster[{'9Cl'}], vmin=300, vmax=1400,
                output='Fig.clearsky.estimation.png', cbar_label=f'clear_sky estimation', bias=0,
                cmap=plt.cm.GnBu, only_sig=False)

            cfc = GEO_PLOT.read_to_standard_da(cfg.input.cfc_cmsaf_pauline, 'CFC')
            Final_Figure.figure_1_ssr_classification(
                field=cfc, classif=ssr_cluster[{'9Cl'}], vmin=0, vmax=100, output='Fig.A2.cloud.png',
                cbar_label=f'cloud fraction to 2015', bias=0, cmap=plt.cm.Greens, only_sig=False)

            print('good')

        if cfg.job.ssr.classification.monthly_distribution:
            # ==================================================================== ssr cluster:

            MIALHE_2021.histogram_classification(ssr_cluster)
            # MIALHE_2021.matrix_classification_at_year_and_month(pclass_ssr)
            MIALHE_2021.table_classification_at_month(ssr_cluster)

            print(f'good')

        if cfg.job.ssr.classification.field_in_class:
            C9_class = pd.DataFrame(ssr_cluster['9Cl'])
            # q_file = f'~/local_data/era5/q/q.hourly.era5.1999-2016.bigreu.local_daytime.nc'
            qs_file = f'/Users/ctang/local_data/era5/q_specific/sp.era5.hourly.1999-2016.bigreu.local_daytime.nc'
            qs = GEO_PLOT.read_to_standard_da(qs_file, 'sp') / 10 ** 5
            qs = qs.assign_attrs({'units': 'g/kg 10E7',
                                  'long_name': 'specific humidity normalised in 10^5'})
            qs_hourly_anomaly = GEO_PLOT.anomaly_hourly(qs)

            cfc_hourly_anomaly = None
            list_da_plot = [cfc_hourly_anomaly, qs_hourly_anomaly, ]
            maxs = [40, 0.001, 200]
            max_moisture_mean = 1.03
            min_moisture_mean = 1.00

            GEO_PLOT.plot_diurnal_cycle_field_in_classif(
                classif=C9_class, field=qs, area='bigreu',
                vmax=max_moisture_mean, vmin=min_moisture_mean, field_bias=False,
                plot_circulation=1, circulation_anomaly=0,
                plot_moisture_flux=1, circulation_name='moisture',
                only_significant_points=True,
                suptitle_add_word='cm_saf cloud fraction',
                row_headers=[f'CL{x + 1:g}' for x in range(9)],
                col_headers=[f'{x + 8:g}:00' for x in range(10)],
                plt_type='pcolormesh',
                test_run=0)
            # cloud
            cfc_file = f'~/local_data/cmsaf/CFC.hourly.cmsaf.1999-2015.bigreu.local_daytime.nc'
            cfc = GEO_PLOT.read_to_standard_da(cfc_file, var='CFC')
            cfc_hourly_anomaly = GEO_PLOT.anomaly_hourly(cfc)

            GEO_PLOT.plot_diurnal_cycle_field_in_classif(
                classif=C9_class, field=cfc_hourly_anomaly, area='bigreu',
                cmap=plt.cm.get_cmap('Spectral', 40 + 1),
                vmax=40, vmin=-40, field_bias=True,
                plot_circulation=1, circulation_anomaly=1,
                plot_moisture_flux=0, circulation_name='wind',
                only_significant_points=True,
                suptitle_add_word='cm_saf cloud fraction',
                row_headers=[f'CL{x + 1:g}' for x in range(9)],
                col_headers=[f'{x + 8:g}:00' for x in range(10)],
                plt_type='pcolormesh',
                test_run=0)

            # mean wind
            GEO_PLOT.plot_diurnal_cycle_field_in_classif(
                classif=C9_class, field=cfc_hourly_anomaly, area='bigreu',
                cmap=plt.cm.get_cmap('Spectral', 40 + 1),
                vmax=40, vmin=-40, field_bias=True,
                plot_circulation=1, circulation_anomaly=0,
                plot_moisture_flux=0, circulation_name='wind',
                only_significant_points=True,
                suptitle_add_word='cm_saf cloud fraction',
                row_headers=[f'CL{x + 1:g}' for x in range(9)],
                col_headers=[f'{x + 8:g}:00' for x in range(10)],
                plt_type='pcolormesh',
                test_run=0)

            # wind anomaly.

            GEO_PLOT.plot_diurnal_cycle_field_in_classif(
                classif=C9_class, field=ssr_hour_anomaly_pauline, area='bigreu',
                vmax=400, vmin=-800, field_bias=True,
                plot_circulation=1, circulation_anomaly=1,
                plot_moisture_flux=0, circulation_name='wind',
                only_significant_points=True,
                suptitle_add_word='sarah_e SSR',
                row_headers=[f'CL{x + 1:g}' for x in range(9)],
                col_headers=[f'{x + 8:g}:00' for x in range(10)],
                plt_type='pcolormesh',
                test_run=1)

        if cfg.job.ssr.variability.daily_energy:
            ssr_land, ssr_land_clear = DATA.reading_sarah_e_hour_reu(
                clearsky=1, select_day_time_hours=0, for_test=1, select_only_NDJF=0, select_only_land=1)

            # calculate daily energy using 24 hours data
            solar_energy = GEO_PLOT.cal_daily_total_energy(ssr_land)
            # remove the ocean 0 values for plot
            solar_energy = GEO_PLOT.select_land_only_reunion_by_altitude(solar_energy)

            ssr_spatial_mean = ssr_land.mean('x', keep_attrs=True).mean('y', keep_attrs=True)
            ssr_spatial_mean_clear = ssr_land_clear.mean('x', keep_attrs=True).mean('y', keep_attrs=True)

            GEO_PLOT.plot_join_heatmap_boxplot(ssr_spatial_mean_clear)

            print(f'starting to load data over land...')

    # ==================================================================== cyclone:

    if any(GEO_PLOT.get_values_multilevel_dict(dict(cfg.job.cyclone))):

        if cfg.job.cyclone.cyclone_in_ssr_class:
            # Final figure:
            GEO_PLOT.plot_cyclone_in_classif(classif=ssr_cluster[{'9Cl'}],
                                             # radius=cfg.param.cyclone_radius,
                                             radius=5,
                                             tag_subplot='CL_',
                                             suptitle_add_word='final figure 1996-2016 SSR clusters')

        # attention: set the period of SSR and thus the cyclone study:
        if cfg.job.cyclone.plot_diurnal_cycle_curve_classif:
            percentage = GEO_PLOT.read_to_standard_da(f'./dataset/ssr_hour_anomaly_ratio_ssr_raw_NDJF_space_mean.nc',
                                                      'SIS')

            Final_Figure.figure_19_diurnal_curve_in_classif(
                classif=cyclone_class, field_1D=percentage,
                anomaly=1, percent=1,
                suptitle_add_word='OLR regime reu sarah_e',
                ylimits=[-0.35, 0.12],
                plot_big_data_test=0)

            GEO_PLOT.plot_diurnal_curve_in_classif(
                classif=cyclone_class,
                field_1D=ssr_hour_anomaly,
                anomaly=1, percent=1,
                ylimits=[-0.4, 0.4],
                suptitle_add_word='TCs bigreu sarah_e',
                plot_big_data_test=cfg.param.plot_big_data_test
            )

        if cfg.job.cyclone.ssr_in_cyclone_day:
            ssr_cyc_day = ssr_DJFM.where(
                ssr_DJFM.time.dt.strftime('%Y-%m-%d').isin(cyc_day.strftime('%Y-%m-%d')),
                drop=True)

            ssr_non_cyc_day = ssr_DJFM.where(
                ssr_DJFM.time.dt.strftime('%Y-%m-%d').isin(non_cyc_day.strftime('%Y-%m-%d')),
                drop=True)

            # cyc season: all the days in summer
            summer_cyclone = cyclone_class[cyclone_class.index.month.isin([1, 2, 3, 12])]
            print(f'total summer nearby cyclone days (considering the missing days of sarah_e) = '
                  f'{len(set(summer_cyclone[summer_cyclone["class"] > 0].index.date)):g}')

            # days with cyc records and ssr records summer:
            common_cyc_day_ssr = ssr_DJFM.where(
                ssr_DJFM.time.dt.strftime('%Y-%m-%d').isin(summer_cyclone.index.strftime('%Y-%m-%d')),
                drop=True)
            print(f'all avail days with cyc records and ssr in summer DJFM is {len(common_cyc_day_ssr) / 10:g}')

            # ----------------------------- data -----------------------------

            tc1 = Final_Figure.nearby_TC_with_var(
                ssr_cluster=ssr_cluster,
                # raw_cyclone_data=cfg.input.tc_raw,
                raw_cyclone_data='./dataset/cyc_df.csv',  # this file has dist_reu
                daily_tc_output=cfg.input.tc_day,
                ssr_hour_anomaly_pauline=ssr_hour_anomaly_pauline,
                nearby_radius=nearby_radius)

            tc2 = Final_Figure.nearby_TC_passtime_ssr(
                daily_tc_output=cfg.input.tc_day,
                radius_3=False,  # to do the same thing within 3 degree, using the same 5 deg data from previous function
                nearby_radius=nearby_radius)


            # ---------------
            # Final plot: east and west cyc:
            # prepare tc class:

            # read cyclone
            df_cyclone = tc1
            df_cyclone['class'] = np.zeros(len(df_cyclone))
            df_cyclone = df_cyclone.set_index(pd.to_datetime(df_cyclone.date))

            df_cyclone.loc[df_cyclone['LON'] >= central_lon, 'class'] = 1
            df_cyclone.loc[df_cyclone['LON'] < central_lon, 'class'] = -1

            # input tc = 5 deg 1999 to 2016, west and east

            # use land only anomaly to compare with previous figure.

            tc_east_west = Final_Figure.east_west_tc_ssr(field=ssr_hour_anomaly_pauline,
                                                         classif=df_cyclone[{'class'}], radius=nearby_radius,
                                                         vmin=-800, vmax=400,
                                                         output='east-west-tc.png',
                                                         bias=1,
                                                         cbar_label='SSR anomaly (W m**-2)',
                                                         only_sig=True,
                                                         cmap=plt.cm.seismic)

            tc_in_class_NDJF = Final_Figure.figure_1_ssr_classification_for_cyclone(field=ssr_DJFM,
                                                                                    classif=summer_cyclone,
                                                                                    vmin=200, vmax=1000,
                                                                                    output=f'Fig.cyclone.nearby_radius_{nearby_radius:g}.png',
                                                                                    cbar_label=f'SSR (W m**-2)',
                                                                                    bias=0,
                                                                                    radius=nearby_radius,
                                                                                    cmap=plt.cm.Reds,
                                                                                    only_sig=True)

            # ===========================================================================

            # cyclone impact on daily mean SSR anomaly

            # ===========================================================================

            Final_Figure.nearby_TC_with_var(ssr_hour_anomaly_pauline=ssr_hour_anomaly_pauline,
                                            nearby_redius=15, nearby_cyclone_class=cyclone_class,
                                            )

            GEO_PLOT.plot_diurnal_cycle_field_in_classif(
                classif=summer_cyclone, field=ssr_DJFM,
                area='bigreu', vmax=1000, vmin=200,
                field_bias=False,
                only_significant_points=False,
                plot_circulation=False,
                suptitle_add_word='ssr in cyclone day',
                test_run=0)

            # plot ssr anomaly NDJFM, ssr have missing data.
            GEO_PLOT.plot_diurnal_cycle_field_in_classif(
                classif=summer_cyclone, field=ssr_DJFM,
                area='bigreu', vmax=350, vmin=-315,
                field_bias=True,
                only_significant_points=True,
                plot_circulation=True,
                suptitle_add_word='ssr in cyclone day',
                test_run=0)

        if cfg.job.cyclone.ssr_in_cyclone_day_welch_test:
            # ================================== test Welch's test

            #
            # sig = GEO_PLOT.value_significant_of_anomaly_2d_mask(ssr_non_cyc_day)
            #
            # sig = GEO_PLOT.value_significant_of_anomaly_2d_mask(ssr_cyc_day)

            # cyc_1h = ssr_non_cyc_day.loc[ssr_non_cyc_day.time.dt.hour == 10]
            # sig = GEO_PLOT.value_significant_of_anomaly_2d_mask(cyc_1h)

            import cartopy.crs as ccrs
            fig, axs = plt.subplots(nrows=6, ncols=10, sharex='row', sharey='col', figsize=(18, 12), dpi=220,
                                    subplot_kw={'projection': ccrs.PlateCarree()})
            fig.subplots_adjust(left=0.1, right=0.85, bottom=0.12, top=0.9, wspace=0.09, hspace=0.01)

            hours = list(range(8, 18))

            colorbars = [i == len(hours) - 1 for i in range(len(hours))]
            for i in range(len(hours)):
                ssr_cyc_1h = ssr_cyc_day.loc[ssr_cyc_day.time.dt.hour == hours[i]]
                ssr_non_cyc_1h = ssr_non_cyc_day.loc[ssr_non_cyc_day.time.dt.hour == hours[i]]

                # plot non_cyc_day:
                GEO_PLOT.plot_geo_subplot_map(
                    ax=axs[0, i],
                    geomap=ssr_non_cyc_1h.mean('time', keep_attrs=True),
                    domain='bigreu', vmin=200, vmax=1200, tag='non_cyc',
                    bias=0,
                    plot_cbar=False, statistics=1
                )

                GEO_PLOT.plot_geo_subplot_map(
                    ax=axs[1, i],
                    geomap=ssr_cyc_1h.mean('time', keep_attrs=True),
                    domain='bigreu', vmin=30, vmax=1000, tag='cyc',
                    bias=0,
                    plot_cbar=0, statistics=1
                )

                # plot Welch's test:
                t_sta, p_2side = scipy.stats.ttest_ind(ssr_cyc_1h, ssr_non_cyc_1h, equal_var=False)

                da1 = xr.zeros_like(ssr_cyc_1h[-1])
                da1[:] = p_2side

                # plot 0.05 level
                conf_level = 0.05
                sig1 = da1 < conf_level
                sig1 = sig1.assign_attrs(dict(units='significance'))

                nan_pixel_1 = GEO_PLOT.count_nan_2d_map(sig1)

                GEO_PLOT.plot_geo_subplot_map(
                    ax=axs[2, i],
                    geomap=sig1,
                    domain='bigreu', vmin=0, vmax=1, tag=f'{conf_level:4.2f}',
                    bias=0,
                    plot_cbar=0, statistics=0
                )

                conf_level = 0.01
                sig2 = da1 < conf_level
                sig2 = sig2.assign_attrs(dict(units='significance'))
                nan_pixel_2 = GEO_PLOT.count_nan_2d_map(sig2)
                GEO_PLOT.plot_geo_subplot_map(
                    ax=axs[3, i],
                    geomap=sig2,
                    domain='bigreu', vmin=0, vmax=1, tag=f'{conf_level:4.2f}',
                    bias=0,
                    plot_cbar=False, statistics=0
                )
                # plot standard T-test:
                t_sta, p_2side2 = scipy.stats.ttest_ind(ssr_cyc_1h, ssr_non_cyc_1h, equal_var=True)

                da2 = xr.zeros_like(ssr_cyc_1h[-1])
                da2[:] = p_2side2

                # plot 0.05 level
                conf_level = 0.05
                sig3 = da2 < conf_level
                sig3 = sig3.assign_attrs(dict(units='significance'))
                nan_pixel_3 = GEO_PLOT.count_nan_2d_map(sig3)
                GEO_PLOT.plot_geo_subplot_map(
                    ax=axs[4, i],
                    geomap=sig3,
                    domain='bigreu', vmin=0, vmax=1, tag=f'{conf_level:4.2f}',
                    bias=0,
                    plot_cbar=0, statistics=0
                )

                conf_level = 0.01
                sig4 = da2 < conf_level
                sig4 = sig4.assign_attrs(dict(units='significance'))
                nan_pixel_4 = GEO_PLOT.count_nan_2d_map(sig4)
                GEO_PLOT.plot_geo_subplot_map(
                    ax=axs[5, i],
                    geomap=sig4,
                    domain='bigreu', vmin=0, vmax=1, tag=f'{conf_level:4.2f}',
                    bias=0,
                    plot_cbar=False, statistics=0
                )

                print(i, hours[i], nan_pixel_1, nan_pixel_2, nan_pixel_3, nan_pixel_4)

            plt.savefig('./plot/test_significant.png', dpi=300)
            plt.show()

            print(f'done')

            # idea: ?

        if cfg.job.cyclone.ssr_in_cyclone_day_Permutation_test:
            # see the yaml for more info
            # Permutation test
            permutation = 1000
            hours = [9, 11, 12, 15, 17]

            import cartopy.crs as ccrs
            fig, axs = plt.subplots(nrows=4, ncols=len(hours), sharex='row', sharey='col',
                                    figsize=(3 * len(hours), 3 * 3), dpi=220,
                                    subplot_kw={'projection': ccrs.PlateCarree()})
            fig.subplots_adjust(left=0.1, right=0.85, bottom=0.12, top=0.9, wspace=0.09, hspace=0.01)

            for i in range(len(hours)):
                ssr_cyc_1h = ssr_cyc_day.loc[ssr_cyc_day.time.dt.hour == hours[i]]
                ssr_non_cyc_1h = ssr_non_cyc_day.loc[ssr_non_cyc_day.time.dt.hour == hours[i]]

                ssr_cyc_1h = ssr_cyc_1h[:, 32:45, 22:42]
                ssr_non_cyc_1h = ssr_non_cyc_1h[:, 32:45, 22:42]

                lon_len = len(ssr_non_cyc_1h.lon)
                lat_len = len(ssr_non_cyc_1h.lat)

                # plot 1

                GEO_PLOT.plot_geo_subplot_map(
                    ax=axs[0, i],
                    plt_type='pcolormesh',
                    geomap=ssr_cyc_1h.mean("time", keep_attrs=True),
                    domain='reu', vmin=200, vmax=1200, tag=f'cyc',
                    bias=0, plot_cbar=False, statistics=1)

                axs[0, i].set_title(f'{hours[i]:g}H00')

                # random choose the non-cyc days
                # plot 2
                from random import randint
                random_index = [randint(0, len(ssr_non_cyc_1h) - 1) for x in range(len(ssr_cyc_1h))]

                ssr_random_1h = ssr_non_cyc_1h[random_index, :, :]
                GEO_PLOT.plot_geo_subplot_map(
                    ax=axs[1, i],
                    plt_type='pcolormesh',
                    geomap=ssr_random_1h.mean("time", keep_attrs=True),
                    domain='reu', vmin=200, vmax=1200, tag=f'random',
                    bias=0, plot_cbar=False, statistics=1)

                # plot 3
                p_map = xr.zeros_like(ssr_cyc_1h[0])
                conf_level = 0.01
                print('done')

                sig_plot_done = 0
                non_sig_plot_done = 0
                for j in range(lon_len):
                    for k in range(lat_len):

                        print(f'hour={hours[i]:g}', k, j)

                        diff, list_diff, p = GEO_PLOT.test_exact_mc_permutation(
                            small=ssr_cyc_1h[:, k, j], big=ssr_non_cyc_1h[:, k, j],
                            nmc=permutation, show=False)
                        p_map[k, j] = p

                        # find a significant point plot the list_diff
                        # here the first sig or not sig point is plotted  using the built-in function
                        # of the test, separately, so the distribution will be different.
                        if sig_plot_done == 0:
                            if p < 0.05:
                                plot_it = True
                                print(f'sig: lon={j:g}, lat={k:g}, p={p:4.2f}')
                                sig_plot_done = 1
                                print('plot sig done')
                            else:
                                plot_it = False

                            if plot_it:
                                diff, list_diff, p = GEO_PLOT.test_exact_mc_permutation(
                                    small=ssr_cyc_1h[:, k, j], big=ssr_non_cyc_1h[:, k, j],
                                    nmc=permutation, show=plot_it)

                        if non_sig_plot_done == 0:
                            if p > 0.05:
                                plot_it = True
                                print(f'non-sig: j={j:g}, k={k:g}, p={p:4.2f}')
                                non_sig_plot_done = 1
                            else:
                                plot_it = False

                            if plot_it:
                                diff, list_diff, p = GEO_PLOT.test_exact_mc_permutation(
                                    small=ssr_cyc_1h[:, k, j], big=ssr_non_cyc_1h[:, k, j],
                                    nmc=permutation, show=plot_it)
                                print('plot non sig done')

                # plot sig_map ?
                conf_level = 0.05
                sig_map = p_map < conf_level
                GEO_PLOT.plot_geo_subplot_map(
                    ax=axs[2, i],
                    geomap=sig_map,
                    domain='reu', vmin=0, vmax=1, tag=f'{conf_level:4.2f}',
                    bias=0,
                    plt_type='pcolormesh',
                    plot_cbar=False, statistics=0
                )

                # plot 4
                conf_level = 0.01
                sig_map2 = p_map < conf_level
                GEO_PLOT.plot_geo_subplot_map(
                    ax=axs[3, i],
                    geomap=sig_map2,
                    domain='reu', vmin=0, vmax=1, tag=f'{conf_level:4.2f}',
                    bias=0,
                    plt_type='pcolormesh',
                    plot_cbar=False, statistics=0
                )

            plt.savefig('./plot/test_permutation.png', dpi=300)
            plt.show()

            print('done')

    # ============================ OLR regimes pattern class (Benjamin Pohl)
    if any(GEO_PLOT.get_values_multilevel_dict(dict(cfg.job.ttt))):

        ssr_hour_anomaly_pauline = GEO_PLOT.read_to_standard_da(cfg.input.ssr_anomaly_pauline, 'SIS')
        ssr_hour_anomaly_pauline = GEO_PLOT.convert_da_shifttime(ssr_hour_anomaly_pauline, 3600 * 4)

        ssr_hour = GEO_PLOT.read_to_standard_da(cfg.input.ssr_raw_pauline, 'SIS')

        if cfg.job.ttt.diurnal_cycle.plot_diurnal_cycle_map:
            # final plot moisture in OLR:

            sp = GEO_PLOT.read_to_standard_da('~/local_data/era5/q_specific/'
                                              'sp.era5.hourly.1999-2016.bigreu.local_daytime.nc', 'sp')
            d2m = GEO_PLOT.read_to_standard_da('~/local_data/era5/q_specific/'
                                               'd2m.era5.hourly.1999-2016.bigreu.local_daytime.nc', 'd2m')

            q_specific = GEO_PLOT.value_humidity_specific_era5_Bolton(
                dew_point_2m_temp=d2m, surface_pressure=sp, test=0)

            # q_reu = q_specific.where(np.logical_and(q_specific.lon >= 55.05, q_specific.lon <= 56), drop=True)
            # q_reu = q_reu.where(np.logical_and(q_reu.lat <= -20.688, q_reu.lat >= -21.55), drop=True)

            GEO_PLOT.plot_diurnal_cycle_field_in_classif(
                classif=olr_regime, field=q_specific, area='bigreu',
                vmax=0.0174, vmin=0.016, field_bias=False, plot_circulation=1, circulation_anomaly=0,
                plot_moisture_flux=1, field_flux_data=q_specific, circulation_name='moisture flux',
                only_significant_points=True,
                suptitle_add_word='OLR',
                plt_type='pcolormesh',
                test_run=0)

            # final plot: SSR anomaly in the OLR regimes
            Final_Figure.figure_1_ssr_classification_OLR(field=ssr_hour_anomaly_pauline,
                                                         classif=olr_regime,
                                                         # for std:
                                                         # vmin=100, vmax=300,
                                                         # output='Fig.5.std.png',
                                                         # for hourly mean:
                                                         output='Fig.5.png',
                                                         vmin=-120, vmax=80,
                                                         cbar_label=f'SSR (W m**-2)',
                                                         bias=1,
                                                         remove_tc_day=0,
                                                         only_tc_day=0,
                                                         cmap=plt.cm.coolwarm,
                                                         only_sig=True)

            # why Regime 1 has low significant SSR anomaly than Regime 2 :

            da = GEO_PLOT.daily_mean_da(ssr_hour_anomaly_pauline.mean(axis=1).mean(axis=1)).assign_attrs(
                {'units': 'W/m2'})
            df = GEO_PLOT.get_df_of_da_in_classif(da=da, classif=olr_regime)
            df.astype({'class': 'int'})
            GEO_PLOT.plot_violin_boxen_df_1D(df=df, x='class', y='SIS',
                                             x_label='OLR_Regime', y_label='daily SSR anomaly', y_unit=f'W m**-2',
                                             suptitle_add_word='daily SSR anomaly in OLR regimes')

            # the violin plot about shows that: Regime 1 has more positive impact tcs so less significant, and
            # the mean is less pronounced.

            GEO_PLOT.plot_diurnal_cycle_field_in_classif(
                classif=olr_regime, field=ssr_hour_anomaly_pauline,
                area='reu', vmax=80, vmin=-80,
                field_bias=True,
                only_significant_points=True,
                plot_circulation=False,
                suptitle_add_word='OLR',
                test_run=cfg.param.plot_big_data_test)

        if cfg.job.ttt.ssr_olr_class_matrix:
            ssr_cluster = DATA.load_ssr_cluster(data=cfg.input.ssr_clusters, check_missing=1)
            n_ssr_class = '9Cl'
            ssr_class = ssr_cluster[[n_ssr_class]].rename(columns={n_ssr_class: 'SSR_cluster'})
            olr_class = olr_regime.rename(columns={'class': 'OLR_regime'})

            # final final Figure: Fig. 4
            contingency = GEO_PLOT.contingency_2df_table(class_x=ssr_class, class_y=olr_class, plot=True,
                                                         output_figure='Fig.4.png')

        if cfg.job.ttt.olr_regime_statistics:
            from collections import Counter
            print('dominant:')
            Counter(sorted(olr_regime.values.ravel()))

            # dominant:
            # Counter({1: 547, 2: 589, 3: 943, 4: 527, 5: 715, 6: 579, 7: 780})
            # 1st_ens_num:
            # Counter({1: 530, 2: 569, 3: 940, 4: 523, 5: 714, 6: 596, 7: 808})

        olr = GEO_PLOT.read_to_standard_da(cfg.input.olr_ens_swio, 'OLR')

        if cfg.job.ttt.plot_ttt_regimes:
            print(f'loading')
            GEO_PLOT.plot_ttt_regimes(olr_regimes=olr_regime, olr=olr,
                                      area='SA_swio', contour=False,
                                      paper_plot=1,
                                      only_significant_points=0)

        if cfg.job.ttt.statistics:
            GEO_PLOT.plot_matrix_classification_at_year_and_month(class_df=olr_regime,
                                                                  output_plot=cfg.output.ttt_class_monthly)

        if cfg.job.ttt.field_in_ttt.large_scale:
            print(f'loading ssr era5')
            ssr = GEO_PLOT.read_to_standard_da(cfg.input.ssr_era5, var='ssrd')
            ssr = GEO_PLOT.convert_unit_era5_flux(ssr, is_ensemble=False)
            ssr_anomaly = GEO_PLOT.anomaly_daily(ssr)

            GEO_PLOT.plot_field_in_classif(
                field=ssr_anomaly, classif=olr_regime, area='SA_swio',
                suptitle_add_word=f'era5, only_robust_class={cfg.param.remove_uncertain_days:g}',
                vmax=40, vmin=-40, plot_wind=False, bias=False, only_significant_points=False)

        if cfg.job.ttt.field_in_ttt.local_scale:
            ssr_day = ssr.groupby(ssr.time.dt.date).mean(keep_attrs=True).rename(date='time')
            from datetime import time, datetime
            dt = [datetime.combine(x, time()) for x in ssr_day.time.values]
            ssr_day = ssr_day.assign_coords(time=('time', dt))

            ssr_daily_anomaly = GEO_PLOT.anomaly_daily(ssr_day)

            GEO_PLOT.plot_field_in_classif(
                field=ssr_daily_anomaly, classif=olr_regime, area='bigreu',
                suptitle_add_word=f'SARAH-E, only_robust_class={cfg.param.remove_uncertain_days:g}',
                vmax=40, vmin=-40, plot_wind=False, bias=1, only_significant_points=1)

        if cfg.job.ttt.temporal_correlation:
            print(f'loading ssr sarah_e')
            ssr_anomaly = GEO_PLOT.read_to_standard_da(cfg.input.ssr_sarah_e, var='SIS')
            season_reu_mean_ssr: np.ndarray = MIALHE_2021.get_spatial_mean_series_each_ttt_season(
                da=ssr_anomaly, season='NDJF', year_start=cfg.param.sarah_e_start_year,
                year_end=cfg.param.sarah_e_end_year, remove_29_feb=True)

            class_name = [5, 6, 7]

            print(f'class in each season')

            freq_df: pd.DataFrame = MIALHE_2021.get_class_occurrence_ttt_seasonal_series(
                class_name=class_name, ttt_classif=olr_regime, year_start=cfg.param.sarah_e_start_year,
                year_end=cfg.param.sarah_e_end_year)

            GEO_PLOT.plot_class_occurrence_and_anomaly_time_series(classif=freq_df, anomaly=season_reu_mean_ssr)

            print(f'good')

        if cfg.job.ttt.diurnal_cycle.plot_diurnal_cycle_boxplot:
            ssr_spatial_mean = ssr.mean('lon', keep_attrs=True).mean('lat', keep_attrs=True)
            ssr_hour_anomaly = GEO_PLOT.anomaly_hourly(da=ssr_spatial_mean, percent=0)

            GEO_PLOT.plot_diurnal_boxplot_in_classif(
                classif=olr_regime, field=ssr_hour_anomaly,
                anomaly=1, ylimits=[-0.5, 0.5],
                suptitle_add_word='OLR bigreu sarah_e',
                plot_big_data_test=cfg.param.plot_big_data_test
            )

        if cfg.job.ttt.diurnal_cycle.plot_diurnal_curve_in_class:
            # ssr_hour raw in

            # multiyear mean diurnal SSR in NDJF

            ssr_hour = GEO_PLOT.read_to_standard_da(cfg.input.ssr_raw_pauline, 'SIS')
            ssr_hour_NDJF = ssr_hour.where(ssr_hour.time.dt.month.isin([11, 12, 1, 2]), drop=True)
            ssr_hour_NDJF_mean = ssr_hour_NDJF.groupby(ssr_hour_NDJF.time.dt.hour).mean('time', keep_attrs=True)

            ssr_hour_NDJF_space_mean = ssr_hour_NDJF_mean.mean('x', keep_attrs=True).mean('y', keep_attrs=True)

            # plot climatology:
            da = ssr_hour_NDJF_space_mean
            da.plot()
            plt.title('SSR mean NDJF')
            plt.grid(True)
            plt.show()

            ssr_hour_anomaly_pauline_space_mean = ssr_hour_anomaly_pauline.mean(
                'x', keep_attrs=True).mean('y', keep_attrs=True)

            percentage = ssr_hour_anomaly_pauline_space_mean.groupby(
                ssr_hour_anomaly_pauline_space_mean.time.dt.hour) / ssr_hour_NDJF_space_mean

            percentage = percentage * 100
            percentage = percentage.assign_attrs({'units': '', 'long_name': 'SSR'})

            percentage.to_netcdf(f'./dataset/ssr_hour_anomaly_ratio_ssr_raw_NDJF_space_mean.nc')

            Final_Figure.figure_19_diurnal_curve_in_classif(
                classif=olr_regime, field_1D=percentage,
                anomaly=1, percent=1,
                suptitle_add_word='OLR regime reu sarah_e',
                ylimits=[-15, 12],
                plot_big_data_test=0)

        if cfg.job.ttt.cyclone_in_ttt:
            df = Final_Figure.TC_day_regime_ssr(regime=olr_regime, ssr=ssr_hour_anomaly_pauline, tc=cyclone_class)

            GEO_PLOT.plot_cyclone_in_classif(classif=olr_regime,
                                             radius=5,
                                             suptitle_add_word='1999-2016 OLR regimes NDJF'
                                             )
            all_regime = olr_regime * 0
            GEO_PLOT.plot_cyclone_in_classif(classif=all_regime,
                                             radius=5,
                                             tag_subplot='All Regimes',
                                             suptitle_add_word='1999-2016 OLR regimes NDJF'
                                             )

            Final_Figure.statistics_tc_olr(olr=olr_regime, nearby_radius=5)
            # the above function is based on the following functions

            Final_Figure.plot_all_cyclone_path_in_OLR_2(classif=olr_regime, radius=5,
                                                        suptitle_add_word='1981-2016 OLR regimes in NDJF swio')

            Final_Figure.plot_all_cyclone_path_in_OLR(classif=all_regime, radius=30,
                                                      subplot_tag='All Regimes',
                                                      suptitle_add_word='1981-2016 OLR regimes in NDJF swio')

        if cfg.job.ttt.statistics_for_pv.violin_plot:

            # note: the multi gaussian distribution is from the complexity of space and time:
            # note: daily: more months, closer to 1 gaussian
            # note: daily: more stations, closer to 1 gaussian
            # note: hourly: fewer stations, closer to 1 gaussian
            # note: hourly: fewer months, closer to 1 gaussian

            if cfg.job.ttt.statistics_for_pv.st_denis_only:
                space_tag = 'st_denis_only'
                index_lon = int(np.abs(ssr.lon - 55.45).argmin())
                index_lat = int(np.abs(ssr.lat - (-20.89)).argmin())
                ssr = ssr[:, index_lat, index_lon]
            else:
                space_tag = 'reunion_mean'
                ssr = ssr.mean(dim=['x', 'y'], keep_attrs=True)

            # for daily mean
            ssr_day_mean = GEO_PLOT.daily_mean_da(ssr)

            ssr_day_mean = ssr_day_mean.where(ssr_day_mean.time.dt.month.isin([1, 2, 11, 12]), drop=True)
            df = GEO_PLOT.get_df_of_da_in_classif(da=ssr_day_mean, classif=olr_regime)

            list_mon = set(list(ssr_day_mean.time.dt.month.values))
            GEO_PLOT.plot_violin_boxen_df_1D(x='class', y=ssr_day_mean.name,
                                             y_unit=ssr_day_mean.units,
                                             hue=0,
                                             df=df,
                                             suptitle_add_word=f'OLR regimes daily mean\n'
                                                               f'{space_tag:s}\n'
                                                               f'month={str(list_mon):s}')

            # for hourly mean
            ssr_hour_mean = ssr
            ssr_hour_mean = ssr_hour_mean.where(ssr_hour_mean.time.dt.month.isin([1, 2, 11, 12]), drop=True)

            df_hour_mean = GEO_PLOT.get_df_of_da_in_classif(da=ssr_hour_mean, classif=olr_regime)

            list_mon = set(list(ssr_hour_mean.time.dt.month.values))
            GEO_PLOT.plot_violin_boxen_df_1D(x='class', y=ssr_hour_mean.name,
                                             y_unit=ssr_hour_mean.units,
                                             hue=0,
                                             df=df_hour_mean,
                                             suptitle_add_word=f'OLR regimes hourly mean\n'
                                                               f'{space_tag:s}\n'
                                                               f'month={str(list_mon):s}')

            # for hourly anomaly
            ssr_hour_anomaly = GEO_PLOT.anomaly_hourly(da=ssr, percent=1)

            df_hour_anomaly = GEO_PLOT.get_df_of_da_in_classif(da=ssr_hour_anomaly, classif=olr_regime)

            GEO_PLOT.plot_violin_boxen_df_1D(x='class', y=ssr_hour_anomaly.name,
                                             y_unit=ssr_hour_anomaly.units,
                                             hue=0,
                                             df=df_hour_anomaly,
                                             suptitle_add_word=f'OLR regimes hourly normalized anomaly\n'
                                                               f'{space_tag:s}\n'
                                                               f'month={str(list_mon):s}')

            print(f'convert to pandas df')

    if any(GEO_PLOT.get_values_multilevel_dict(dict(cfg.job.mjo))):

        print(f'starting to read data for MJO analysis ...')

        ssr_hour_anomaly_pauline = GEO_PLOT.read_to_standard_da(cfg.input.ssr_anomaly_pauline, 'SIS')
        ssr_hour_anomaly_pauline = GEO_PLOT.convert_da_shifttime(ssr_hour_anomaly_pauline, 3600 * 4)

        ssr_cluster = DATA.load_ssr_cluster(data=cfg.input.ssr_clusters, check_missing=False)
        n_ssr_class = '9Cl'
        ssr_class = ssr_cluster[[n_ssr_class]].rename(columns={n_ssr_class: 'SSR_cluster'})

        # ------------------------------ mjo class
        print(f'prepare MJO in seasons and amplitude')
        # read mjo: ( all amplitude 6575 days) for statistics:
        mjo_12m = GEO_PLOT.read_mjo(match_ssr_avail_day=0)
        mjo_12m['month'] = mjo_12m.index.month
        # mjo_12m_class = mjo_12m[{'phase'}].rename(columns={'phase': 'class'})

        # impact study are done for high amplitude mjo events only:
        mjo_12m_high = mjo_12m[mjo_12m['amplitude'] > 1]
        # mjo_12m_high_class = mjo_12m_high[{'phase'}].rename(columns={'phase': 'class'})

        # mjo_DJF = mjo_12m[mjo_12m.index.month.isin([12, 1, 2])]
        # mjo_DJF_high = mjo_DJF[mjo_DJF['amplitude'] > 1]
        # mjo_DJF_high_class = pd.DataFrame(mjo_DJF_high['phase']).rename(columns={'phase': 'MJO_phase'})
        # # high DJF: 1087 days

        mjo_NDJF = mjo_12m[mjo_12m.index.month.isin([11, 12, 1, 2])]
        mjo_NDJF_high = mjo_NDJF[mjo_NDJF['amplitude'] > 1]
        mjo_NDJF_high_class = pd.DataFrame(mjo_NDJF_high['phase']).rename(columns={'phase': 'MJO_phase'})
        # high NDJF: 1391 days
        #
        # # low amplitude: mjo noise:
        # mjo_DJF_low = mjo_DJF[mjo_DJF['amplitude'] < 1]
        #
        # mjo_JJA = mjo_12m[mjo_12m.index.month.isin([6, 7, 8])]
        # mjo_JJA_high = mjo_JJA[mjo_JJA['amplitude'] > 1]
        # mjo_JJA_high_class = pd.DataFrame(mjo_JJA_high['phase']).rename(columns={'phase': 'MJO_phase'})
        # # high JJA 989 days
        #
        # mjo_JJAS = mjo_12m[mjo_12m.index.month.isin([6, 7, 8, 9])]
        # mjo_JJAS_high = mjo_JJAS[mjo_JJAS['amplitude'] > 1]
        # mjo_JJAS_high_class = pd.DataFrame(mjo_JJAS_high['phase']).rename(columns={'phase': 'MJO_phase'})
        # # high JJAS: 1312 days

        # ------------------------------

        if cfg.job.mjo.cyclone_in_mjo:
            olr_swio = GEO_PLOT.read_to_standard_da(cfg.input.olr_day_swio, 'OLR')
            swio, reu = Final_Figure.plot_all_cyclone_path_in_MJO_2(classif=mjo_NDJF_high_class, radius=5,
                                                                    plot_mjo_phases=1,
                                                                    olr=olr_swio,
                                                                    suptitle_add_word='1981-2016 MJO in NDJF swio')

            GEO_PLOT.plot_cyclone_in_classif(classif=mjo_NDJF_high_class,
                                             radius=5,
                                             suptitle_add_word='1981-2016 strong MJO NDJF'
                                             )
        if cfg.job.mjo.mjo_ssr_class_matrix:
            # final plot: contingency: Fig. 6
            contingency = GEO_PLOT.contingency_2df_table(class_x=ssr_class, class_y=mjo_NDJF_high_class, plot=True,
                                                         output_figure='Fig.6.png')
            #
            GEO_PLOT.plot_matrix_class_vs_class(class_x=mjo_NDJF_high_class,
                                                class_y=ssr_class,
                                                occurrence=1,
                                                suptitle_add_word=f'(N_SSR_class={n_ssr_class:s})',
                                                output_plot=f'plot/{n_ssr_class:s}_matrix_ssr_MJO_NDJF_high.png')

        # season will be controlled by mjo, so here ssr in all month
        ssr_hour_12m = DATA.reading_sarah_e_hour_reu(
            select_only_land=False, select_only_NDJF=False,
            select_day_time_hours=True, for_test=False, clearsky=False)
        # ssr_hour_anomaly = GEO_PLOT.read_to_standard_da(cfg.output.ssr_total_sky_hour_reu_anomaly, 'SIS')

        # ssr_day = GEO_PLOT.read_to_standard_da(cfg.output.ssr_total_sky_day_reu, 'SIS')
        # ssr_day_anomaly = GEO_PLOT.read_to_standard_da(cfg.output.ssr_total_sky_day_reu_anomaly, 'SIS')

        if cfg.job.mjo.statistics:

            # daily ssr anomaly in month and phase of MJO:
            seasons = ['NDJF', 'JJAS', 'DJF', 'JJA', '12M']
            seasonal_data = [mjo_NDJF_high, mjo_12m_high]

            for i in range(len(seasons)):
                GEO_PLOT.plot_matrix_class_vs_class_field(
                    class_x=seasonal_data[i][{'month'}],
                    class_y=seasonal_data[i][{'phase'}],
                    field=ssr_day_anomaly,
                    plt_type='pcolormesh',
                    vmax=50, vmin=-50, bias=1,
                    occurrence=1,
                    only_significant_points=1,
                    suptitle_add_word=f'high amplitude daily SSR anomaly in '
                                      f'{seasons[i]:s}')

            GEO_PLOT.plot_matrix_2d_df(
                df=mjo_12m,
                x_column='month', x_label='Month',
                y_column='phase', y_label='MJO_phase',
                z_label='amplitude', z_column='amplitude',
                cut_off=True, cut_value=1,
                x_plt_limit=[0, 13], y_plt_limit=[0, 4], z_plt_limit=[0, 4],
                statistics=1, occurrence=1,
                suptitle_add_word='MJO all month')

            # violin, to see amplitude in phase and time:
            GEO_PLOT.plot_violin_boxen_df_1D(
                x='month', y='amplitude', hue='phase',
                x_label='month', y_label='amplitude',
                y_unit='',
                df=mjo_12m_high,
                suptitle_add_word=f'intense_1'
            )

            # monthly distribution:
            GEO_PLOT.plot_mjo_monthly_distribution(mjo_12m_high, instense=1)

            GEO_PLOT.plot_matrix_classification_at_year_and_month(
                class_df=mjo_12m_high[{'phase'}].rename(columns={'phase': 'class'}),
                output_plot=cfg.output.mjo_class_monthly)

        if cfg.job.mjo.plot_mjo_phase:
            olr_swio = GEO_PLOT.read_to_standard_da(cfg.input.olr_day_swio, 'OLR')
            GEO_PLOT.plot_mjo_phase(mjo_phase=mjo_DJF_high, olr=olr_swio, high_amplitude=1,
                                    month='DJF', only_significant_points=1)

            GEO_PLOT.plot_mjo_phase(mjo_phase=mjo_NDJF_high, olr=olr_swio, high_amplitude=1,
                                    month='NDJF', only_significant_points=1)

        if cfg.job.mjo.plot_field_in_mjo:

            for season in ["austral_summer", ]:
                # SSR
                # MIALHE_2021.plot_fields_in_mjo_phases(mjo_phase=mjo, field=ssr_reu, area='bigreu', season=season,
                #                                       only_significant_points=1, consistency=0)
                # MIALHE_2021.plot_fields_in_mjo_phases(mjo_phase=mjo, field=ssr_reu, area='bigreu', season=season,
                #                                       only_significant_points=0, consistency=1)
                # OLR
                # MIALHE_2021.plot_fields_in_mjo_phases(mjo_phase=mjo, field=olr_reu, area='bigreu', season=season,
                #                                       only_significant_points=1, consistency=1)
                # MIALHE_2021.plot_fields_in_mjo_phases(
                # mjo_phase=mjo, field=olr_reu, area='bigreu', season=season, consistency=1)

                # MIALHE_2021.plot_fields_in_mjo_phases(
                # mjo_phase=mjo, field=olr_swio, area='SA_swio', season=season, only_significant_points=1)
                # MIALHE_2021.plot_fields_in_mjo_phases(
                # mjo_phase=mjo, field=olr_swio, area='SA_swio', season=season,
                # only_significant_points=1, consistency=1)

                # MIALHE_2021.plot_fields_in_mjo_phases(
                # mjo_phase=mjo, field=olr_swio, area='SA_swio', season=season, percentage=1)

                # MIALHE_2021.plot_fields_in_mjo_phases(
                # mjo_phase=mjo, field=w500_reu, area='reu', season=season)
                # MIALHE_2021.plot_fields_in_mjo_phases(
                # mjo_phase=mjo, field=w500_reu, area='reu', season=season, percentage=1)
                # MIALHE_2021.plot_fields_in_mjo_phases(
                # mjo_phase=mjo, field=w500_sa_swio, area='SA_swio', season=season)
                # MIALHE_2021.plot_fields_in_mjo_phases(
                # mjo_phase=mjo, field=w500_sa_swio, area='SA_swio', season=season, percentage=1)

                # w500 = xr.open_dataset(f'{WORKING_DIR:s}/data/mjo/w500.era5.1999-2016.daily.anomaly.d01.nc')['w']
                # MIALHE_2021.plot_fields_in_mjo_phases(mjo_phase=mjo, field=w500)

                print(f'good')

        if cfg.job.mjo.diurnal_cycle.plot_diurnal_cycle_map:
            # radiation:
            # NDJF high amplitude: final plot:
            # 1. all the phases from 1 to 8:
            mjo_plot = mjo_NDJF_high_class

            # 2. plot only phase 4 to 7, since other phases have almost non point significant
            mjo_plot = mjo_NDJF_high_class[mjo_NDJF_high_class > 3].dropna()
            mjo_plot = mjo_plot[mjo_plot < 8].dropna()

            mjo_plot = mjo_plot.rename(columns={'MJO_phase': 'class'})
            Final_Figure.figure_7_ssr_classification_MJO(field=ssr_hour_anomaly_pauline, classif=mjo_plot,
                                                         # vmax=50, vmin=-400,  #     use this value to plot TC and no-TC
                                                         vmax=50, vmin=-50,   #     use this value to plot all days
                                                         output='Figure.7.all_phase.png', bias=True,
                                                         remove_tc_day=0,
                                                         only_tc_day=0,
                                                         cbar_label=f'SSR anomaly (W m**-2)', cmap=plt.cm.coolwarm,
                                                         only_sig=1)
            # moisture:
            sp = GEO_PLOT.read_to_standard_da('~/local_data/era5/q_specific/'
                                              'sp.era5.hourly.1999-2016.bigreu.local_daytime.nc', 'sp')
            d2m = GEO_PLOT.read_to_standard_da('~/local_data/era5/q_specific/'
                                               'd2m.era5.hourly.1999-2016.bigreu.local_daytime.nc', 'd2m')
            q_specific = GEO_PLOT.value_humidity_specific_era5_Bolton(
                dew_point_2m_temp=d2m, surface_pressure=sp, test=0)

            GEO_PLOT.plot_diurnal_cycle_field_in_classif(
                classif=mjo_NDJF_high_class, field=q_specific, area='bigreu',
                vmax=0.0174, vmin=0.016, field_bias=False, plot_circulation=1, circulation_anomaly=0,
                plot_moisture_flux=1, field_flux_data=q_specific, circulation_name='moisture flux',
                only_significant_points=True,
                suptitle_add_word='MJO',
                plt_type='pcolormesh',
                test_run=0)

            q_specific_anomaly = GEO_PLOT.anomaly_hourly(q_specific)

            GEO_PLOT.plot_diurnal_cycle_field_in_classif(
                classif=mjo_NDJF_high_class, field=q_specific_anomaly, area='bigreu',
                vmax=1.5e-4, vmin=-1.5e-4, field_bias=True, plot_circulation=1, circulation_anomaly=1,
                plot_moisture_flux=1, field_flux_data=q_specific_anomaly,
                circulation_name='moisture flux anomaly',
                only_significant_points=True,
                suptitle_add_word='MJO',
                plt_type='pcolormesh',
                test_run=0)

            # SSR, all month:



            GEO_PLOT.plot_diurnal_cycle_field_in_classif(
                classif=mjo_NDJF_high_class, field=ssr_hour_anomaly, area='bigreu',
                vmax=30, vmin=-30, field_bias=True, plot_circulation=0, circulation_anomaly=0,
                only_significant_points=True,
                suptitle_add_word='MJO NDJF high amplitude',
                plt_type='pcolormesh',
                test_run=0)

            # CF:
            cfc = GEO_PLOT.read_to_standard_da(
                '/Users/ctang/local_data/cmsaf/CFC.hourly.cmsaf.1999-2015.bigreu.local_daytime.nc',
                'CFC')

            cfc_hourly_anomaly = GEO_PLOT.anomaly_hourly(cfc)

            GEO_PLOT.plot_diurnal_cycle_field_in_classif(
                classif=mjo_DJF_high_class, field=cfc_hourly_anomaly, area='bigreu',
                vmax=30, vmin=-30, field_bias=True, plot_circulation=1, circulation_anomaly=1,
                only_significant_points=True,
                suptitle_add_word='MJO',
                test_run=0)

            # OLR:

            # convergence:

        if cfg.job.mjo.diurnal_cycle.plot_diurnal_cycle_boxplot:
            ssr_spatial_mean = ssr_hour_12m.mean('x', keep_attrs=True).mean('y', keep_attrs=True)
            ssr_hour_anomaly_spatial_mean = GEO_PLOT.anomaly_hourly(da=ssr_spatial_mean, percent=0)

            GEO_PLOT.plot_diurnal_boxplot_in_classif(
                classif=mjo_NDJF_high[{'phase'}], field=ssr_hour_anomaly_spatial_mean,
                anomaly=1, ylimits=[-200, 200],
                suptitle_add_word='MJO bigreu sarah_e',
                plot_big_data_test=0)

        if cfg.job.mjo.diurnal_cycle.plot_diurnal_curve_in_class:
            percentage = GEO_PLOT.read_to_standard_da(f'./dataset/ssr_hour_anomaly_ratio_ssr_raw_NDJF_space_mean.nc',
                                                      'SIS')

            Final_Figure.figure_21_diurnal_curve_in_classif_MJO(
                classif=mjo_NDJF_high_class, field_1D=percentage,
                anomaly=1, percent=1,
                suptitle_add_word='mjo ndjf high amplitude reu sarah_e',
                ylimits=[-15, 8],
                plot_big_data_test=0)

        if any(GEO_PLOT.get_values_multilevel_dict(dict(cfg.job.mjo.ttt_vs_mjo))):

            # read OLR regimes in NDJF:
            olr_regime = DATA.load_olr_regimes(
                mat_file=cfg.input.ttt_Ben_class_file, classif=cfg.param.ttt_classif)
            # days(NDJF, 29 Feb suppressed for leap yrs) x 39 seasons (1979-80 to 2017-18) x 10 members.
            olr_class = olr_regime.rename(columns={'class': 'OLR_regime'})

            # SSR:
            ssr_hourly_anomaly = GEO_PLOT.read_to_standard_da(cfg.input.ssr_anomaly_pauline, 'SIS')

            # plot only few hours:
            ssr_hourly_anomaly_5 = ssr_hourly_anomaly.where(
                ssr_hourly_anomaly.time.dt.hour.isin([8, 10, 12, 14, 17]), drop=True)

            if cfg.job.mjo.ttt_vs_mjo.compare_ssr_in_ttt_mjo_class:
                # merge this two, so ONLY in NDJF

                mjo_ttt: pd.DataFrame = mjo_NDJF_high.merge(olr_regime, left_index=True, right_index=True)
                # ['rmm1', 'rmm2', 'phase', 'amplitude', 'class']

                # select target phase according to the level of SSR anomaly (see ppt):
                sel_regime_olr = [2, 6]
                sel_phase_mjo = [5, 6]

                sel_mjo_olr = mjo_ttt.where(
                    np.logical_and(mjo_ttt.phase.isin(sel_phase_mjo), mjo_ttt['class'].isin(sel_regime_olr))).dropna()
                sel_mjo_olr = sel_mjo_olr.rename(columns={'phase': 'mjo', 'class': 'olr'})[{'mjo', 'olr'}]
                # only 126 days totally

                # statistics: num of selected phase/regime
                for i in range(len(sel_phase_mjo)):
                    print(f'mjo: phase {sel_phase_mjo[i]:g} = {len(sel_mjo_olr[sel_mjo_olr.mjo == sel_phase_mjo[i]])}')
                for j in range(len(sel_regime_olr)):
                    print(
                        f'olr: regime {sel_regime_olr[j]:g} = {len(sel_mjo_olr[sel_mjo_olr.olr == sel_regime_olr[j]])}')

                # put the 4 mixing possibilities in class:
                sel_mjo_olr['class'] = 999

                name_class = []

                k = 0
                for i in range(2):
                    for j in range(2):
                        sel_mjo_olr.loc[
                            (sel_mjo_olr.mjo == sel_phase_mjo[i]) &
                            (sel_mjo_olr.olr == sel_regime_olr[j]), 'class'] = range(4)[k]
                        name_class.append(f'mjo{sel_phase_mjo[i]:g}-olr{sel_regime_olr[j]}')
                        k += 1

                # statistics:
                for i in range(4):
                    print(i, len(sel_mjo_olr[sel_mjo_olr['class'] == i]))

                # plot the matrix:

                # some plots are done before, while here to have a single colorbar:
                # this function is adaptable to changing hours and n_class
                # mix OLR_MJO vs SSR:
                GEO_PLOT.plot_diurnal_cycle_field_in_classif(
                    classif=sel_mjo_olr[{'class'}],
                    str_class_names=name_class,
                    field=ssr_hourly_anomaly_5, area='bigreu',
                    vmax=100, vmin=-100, field_bias=True, plot_circulation=0, circulation_anomaly=0,
                    only_significant_points=True,
                    suptitle_add_word='MJO vs OLR',
                    test_run=0)

                # plot OLR vs SSR:
                GEO_PLOT.plot_diurnal_cycle_field_in_classif(
                    classif=olr_regime,
                    field=ssr_hourly_anomaly_5, area='bigreu',
                    vmax=100, vmin=-100, field_bias=True, plot_circulation=0, circulation_anomaly=0,
                    only_significant_points=True,
                    suptitle_add_word='MJO vs OLR olr_class',
                    test_run=0)

                # plot MJO vs SSR:
                GEO_PLOT.plot_diurnal_cycle_field_in_classif(
                    classif=mjo_NDJF_high_class,
                    field=ssr_hourly_anomaly_5, area='bigreu',
                    vmax=100, vmin=-100, field_bias=True, plot_circulation=0, circulation_anomaly=0,
                    only_significant_points=True,
                    suptitle_add_word='MJO vs OLR mjo_class',
                    test_run=0)

                print('good')

            if cfg.job.mjo.ttt_vs_mjo.plot_mjo_olr_matrix_maps:
                ssr_day_anomaly_pauline = GEO_PLOT.read_to_standard_da(cfg.input.ssr_daily_anomaly_pauline, 'SIS')

                # final figure:
                # mjo has feb 29, so have 4 days more than ttt
                # mjo = 1391 days, high amplitude
                # ttt has 2160 days without Feb. 29
                # merge = 1387 days
                contingency2 = GEO_PLOT.plot_matrix_class_vs_class_field(
                    class_x=olr_class, class_y=mjo_NDJF_high_class,
                    field=ssr_day_anomaly_pauline,
                    plt_type='pcolormesh',
                    vmax=100, vmin=-100, bias=1,
                    occurrence=1,
                    only_significant_points=1,
                    suptitle_add_word='NDJF')

                # =====================================================
                # test contingency code Macron 2016
                cross = [
                    [28, 13, 31, 18, 19, 23, 21],
                    [25, 36, 35, 30, 19, 21, 28],
                    [20, 49, 45, 48, 23, 28, 26],
                    [50, 26, 41, 21, 35, 20, 11],
                    [45, 11, 26, 17, 24, 21, 25],
                    [63, 5, 28, 17, 40, 30, 23],
                    [69, 18, 25, 17, 44, 27, 35],
                    [25, 23, 20, 11, 39, 28, 33]
                ]
                a025 = GEO_PLOT.value_sig_neu_test_2d(
                    contingency=np.array(cross), alpha=0.025, output_expected=1)

                a05 = GEO_PLOT.value_sig_neu_test_2d(
                    contingency=np.array(cross), alpha=0.05, output_expected=1)
                # =====================================================
                # an individual column test:

                cross_nd = np.array(cross)
                p_expect = np.array(cross_nd.sum(axis=1) / np.sum(cross_nd))
                cross_1col = np.array(cross)[:, 3].reshape(-1, 1)

                a05 = GEO_PLOT.value_sig_neu_test_2d(
                    contingency=cross_1col, alpha=0.05, p_expected=p_expect, output_expected=1)

                # =====================================================
                # another test data:
                cross = [
                    [125, 42, 98, 84, 170, 68, 21],
                    [40, 31, 64, 76, 190, 138, 52],
                    [114, 86, 175, 90, 60, 168, 157],
                    [191, 97, 115, 69, 16, 14, 131],
                    [207, 119, 103, 117, 137, 25, 90]
                ]
                a05 = GEO_PLOT.value_sig_neu_test_2d(
                    contingency=np.array(cross), alpha=0.05, output_expected=1)
                # =====================================================
                # test: << Analysis of Wildlife Radio-Tracking Data>>, Page 188
                cross = [
                    [302, ],
                    [180, ],
                    [69, ],
                    [49, ],
                    [50, ]
                ]
                expected = [0.5, 0.3, 0.1, 0.05, 0.05]

                a1 = GEO_PLOT.value_sig_neu_test_2d(
                    contingency=np.array(cross), alpha=0.1, p_expected=expected, output_expected=1)
                print(a1)
                # test: << Analysis of Wildlife Radio-Tracking Data>>
                # =====================================================

                a1 = GEO_PLOT.value_sig_neu_test_2d(
                    contingency=np.array(cross), alpha=0.1, output_expected=1)

                a01 = GEO_PLOT.value_sig_neu_test_2d(
                    contingency=np.array(cross), alpha=0.01, output_expected=1)

                a025 = GEO_PLOT.value_sig_neu_test_2d(
                    contingency=np.array(cross), alpha=0.025, output_expected=1)

                a05 = GEO_PLOT.value_sig_neu_test_2d(
                    contingency=np.array(cross), alpha=0.05, output_expected=1)

                a1 = GEO_PLOT.value_sig_neu_test_2d(
                    contingency=np.array(cross), alpha=0.1, output_expected=1)

                print(f'good')

    if any(GEO_PLOT.get_values_multilevel_dict(dict(cfg.job.pv_statistics))):

        # start to read all classifications:
        ssr_cluster = DATA.load_ssr_cluster(cfg.input.ssr_clusters, check_missing=0)
        # not for analysis but for select cyclone days

        mjo: pd.DataFrame = GEO_PLOT.read_mjo(match_ssr_avail_day=1)
        mjo: pd.DataFrame = GEO_PLOT.filter_by_season_name(data=mjo, season_name='austral_summer')

        # read cyclone class into pd.DataFrame, all ssr period:
        cyclone_class: pd.DataFrame = DATA.cal_cyclone_class_in_ssr_period(
            ssr_cluster=ssr_cluster, raw_cyclone_data=cfg.input.cyclone_all,
            nearby_radius=3, find_day_with_more_than_one_cyclone=0)
        # select cyclone season: NDJFM:
        summer_cyclone = cyclone_class[cyclone_class.index.month.isin([1, 2, 3, 11, 12])]

        # statistics of cyc in ssr period summer
        n_d_near_summer = len(summer_cyclone[summer_cyclone['class'] > 0])
        print(f'summer: {n_d_near_summer:g} nearby cyclones')
        # read OLR in NDJF:

        olr_regime = DATA.load_olr_regimes(
            mat_file=cfg.input.ttt_Ben_class_file, classif=cfg.param.ttt_classif)

        print(f'all classif/variability df have to be loaded before this line')

        # read clearsky radiation:
        print(f'\n' f'start to calculate clearsky radiation...')

        # using all hours, so before sun set, total sky (sarah_e) may > clearsky, redirection
        ssr_land, ssr_land_clearsky = DATA.reading_sarah_e_hour_reu(
            select_only_land=True,
            select_day_time_hours=False, select_only_NDJF=False,
            for_test=0,
            clearsky=1)

        clearsky_index = ssr_land / ssr_land_clearsky
        clearsky_index = clearsky_index.assign_attrs({'units': 'W/m2', 'long_name': 'ghi'}).rename('ghi')

        clearsky_index = clearsky_index.where(
            np.logical_and(clearsky_index.time.dt.hour >= 8,
                           clearsky_index.time.dt.hour <= 17), drop=True)

        # when considering all 24 hours,
        # if clearsky value is 0, the index will have some inf values around 18h and 19h,
        # after or before this both all sky and clearsky are zeros, so got only Nan, not inf.
        print(f'so we have some infinite values at these times:\n',
              clearsky_index.where(np.isinf(clearsky_index), drop=True).time)

        clearsky_index = clearsky_index.where(np.logical_not(np.isinf(clearsky_index)), 0)
        # key word: replace, inf,

        print(f'so we also have some very large values at these times:\n',
              clearsky_index.where(clearsky_index > 1.5, drop=True).time)

        # when consider 24 hours, we will have very high values, so make it to 2.
        # if only day time 8am to 5pm, then max =2.7, so not necessary to change it
        # clearsky_index = clearsky_index.where(np.logical_not(clearsky_index > 1.5), 2)
        # note: all index > 1.3 ara in the hour of 16-19h.
        print(f'hours when we have index > 1.2 : \n',
              np.array(set(list(clearsky_index.where(clearsky_index > 1.2, drop=True).time.dt.hour.values))))

        GEO_PLOT.plot_geo_map(clearsky_index[10], cb_limits=[0.4, 1], plt_limits=[55.2, 55.9, -21.4, -20.9],
                              suptitle_add_word='clearsky index')

        # ------ all data are prepared, so statistics in loop: -------------
        var_dfs = [summer_cyclone, olr_regime, mjo[{'phase'}].rename(columns={'phase': 'class'})]
        var_names = ['tropical_cyclone', 'OLR regime', 'MJO index']
        var_months = ['DJFM', 'NDJF', 'NDJF']

        # calculate daily energy using 24 hours data
        solar_energy = GEO_PLOT.cal_daily_total_energy(ssr_land)
        # remove the ocean 0 values for plot:
        solar_energy = GEO_PLOT.select_land_only_reunion_by_altitude(solar_energy)

        solar_energy_clearsky = GEO_PLOT.cal_daily_total_energy(ssr_land_clearsky)
        solar_energy_clearsky = GEO_PLOT.select_land_only_reunion_by_altitude(solar_energy_clearsky)

        energy_index = solar_energy / solar_energy_clearsky
        energy_index = energy_index.assign_attrs({
            'units': 'index',
            'long_name': solar_energy.long_name + ' all vs clear sky'}).rename('daily_total_energy_density')
        energy_index = GEO_PLOT.select_land_only_reunion_by_altitude(energy_index)

        # loop in all variability:
        for i in range(len(var_dfs)):
            classif = var_dfs[i]

            df_hour_clearsky_index = GEO_PLOT.get_df_of_da_in_classif(
                da=clearsky_index, classif=classif)

            # plot clear sky index
            GEO_PLOT.plot_violin_boxen_df_1D(x='class', y=clearsky_index.name,
                                             y_unit='GHI/GHI_clearsky',
                                             hue=0,
                                             df=df_hour_clearsky_index,
                                             suptitle_add_word=f'GHI_clearsky_index '
                                                               f'hourly all pixels over land area '
                                                               f'in {var_names[i]:s} '
                                                               f'over {var_months[i]:s}')
            # field of mean clearsky index:
            GEO_PLOT.plot_field_in_classif(
                field=clearsky_index, classif=classif, area='reu',
                vmax=1, vmin=0.4, bias=0, plot_wind=0,
                only_significant_points=0, plt_type='pcolormesh',
                cmap=plt.cm.Blues,
                suptitle_add_word=f'{var_names[i]:s} {var_months[i]:s} clearsky_index')

            # diurnal clearsky index: boxplot:
            GEO_PLOT.plot_diurnal_boxplot_in_classif(
                classif=classif,
                field=clearsky_index,
                ylimits=[0, 1.4],
                relative_data=1,
                suptitle_add_word=f'{var_names[i]:s} clearsky index all pixel over land')

            # plot total daily energy density
            GEO_PLOT.plot_field_in_classif(
                field=energy_index, classif=classif, area='reu',
                vmax=1, vmin=0.4, bias=0, plot_wind=0,
                only_significant_points=0, plt_type='pcolormesh',
                cmap=plt.cm.Greens,
                suptitle_add_word=f'{var_names[i]:s} {var_months[i]:s}')

            GEO_PLOT.plot_field_in_classif(
                field=solar_energy, classif=classif, area='reu',
                vmax=8, vmin=4, bias=0, plot_wind=0,
                only_significant_points=0, plt_type='pcolormesh',
                cmap=plt.cm.YlOrRd,
                suptitle_add_word=f'{var_names[i]:s} {var_months[i]:s} total sky')

            GEO_PLOT.plot_field_in_classif(
                field=solar_energy_clearsky, classif=classif, area='reu',
                vmax=10, vmin=6, bias=0, plot_wind=0,
                cmap=plt.cm.YlOrRd,
                only_significant_points=0, plt_type='pcolormesh',
                suptitle_add_word=f'{var_names[i]:s} {var_months[i]:s} clear sky')

            print(f'end of plot {var_names[i]:s}')
        print(f'end of pv statistics')

    if cfg.job.interannual_variability.loading:
        # note: here the intra annual time scales are explored together IOD, SIOD, Enso

        # reading: iod
        iod_week = GEO_PLOT.read_to_standard_da(f'{cfg.dir.data:s}/dmi.nc', var='DMI')
        # https: // stateoftheocean.osmc.noaa.gov / sur / ind / dmi.php
        # info: http://la.climatologie.free.fr/iod/iod-english.htm#index

        # reading: siod
        siod_month: xr.DataArray = GEO_PLOT.read_to_standard_da(f'{cfg.dir.data:s}/{cfg.input.siod:s}', 'index')
        # see readme for more info

        # reading: enso
        enso_2mon_mean: xr.DataArray = GEO_PLOT.read_to_standard_da(f'{cfg.dir.data:s}/{cfg.input.enso:s}', 'index')

        ssr_all_pixel = DATA.reading_sarah_e_hour_reu(
            select_only_land=False,
            select_day_time_hours=True, select_only_NDJF=False,
            for_test=0,
            clearsky=0)

        # daily mean of SSR:
        ssr_day = GEO_PLOT.daily_mean_da(ssr_all_pixel)
        del ssr_all_pixel

        ssr_day_anomaly = GEO_PLOT.anomaly_daily(ssr_day)

        # getting clearsky over land and solar energy
        # using all hours, so before sun set, total sky (sarah_e) may > clearsky, redirection
        ssr_land, ssr_land_clearsky = DATA.reading_sarah_e_hour_reu(
            select_only_land=True,
            select_day_time_hours=False, select_only_NDJF=False,
            for_test=0,
            clearsky=1)

        # todo: fix it
        ssr_land_clearsky = ssr_land_clearsky.assign_attrs({'units': 'W/m2'})

        clearsky_index = ssr_land / ssr_land_clearsky
        clearsky_index = clearsky_index.assign_attrs({
            'units': ssr_land.units,
            'long_name': 'ghi'}).rename('ghi')

        # to avoid the inf and nan:
        clearsky_index = clearsky_index.where(np.logical_and(clearsky_index.time.dt.hour >= 8,
                                                             clearsky_index.time.dt.hour <= 17), drop=True)

        # calculate daily energy using 24 hours data
        solar_energy = GEO_PLOT.cal_daily_total_energy(ssr_land)
        # remove the ocean 0 values for plot
        solar_energy = GEO_PLOT.select_land_only_reunion_by_altitude(solar_energy)
        solar_energy_clearsky = GEO_PLOT.cal_daily_total_energy(ssr_land_clearsky)
        solar_energy_clearsky = GEO_PLOT.select_land_only_reunion_by_altitude(solar_energy_clearsky)

        energy_index = solar_energy / solar_energy_clearsky
        energy_index = energy_index.assign_attrs({
            'units': 'index',
            'long_name': solar_energy.long_name + ' all vs clear sky'}).rename('daily_total_energy_density')

        energy_index = GEO_PLOT.select_land_only_reunion_by_altitude(energy_index)

        del ssr_land, ssr_land_clearsky
        del ssr_day

    if any(GEO_PLOT.get_values_multilevel_dict(dict(cfg.job.interannual_variability.loop))):
        # ----------------------------- loop -----------------------------
        climate_indexes = [enso_2mon_mean, siod_month, iod_week]
        climate_index_names = ['ENSO', 'SIOD', 'IOD']

        percentile = cfg.job.interannual_variability.percentile
        percentile = [0.1, 0.1, 0.05]
        # using limits, all the same for these 3:
        limits = [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]

        for i in range(len(climate_indexes)):
            idx = climate_indexes[i]

            # match the period of ssr:
            idx = idx.loc["1999-01-01":"2016-12-31"]

            if cfg.job.interannual_variability.loop.plot_monthly_index:
                # get monthly mean
                idx = GEO_PLOT.monthly_mean_da(idx)

                GEO_PLOT.plot_climate_index(
                    idx.time.values, idx.values,
                    index_name=climate_index_names[i],
                    by_percentage=0, alpha=percentile[i],
                    by_limit=1, limits=limits[i],
                    x_label='year', y_label=f'{climate_index_names[i]:s} {idx.units:s}',
                    title_add_word=f'monthly')

            if cfg.job.interannual_variability.loop.calculate_daily_index_df:
                # interpolate to daily
                start_day = str(idx.time.dt.strftime("%Y-%m-%d")[0].values)
                end_day = str(idx.time.dt.strftime("%Y-%m-%d")[-1].values)

                dates = pd.date_range(start_day, end_day, freq='D')
                idx_daily = idx.interp(time=dates)

                # find the positive and negative and weak phase
                # down, up = GEO_PLOT.get_confidence_interval(idx_daily.values, alpha=percentile[i])

                # if by limits, defined above
                down, up = limits[i]

                idx_positive = idx_daily.loc[idx_daily > up]
                idx_negative = idx_daily.loc[idx_daily < down]
                idx_neutral = idx_daily.loc[(idx_daily < up) & (idx_daily > down)]

                print(f'idx_positive = {len(idx_positive):g}\n'
                      f'idx_negative = {len(idx_negative):g}\n'
                      f'idx_neutral = {len(idx_neutral):g}\n')
                # note: the definition of neutral could be different from literature.

            if cfg.job.interannual_variability.loop.plot_field_in_classif:
                df_idx = idx_daily.to_dataframe()
                df_idx['class'] = np.select(
                    [df_idx <= down, df_idx >= up, True], [-1, 1, 0])

                df_idx_class = df_idx[{'class'}]

                # OLR:
                ttr_era5 = GEO_PLOT.read_to_standard_da('~/local_data/era5/ttr/ttr.era5.1999-2016.day.reu.nc', 'ttr')

                ttr_daily = GEO_PLOT.anomaly_daily(ttr_era5)

                # change unit, etc
                olr_daily_anomaly = GEO_PLOT.convert_ttr_era5_2_olr(ttr=ttr_daily, is_reanalysis=1)

                GEO_PLOT.plot_field_in_classif(
                    field=olr_daily_anomaly, classif=df_idx_class, area='bigreu',
                    vmax=10, vmin=-10, bias=1,
                    cmap=plt.cm.PuOr_r,
                    plot_wind=0, only_significant_points=1,
                    suptitle_add_word=f'{climate_index_names[i]:s} '
                                      f'class limit daily OLR anomaly')
                # SSR anomaly
                GEO_PLOT.plot_field_in_classif(
                    field=ssr_day_anomaly, classif=df_idx_class, area='bigreu',
                    vmax=30, vmin=-30, bias=1,
                    plot_wind=0,
                    only_significant_points=1,
                    suptitle_add_word=f'{climate_index_names[i]:s} '
                                      f'class limits daily anomaly')

            if cfg.job.interannual_variability.loop.two_sample_test:
                # welch's test:
                ssr_idx = GEO_PLOT.get_data_in_classif(da=ssr_day_anomaly, df=df_idx_class, significant=False)

                positive = ssr_idx[:, :, :, list(ssr_idx['class'].values).index(1)].dropna(dim='time')
                negative = ssr_idx[:, :, :, list(ssr_idx['class'].values).index(-1)].dropna(dim='time')
                neutral = ssr_idx[:, :, :, list(ssr_idx['class'].values).index(0)].dropna(dim='time')

                # GEO_PLOT.welch_test(a=positive, b=all_phase, conf_level=0.05, equal_var=False,
                #                     title=f'{climate_index_names[i]:s} positive vs all '
                #                           f' percentile={percentile[i]:4.2}')
                # GEO_PLOT.welch_test(a=negative, b=all_phase, conf_level=0.05, equal_var=False,
                #                     title=f'{climate_index_names[i]:s} negative vs all '
                #                           f' percentile={percentile[i]:4.2}')

                GEO_PLOT.welch_test(a=positive, b=neutral, conf_level=0.05, equal_var=False,
                                    title=f'{climate_index_names[i]:s} positive vs neutral '
                                          f'percentile={percentile[i]:4.2}')
                GEO_PLOT.welch_test(a=negative, b=neutral, conf_level=0.05, equal_var=False,
                                    title=f'{climate_index_names[i]:s} negative vs neutral '
                                          f'percentile={percentile[i]:4.2}')

                GEO_PLOT.welch_test(a=positive, b=negative, conf_level=0.05, equal_var=False,
                                    title=f'{climate_index_names[i]:s} positive vs negative '
                                          f'percentile={percentile[i]:4.2}')

            if cfg.job.interannual_variability.loop.solar_energy:
                classif = df_idx_class
                # plot total daily energy
                GEO_PLOT.plot_field_in_classif(
                    field=energy_index, classif=classif, area='reu',
                    vmax=1, vmin=0.4, bias=0, plot_wind=0,
                    cmap=plt.cm.Greens,
                    only_significant_points=0, plt_type='pcolormesh',
                    suptitle_add_word=f'{climate_index_names[i]:s}')

                GEO_PLOT.plot_field_in_classif(
                    field=solar_energy, classif=classif, area='reu',
                    vmax=6, vmin=4, bias=0, plot_wind=0,
                    cmap=plt.cm.YlOrRd,
                    only_significant_points=0, plt_type='pcolormesh',
                    suptitle_add_word=f'{climate_index_names[i]:s} total sky')

                GEO_PLOT.plot_field_in_classif(
                    field=clearsky_index, classif=classif, area='reu',
                    vmax=1, vmin=0.5, bias=0, plot_wind=0,
                    cmap=plt.cm.Blues,
                    only_significant_points=0, plt_type='pcolormesh',
                    suptitle_add_word=f'{climate_index_names[i]:s} total sky')

                GEO_PLOT.plot_field_in_classif(
                    field=solar_energy_clearsky, classif=classif, area='reu',
                    vmax=8, vmin=4, bias=0, plot_wind=0,
                    only_significant_points=0, plt_type='pcolormesh',
                    suptitle_add_word=f'{climate_index_names[i]:s} clear sky')

            if cfg.job.interannual_variability.loop.violin_clearsky_in_classif:
                df_hour_clearsky_index = GEO_PLOT.get_df_of_da_in_classif(da=clearsky_index.stack(pixel=('x', 'y')),
                                                                          classif=df_idx_class)

                GEO_PLOT.plot_violin_boxen_df_1D(x='class', y=clearsky_index.name,
                                                 y_unit='GHI/GHI_clearsky',
                                                 hue=0,
                                                 df=df_hour_clearsky_index,
                                                 suptitle_add_word=f'GHI_clearsky_index '
                                                                   f'hourly all pixels over land area '
                                                                   f'climate_index {climate_index_names[i]:s}')

                GEO_PLOT.plot_diurnal_boxplot_in_classif(
                    classif=df_idx_class,
                    field=clearsky_index,
                    ylimits=[0, 1.2],
                    relative_data=1,
                    suptitle_add_word=f'{climate_index_names[i]:s} clearsky index all pixel over land'
                )

            print(f'done {climate_index_names[i]:s}')

        print(f'done of loop en climate index')

    print(f'done')


#
# # ----------------------------- clustering SARAH-E daily SSR (Pauline Mialhe) -----------------------------
# # clusters = f'{WORKING_DIR:s}/data/classification/classif.csv'
# # ssr_class = MIALHE_2021.read_ssr_class(clusters)
#
# if test:
#     print('this is a test')
# # ----------------------------- monthly distribution of ssr class -----------------------------
# # if ttt_monthly_distribution:
# #
# #     MIALHE_2021.histogram_classification(olr_regimes)
# #     MIALHE_2021.matrix_classification_at_year_and_month(olr_regimes)
# #     MIALHE_2021.table_classification_at_month(olr_regimes)
# #
# # # ----------------------------- SSR field vs OLR patterns -----------------------------
# #
# # # ================================== SSR class vs OLR class ==================================
# #
# # if ssr_class_vs_olr_regime:
# #
# #     MIALHE_2021.classification_vs_classification(cls1=pclass_olr, cls2=pclass_ssr)
# #
# # # ==================================SSR class link with circulation ==================================
# #
# #
# # if plot_daily_mean_field_in_ttt_regimes:
# #
# #     mjo: pd.DataFrame = GEO_PLOT.read_mjo(match_ssr_avail_day=1)
# #
# #     w500_reu = xr.open_dataset(f'{WORKING_DIR:s}/data/mjo/w500.era5.1999-2016.daily.anomaly.reu.nc')['w']
# #     w500_sa_swio =
# xr.open_dataset(f'{WORKING_DIR:s}/data/mjo/w500.era5.1999-2016.daily.anomaly.sa-swio.nc')['w']
# #
# #     olr_reu = xr.open_dataset(f'{WORKING_DIR:s}/data/era5/ttr.era5.1999-2016.day.anomaly.nc.reu.nc')['ttr']
# #     ssr_reu = xr.open_dataset(f'{WORKING_DIR:s}/data/sarah_e/SIS.sarah_e.daily.1999-2016.reu.anomaly.nc')['SIS']
# #
# #     for season in ["austral_summer", "winter", "all_year"]:
# #         # OLR
# #         MIALHE_2021.plot_fields_in_mjo_phases(mjo_phase=mjo, field=olr_swio, area='swio', season=season,
# #                                               only_significant_points=0, consistency=0)
# #         MIALHE_2021.plot_fields_in_mjo_phases(mjo_phase=mjo, field=olr_reu, area='bigreu',
# #                                               season=season, consistency=1)
# #
# #         # SSR
# #         MIALHE_2021.plot_fields_in_mjo_phases(mjo_phase=mjo, field=ssr_reu, area='bigreu',
# #         season=season, consistency=1)
# #         MIALHE_2021.plot_fields_in_mjo_phases(mjo_phase=mjo, field=ssr_reu, area='bigreu',
# #         season=season, consistency=0)
# #
# #         MIALHE_2021.plot_fields_in_mjo_phases(mjo_phase=mjo, field=olr_swio, area='SA_swio', season=season,
# #                                                  only_significant_points=1)
# #
# #         MIALHE_2021.plot_fields_in_mjo_phases(mjo_phase=mjo, field=olr_swio, area='SA_swio', season=season,
# #                                               percentage=1)
# #
# #
# #         # MIALHE_2021.plot_fields_in_mjo_phases(mjo_phase=mjo, field=w500_reu, area='reu', season=season)
# #         # MIALHE_2021.plot_fields_in_mjo_phases(mjo_phase=mjo, field=w500_reu, area='reu', season=season,
# #                   percentage=1)
# #         # MIALHE_2021.plot_fields_in_mjo_phases(
# mjo_phase=mjo, field=w500_sa_swio, area='SA_swio', season=season)
# #         # MIALHE_2021.plot_fields_in_mjo_phases(mjo_phase=mjo, field=w500_sa_swio, area='SA_swio',
# #                   season=season, percentage=1)
# #
# #     w500 = xr.open_dataset(f'{WORKING_DIR:s}/data/mjo/w500.era5.1999-2016.daily.anomaly.d01.nc')['w']
# #     MIALHE_2021.plot_fields_in_mjo_phases(mjo_phase=mjo, field=w500)
# #
#     print('good')
# #
# # if ttt_in_ssr_class:
# #
# #     mjo: pd.DataFrame = GEO_PLOT.read_mjo(match_ssr_avail_day=1)
# #
# #     mjo_dec_feb = mjo[(mjo.index.month == 1) |
# #                       (mjo.index.month == 12) |
# #                       (mjo.index.month == 2)]
# #
# #     mjo_may_jun = mjo[(mjo.index.month == 6) |
# #                       (mjo.index.month == 7) |
# #                       (mjo.index.month == 8)]
# #
# #     MIALHE_2021.plot_classification_vs_mjo(class_ssr=pclass_ssr, mjo=mjo_dec_feb, tag='DJF')
# #     MIALHE_2021.plot_classification_vs_mjo(class_ssr=pclass_ssr, mjo=mjo_may_jun, tag='JJA')
# #
# #     MIALHE_2021.plot_distribution_a_in_b(df_ab=mjo_and_ssr[{'ssr_class', 'phase', 'amplitude'}],
# #                                          column_a='phase', column_b='ssr_class')
# #
# # if ttt_monthly_distribution:
# #
# #     MIALHE_2021.plot_mjo_monthly_distribution(GEO_PLOT.read_mjo(match_ssr_avail_day=1))


if __name__ == "__main__":
    sys.exit(interaction())
