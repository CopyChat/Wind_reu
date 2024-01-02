"""
data processing file
"""

__version__ = f'Version 2.0  \nTime-stamp: <2021-05-15>'
__author__ = "ChaoTANG@univ-reunion.fr"

import numpy as np
import pandas as pd
import xarray as xr
from omegaconf import DictConfig
import scipy.io as sio
import GEO_PLOT


# ----------------------------- functions -----------------------------

def prepare_tc_day_data(input: str, output: str, cyclone_class: pd.DataFrame, ssr_hour_anomaly_pauline):

    central_lat = -21.1
    central_lon = 55.5

    cyc = pd.read_csv(input)

    # 1. calculate daily mean tc position:
    cyc_df = GEO_PLOT.read_csv_into_df_with_header(input)

    # have to use SSR, since we need ssr day mean
    cyclone_day = cyclone_class[cyclone_class > 0].dropna().index
    # in cyc_df there are days with a nearby tc and a distancial tc,
    # such as 2006-03-03:
    # nearby_tc['2006-03-03']

    b = cyc_df[cyc_df.index.strftime('%Y-%m-%d').isin(cyclone_day.strftime('%Y-%m-%d'))]

    b['date'] = b.index.date
    nearby_tc_daymean = b.groupby(by=[b.NOM_CYC, b['date']]).mean()

    import math

    # position in the right:

    right = nearby_tc_daymean[nearby_tc_daymean.LON > central_lon]
    left = nearby_tc_daymean[nearby_tc_daymean.LON < central_lon]

    # calculate azimuth angle to the right:
    lat = right.LAT
    lon = right.LON
    cos_right = (lat - central_lat) / np.sqrt((lon - central_lon) ** 2 + (lat - central_lat) ** 2)
    radians_right = np.arccos(cos_right)
    azimuth_right = (radians_right / math.pi) * 180

    # calculate azimuth angle to the left:
    lat = left.LAT
    lon = left.LON
    cos_left = (lat - central_lat) / np.sqrt((lon - central_lon) ** 2 + (lat - central_lat) ** 2)
    radians_left = np.arccos(cos_left)
    azimuth_left = 360 - (radians_left / math.pi) * 180

    azimuth = pd.concat([azimuth_left, azimuth_right]).sort_index()
    nearby_tc_daymean['azimuth'] = azimuth

    # calculate mean ssr anomaly:

    # land area mean:

    land_mask = np.load('./dataset/sarah_e.land_mask.format_pauline.mat.npy')
    lookup = xr.DataArray(land_mask, dims=('y', 'x'))
    geo_land = ssr_hour_anomaly_pauline.where(lookup)
    # show
    geo_land.mean(axis=0).plot()
    import matplotlib.pyplot as plt
    plt.show()

    land_ocean = ssr_hour_anomaly_pauline

    ssr_land_mean_anomaly = geo_land.mean(axis=1).mean(axis=1)
    ssr_land_ocean_mean_anomaly = land_ocean.mean(axis=1).mean(axis=1)

    da1 = ssr_land_mean_anomaly
    hourly1 = da1.where(da1.time.dt.strftime('%Y-%m-%d').isin(cyclone_day.strftime('%Y-%m-%d')), drop=True)
    ssr_land_daymean_anomaly = hourly1.groupby(hourly1.time.dt.date).mean()

    da2 = ssr_land_ocean_mean_anomaly
    hourly2 = da2.where(da2.time.dt.strftime('%Y-%m-%d').isin(cyclone_day.strftime('%Y-%m-%d')), drop=True)
    ssr_land_ocean_daymean_anomaly = hourly2.groupby(hourly2.time.dt.date).mean()

    # merge with cyclone azimuth:

    cyc1 = nearby_tc_daymean

    left1 = cyc1
    right1 = ssr_land_daymean_anomaly.to_dataframe()

    cyc1 = pd.merge(left1.reset_index(), right1.reset_index(), on=['date'], how='inner')
    cyc1 = cyc1.rename(columns={'SIS': 'SIS_land'})

    left2 = cyc1
    right2 = ssr_land_ocean_daymean_anomaly.to_dataframe()

    cyc = pd.merge(left2.reset_index(), right2.reset_index(), on=['date'], how='inner')

    cyc['dt'] = pd.to_datetime(cyc['date'])
    cyc.sort_values(by='dt', inplace=True)

    cyc = cyc.reset_index()

    cyc_df = cyc[{'NOM_CYC', 'date', 'LAT', 'LON', 'RPRESS',
       'RVENT_MAX', 'dist_reu', 'azimuth', 'SIS_land', 'SIS', 'dt'}]

    # attention: can not use date as index, there're more same value
    # attention: can not use date as index, there're more same value
    # cyc_df = cyc.set_index('date')
    cyc_df.to_csv(output)

    print(f'done')

    return cyc_df


def prepare_tc_6h_data(raw: str, output: str):

    cen_lat = -21.1
    cen_lon = 55.5

    cyclone_file = f'./dataset/cyc_df.csv'

    cyc = pd.read_csv(raw)
    cyc['dt'] = pd.to_datetime(cyc['DateTime'])
    df_tc = cyc.set_index('dt')

    df_tc['dist_reu'] = GEO_PLOT.distance_two_point_deg(
        lon1=cen_lon,
        lon2=df_tc['LON'],
        lat1=cen_lat,
        lat2=df_tc['LAT'])

    tc = df_tc[{'NOM_CYC', 'LAT', 'LON', 'RPRESS', 'RVENT_MAX', 'NUM_CYC', 'dist_reu'}]
    tc.to_csv(output)

    return df_tc


def reading_sarah_e_hour_reu(clearsky: bool = 0,
                             clearsky_file: str =
                             './dataset/ssr_clearsky_hourly.sarah_e_dim.nc',
                             select_only_NDJF: bool = False,
                             select_day_time_hours: bool = False,
                             for_test: bool = False,
                             select_only_land: bool = False):
    """
    ex. ssr = reading_sarah_e_hour_reu(**cfg.job.ssr.ssr_reading_param), which could also in the code directly
    Args:
        clearsky (): to read or not
        clearsky_file ():
        select_only_NDJF ():
        select_day_time_hours ():
        for_test ():
        select_only_land ():

    Returns:

    """
    ssr_sarah_e = GEO_PLOT.read_to_standard_da(
        '/Users/ctang/local_data/sarah_e/SIS.sarah-e.1999-2016.hour.reu.nc', 'SIS')
    ssr_sarah_e = GEO_PLOT.convert_da_shifttime(ssr_sarah_e, second=4 * 3600)

    print(f'process for total sky flux...')
    if for_test:
        ssr_sarah_e = ssr_sarah_e.sel(time=slice('19990101', '20001201'))

    if select_only_NDJF:
        ssr_sarah_e = ssr_sarah_e.where(np.logical_or(ssr_sarah_e.time.dt.month >= 11,
                                                      ssr_sarah_e.time.dt.month <= 2), drop=True)
    if select_day_time_hours:
        ssr_sarah_e = ssr_sarah_e.where(np.logical_and(ssr_sarah_e.time.dt.hour >= 8,
                                                       ssr_sarah_e.time.dt.hour <= 17), drop=True)
    if select_only_land:
        ssr_sarah_e = GEO_PLOT.select_land_only_reunion_by_altitude(ssr_sarah_e)

    # ----------------------------- -----------------------------
    if clearsky:
        print(f'process for clearsky flux...')
        ssr_clear = GEO_PLOT.read_to_standard_da(clearsky_file, 'ghi')
        if for_test:
            ssr_clear = ssr_clear.sel(time=slice('19990101', '20001201'))

        if select_only_NDJF:
            ssr_clear = ssr_clear.where(np.logical_or(ssr_clear.time.dt.month >= 11,
                                                      ssr_clear.time.dt.month <= 2), drop=True)
        if select_day_time_hours:
            ssr_clear = ssr_clear.where(np.logical_and(ssr_clear.time.dt.hour >= 8,
                                                       ssr_clear.time.dt.hour <= 17), drop=True)
        if select_only_land:
            ssr_clear = GEO_PLOT.select_land_only_reunion_by_altitude(ssr_clear)

        return ssr_sarah_e, ssr_clear
    else:
        return ssr_sarah_e


def load_ssr_cluster(data: str, check_missing: bool = False):
    # ---------- clustering SARAH-E daily SSR (by Pauline Mialhe) -----------------------------
    ssr_cluster = GEO_PLOT.read_csv_into_df_with_header(data)
    # todo: cluster has some missing day for example 20000116
    #  df.loc[slice('20000115', '20000118')]
    # key word: slice, date dataframe

    if check_missing:
        missing_day = len(pd.date_range(ssr_cluster.index.date[0], ssr_cluster.index.date[-1])) - len(ssr_cluster)
        print(f'found missing day = {missing_day:g}, between '
              f'{ssr_cluster.index.date[0].strftime("%Y-%m-%d"):s} and '
              f'{ssr_cluster.index.date[-1].strftime("%Y-%m-%d"):s}. '
              f'total size = {len(ssr_cluster):g}')  # 154 days

    return ssr_cluster


def get_date():
    """
    to get date range for the NDJF from 1979-2018
    :return: list of DateTimeIndex
    """
    import datetime as dt
    dates = []
    for start_year in range(1979, 2018):
        date_time_str = f'{start_year:g}-11-01'
        date_time_obj = dt.datetime.strptime(date_time_str, '%Y-%m-%d')
        winter = pd.date_range(start=date_time_obj, periods=120, freq='24H')

        dates.append(winter)
    dates = [y for x in dates for y in x]

    return dates


def load_olr_regimes(mat_file: str, classif: str = 'robust'):
    """
    to read a mat, the file from B.P.
    :param mat_file:
    :type mat_file: str
    :param classif:  defined in the configs
        ttt_classif: 'robust'   # dominant, and remove days with ensemble convergence < 8
        ttt_classif: 'dominant' # all the days with dominant class, while plot is based on 1st ens number
        ttt_classif: '1st_ens_num' # all the days from 1st ens number
        ttt_classif: 'not_robust'   # not robust days
    :type classif:
    :return:
    :rtype: Union[None, pandas.core.frame.DataFrame]
    """
    """
    :return:
    """

    mat = sio.loadmat(mat_file)
    pclass = np.array(mat["CLASS"])
    # B.P.: 10 ensemble members: 120 days (NDJF) x 39 years (1979-2018)

    # convert to 2D:
    pclass = np.moveaxis(pclass, 0, -1)
    # same as:
    # reshape = np.moveaxis(pclass, [0, 1, 2], [2, 0, 1])

    # permutation des numeros // Fauchereau 2009
    pclass[np.where(pclass == 1)] = 8
    pclass[np.where(pclass == 4)] = 1
    pclass[np.where(pclass == 6)] = 4
    pclass[np.where(pclass == 5)] = 6
    pclass[np.where(pclass == 7)] = 5
    pclass[np.where(pclass == 8)] = 7
    pclass[np.where(pclass == 2)] = 8
    pclass[np.where(pclass == 3)] = 2
    pclass[np.where(pclass == 8)] = 3

    reshape = np.reshape(pclass, (-1, pclass.shape[2]), order='F')
    df2 = pd.DataFrame(data=reshape, columns=list(range(1, 11)), dtype='int32')

    df2['DateTime'] = get_date()

    df = df2.set_index(pd.to_datetime(df2['DateTime']))
    df = df.drop(['DateTime'], axis=1)

    # find the dominant class
    most_common: pd.DataFrame = df.mode(axis=1).iloc[:, 0]

    freq = np.zeros((len(df)))
    for i in range(len(df)):
        freq[i] = list(df.iloc[i, :]).count(most_common[i])
        df['most_common_class'] = most_common
        df['freq'] = freq

    # selecting:
    dd = pd.DataFrame()
    if classif == 'robust':
        #  remove the most uncertain days and we 'd keep robust regime definitions with N_class > 8 ensemble numbers
        robust = df.loc[df['freq'] > 7]
        dd['class'] = robust['most_common_class']

    if classif == 'dominant':
        #  remove the most uncertain days and we 'd keep robust regime definitions with N_class > 8 ensemble numbers
        dd['class'] = df['most_common_class']

    if classif == '1st_ens_num':
        dd = df.iloc[:, 0]

    if classif == 'not_robust':
        not_robust = df.loc[df['freq'] <= 7]
        dd['class'] = not_robust['most_common_class']

    print(f'selecting {len(dd):g} out of {len(df):g} days: {classif:s}')

    return dd


def cal_cyclone_class_in_olr_period(ssr_cluster: pd.DataFrame, nearby_radius: float,
                                    raw_cyclone_data: str,
                                    find_day_with_more_than_one_cyclone: bool = 0,
                                    monthly_num_cyclone: bool = 0
                                    ):
    """
    as the name
    Args:
        ssr_cluster:
        nearby_radius:
        raw_cyclone_data:
        find_day_with_more_than_one_cyclone:
        monthly_num_cyclone:

    Returns:

    """
    # raw cyc data 1981-01 - 2016-04:
    cyc_df = GEO_PLOT.read_csv_into_df_with_header(raw_cyclone_data)
    # TODO: local time or UTC ? assuming local time to be confirmed.
    #  while it doesn't make any difference in daily analysis of zone within 6 hours time difference to UTC00:
    #  since in reunion, the record times 0, 6, 12, 18 -> 4, 10, 16, 22, still in the same day.

    # select nearby cyc: 1982 - 2015-03:
    nearby_cyc: pd.DataFrame = GEO_PLOT.select_nearby_cyclone(
        cyc_df=cyc_df, lon_name='LON', lat_name='LAT', radius=nearby_radius,
        cen_lat=-21.1, cen_lon=55.5)

    num_day = len(set(nearby_cyc.index.date))  # 144 days
    num_tc = len(set(nearby_cyc['NOM_CYC']))
    print(f'totally {num_day:g} days with nearby cyclone between Feb 1982 and Feb 2016')
    print(f'totally {num_tc:g} TCs with nearby cyclone between Feb 1982 and Feb 2016')
    print(f'totally nearby cyclone days in ssr_cluster period (1999 - 2016)'
          f' = {len(set(nearby_cyc[nearby_cyc.index.year > 1998].index.date)):g}')

    nearby_cyc_99_16 = nearby_cyc[nearby_cyc.index.year > 1998]

    if monthly_num_cyclone:
        # check num of cyclone in months: in 1999-2016
        for m in list(set(nearby_cyc_99_16.index.month.values)):
            # num of cyc_99_16
            n_c = len(set(nearby_cyc_99_16[nearby_cyc_99_16.index.month == m].NOM_CYC))
            # num of days
            n_d = len(set(nearby_cyc_99_16[nearby_cyc_99_16.index.month == m].index.date))
            print(f'in month {m:g}, {n_c:g} nearby cyclones, covering {n_d:g} days')

    # statistics of cyc in 99 - 16 period, all months
    n_c_all = len(set(nearby_cyc_99_16.NOM_CYC))
    n_d_all = len(set(nearby_cyc_99_16.index.date))
    print(f'totally, {n_c_all:g} nearby cyclones, covering {n_d_all:g} days')

    # merge ssr class and cyc, to get common days, ssr has missing data
    ssr_cluster['cyclone'] = np.zeros(ssr_cluster.index.shape)
    for i in range(len(ssr_cluster)):
        cyc_in_day = nearby_cyc.iloc[nearby_cyc.index.date == ssr_cluster.index.date[i]]
        ssr_cluster.iloc[i, -1] = len(list(set(cyc_in_day['NOM_CYC'])))

        if find_day_with_more_than_one_cyclone:
            if len(list(set(cyc_in_day['NOM_CYC']))) > 1:
                print(ssr_cluster.index.date[i], cyc_in_day.NOM_CYC)

    ssr_cyclone: pd.DataFrame = ssr_cluster[['class', 'cyclone']]
    # note: there's ONLY one day 2008-01-31  and 2008-02-01 with two cyclone within 5 deg of reunion: FAME and "GULA".
    # this day belongs to SSR cluster CL2, when the total number of class is 9.

    # check why some nearby cyclone days do not included in ssr_cluster:
    # since SSR has missing date

    print(f'days with tc but not in the class:')
    nearby_cyc_ssr_period = nearby_cyc[nearby_cyc.index.year > 1998]
    for i in range(len(nearby_cyc_ssr_period)):
        cyc_day = nearby_cyc_ssr_period.index.date[i]
        ssr_cluster_in_day = ssr_cluster.iloc[ssr_cluster.index.date == cyc_day]
        if len(ssr_cluster_in_day) < 1:
            print(cyc_day)

    cyclone_class_ssr_period = ssr_cyclone[['cyclone']].rename(columns={'cyclone': 'class'})
    # df.rename(columns={"A": "a", "B": "c"})

    # make record > 0 as cyclone day:
    cyclone_class_ssr_period.loc[cyclone_class_ssr_period['class'] > 0] = 1

    # selecting cyc day:
    # ==============================================
    # cyc_day = cyclone_class[cyclone_class > 0].dropna().index
    # non_cyc_day = cyclone_class[cyclone_class['class'] == 0].index

    return cyclone_class_ssr_period


def cal_cyclone_class_in_ssr_period(ssr_cluster: pd.DataFrame, nearby_radius: float,
                                    raw_cyclone_data: str,
                                    find_day_with_more_than_one_cyclone: bool = 0,
                                    monthly_num_cyclone: bool = 0
                                    ):
    """
    as the name
    Args:
        ssr_cluster:
        nearby_radius:
        raw_cyclone_data:
        find_day_with_more_than_one_cyclone:
        monthly_num_cyclone:

    Returns:

    """
    # raw cyc data 1981-01 - 2016-04:
    cyc_df = GEO_PLOT.read_csv_into_df_with_header(raw_cyclone_data)
    # TODO: local time or UTC ? assuming local time to be confirmed.
    # while it doesn't make any difference in daily analysis of zone within 6 hours time difference to UTC00:
    # since in reunion, the record times 0, 6, 12, 18 -> 4, 10, 16, 22, still in the same day.

    # select nearby cyc: 1982 - 2016-04:
    nearby_cyc: pd.DataFrame = GEO_PLOT.select_nearby_cyclone(
        cyc_df=cyc_df, lon_name='LON', lat_name='LAT', radius=nearby_radius,
        cen_lat=-21.1, cen_lon=55.5)

    num_day = len(set(nearby_cyc.index.date))  # 319 days
    print(f'totally {num_day:g} days with nearby cyclone between Feb 1982 and Apr 2016')
    print(f'totally nearby cyclone days in ssr_cluster period (1999 - Apr 2016), '
          f'num will be smaller if merge with sarah-e '
          f'available days.'
          f' = {len(set(nearby_cyc[nearby_cyc.index.year > 1998].index.date)):g} \n'
          f'that is ok nearly 50%')

    nearby_cyc_99_16 = nearby_cyc[nearby_cyc.index.year > 1998]

    print(f'in 1999 to 2016 apr, {len(set(nearby_cyc_99_16.index.date)):g} TCs have nearby TCs.\n'
          f'{len(set(nearby_cyc_99_16.NOM_CYC)):g} nearby TC records\t'
          f'distributed in {len(set(nearby_cyc_99_16.index.date)):g} days')

    if monthly_num_cyclone:
        print(f'monthly distribution of nearby TCs within {nearby_radius:g} degrees')
        # check num of cyclone in months: in 1999-2016
        for m in list(set(nearby_cyc_99_16.index.month.values)):
            # num of cyc_99_16
            n_c = len(set(nearby_cyc_99_16[nearby_cyc_99_16.index.month == m].NOM_CYC))
            # num of days
            n_d = len(set(nearby_cyc_99_16[nearby_cyc_99_16.index.month == m].index.date))
            print(f'in month {m:g}, {n_c:g} nearby cyclones, covering {n_d:g} days')

    # statistics of cyc in 99 - 16 apr period, all months
    n_c_all = len(set(nearby_cyc_99_16.NOM_CYC))
    n_d_all = len(set(nearby_cyc_99_16.index.date))
    print(f'totally, {n_c_all:g} nearby cyclones, covering {n_d_all:g} days')

    # merge ssr class and cyc, to get common days, ssr has missing data
    ssr_cluster['cyclone'] = np.zeros(ssr_cluster.index.shape)
    all_tc_names = list([])
    for i in range(len(ssr_cluster)):
        cyc_in_day = nearby_cyc.iloc[nearby_cyc.index.date == ssr_cluster.index.date[i]]
        ssr_cluster.iloc[i, -1] = len(set(cyc_in_day['NOM_CYC']))
        all_tc_names.append(cyc_in_day['NOM_CYC'])

        if find_day_with_more_than_one_cyclone:
            if len(cyc_in_day) > 1:
                if len(set(list(cyc_in_day.NOM_CYC))) > 1:
                    print(i, cyc_in_day.NOM_CYC)

    ssr_cyclone: pd.DataFrame = ssr_cluster[['9Cl', 'cyclone']]
    # note: there's ONLY one day 2008-01-31 and 2008-02-01 with two cyclone within 5 deg of reunion: FAME and "GULA".
    # this day belongs to SSR cluster CL2, when the total number of class is 9.

    # intotal 165 tc days, with ssr (missing) = 162 days, total tc day = 162
    # while in the function plot_cyclone_in_class, which is to plot every cyclone within a radius,
    # so the number of cyclones within the ssr period is ONE day more,

    # check why some nearby cyclone days do not included in ssr_cluster:
    # since SSR has missing date
    nearby_cyc_ssr_period = nearby_cyc[nearby_cyc.index.year > 1998]
    for i in range(len(nearby_cyc_ssr_period)):
        cyc_day = nearby_cyc_ssr_period.index.date[i]
        ssr_cluster_in_day = ssr_cluster.iloc[ssr_cluster.index.date == cyc_day]
        if len(ssr_cluster_in_day) < 1:
            print(i, cyc_day)

    # 3 days are missing in ssr_cluster, so 165-3 = 162 days with ssr:
    # 1999 - 03 - 13
    # 2004 - 01 - 01
    # 2015 - 01 - 11

    # output:
    cyclone_class_ssr_period = ssr_cyclone[['cyclone']].rename(columns={'cyclone': 'class'})
    # df.rename(columns={"A": "a", "B": "c"})
    # make record > 0 as cyclone day:
    cyclone_class_ssr_period.loc[cyclone_class_ssr_period['class'] > 0] = 1

    cyclone_num_ssr_period = ssr_cyclone[['cyclone']].rename(columns={'cyclone': 'num'})

    # selecting cyc day:
    # ==============================================
    # cyc_day = cyclone_class[cyclone_class > 0].dropna().index
    # non_cyc_day = cyclone_class[cyclone_class['class'] == 0].index

    # get duration of tc:
    # ==============================================
    all_tc_names = list(set(nearby_cyc_99_16.NOM_CYC))

    n_day_list = []
    for b in range(len(all_tc_names)):
        n_day = len(set(nearby_cyc_99_16[nearby_cyc_99_16.NOM_CYC==all_tc_names[b]].index.date))
        print(b, all_tc_names[b], n_day)
        n_day_list.append(n_day)

    n_day_df = pd.DataFrame(data=n_day_list)
    n_day_df.hist(bins=1)
    import matplotlib.pyplot as plt
    plt.show()

    return cyclone_class_ssr_period, cyclone_num_ssr_period, nearby_cyc_99_16

