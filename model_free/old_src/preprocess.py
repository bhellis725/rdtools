

import os

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt


def main():
    filename = os.path.expanduser('~/data_sets/snl_raw_data/1429_1405/raw_1405_weather_for_1429.csv')
    cols = ['Global_Wm2', 'Date-Time']
    data = pd.read_csv(filename, parse_dates=['Date-Time'], usecols=cols, index_col=['Date-Time'])
    data.index = data.index.tz_localize('Etc/GMT+7')
    data = data.reindex(pd.date_range(start=data.index[0], end=data.index[-1], freq='1min')).fillna(0)
    data = pd.Series(data['Global_Wm2'], index=data.index)
    data[data < 0] = 0
    sample = data[(data.index >= '2016-10-01') & (data.index < '2016-10-15')]

    filtered = deviation_time_filter(sample, viz=True)


def series_to_df_by_time(data):
    '''Transforms a pandas time series (spanning several days) into a data frame
    with each row being time of a day and each column as a date.

    Arguments
    ---------
    data: pd.Series
        time series data

    Returns
    -------
    by_time: pd.DataFrame
        data frame with time of day as rows and dates as columns
    '''
    day_list = []
    for day, group in data.groupby(data.index.date):
        ser = pd.Series(group.values, index=group.index.time, name=day)
        day_list.append(ser)
    by_time = pd.concat(day_list, axis=1)
    return by_time


# def zscore_time_filter(data, zscore_range=(-1, np.inf), replace_val=(-np.inf, np.inf), verbose=False, viz=False):
#     '''Filter measurements by time of day based on zscore at that time.
#     It is suggested data be on the order of a couple weeks to a month.  This method
#     will be applied to data that has seasonal variations.  In order to minimize the impact
#     of seasonal variations, smaller samples should be used.
#
#     Arguments
#     ---------
#     data: pd time series
#         data to be filtered based on z score
#     zscore_range: tuple(float, float)
#         z score range outside of which values will be filtered
#     replace_val: tuple(float, float)
#         values to set data to if outside the zscore range
#     verbose: bool, optional
#         return components of calculation
#     viz: bool, optional
#         produce visualization with outliers marked
#
#     Returns
#     -------
#     filtered: pd.Series
#         data with values outside of zscore_range replaced
#     mask: pd.Series
#         indices where data was out of range
#     '''
#     if len(pd.unique(data.index.date)) > 31:
#         warnings.warn('Using more than one month of data may give suspect results.', RuntimeWarning)
#
#     by_time = series_to_df_by_time(data)
#     avg = pd.Series(by_time.replace(0, np.nan).mean(axis=1), name='avg') # values of 0 will be ignored - assumed missing data
#     std = pd.Series(by_time.replace(0, np.nan).std(axis=1), name='std')  # values of 0 will be ignored - assumed missing data
#     stats = pd.concat([avg, std], axis=1)
#
#     filtered = data.copy()
#     mask_list = []
#     zscore_list = []
#     for day, group in filtered.groupby(filtered.index.date):
#         zscores = pd.Series((group.values - stats['avg'].values) / stats['std'].values, index=group.index)
#         zscores_vals = pd.Series(zscores.values, index=group.index.time, name=day)
#         zscore_list.append(zscores_vals)
#         mask_vals = pd.Series(((zscores < zscore_range[0]) | (zscores > zscore_range[1])).values, name=day, index=group.index.time)
#         mask_list.append(mask_vals.copy())
#         group[zscores < zscore_range[0]] = replace_val[0]
#         group[zscores > zscore_range[1]] = replace_val[1]
#         filtered[group.index] = group
#
#     if viz or verbose:
#         mask = pd.concat(mask_list, axis=1)
#         zscore = pd.concat(zscore_list, axis=1)
#
#     if viz:
#         _zscore_filter_viz(stats, by_time, mask)
#
#     if verbose:
#         components = {'by_time': by_time,
#                       'stats': stats,
#                       'mask': mask,
#                       'zscore': zscore}
#         return filtered, components
#     return filtered


def deviation_time_filter(data, central_tendency_fxn=np.mean, mode='relative',
                          dev_range=(.8, np.inf), replace_val=(-np.inf, np.inf), verbose=False, viz=False):
    '''Filter measurements by time of day based on deviation from fxn.  Likely candidates
    for fxn would be np.mean or np.nanmedian.  The deviation can be relative or direct.

    Note: using np.median will give missing values for nan periods.  use np.nanmedian if this is a concern

    Arguments
    ---------
    data: pd time series
        data to be filtered based on z score
    central_tendency_fxn: callable, optional
        function from which deviation will be measured (usually mean or median), default=np.mean
    mode: str
        deviation mode [relative, direct, zscore]
        relative devitation deviation relative to the central_tendency_fxn result at that point
        direct devitaion compares deviation by direct value comparison at each time
        zscore filters points based off of zscore at a given time
    dev_range: tuple(float, float)
        range outside which values will be filtered -  must consider mode
        and central_tendency_fxn when setting parameters
    replace_val: tuple(float, float)
        values to set data to if outside the dev_range
    verbose: bool, optional
        return components of calculation
    viz: bool, optional
        produce visualization with outliers marked

    Returns
    -------
    filtered: pd.Series
        data with values outside of zscore_range replaced
    mask: pd.Series
        indices where data was out of range
    '''
    if len(pd.unique(data.index.date)) > 31:
        warnings.warn('Using more than one month of data may give suspect results.', RuntimeWarning)
    if mode not in ('relative', 'direct', 'zscore'):
        raise ValueError('Unrecognized mode <{}>.  Select either relative, direct, or zscore'.format(mode))

    by_time = series_to_df_by_time(data)
    if mode in ('relative', 'direct'):
        central_vals = pd.Series(by_time.replace(0, np.nan).apply(central_tendency_fxn, axis=1), name='central')
        deviation_vals = by_time.subtract(central_vals.values, axis=0)
        if mode == 'relative':
            deviation_vals = deviation_vals.divide(central_vals, axis=0)
            lower_lim = pd.Series(dev_range[0] * central_vals, index=deviation_vals.index)
            upper_lim = pd.Series(dev_range[1] * central_vals, index=deviation_vals.index)
        else:
            lower_lim = pd.Series(np.maximum(central_vals + dev_range[0], 0), index=deviation_vals.index)
            upper_lim = pd.Series(central_vals + dev_range[1], index=deviation_vals.index)
    else:
        central_vals = pd.Series(by_time.replace(0, np.nan).apply(np.mean, axis=1), name='central')
        deviation_vals = by_time.subtract(central_vals.values, axis=0)
        std_dev_vals = pd.Series(by_time.replace(0, np.nan).std(axis=1), name='central')
        deviation_vals = by_time.subtract(central_vals.values, axis=0).divide(std_dev_vals, axis=1)
        lower_lim = central_vals + (dev_range[0] * std_dev_vals)
        upper_lim = central_vals + (dev_range[1] * std_dev_vals)

    filtered = data.copy()
    mask_list = []
    for day, group in filtered.groupby(filtered.index.date):
        mask_vals = pd.Series(((group.values < lower_lim.values) | (group.values > upper_lim.values)),
                              name=day, index=group.index.time)
        mask_list.append(mask_vals.copy())
        group[group < lower_lim.values] = replace_val[0]
        group[group > upper_lim.values] = replace_val[1]
        filtered[group.index] = group

    if viz or verbose:
        mask = pd.concat(mask_list, axis=1)

    if viz:
        _deviation_filter_viz(by_time, central_vals, mask, lower_lim, upper_lim)

    if verbose:
        components = {'by_time': by_time,
                      'mask': mask,
                      'deviation_vals': deviation_vals,
                      'central_vals': central_vals}
        return filtered, components
    return filtered


def _deviation_filter_viz(by_time, central_vals, mask, lower_vals, upper_vals):
    '''Generate visualization of deviation_time_filter result.

    Arguments
    ---------
    by_time: pd.DataFrame
        measured irradiance values indices are time of the day, columns are the date
    central_vals: pd.Series
        central values of data
    mask: pd.DataFrame
        value that is filtered based on deviance values and limits - organized same as by_time
    lower_vals: pd.Series
        time series data of central tendency minues deviance allowance
    upper_vals: pd.Series
        time series data of central tendency plus deviance allowance
    '''
    fig, ax = plt.subplots(figsize=(8, 2.5))

    by_time.plot(ax=ax, legend=False)
    central_vals.plot(ax=ax, color='black', legend=False)
    upper_vals.plot(ax=ax, color='black', linestyle='--', legend=False)
    lower_vals.plot(ax=ax, color='black', linestyle='--', legend=False)
    for col_data, col_mask in zip(by_time, mask):
        tmp_data = by_time[col_data]
        tmp_mask = mask[col_mask]
        if len(tmp_data[tmp_mask] > 0):
            ax.scatter(tmp_data[tmp_mask].index, tmp_data[tmp_mask],
                       edgecolor='black', facecolor='none', alpha=.25, zorder=100)
    plt.show()


if __name__ == '__main__':
    main()




