

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

    filtered = zscore_time_filter(sample, verbose=True, viz=True)


def zscore_time_filter(data, zscore_range=(-1, np.inf), replace_val=(-np.inf, np.inf), verbose=False, viz=False):
    '''Filter measurements by time of day based on zscore at that time.
    It is suggested data be on the order of a couple weeks to a month.  This method
    will be applied to data that has seasonal variations.  In order to minimize the impact
    of seasonal variations, smaller samples should be used.

    Arguments
    ---------
    data: pd time series
        data to be filtered based on z score
    zscore_range: tuple(float, float)
        z score range outside of which values will be filtered
    replace_val: tuple(float, float)
        values to set data to if outside the zscore range
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

    day_list = []
    for day, group in data.groupby(data.index.date):
        ser = pd.Series(group.values, index=group.index.time, name=day)
        day_list.append(ser)
    by_time = pd.concat(day_list, axis=1)
    avg = pd.Series(by_time.replace(0, np.nan).mean(axis=1), name='avg') # values of 0 will be ignored - assumed missing data
    std = pd.Series(by_time.replace(0, np.nan).std(axis=1), name='std')  # values of 0 will be ignored - assumed missing data
    stats = pd.concat([avg, std], axis=1)

    filtered = data.copy()
    mask_list = []
    zscore_list = []
    for day, group in filtered.groupby(filtered.index.date):
        zscores = pd.Series((group.values - stats['avg'].values) / stats['std'].values, index=group.index)
        zscores_vals = pd.Series(zscores.values, index=group.index.time, name=day)
        zscore_list.append(zscores_vals)
        mask_vals = pd.Series(((zscores < zscore_range[0]) | (zscores > zscore_range[1])).values, name=day, index=group.index.time)
        mask_list.append(mask_vals.copy())
        group[zscores < zscore_range[0]] = replace_val[0]
        group[zscores > zscore_range[1]] = replace_val[1]
        filtered[group.index] = group

    if viz or verbose:
        mask = pd.concat(mask_list, axis=1)
        zscore = pd.concat(zscore_list, axis=1)

    if viz:
        _zscore_filter_viz(stats, by_time, mask)

    if verbose:
        components = {'by_time': by_time,
                      'stats': stats,
                      'mask': mask,
                      'zscore': zscore}
        return filtered, components
    return filtered


def _zscore_filter_viz(stats, by_time, mask):
    '''Generate visualization of zscore filtered data.  Plots all data (by day) and
    marks points that were outside the specified zscore range (from zscore_time_filter).

    Arguments
    ---------
    stats: pd.Series
        Time series of mean and std for measurements (generated from by_time).  Indices are times
        throughout a given day.
    by_time: pd.DataFrame
        Time series data of measured values.  Columns are the day, indices are the time of day
        of the measurement.
    mask: pd.DataFrame
        Time series data.  Bools where data has been filtered based on zscore from zscore_time_filter.
        Has same structure as by_time.

    Returns
    -------
    None
    '''
    fig, ax = plt.subplots(figsize=(10, 2.5))

    by_time.plot(ax=ax, legend=False)
    stats['avg'].plot(ax=ax, legend=False, color='black')
    (stats['avg'] - stats['std']).plot(ax=ax, legend=False, color='black', linestyle='--')
    (stats['avg'] + stats['std']).plot(ax=ax, legend=False, color='black', linestyle='--')

    # quantile = by_time.quantile(q=.9, axis=1)
    # quantile = quantile.rolling(30, center=True).mean()
    # quantile.plot(ax=ax, legend=False, color='black', linestyle='--')

    for col_data, col_mask in zip(by_time, mask):
        tmp_data = by_time[col_data]
        tmp_mask = mask[col_mask]
        if len(tmp_data[tmp_mask] > 0):
            ax.scatter(tmp_data[tmp_mask].index, tmp_data[tmp_mask],
                       edgecolor='black', facecolor='none', alpha=.25, zorder=100)

    plt.show()




if __name__ == '__main__':
    main()




