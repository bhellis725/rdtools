
import os

import pandas as pd
import numpy as np


def main():
    file_path = os.path.expanduser('~/data_sets/snl_raw_data/1429_1405/raw_1405_weather_for_1429.csv')
    cols = ['Global_Wm2', 'Date-Time']
    data = pd.read_csv(file_path, parse_dates=['Date-Time'], usecols=cols, index_col=['Date-Time'])

    data = data.reindex(pd.date_range(start=data.index[0], end=data.index[-1], freq='1min')).fillna(0)
    data = data[(data.index >= '2016-07-01 12:00:00') & (data.index < '2016-07-02')]
    data = pd.Series(data['Global_Wm2'], index=data.index)

    clear_times = model_free_detect(data)
    clear_times = model_free_detect_democratic(data)
    clear_times = model_free_detect_meanval(data)


def model_free_detect(data, window=10, verbose=False, metric_tol=.05):
    '''Determine clear sky periods based on irradiance measurements.

    Arguments
    ---------
    data: pd.Series
        time series irradiance data
    window: int
        number of samples to include in each window
    verbose: bool
        whether or not to return components used to determine is_clear
    tol: float
        tolerance for determining clear skies

    Returns
    -------
    is_clear: pd.Series
        boolean time series of clear times
    components: dict, optional
        contains series of normalized lengths, local integrals, and calculated metric
    '''
    # data quality check
    if len(pd.unique(data.index.to_series().diff().dropna())) != 1:
        raise NotImplementedError('You must use evenly spaced time series data.')
    if data.isnull().values.any():
        raise ValueError('NaN/infinity in data.  Process these first.')

    is_clear = pd.Series(False, data.index)

    components = calc_distances_integrals(data, window)
    distances = components['local_distances']
    integrals = components['local_integrals']
    metric = pd.Series(np.log(distances) / np.log(integrals), index=data.index)

    is_clear = (metric <= metric_tol) & (data > 0.0)

    # is_clear[data <= 0.0] = False

    if verbose:
        components['metric'] = metric
        return is_clear, components
    else:
        return is_clear


def model_free_detect_democratic(data, window=10, verbose=False, metric_tol=.05, pct_tol=0.5):
    '''Determine clear sky periods based on irradiance measurements.

    Arguments
    ---------
    data: pd.Series
        time series irradiance data
    window: int
        number of samples to include in each window
    verbose: bool
        whether or not to return components used to determine is_clear
    metric_tol: float
        tolerance for determining clear skies
    pct_tol: float
        percent of passes in order to grant pass/fail

    Returns
    -------
    is_clear: pd.Series
        boolean time series of clear times
    components: dict, optional
        contains series of normalized lengths, local integrals, and calculated metric
    '''
    if len(pd.unique(data.index.to_series().diff().dropna())) != 1:
        raise NotImplementedError('You must use evenly spaced time series data.')
    if data.isnull().values.any():
        raise ValueError('NaN/infinity in data.  Process these first.')

    is_clear = pd.Series(False, data.index)

    components = calc_distances_integrals(data, window)
    distances = components['local_distances']
    integrals = components['local_integrals']
    metric = pd.Series(np.log(distances) / np.log(integrals), index=data.index)

    for i in np.arange(0, len(metric) - window):
        group = metric.iloc[i: i + window]
        midpoint = group.iloc[[len(group) // 2 + 1]].index

        bools = (group <= metric_tol).astype(int)
        pct_pass = np.sum(bools) / window

        if pct_pass >= pct_tol and data[midpoint].values > 0.0:
                is_clear[midpoint] = True

    # is_clear[data <= 0.0] = False

    if verbose:
        components['metric'] = metric
        return is_clear, components
    else:
        return is_clear


def model_free_detect_meanval(data, window=10, verbose=False, metric_tol=.05):
    '''Determine clear sky periods based on irradiance measurements.

    Arguments
    ---------
    data: pd.Series
        time series irradiance data
    window: int
        number of samples to include in each window
    verbose: bool
        whether or not to return components used to determine is_clear
    metric_tol: float
        tolerance for determining clear skies
    pct_tol: float
        percent of passes in order to grant pass/fail

    Returns
    -------
    is_clear: pd.Series
        boolean time series of clear times
    components: dict, optional
        contains series of normalized lengths, local integrals, and calculated metric
    '''
    if len(pd.unique(data.index.to_series().diff().dropna())) != 1:
        raise NotImplementedError('You must use evenly spaced time series data.')
    if data.isnull().values.any():
        raise ValueError('NaN/infinity in data.  Process these first.')

    is_clear = pd.Series(False, data.index)

    components = calc_distances_integrals(data, window)
    distances = components['local_distances']
    integrals = components['local_integrals']
    metric = pd.Series(np.log(distances) / np.log(integrals), index=data.index)

    for i in np.arange(0, len(metric) - window):
        group = metric.iloc[i: i + window]
        midpoint = group.iloc[[len(group) // 2 + 1]].index

        meanval = np.mean(group)

        if meanval <= metric_tol and data[midpoint].values > 0.0:
            is_clear[midpoint] = True

    # is_clear[data <= 0.0] = False

    if verbose:
        components['metric'] = metric
        return is_clear, components
    else:
        return is_clear


def calc_distances_integrals(data, window=10):
    '''Calculate distances of line and integrals of moving window.  Values
    are reported at the central index of the window.

    Arguments
    ---------
    data: pd.Series
        time series irradiance data
    window: int
        number of samples to include in each window

    Returns
    -------
    result: dict
        time series data frame with distances and integrals
    '''
    if len(pd.unique(data.index.to_series().diff().dropna())) != 1:
        raise NotImplementedError('You must use evenly spaced time series data.')
    if data.isnull().values.any():
        raise ValueError('NaN/infinity in data.  Process these first.')

    local_distances = pd.Series(np.nan, index=data.index, name='local_distances')
    local_integrals = pd.Series(np.nan, index=data.index, name='local_integrals')

    for i in np.arange(0, len(data) - window):
        group = data.iloc[i: i + window]
        midpoint = group.iloc[[len(group) // 2 + 1]].index

        d_total = ts_distance(group)
        d_min = ts_distance(group.iloc[[0, -1]])
        d_norm = d_total / d_min
        integral = ts_integral(group)

        local_distances[midpoint] = d_norm
        local_integrals[midpoint] = integral

    result = {'local_distances': local_distances,
              'local_integrals': local_integrals}

    return result


def ts_distance(series):
    d_index = series.index.to_series().diff().dt.seconds.div(60, fill_value=0).astype(int).values
    d_value = series.diff().values
    distance = np.sum(np.sqrt(np.square(d_index[1:]) + np.square(d_value[1:])))
    return distance


def ts_integral(series):
    return np.trapz(series.values.ravel())


if __name__ == '__main__':
    main()
