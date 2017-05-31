
import os

import pandas as pd
import numpy as np
import scipy.linalg as la


def main():
    file_path = os.path.expanduser('~/data_sets/snl_raw_data/1429_1405/raw_1405_weather_for_1429.csv')
    cols = ['Global_Wm2', 'Date-Time']
    data = pd.read_csv(file_path, parse_dates=['Date-Time'], usecols=cols, index_col=['Date-Time'])

    data = data.reindex(pd.date_range(start=data.index[0], end=data.index[-1], freq='1min')).fillna(0)
    data = data[(data.index >= '2016-07-01 12:00:00') & (data.index < '2016-07-02')]
    data = pd.Series(data['Global_Wm2'], index=data.index)

    clear_times = model_free_detect(data)
    clear_times = model_free_detect_meanval(data)
    clear_times = model_free_detect_democratic(data)


def model_free_detect(data, window=30, metric_tol=.05, verbose=False):
    '''Determine clear sky periods based on irradiance measurements.

    Arguments
    ---------
    data: pd.Series
        time series irradiance data
    window: int
        number of samples to include in each window
    metric_tol: float
        tolerance for determining clear skies
    verbose: bool
        whether or not to return components used to determine is_clear

    Returns
    -------
    is_clear: pd.Series
        boolean time series of clear times
    components: dict, optional
        contains series of normalized lengths, local integrals, and calculated metric
    '''
    if len(pd.unique(data.index.to_series().diff().dropna())) != 1:
        raise NotImplementedError('You must use evenly spaced time series data.')

    is_clear = pd.Series(False, data.index)

    components = calc_components(data, window)
    distances = components['local_distances']
    integrals = components['local_integrals']
    metric = pd.Series(np.log(distances) / np.log(integrals), index=data.index)

    is_clear = (metric <= metric_tol) & (data > 0.0)

    if verbose:
        components['metric'] = metric
        return is_clear, components
    else:
        return is_clear


def model_free_detect_democratic(data, window=30, metric_tol=.05, vote_pct=0.6, verbose=False):
    '''Determine clear sky periods based on irradiance measurements.

    Arguments
    ---------
    data: pd.Series
        time series irradiance data
    window: int
        number of samples to include in each window
    metric_tol: float
        tolerance for determining clear skies
    vote_pct: float
        percent of passes in order to grant pass/fail
    verbose: bool
        whether or not to return components used to determine is_clear

    Returns
    -------
    is_clear: pd.Series
        boolean time series of clear times
    components: dict, optional
        contains series of normalized lengths, local integrals, and calculated metric
    '''
    if len(pd.unique(data.index.to_series().diff().dropna())) != 1:
        raise NotImplementedError('You must use evenly spaced time series data.')

    is_clear = pd.Series(False, data.index)

    components = calc_components(data, window)
    distances = components['local_distances']
    integrals = components['local_integrals']
    metric = pd.Series(np.log(distances.values) / np.log(integrals.values), index=data.index)

    H = la.hankel(np.arange(0, len(data) - window + 1),
                  np.arange(len(data) - window, len(data)))

    midpoints = np.apply_along_axis(get_midval, 1, H)
    pcts = np.apply_along_axis(calc_pct, 1, metric.values[H], metric_tol)

    is_clear.iloc[midpoints] = (pcts >= vote_pct) & (data[midpoints] > 0.0)

    if verbose:
        components['metric'] = metric
        return is_clear, components
    else:
        return is_clear


def model_free_detect_meanval(data, window=30, metric_tol=.05, verbose=False):
    '''Determine clear sky periods based on irradiance measurements.

    Arguments
    ---------
    data: pd.Series
        time series irradiance data
    window: int
        number of samples to include in each window
    metric_tol: float
        tolerance for determining clear skies
    verbose: bool
        whether or not to return components used to determine is_clear

    Returns
    -------
    is_clear: pd.Series
        boolean time series of clear times
    components: dict, optional
        contains series of normalized lengths, local integrals, and calculated metric
    '''
    if len(pd.unique(data.index.to_series().diff().dropna())) != 1:
        raise NotImplementedError('You must use evenly spaced time series data.')

    is_clear = pd.Series(False, data.index)

    components = calc_components(data, window)
    distances = components['local_distances']
    integrals = components['local_integrals']
    metric = pd.Series(np.log(distances.values) / np.log(integrals.values), index=data.index)

    H = la.hankel(np.arange(0, len(data) - window + 1),
                  np.arange(len(data) - window, len(data)))

    midpoints = np.apply_along_axis(get_midval, 1, H)
    means = np.apply_along_axis(np.mean, 1, metric.values[H])

    is_clear.iloc[midpoints] = (means <= metric_tol) & (data[midpoints] > 0.0)

    if verbose:
        components['metric'] = metric
        return is_clear, components
    else:
        return is_clear


def calc_components(data, window):
    '''Calculate normalized distances and integrals of moving window.  Values
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

    local_distances = pd.Series(np.nan, index=data.index, name='local_distances')
    local_integrals = pd.Series(np.nan, index=data.index, name='local_integrals')

    H = la.hankel(np.arange(0, len(data) - window + 1),
                  np.arange(len(data) - window, len(data)))

    midpoints = np.apply_along_axis(get_midval, 1, H)
    distances = np.apply_along_axis(calc_distance, 1, data.values[H])
    integrals = np.apply_along_axis(calc_integral, 1, data.values[H])

    local_distances.iloc[midpoints] = distances[:]
    local_integrals.iloc[midpoints] = integrals[:]

    result = {'local_distances': local_distances,
              'local_integrals': local_integrals}

    return result


def get_midval(array):
    '''Returns element at the midpoint of an array.

    Arguments
    ---------
    array: numpy array

    Returns
    -------
    midval: element at midpoint of array
        type depends on array elements
    '''
    midpoint = (len(array) // 2) + 1
    midval = array[midpoint]
    return midval


def calc_distance(array):
    '''Calculate normalized distance of array.

    Normalized distance is sum of the distsances of line segments divided
    by the distance of the line segment of the end points.  The distance between
    points in the x direction is assumed to be one.

    Arguments
    ---------
    array: numpy array

    Returns
    -------
    d_norm: float
        normalized distance
    '''
    d_value = np.diff(array)
    d_total = np.sum(np.sqrt(np.square(d_value[:]) + 1))
    d_line  = np.sqrt(np.square(array[-1] - array[0]) + np.square(len(array) - 1))
    d_norm = d_total / d_line
    return d_norm


def calc_integral(array):
    '''Calculate integral of array values.

    Array values are assumed to be y values and dx is assumed to be one.

    Arguments
    ---------
    array: numpy array

    Returns
    -------
    val: float
        integral of array

    '''
    val = np.trapz(array)
    return val


def calc_pct(array, metric_tol):
    '''Calculate percent of array elements that are less than or equal to a tolerance.

    Arguments
    ---------
    array: numpy array
    metric_tol: numeric
        tolerance to be less than/equal to

    Returns
    -------
    pct: float
        percent of values that are <= metric_tol
    '''
    pct = np.sum((array <= metric_tol).astype(int)) / len(array)
    return pct


if __name__ == '__main__':
    main()



