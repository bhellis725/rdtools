
import os
import warnings

import pandas as pd
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt


def main():
    file_path = os.path.expanduser('~/data_sets/snl_raw_data/1429_1405/raw_1405_weather_for_1429.csv')
    cols = ['Global_Wm2', 'Date-Time']
    data = pd.read_csv(file_path, parse_dates=['Date-Time'], usecols=cols, index_col=['Date-Time'])

    data = data.reindex(pd.date_range(start=data.index[0], end=data.index[-1], freq='1min')).fillna(0)
    data = data[(data.index >= '2016-07-01 12:00:00') & (data.index < '2016-07-02')]
    data = pd.Series(data['Global_Wm2'], index=data.index)

    clear_times = model_free_detect(data)
    print('passed standard')
    clear_times = model_free_detect_meanval(data)
    print('passed meanval')
    clear_times = model_free_detect_democratic(data)
    print('passed democratic')
    clear_times = model_comparison(data, data)
    print('passed comparison')


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
    metric = calc_cloudiness_metric(distances.values, integrals.values, data.index)

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
    metric = calc_cloudiness_metric(distances.values, integrals.values, data.index)

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
    metric = calc_cloudiness_metric(distances.values, integrals.values, data.index)

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
    distances = np.apply_along_axis(calc_window_line_length_norm, 1, data.values[H])
    integrals = np.apply_along_axis(calc_window_integral, 1, data.values[H])

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
    midpoint = (len(array) // 2)
    midval = array[midpoint]
    return midval


# def calc_window_distance(array):
#     '''Calculate normalized distance of array.
#
#     Normalized distance is sum of the distsances of line segments divided
#     by the distance of the line segment of the end points.  The distance between
#     points in the x direction is assumed to be one.
#
#     Arguments
#     ---------
#     array: numpy array
#
#     Returns
#     -------
#     d_norm: float
#         normalized distance
#     '''
#     d_value = np.diff(array)
#     d_total = np.sum(np.sqrt(np.square(d_value[:]) + 1))
#     d_line  = np.sqrt(np.square(array[-1] - array[0]) + np.square(len(array) - 1))
#     d_norm = d_total / d_line
#     return d_norm


def calc_window_integral(array):
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


def calc_cloudiness_metric(distances, integrals, index):
    '''Calculate the cloudiness metric.

    Cloudiness = log(distances) / log(integrals)

    Arguments
    ---------
    distances: np.array
        normalized distsances of windows
    integrals: np.array
        local integral of irradiance of a window
    index: array like
        indices to use for metric series

    Returns
    -------
    metric: pd.Series
        series of metric calculations

    '''
    metric = pd.Series(np.log(distances) / np.log(integrals), index=index)
    return metric


def calc_window_avg(array):
    '''Calculate average value of array.

    Arugments
    ---------
    array: np.array

    Returns
    -------
    val: float
        mean value of array
    '''
    return np.mean(array)


def calc_window_max(array):
    '''Calculate average value of array.

    Arugments
    ---------
    array: np.array

    Returns
    -------
    val: float
        mean value of array
    '''
    return np.max(array)


def calc_window_derivative_avg(array):
    '''Calculate average derivative of array.

    Arguments
    ---------
    array: np.array

    Returns
    -------
    avg_dy_dx: float
        average derivative of array
    '''
    # dy = np.gradient(array)
    dy = np.diff(array)
    avg_dy_dx = np.mean(np.abs(dy)) # dx always 1
    return avg_dy_dx


def calc_window_derivative_max(array):
    '''Calculate max derivative of array.

    Arguments
    ---------
    array: np.array

    Returns
    -------
    max_dy_dx: float
        average derivative of array
    '''
    # dy = np.gradient(array)
    dy = np.diff(array)
    max_dy_dx = np.max(np.abs(dy)) # dx always 1
    return max_dy_dx


def calc_window_derivative_std(array):
    '''Calculate std deviation of derivatives of array.

    Arguments
    ---------
    array: np.array

    Returns
    -------
    max_dy_dx: float
        average derivative of array
    '''
    # dy = np.gradient(array)
    dy = np.diff(array) # diff as that's what Reno-Hansen indicate
    std_dy = np.std(dy) # dx always 1
    return std_dy


def calc_window_derivative_std_normed(array):
    '''Calculate std deviation of derivatives of array.  This is normalized
    by the average value of irradiance over the time interval.  This metric
    is used in Reno-Hansen detection.

    Arguments
    ---------
    array: np.array

    Returns
    -------
    norm_std: float
        std devation of derivatives divided by average array value
    '''
    window_mean = calc_window_avg(array)
    dy = np.diff(array) # diff as that's what Reno-Hansen indicate
    std_dy = np.std(dy, ddof=1) # dx always 1
    return std_dy / window_mean


def calc_window_max_diff(meas, model):
    '''Calculate the maximum difference between measured and modeled
    irradiance slopes for a given time window.

    Arugments
    ---------
    meas: np.array
        measured irradiance values
    model: mp.array
        modeled irradiance values

    Returns
    -------
    max_abs_diff: float
        maximum absolute difference between two arrays
    '''
    meas_diff = np.diff(meas)
    model_diff = np.diff(model)
    max_abs_diff = np.max(np.abs(meas_diff - model_diff))
    return max_abs_diff


def calc_window_line_length(array):
    '''Calculate line length of an array.  The points are assumed to
    be evenly spaced (dx=1).

    Arguments
    ---------
    array: np.array

    Returns
    -------
    line_length: float
    '''
    diffs = np.diff(array)
    line_length = np.sum(np.sqrt(np.square(diffs[:]) + 1)) # 1 for dx
    return line_length


def calc_window_line_length_norm(array):
    '''Calculate normalizedline length of an array.  The points are assumed to
    be evenly spaced (dx=1).  Line length ar normalized by the
    straight line distances between the first and last array elements.

    Arguments
    ---------
    array: np.array

    Returns
    -------
    line_length_norm: float
    '''
    line_length = calc_window_line_length(array)
    endpoint_line_length  = np.sqrt(np.square(array[-1] - array[0]) + np.square(len(array) - 1))
    line_length_norm = line_length / endpoint_line_length
    return line_length_norm


def properties_calculator(data, window):
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
    result: pd.DataFrame
        dataframe of local distances, integrals, average derivative, std deviation of derivative,
        and local average value of GHI as columns
    '''
    if len(pd.unique(data.index.to_series().diff().dropna())) != 1:
        raise NotImplementedError('You must use evenly spaced time series data.')

    local_distances = pd.Series(np.nan, index=data.index, name='local_distances')
    local_integrals = pd.Series(np.nan, index=data.index, name='local_integrals')
    avg_derivative = pd.Series(np.nan, index=data.index, name='avg_derivative')
    max_derivative = pd.Series(np.nan, index=data.index, name='max_derivative')
    std_derivative = pd.Series(np.nan, index=data.index, name='std_derivative')
    local_avg_val = pd.Series(np.nan, index=data.index, name='avg_val')

    H = la.hankel(np.arange(0, len(data) - window + 1),
                  np.arange(len(data) - window, len(data)))

    midpoints = np.apply_along_axis(get_midval, 1, H)

    arr_distances = np.apply_along_axis(calc_window_line_length_norm, 1, data.values[H])
    arr_integrals = np.apply_along_axis(calc_window_integral, 1, data.values[H])
    arr_derivative_avg = np.apply_along_axis(calc_window_derivative_avg, 1, data.values[H])
    arr_derivative_std = np.apply_along_axis(calc_window_derivative_std, 1, data.values[H])
    arr_derivative_max = np.apply_along_axis(calc_window_derivative_max, 1, data.values[H])
    arr_avg_val = np.apply_along_axis(calc_window_avg, 1, data.values[H])

    local_distances.iloc[midpoints] = arr_distances[:]
    local_integrals.iloc[midpoints] = arr_integrals[:]
    avg_derivative.iloc[midpoints] = arr_derivative_avg[:]
    std_derivative.iloc[midpoints] = arr_derivative_std[:]
    max_derivative.iloc[midpoints] = arr_derivative_max[:]
    local_avg_val.iloc[midpoints] = arr_avg_val[:]

    return pd.concat([local_distances, local_integrals,
                      avg_derivative, std_derivative, max_derivative, local_avg_val], axis=1)


# def model_comparison(meas, model, window=10, mean_diff=75, max_diff=75, lower_line_length=-5,
#                      upper_line_length=10, var_diff=.005, slope_dev=8, verbose=False):
#     '''Detect clear sky periods by comparing to modeled values.  This is meant to be a recreation of
#     the Reno-Hansen method already implemented in PVLib.
#
#     Arguments
#     ---------
#     meas: pd.Series
#         time series of measured irradiance values
#     model: pd.Series
#         time series of modeled irradiance values
#     window: int
#         size of window for comparison parameters
#     mean_diff: float
#         agreement tolerance for mean value between meas and modeled
#     max_diff: float
#         agreement tolerance for max value between meas and modeled
#     lower_line_length: float
#         lower agreement tolerance for meas line length - modeled line length
#     upper_line_length: float
#         upper agreement tolerance for meas line length + modeled line length
#     var_diff: float
#         agreement tolerance for normalized standard deviations of slopes
#     slope_dev: float
#         agreement tolerance for maximum difference of successive points between meas and modeled
#
#     Returns
#     -------
#     clear_skies: pd.Series
#         time series bool mask for clear (True) and obscured (False) skies
#     '''
#     for ser in [meas, model]:
#         if len(pd.unique(ser.index.to_series().diff().dropna())) != 1:
#             raise NotImplementedError('You must use evenly spaced time series data.')
#     if not meas.index.equals(model.index):
#         raise NotImplementedError('Indices for meas and modeled series must be identical.')
#
#     components = calc_model_comparison_components(meas, model, window)
#
#     mask = pd.DataFrame(False, index=meas.index,
#                         columns=['mean_diff', 'max_diff', 'line_length', 'var_diff', 'slope_dev', 'zero_test'])
#
#     mask['mean_diff'] = np.abs(components['meas_avg'] - components['model_avg']) < mean_diff
#     mask['max_diff'] = np.abs(components['meas_max'] - components['model_max']) < max_diff
#     mask['line_length'] = ((components['meas_line_length'] - components['model_line_length']) > lower_line_length) & \
#                           ((components['meas_line_length'] - components['model_line_length']) < upper_line_length)
#     mask['var_diff'] = components['meas_slope_std'] < var_diff
#     mask['slope_dev'] = components['max_slope_diff'] < slope_dev
#     mask['zero_test'] = meas > 0.0
#
#     clear_skies = mask.all(axis=1)
#
#     if verbose:
#         return clear_skies, mask, components
#     return clear_skies
#
#
# def calc_model_comparison_components(meas, model, window):
#     '''Calculate components needed for model comparions method.
#
#     Arguments
#     ---------
#     meas: pd.Series
#         time series of measured irradiance values
#     model: pd.Series
#         time series of modeled irradiance values
#     window: int
#         size of window for comparison parameters
#
#     Returns
#     -------
#     xyz
#     '''
#     H = la.hankel(np.arange(0, len(meas) - window + 1),
#                   np.arange(len(meas) - window, len(meas)))
#     meas_avg = np.apply_along_axis(calc_window_avg, 1, meas.values[H])
#     meas_max = np.apply_along_axis(calc_window_max, 1, meas.values[H])
#     meas_line_length = np.apply_along_axis(calc_window_line_length, 1, meas.values[H])
#     meas_slope_std = np.apply_along_axis(calc_window_derivative_std_normed, 1, meas.values[H])
#
#     model_avg = np.apply_along_axis(calc_window_avg, 1, model.values[H])
#     model_max = np.apply_along_axis(calc_window_max, 1, model.values[H])
#     model_line_length = np.apply_along_axis(calc_window_line_length, 1, model.values[H])
#     model_slope_std = np.apply_along_axis(calc_window_derivative_std_normed, 1, model.values[H])
#
#     max_slope_diff = np.empty(H.shape[0])
#     for i in H:
#         max_slope_diff[i[0]] = calc_window_max_diff(meas.values[i], model.values[i])
#
#     result = pd.DataFrame(np.nan, index=meas.index,
#                           columns=['meas_avg', 'meas_max', 'meas_line_length', 'meas_slope_std',
#                                   'model_avg', 'model_max', 'model_line_length', 'model_slope_std', 'max_slope_diff'])
#
#     midpoints = np.apply_along_axis(get_midval, 1, H)
#
#     result['meas_avg'].iloc[midpoints] = meas_avg[:]
#     result['meas_max'].iloc[midpoints] = meas_max[:]
#     result['meas_line_length'].iloc[midpoints] = meas_line_length[:]
#     result['meas_slope_std'].iloc[midpoints] = meas_slope_std[:]
#
#     result['model_avg'].iloc[midpoints] = model_avg[:]
#     result['model_max'].iloc[midpoints] = model_max[:]
#     result['model_line_length'].iloc[midpoints] = model_line_length[:]
#     result['model_slope_std'].iloc[midpoints] = model_slope_std[:]
#
#     result['max_slope_diff'].iloc[midpoints] = max_slope_diff[:]
#
#     return result


def calc_model_comparison_components(data, window):
    '''Calculate components needed for model comparions method.

    Arguments
    ---------
    data pd.Series
        time series of irradiance values
    window: int
        size of window for comparison parameters

    Returns
    -------
    result: pd.DataFrame
        data frame with columns of avg, max, line_length, and slope_std
    '''
    H = la.hankel(np.arange(0, len(data) - window + 1),
                  np.arange(len(data) - window, len(data)))
    data_avg = np.apply_along_axis(calc_window_avg, 1, data.values[H])
    data_max = np.apply_along_axis(calc_window_max, 1, data.values[H])
    data_line_length = np.apply_along_axis(calc_window_line_length, 1, data.values[H])
    data_slope_std = np.apply_along_axis(calc_window_derivative_std_normed, 1, data.values[H])

    result = pd.DataFrame(np.nan, index=data.index, columns=['avg', 'max', 'line_length', 'slope_std'])

    midpoints = np.apply_along_axis(get_midval, 1, H)

    result['avg'].iloc[midpoints] = data_avg[:]
    result['max'].iloc[midpoints] = data_max[:]
    result['line_length'].iloc[midpoints] = data_line_length[:]
    result['slope_std'].iloc[midpoints] = data_slope_std[:]

    return result


def calc_model_comparison_slope_dev(meas, model, window):
    '''Calculate slope_dev for RH reimplementation.  This is metric #5 from their paper.

    Arguments
    ---------
    meas: pd.Series
        time series measured irradiance data
    model: pd.Series
        time series modeled irradiance data
    window: int
        size of moving window

    Returns
    -------
    max_slope_diff_series: pd.Series
        time series of metric
    '''
    H = la.hankel(np.arange(0, len(meas) - window + 1),
                  np.arange(len(meas) - window, len(meas)))

    max_slope_diff = np.empty(H.shape[0])
    for i in H:
        max_slope_diff[i[0]] = calc_window_max_diff(meas.values[i], model.values[i])

    midpoints = np.apply_along_axis(get_midval, 1, H)
    max_slope_diff_series = pd.Series(np.nan, index=meas.index)
    max_slope_diff_series.iloc[midpoints] = max_slope_diff[:]

    return max_slope_diff_series


def model_comparison(meas, model, window=10, max_iter=10, mean_diff=75,
                     max_diff=75, lower_line_length=-5, upper_line_length=10,
                     var_diff=.005, slope_dev=8, verbose=False):
    '''Detect clear sky periods by comparing to modeled values.  This is meant to be a recreation of
    the Reno-Hansen method already implemented in PVLib.  This is much slower than the RH implementation,
    though it is easier to extract components for analysis.

    Arguments
    ---------
    meas: pd.Series
        time series of measured irradiance values
    model: pd.Series
        time series of modeled irradiance values
    window: int
        size of window for comparison parameters
    mean_diff: float
        agreement tolerance for mean value between meas and modeled
    max_diff: float
        agreement tolerance for max value between meas and modeled
    lower_line_length: float
        lower agreement tolerance for meas line length - modeled line length
    upper_line_length: float
        upper agreement tolerance for meas line length + modeled line length
    var_diff: float
        agreement tolerance for normalized standard deviations of slopes
    slope_dev: float
        agreement tolerance for maximum difference of successive points between meas and modeled

    Returns
    -------
    clear_skies: pd.Series
        time series bool mask for clear (True) and obscured (False) skies
    '''
    for ser in [meas, model]:
        if len(pd.unique(ser.index.to_series().diff().dropna())) != 1:
            raise NotImplementedError('You must use evenly spaced time series data.')
    if not meas.index.equals(model.index):
        raise NotImplementedError('Indices for meas and modeled series must be identical.')

    meas_components = calc_model_comparison_components(meas, window)

    alpha = 1
    for i in range(max_iter):
        model_components = calc_model_comparison_components(alpha * model, window)
        slope_dev_component = calc_model_comparison_slope_dev(meas, alpha * model, window)
        mask = pd.DataFrame()
        mask['mean_diff'] = np.abs(meas_components['avg'] - model_components['avg']) < mean_diff
        mask['max_diff'] = np.abs(meas_components['max'] - model_components['max']) < max_diff
        mask['line_length'] = ((meas_components['line_length'] - model_components['line_length']) > lower_line_length) & \
                              ((meas_components['line_length'] - model_components['line_length']) < upper_line_length)
        mask['var_diff'] = meas_components['slope_std'] < var_diff
        mask['slope_dev'] = slope_dev_component < slope_dev
        mask['finite_gt_zero'] = meas > 0.0
        clear_skies = mask.all(axis=1)
        clear_meas = meas[clear_skies].values
        clear_model = model[clear_skies].values
        last_alpha = alpha
        def rmse(alpha):
            return np.sqrt(np.mean(np.square(clear_meas - alpha * clear_model)))
        alpha = opt.minimize_scalar(rmse).x
        if np.isclose(alpha, last_alpha):
            break
    else:
        warnings.warn('Calculation did not converge.', RuntimeWarning)

    if verbose:
        total_components = pd.DataFrame()
        for meas_key, model_key in zip(meas_components, model_components):
            total_components['meas_' + meas_key] = meas_components[meas_key]
            total_components['model_' + model_key] = model_components[model_key]
        total_components['max_slope_dev'] = slope_dev_component
        return clear_skies, {'alpha': alpha,
                             'component_vals': total_components,
                             'component_mask': mask}
    return clear_skies


if __name__ == '__main__':
    main()



