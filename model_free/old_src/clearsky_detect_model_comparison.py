
import os

import pandas as pd
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt

import clearsky_detect_model_free


def main():
    file_path = os.path.expanduser('~/data_sets/snl_raw_data/1429_1405/raw_1405_weather_for_1429.csv')
    cols = ['Global_Wm2', 'Date-Time']
    data = pd.read_csv(file_path, parse_dates=['Date-Time'], usecols=cols, index_col=['Date-Time'])

    data = data.reindex(pd.date_range(start=data.index[0], end=data.index[-1], freq='1min')).fillna(0)
    data = data[(data.index >= '2016-07-01 12:00:00') & (data.index < '2016-07-02')]
    data = pd.Series(data['Global_Wm2'], index=data.index)

    comparison_detect(data, data)


def comparison_detect(meas, model, window=30, error_tol=.10, verbose=False):
    '''Determine clear sky periods based on irradiance measurements by comparing
    measurement data to modeled data using a moving window scheme.

    Arguments
    ---------
    meas: pd.Series
        measured time series irradiance data
    model: pd.Series
        modeled time series irradiance data
    window: int
        number of samples to include in each window
    error_tol: float
        error tolerance between measured and modeled (as decimal, .10 = 10%)
    verbose: bool
        whether or not to return components used to determine sky clarity

    Returns
    -------
    is_clear: pd.Series
        boolean time series of clear times
    components: dict, optional
        contains series of normalized lengths, local integrals, and calculated metric
    '''
    if len(pd.unique(meas.index.to_series().diff().dropna())) != 1:
        raise NotImplementedError('You must use evenly spaced time series data.')

    meas_components = clearsky_detect_model_free.calc_components(meas, window)
    meas_distances = meas_components['local_distances']
    meas_integrals = meas_components['local_integrals']
    meas_metric = pd.Series(np.log(meas_distances.values) / np.log(meas_integrals.values), index=meas.index)


    alpha = 1
    for i in range(10):
        model_components = clearsky_detect_model_free.calc_components(alpha * model, window)
        model_distances = model_components['local_distances']
        model_integrals = model_components['local_integrals']
        model_metric = pd.Series(np.log(model_distances.values) / np.log(model_integrals.values), index=meas.index)
        def rmse(alpha):
            return np.sqrt(np.mean(np.square(meas_metric - model_metric)))
        alpha = opt.minimize_scalar(rmse).x
        print(alpha)

        # pct_errors = np.square(meas_metric - model_metric) # / model_metric
        # pct_errors = np.abs(meas_metric - model_metric) / model_metric

    # is_clear = (pct_errors <= error_tol) & (meas > 0.0)
    is_clear = (np.abs(meas_metric - model_metric) <= error_tol) & (meas > 0)

    if verbose:
        components = {}
        meas_components['metric'] = meas_metric
        model_components['metric'] = model_metric
        for meas_key, model_key in zip(meas_components, model_components):
            components['meas_' + meas_key] = meas_components[meas_key]
            components['meas_' + model_key] = model_components[model_key]
        # components['pct_errors'] = pct_errors
        return is_clear, components
    else:
        return is_clear


if __name__ == '__main__':
    main()
