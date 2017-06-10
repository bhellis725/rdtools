
import os
import warnings

import pandas as pd
import numpy as np
import scipy.optimize as opt

from detection_template import DetectionTemplate


def main():
    file_path = os.path.expanduser('~/data_sets/snl_raw_data/1429_1405/raw_1405_weather_for_1429.csv')
    cols = ['Global_Wm2', 'Date-Time']
    data = pd.read_csv(file_path, parse_dates=['Date-Time'], usecols=cols, index_col=['Date-Time'])

    data = data.reindex(pd.date_range(start=data.index[0], end=data.index[-1], freq='1min')).fillna(0)
    data = data[(data.index >= '2016-07-01') & (data.index < '2016-07-08')]
    data = pd.Series(data['Global_Wm2'], index=data.index)

    mc = ModelCompareDetect(data, data)
    _ = mc.reno_hansen_detection()


class ModelCompareDetect(DetectionTemplate):
    '''Implementation of model comparison clear sky detection algorithms.  Inhereits from
    clearsky_detect_template.ClearskyDetection which implements the moving window
    properties calculations used here.
    '''

    def __init__(self, data, model, window=10, max_iter=10, mean_diff=75, max_diff=75,
                 lower_line_length=-5, upper_line_length=10, var_diff=.005, slope_dev=8, copy=True):
        '''Initialize class with data.  Data may be copied (it can be modified during detection).
        Window size and the tolerance for cloudiness metric are also set here.

        Arguments
        ---------
        data: pd.Series
            time series data (measured)
        model: pd.Series
            time series data (modeled)
        window, optional: int
            numer of measurements per window
        max_iter, optional: int
            number of allowed iterations for detection to converge
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
        copy, optional: bool
            copy data

        Returns
        -------
        None
        '''
        if not data.index.equals(model.index):
            raise NotImplementedError('Indices for series must be identical.')

        super(ModelCompareDetect, self).__init__(data, window, copy)

        if len(pd.unique(model.index.to_series().diff().dropna())) != 1:
            raise NotImplementedError('You must use evenly spaced time series data.')
        if copy:
            self.model = model.copy()
        else:
            self.model = model
        self.max_iter = max_iter
        self.mean_diff = mean_diff
        self.max_diff = max_diff
        self.lower_line_length = lower_line_length
        self.upper_line_length = upper_line_length
        self.var_diff = var_diff
        self.slope_dev = slope_dev

    def reno_hansen_detection(self, verbose=False):
        '''Detect clear sky periods by comparing to modeled values.  This is meant to be a recreation of
        the Reno-Hansen method already implemented in PVLib.  This is slower than the RH implementation,
        though it is easier to extract components for analysis.

        Arguments
        ---------
        verbose, optional: bool
            return components for calculation

        Returns
        -------
        clear_skies: pd.Series
            time series bool mask for clear (True) and obscured (False) skies
        '''

        slices = self.generate_window_slices()

        meas_comp = self.calc_components(self.data, slices=slices)

        alpha = 1
        for i in range(self.max_iter):
            model_comp = self.calc_components(alpha * self.model, slices=slices)
            slope_dev_comp = self.calc_slope_dev(alpha, slices=slices)
            mask = pd.DataFrame()
            mask['mean_diff'] = np.abs(meas_comp['avg'] - model_comp['avg']) < self.mean_diff
            mask['max_diff'] = np.abs(meas_comp['max'] - model_comp['max']) < self.max_diff
            mask['line_length'] = ((meas_comp['line_length'] - model_comp['line_length']) > self.lower_line_length) & \
                                  ((meas_comp['line_length'] - model_comp['line_length']) < self.upper_line_length)
            mask['var_diff'] = meas_comp['slope_std'] < self.var_diff
            mask['slope_dev'] = slope_dev_comp < self.slope_dev
            mask['finite_gt_zero'] = self.data > 0.0
            clear_skies = mask.all(axis=1)
            clear_meas = self.data[clear_skies].values
            clear_model = self.model[clear_skies].values
            last_alpha = alpha
            def rmse(alpha):
                return np.sqrt(np.mean(np.square(clear_meas - alpha * clear_model)))
            alpha = opt.minimize_scalar(rmse).x
            if np.isclose(alpha, last_alpha):
                break
        else:
            msg = 'Calculation did not converge. You can try increasing max_iter and rerunning.'
            warnings.warn(msg, RuntimeWarning)

        if verbose:
            total_components = pd.DataFrame()
            for meas_key, model_key in zip(meas_comp, model_comp):
                total_components['meas_' + meas_key] = meas_comp[meas_key]
                total_components['model_' + model_key] = model_comp[model_key]
            total_components['max_slope_dev'] = slope_dev_comp
            return clear_skies, {'alpha': alpha,
                                 'component_vals': total_components,
                                 'component_mask': mask}
        return clear_skies

    def calc_components(self, ser, slices=None):
        '''Calculate components needed for model comparions method.

        Arguments
        ---------
        data pd.Series
            time series of irradiance values

        Returns
        -------
        result: pd.DataFrame
            data frame with columns of avg, max, line_length, and slope_std
        '''
        if slices is None:
            slices = self.generate_window_slices()

        data_avg = np.apply_along_axis(self.calc_window_avg, 1, ser.values[slices])
        data_max = np.apply_along_axis(self.calc_window_max, 1, ser.values[slices])
        data_line_length = np.apply_along_axis(self.calc_window_line_length, 1, ser.values[slices])
        data_slope_std = np.apply_along_axis(self.calc_window_derivative_std_normed, 1, ser.values[slices])

        result = pd.DataFrame(np.nan, index=ser.index, columns=['avg', 'max', 'line_length', 'slope_std'])

        midpoints = np.apply_along_axis(self.get_midval, 1, slices)

        result['avg'].iloc[midpoints] = data_avg[:]
        result['max'].iloc[midpoints] = data_max[:]
        result['line_length'].iloc[midpoints] = data_line_length[:]
        result['slope_std'].iloc[midpoints] = data_slope_std[:]

        return result

    def calc_slope_dev(self, alpha, slices=None):
        '''Calculate slope_dev for RH reimplementation.  This is metric #5 from their paper.

        Arguments
        ---------
        meas: pd.Series
            tme series measured irradiance data
        model: pd.Series
            time series modeled irradiance data
        window: int
            size of moving window

        Returns
        -------
        max_slope_diff_series: pd.Series
            time series of metric
        '''
        if slices is None:
            slices = self.generate_window_slices()

        max_slope_diff = np.empty(slices.shape[0])
        for i in slices:
            max_slope_diff[i[0]] = self.calc_window_max_diff(self.data.values[i], alpha * self.model.values[i])

        midpoints = np.apply_along_axis(self.get_midval, 1, slices)
        max_slope_diff_series = pd.Series(np.nan, index=self.data.index)
        max_slope_diff_series.iloc[midpoints] = max_slope_diff[:]

        return max_slope_diff_series


if __name__ == '__main__':
    main()


