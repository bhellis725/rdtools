
import os
import warnings

import pandas as pd
import numpy as np

from detection_template import DetectionTemplate


def main():
    file_path = os.path.expanduser('~/data_sets/snl_raw_data/1429_1405/raw_1405_weather_for_1429.csv')
    cols = ['Global_Wm2', 'Date-Time']
    data = pd.read_csv(file_path, parse_dates=['Date-Time'], usecols=cols, index_col=['Date-Time'])

    data = data.reindex(pd.date_range(start=data.index[0], end=data.index[-1], freq='1min')).fillna(0)
    data = data[(data.index >= '2016-07-01') & (data.index < '2016-07-08')]
    data = pd.Series(data['Global_Wm2'], index=data.index)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mf = ModelFreeDetect(data)
        _ = mf.standard_detection()
        print('standard detection')
        _ = mf.mean_detection()
        print('mean detection')
        _ = mf.democratic_detection()
        print('democratic detection')
        _ = mf.deviation_time_filter()
        print('filtered')
        _ = mf.standard_detection()
        print('filtered standard detection')


class ModelFreeDetect(DetectionTemplate):
    '''Implementation of model-free clear sky detection algorithms.  Inhereits from
    clearsky_detect_template.ClearskyDetection which implements the moving window
    properties calculations used here.
    '''

    def __init__(self, data, window=30, metric_tol=.05, copy=True):
        '''Initialize class with data.  Data may be copied (it can be modified during detection).
        Window size and the tolerance for cloudiness metric are also set here.

        Arguments
        ---------
        data: pd.Series
            time series data
        window, optional: int
            numer of measurements per window
        metric_tol, optional: float
            tolerance for determining clear/cloudy skies
        copy, optional: bool
            copy data

        Returns
        -------
        None
        '''
        super(ModelFreeDetect, self).__init__(data, window, copy)
        self.metric_tol = metric_tol

    def standard_detection(self, verbose=False):
        '''Determine clear sky periods based on irradiance measurements.  Central value of window
        is labeled clear if window passes test.

        Arguments
        ---------
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
        components, optional: dict
            contains series of normalized lengths, local integrals, and calculated metric
        '''
        is_clear = pd.Series(False, self.data.index)

        components = self.calc_components()

        is_clear = (components['metric'] <= self.metric_tol) & (self.data > 0.0)

        if verbose:
            return is_clear, components
        else:
            return is_clear

    def mean_detection(self, verbose=False):
        '''Determine clear sky periods based on irradiance measurements.  Central value
        of window is labeled clear if the average value of the window is at or below metric_tol.

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
        components, optional: dict
            contains series of normalized lengths, local integrals, and calculated metric
        '''
        is_clear = pd.Series(False, self.data.index)

        slices = self.generate_window_slices()

        components = self.calc_components(slices=slices)

        midpoints = np.apply_along_axis(self.get_midval, 1, slices)
        means = np.apply_along_axis(np.mean, 1, components['metric'].values[slices])

        is_clear.iloc[midpoints] = (means <= self.metric_tol) & (self.data[midpoints] > 0.0)

        if verbose:
            return is_clear, components
        else:
            return is_clear

    def democratic_detection(self, vote_pct=.75, verbose=False):
        '''Determine clear sky periods based on irradiance measurements.  Central value of window
        will be labeled clear if vote_pct or more points are below the metric_tol threshold.

        Arguments
        ---------
        vote_pct: float
            percent of passes in order to grant pass/fail
        verbose: bool
            whether or not to return components used to determine is_clear

        Returns
        -------
        is_clear: pd.Series
            boolean time series of clear times
        components, optional: dict
            contains series of normalized lengths, local integrals, and calculated metric
        '''
        is_clear = pd.Series(False, self.data.index)

        slices = self.generate_window_slices()

        components = self.calc_components(slices=slices)

        midpoints = np.apply_along_axis(self.get_midval, 1, slices)
        pcts = np.apply_along_axis(self.calc_pct, 1, components['metric'].values[slices])

        is_clear.iloc[midpoints] = (pcts >= vote_pct) & (self.data[midpoints] > 0.0)

        if verbose:
            return is_clear, components
        else:
            return is_clear

    def calc_components(self, slices=None):
        '''Calculate normalized distances and integrals of moving window.  Values
        are reported at the central index of the window.

        Arguments
        ---------
        slices: np.ndarray
            slices for windows

        Returns
        -------
        result: dict
            time series data frame with distances and integrals
        '''
        if slices is None:
            slices = self.generate_window_slices()

        midpoints = np.apply_along_axis(self.get_midval, 1, slices)
        distances = np.apply_along_axis(self.calc_window_line_length_norm, 1, self.data.values[slices])
        integrals = np.apply_along_axis(self.calc_window_integral, 1, self.data.values[slices])

        metric = pd.Series(np.nan, index=self.data.index, name='metric')
        metric.iloc[midpoints] = self.calc_cloudiness_metric(distances, integrals)

        local_distances = pd.Series(np.nan, index=self.data.index, name='local_distances')
        local_distances.iloc[midpoints] = distances[:]

        local_integrals = pd.Series(np.nan, index=self.data.index, name='local_integrals')
        local_integrals.iloc[midpoints] = integrals[:]

        result = pd.concat([local_distances, local_integrals, metric], axis=1)

        return result


if __name__ == '__main__':
    main()


