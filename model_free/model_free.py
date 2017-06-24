
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
    data = data[(data.index >= '2016-07-01') & (data.index < '2016-07-15')]
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

        _ = ModelFreeDetect(data)
        _ = mf.deviation_time_filter(central_tendency_fxn=np.percentile)
        print('filtered percentile')

        _ = mf.standard_detection()
        print('standard filtered percentile')



class ModelFreeDetect(DetectionTemplate):
    '''Implementation of model-free clear sky detection algorithms.  Inhereits from
    clearsky_detect_template.ClearskyDetection which implements the moving window
    properties calculations used here.
    '''

    def __init__(self, data, window=30, metric_tol=.01, copy=True):
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

    def standard_detection(self, verbose=False, splines=False, spline_window=None):
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

        # if splines:
        #     components = self.calc_components_spline(spline_window=spline_window)
        # else:
        #     components = self.calc_components()
        components = self.calc_components(splines=splines, spline_window=spline_window)

        is_clear = (components['metric'] <= self.metric_tol) & (self.data > 0.0)
        # is_clear = (components['metric'] <= self.metric_tol) & (self.data > 0.0)

        if verbose:
            return is_clear, components
        else:
            return is_clear

    def mean_detection(self, verbose=False, splines=False, spline_window=None):
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

        # if splines:
        #     components = self.calc_components_spline(spline_window=spline_window)
        # else:
        #     components = self.calc_components()
        components = self.calc_components(splines=splines, spline_window=spline_window)

        slices = self.generate_window_slices(self.data)

        components = self.calc_components(slices=slices)

        midpoints = np.apply_along_axis(self.get_midval, 1, slices)
        means = np.apply_along_axis(self.calc_window_avg, 1, components['metric'].values[slices], weights=None)
        # means = np.apply_along_axis(np.mean, 1, components['metric'].values[slices])

        is_clear.iloc[midpoints] = (means <= self.metric_tol) & (self.data[midpoints] > 0.0)

        if verbose:
            return is_clear, components
        else:
            return is_clear

    def democratic_detection(self, vote_pct=.75, splines=False, spline_window=None, verbose=False):
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

        # if splines:
        #     components = self.calc_components_spline(spline_window=spline_window)
        # else:
        #     components = self.calc_components()
        components = self.calc_components(splines=splines, spline_window=spline_window)

        slices = self.generate_window_slices(self.data)

        # components = self.calc_components(slices=slices)

        midpoints = np.apply_along_axis(self.get_midval, 1, slices)
        pcts = np.apply_along_axis(self.calc_pct, 1, components['metric'].values[slices])

        is_clear.iloc[midpoints] = (pcts >= vote_pct) & (self.data[midpoints] > 0.0)

        if verbose:
            return is_clear, components
        else:
            return is_clear

    def calc_components(self, splines=False, spline_window=None, slices=None):
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
            slices = self.generate_window_slices(self.data)

        # local_distances = self.calc_window_line_length_norm_spline(self.data, slices=slices)
        # local_distances.name = 'local_distances'
        # local_integrals = self.calc_property(self.data, self.calc_window_avg)
        # local_integrals.name = 'local_integrals'
        # integrals = np.apply_along_axis(self.calc_window_avg, 1, self.data.values[slices])
        # distances = np.apply_along_axis(self.calc_window_line_length_norm, 1, self.data.values[slices])
        # integrals = np.apply_along_axis(self.calc_window_integral, 1, self.data.values[slices])
        # integrals = np.apply_along_axis(self.calc_window_avg, 1, self.data.values[slices])
        # metric.iloc[midpoints] = self.calc_cloudiness_metric(local_distances.dropna().values,
        #                                                      local_integrals.dropna().values)
        # midpoints = np.apply_along_axis(self.get_midval, 1, slices)
        # local_distances = pd.Series(np.nan, index=self.data.index, name='local_distances')
        # local_distances.iloc[midpoints] = distances[:]
        # local_integrals = pd.Series(np.nan, index=self.data.index, name='local_integrals')
        # local_integrals.iloc[midpoints] = integrals[:]

        local_integrals = self.calc_property(self.data, self.calc_window_avg)
        local_integrals.name = 'local_integrals'
        if splines:
            local_distances = self.calc_window_line_length_norm_spline(self.data, slices=slices, spline_window=spline_window)
        else:
            local_distances = self.calc_property(self.data, self.calc_window_line_length_norm)
        local_distances.name = 'local_distances'

        metric = self.calc_cloudiness_metric(local_distances.values,
                                             local_integrals.values)
        metric = pd.Series(metric, index=self.data.index, name='metric')

        result = pd.concat([local_distances, local_integrals, metric], axis=1)

        return result

    # def calc_components_spline(self, slices=None, spline_window=None):
    #     '''Calculate normalized distances and integrals of moving window.  Values
    #     are reported at the central index of the window.

    #     Arguments
    #     ---------
    #     slices: np.ndarray
    #         slices for windows

    #     Returns
    #     -------
    #     result: dict
    #         time series data frame with distances and integrals
    #     '''
    #     if slices is None:
    #         slices = self.generate_window_slices(self.data)

    #     local_distances = self.calc_window_line_length_norm_spline(self.data, slices=slices, spline_window=spline_window)
    #     local_distances.name = 'local_distances'
    #     local_integrals = self.calc_property(self.data, self.calc_window_avg)
    #     local_integrals.name = 'local_integrals'

    #     metric = self.calc_cloudiness_metric(local_distances.values,
    #                                          local_integrals.values)
    #     metric = pd.Series(metric, index=self.data.index, name='metric')

    #     result = pd.concat([local_distances, local_integrals, metric], axis=1)

    #     return result

    def calc_cloudiness_metric(self, distances, integrals):
        '''Calculate the cloudiness metric.

        Cloudiness = log(distances) / log(integrals)

        Arguments
        ---------
        distances: np.array
            normalized distsances of windows
        integrals: np.array
            local integral of irradiance of a window

        Returns
        -------
        metric: np.array
            metric values

        '''
        #metric = np.log(distances) / np.log(integrals)
        # metric = np.log(distances) / np.log(integrals)
        metric = distances**2 / integrals
        return metric


if __name__ == '__main__':
    main()


