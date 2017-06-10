
import os
import warnings

import pandas as pd
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


def main():
    pass


class DetectionTemplate(object):
    '''This class provides a template for implementing clearsky detection strategies.
    This assumes that at least one time series data object will be stored (data) and
    that a moving window approach will be used (and stores window size).  Detection
    strategies and algorithms should not be implemented here.  Methods here should be
    useful across different strategies.  Examples are calculating the line length of
    a window, calculating the average derivative, and so on.  These methods are appropriate
    to be implemented here because they are applicable to multiple strategies and algorithms.
    '''

    def __init__(self, data, window=30, copy=True):
        '''Initialize class with data.  Data may be copied (it can be modified during detection).
        Window size is also set here for calculating properties of measured curve.

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
        if len(pd.unique(data.index.to_series().diff().dropna())) != 1:
            raise NotImplementedError('You must use evenly spaced time series data.')
        if copy:
            self.data = data.copy()
        else:
            self.data = data
        self.window = window
        self.__data_filtered = False

    def generate_window_slices(self):
        '''Generate arrays for slicing data into windows.

        Arguments
        ---------
        None

        Returns
        -------
        slices: np.ndarray
            hankel matrix for slicing data into windows
        '''
        slices = la.hankel(np.arange(0, len(self.data) - self.window + 1),
                           np.arange(len(self.data) - self.window, len(self.data)))
        return slices

    def calc_window_integral(self, array):
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

    def calc_window_line_length_norm(self, array):
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
        line_length = self.calc_window_line_length(array)
        endpoint_line_length  = np.sqrt(np.square(array[-1] - array[0]) + np.square(len(array) - 1))
        line_length_norm = line_length / endpoint_line_length
        return line_length_norm

    def calc_window_line_length(self, array):
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

    def get_midval(self, array):
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
        metric = np.log(distances) / np.log(integrals)
        return metric

    def calc_pct(self, array):
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
        pct = np.sum((array <= self.metric_tol).astype(int)) / len(array)
        return pct

    def calc_window_avg(self, array):
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

    def calc_window_max(self, array):
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

    def calc_window_derivative_std_normed(self, array):
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
        window_mean = self.calc_window_avg(array)
        dy = np.diff(array) # diff as that's what Reno-Hansen indicate
        std_dy = np.std(dy, ddof=1) # dx always 1
        return std_dy / window_mean

    def calc_window_max_diff(self, arr1, arr2):
        '''Calculate the maximum difference between measured and modeled
        irradiance slopes for a given time window.

        Arugments
        ---------
        arr1: np.array
            measured irradiance values
        arr2: mp.array
            modeled irradiance values

        Returns
        -------
        max_abs_diff: float
            maximum absolute difference between two arrays
        '''
        meas_diff = np.diff(arr1)
        model_diff = np.diff(arr2)
        max_abs_diff = np.max(np.abs(meas_diff - model_diff))
        return max_abs_diff

    def by_time_of_day_transform(self):
        '''Transforms a pandas time series (spanning several days) into a data frame
        with each row being time of a day and each column as a date.

        Arguments
        ---------
        None

        Returns
        -------
        by_time: pd.DataFrame
            data frame with time of day as rows and dates as columns
        '''
        day_list = []
        for day, group in self.data.groupby(self.data.index.date):
            ser = pd.Series(group.values, index=group.index.time, name=day)
            day_list.append(ser)
        by_time = pd.concat(day_list, axis=1)
        return by_time

    def deviation_time_filter(self, central_tendency_fxn=np.mean, mode='relative', quant=.9,
                              dev_range=(.8, np.inf), replace_val=(-np.inf, np.inf), verbose=False, viz=False):
        '''Filter measurements by time of day based on deviation from fxn.  Likely candidates
        for fxn would be np.mean or np.nanmedian.  The deviation can be relative or direct.

        Note: using np.median will give missing values for nan periods.  use np.nanmedian if this is a concern

        Arguments
        ---------
        central_tendency_fxn, optional: callable
            function from which deviation will be measured (usually mean or median), default=np.mean
        mode, optional: str
            deviation mode [relative, direct, zscore]
            relative devitation deviation relative to the central_tendency_fxn result at that point
            direct devitaion compares deviation by direct value comparison at each time
            zscore filters points based off of zscore at a given time
        dev_range, optional: tuple(float, float)
            range outside which values will be filtered -  must consider mode
            and central_tendency_fxn when setting parameters
        replace_val, optional: tuple(float, float)
            values to set data to if outside the dev_range
        verbose, optional: bool
            return components of calculation
        viz, optional: bool
            produce visualization with outliers marked

        Returns
        -------
        components, optional: pd.DataFrame
            components used for calculating removal of points
        '''
        if len(pd.unique(self.data.index.date)) > 31:
            warnings.warn('Using more than one month of data may give suspect results.', RuntimeWarning)
        if len(pd.unique(self.data.index.date)) < 3:
            warnings.warn('Using less than three days of data may give suspect results.', RuntimeWarning)
        if mode not in ('relative', 'direct', 'zscore'):
            raise ValueError('Unrecognized mode {}.  Select either relative, direct, or zscore'.format(mode))
        if central_tendency_fxn not in (np.mean, np.nanmedian):
            warnings.warn('Using an untested central_tendency_fxn.', RuntimeWarning)
        if self.__data_filtered:
            warnings.warn('You are performance deviance filtering again.', RuntimeWarning)


        by_time = self.by_time_of_day_transform()
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

        mask_list = []
        for day, group in self.data.groupby(self.data.index.date):
            mask_vals = pd.Series(((group.values < lower_lim.values) | (group.values > upper_lim.values)),
                                  name=day, index=group.index.time)
            mask_list.append(mask_vals.copy())
            group[group < lower_lim.values] = replace_val[0]
            group[group > upper_lim.values] = replace_val[1]
            self.data[group.index] = group

        self.__data_filtered = True

        if viz or verbose:
            mask = pd.concat(mask_list, axis=1)

        if viz:
            self.__deviation_filter_viz(by_time, central_vals, mask, lower_lim, upper_lim)

        if verbose:
            components = {'by_time': by_time,
                          'mask': mask,
                          'deviation_vals': deviation_vals,
                          'central_vals': central_vals}
            return components

    def __deviation_filter_viz(self, by_time, central_vals, mask, lower_vals, upper_vals):
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

        Returns
        -------
        None
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


