
import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg
from scipy import interpolate


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

    def generate_window_slices(self, arr):
        '''Generate arrays for slicing data into windows.

        Arguments
        ---------
        arr: np.array

        Returns
        -------
        slices: np.ndarray
            hankel matrix for slicing data into windows
        '''
        slices = linalg.hankel(np.arange(0, len(arr) - self.window + 1),
                               np.arange(len(arr) - self.window, len(arr)))
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

    def calc_window_diff_coeff_variation(self, array):
        '''Calculate coefficient of variation of differences for a given window.
        This is meant to measure the noise/smoothness of the window.

        cv = stdev / |mean|

        Arguments
        ---------
        array: np.array

        Returns
        -------
        cv: float
            coeff of variation for array
        '''
        y_diff = np.diff(array)
        cv = np.std(y_diff) / np.abs(np.mean(array))
        return cv

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

    def calc_window_avg(self, array, weights=None):
        '''Calculate average value of array.

        Arugments
        ---------
        array: np.array

        Returns
        -------
        val: float
            mean value of array
        '''
        if weights == 'gaussian':
            center = len(array) // 2
            weights = np.asarray([np.exp(-(i - center)**2 / (2 * 1**2)) for i in range(len(array))])
        else:
            weights = np.ones(len(array))

        return np.average(array, weights=weights)

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

    def calc_window_derivative_avg(self, array):
        '''Calculate average derivative of array.

        Arguments
        ---------
        array: np.array

        Returns
        -------
        val: float
            average rate of change
        '''
        val = np.mean(np.gradient(array))
        return val

    def calc_window_derivative_std(self, array):
        '''Calculate average derivative of array.

        Arguments
        ---------
        array: np.array

        Returns
        -------
        val: float
            average rate of change
        '''
        val = np.std(np.diff(array))
        return val

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

    def by_time_of_day_transform(self, specific_days=None):
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
        if specific_days is None:
            for day, group in self.data.groupby(self.data.index.date):
                ser = pd.Series(group.values, index=group.index.time, name=day)
                day_list.append(ser)
        else:
            for day in specific_days:
                group = self.data[self.data.index.date == day]
                ser = pd.Series(group.values, index=group.index.time, name=day)
                day_list.append(ser)
        by_time = pd.concat(day_list, axis=1)
        return by_time


    def deviation_time_filter(self, window_size=30, central_tendency_fxn=np.nanmean, mode='relative', percentile=90,
                               dev_range=(.8, np.inf), replace_val=(np.nan, np.nan), verbose=False, inplace=True):
        '''Filter measurements by time of day based on deviation from fxn.  Likely candidates
        for fxn would be np.mean or np.nanmedian.  The deviation can be relative or direct.

        Note: using np.median will give missing values for nan periods.  use np.nanmedian if this is a concern

        Arguments
        ---------
        window_size, optional: int
            size of window (in days) for rejecting points
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
        inplace, optional: bool
            alter data internally

        Returns
        -------
        filtered_days, optional: pd.Series
            time series with filtered values replaced
        components, optional: dict
            contains various components used in calculation
            components used for calculating removal of points
                by_time: time series transformed to DataFrame with columns as dates and rows as times
                mask: bool values of filtered/not filtered data points organized like by_time
                deviation_vals: deviation of each point based on metric chosen organized like by_time
                central_vals: value as calculated by time based on central_tendency_fxn
        '''
        filtered_days = []
        components_list = []

        for day_to_filter, sample in self.generate_day_range(window_size=window_size):
            if verbose:
                filtered_day, components = self.deviation_by_time_per_sample(day_to_filter, sample,
                                                                             model_fxn=central_tendency_fxn,
                                                                             mode=mode, percentile=percentile,
                                                                             dev_range=dev_range, verbose=verbose)
                components_list.append(components)
                filtered_days.append(filtered_day)
            else:
                filtered_day = self.deviation_by_time_per_sample(day_to_filter, sample,
                                                                 model_fxn=central_tendency_fxn,
                                                                 mode=mode, percentile=percentile, dev_range=dev_range)
                filtered_days.append(filtered_day)
        filtered_days = pd.concat(filtered_days, axis=0)

        if verbose:
            components = {}
            components['by_time'] = pd.concat([i['by_time'] for i in components_list], axis=1)
            components['by_time'] = components['by_time'].loc[:,~components['by_time'].columns.duplicated()]
            components['mask'] = pd.concat([i['mask'] for i in components_list], axis=1)
            components['deviation_vals'] = pd.concat([i['deviation_vals'] for i in components_list], axis=1)
            components['central_vals'] = pd.concat([i['central_vals'] for i in components_list], axis=1)

        if inplace:
            self.__data_filtered = True
            self.data = filtered_days
            if verbose:
                return components
        else:
            if verbose:
                return filtered_days, components
            return filtered_days


    def deviation_by_time_per_sample(self, day_to_filter, sample_days, model_fxn=np.nanmean,
                                     mode='relative', percentile=90, dev_range=(.8, np.inf),
                                     replace_val=(-np.inf, np.inf), verbose=False):
        '''Filter measurements by time of day based on deviation from fxn.  Likely candidates
        for fxn would be np.mean or np.nanmedian.  The deviation can be relative or direct.

        Note: using np.median will give missing values for nan periods.  use np.nanmedian if this is a concern

        Arguments
        ---------
        day_to_filter: datetime.date
            day which will be filtered
        sample_days: list-like (of dates)
            dates to apply model_fxn to
        model_fxn, optional: callable
            function from which deviation will be measured (usually mean or median), default=np.mean
        mode, optional: str
            deviation mode [relative, direct, zscore]
            relative devitation deviation relative to the central_tendency_fxn result at that point
            direct devitaion compares deviation by direct value comparison at each time
            zscore filters points based off of zscore at a given time
                choosing this will set central_tendency_fxn to be np.mean
        percentile, optional: float
            percentile value if percentile model_fxn chosen, ignored otherwise
        dev_range, optional: tuple(float, float)
            range outside which values will be filtered -  must consider mode
            and central_tendency_fxn when setting parameters
        replace_val, optional: tuple(float, float)
            values to set data to if outside the dev_range
        verbose, optional: bool
            return components of calculation

        Returns
        -------
        group: pd.Series
            filtered day_to_filter values
        components, optional: dict
            components used for calculating removal of points
                by_time: time series transformed to DataFrame with columns as dates and rows as times
                mask: bool values of filtered/not filtered data points organized like by_time
                deviation_vals: deviation of each point based on metric chosen organized like by_time
                central_vals: value as calculated by time based on central_tendency_fxn
        '''
        by_time = self.by_time_of_day_transform(specific_days=sample_days)
        ser_filter = by_time[day_to_filter]

        if model_fxn in (np.percentile, np.nanpercentile):
            args = ([percentile])
        else:
            args = ()

        if mode in ('relative', 'direct'):
            central_vals = pd.Series(by_time.replace(0, np.nan).apply(model_fxn, axis=1, args=args),
                                     name='central')
            deviation_vals = ser_filter.subtract(central_vals.values, axis=0)
            deviation_vals.rename('deviation')
            if mode == 'relative':
                deviation_vals = deviation_vals.divide(central_vals)
                lower_lim = pd.Series(dev_range[0] * central_vals, index=deviation_vals.index)
                upper_lim = pd.Series(dev_range[1] * central_vals, index=deviation_vals.index)
            else:
                lower_lim = pd.Series(np.maximum(central_vals + dev_range[0], 0), index=deviation_vals.index)
                upper_lim = pd.Series(central_vals + dev_range[1], index=deviation_vals.index)
        else:
            central_vals = pd.Series(by_time.replace(0, np.nan).apply(np.nanmean, axis=1), name='central')
            deviation_vals = by_time.subtract(central_vals.values)
            std_dev_vals = pd.Series(ser_filter.replace(0, np.nan).std(), name='deviation')
            deviation_vals = ser_filter.subtract(central_vals.values).divide(std_dev_vals)
            deviation_vals.rename('deviation')
            lower_lim = central_vals + (dev_range[0] * std_dev_vals)
            upper_lim = central_vals + (dev_range[1] * std_dev_vals)

        mask = pd.Series(((ser_filter.values < lower_lim.values) | (ser_filter.values > upper_lim.values)),
                          name=str(day_to_filter), index=ser_filter.index)
        group = self.data[self.data.index.date == day_to_filter]
        group[group < lower_lim.values] = replace_val[0]
        group[group > upper_lim.values] = replace_val[1]

        if verbose:
            components = {'by_time': by_time,
                          'mask': mask,
                          'deviation_vals': deviation_vals,
                          'central_vals': central_vals}
            return group, components
        return group

    def generate_day_range(self, window_size=30):
        '''Generates groups of days for statistical analysis.

        Arguments
        ---------
        window_size, optional: int
            size of window (in days) for rejecting points (will be +/- (window_size / 2))

        Returns
        -------
        None

        Yields
        ------
        day_range: tuple
            (day of interest, date of days +/- window_size / 2)
        '''
        if window_size > 31:
            warnings.warn('Using a large window of days may give suspect results.', RuntimeWarning)
        if window_size < 3:
            warnings.warn('Using a very small window of days give suspect results.', RuntimeWarning)

        days = pd.unique(self.data.index.date)
        if len(days) <= window_size:
            warnings.warn('Data is smaller than specified window size.', RuntimeWarning)
            for i in range(len(days)):
                yield days[i], days
        else:
            plus_minus = (window_size // 2) + 1
            for i in range(len(days)):
                if i - plus_minus < 0:
                    day_range = days[:window_size]
                elif i + plus_minus >= len(days):
                    day_range = days[len(days) - window_size: len(days)]
                else:
                    day_range = days[i - plus_minus + 1: i + plus_minus]
                yield days[i], day_range

    def __deviation_filter_viz(self, day, by_time, central_vals, mask, lower_vals, upper_vals):
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
        day_of_interest = by_time[day]
        ax.scatter(day_of_interest[mask].index, day_of_interest[mask],
                   edgecolor='black', facecolor='none', alpha=.25, zorder=100)
        plt.show()

    def calc_property(self, data, fxn, slices=None):
        '''Calculate properties of windows.  The acceptable functions for calculating
        properties are
            self.calc_window_max, self.calc_window_line_length_norm,
            self.calc_window_integral, self.calc_window_line_length,
            self.calc_window_avg, self.calc_window_derivative_std_normed,
            self.calc_window_derivative_avg, self.calc_window_derivative_std,
            self.calc_window_diff_coeff_variation

        Arguments
        ---------
        data: pd.Series
            time series data to calculate properties of
        fxn: callable
            listed in choices above
        slices, optional: np.array
            slices for windows - will be generated based on self.window if None

        Returns
        -------
        ser_vals: pd.Series
            time series data of calculated property - the window value is centered at the central
            index of the window
        '''
        if fxn not in (self.calc_window_max, self.calc_window_line_length_norm,
                       self.calc_window_integral, self.calc_window_line_length,
                       self.calc_window_avg, self.calc_window_derivative_std_normed,
                       self.calc_window_derivative_avg, self.calc_window_derivative_std,
                       self.calc_window_diff_coeff_variation):
            raise ValueError('You have chosen an invalid function.')
        if slices is None:
            slices = self.generate_window_slices(self.data)

        midpoints = np.apply_along_axis(self.get_midval, 1, slices)
        vals = np.apply_along_axis(fxn, 1, data.values[slices])

        ser_vals = pd.Series(np.nan, index=data.index)
        ser_vals.iloc[midpoints] = vals[:]

        return ser_vals

    def calc_spline(self, series, spline_window=None, kind=2):
        '''Calculate smoothing spline for a given data set.  Data will be broken up
        by day and resampled based on spline_window or on self.window.

        Arguments
        ---------
        series: pd.Series
            time series data
        spline_window, optional: int
            length of time in between points with which splines are calculated
            if None, will use self.window as size.
        kind, optional: int
            order of spline

        Returns
        -------
        spline: pd.Series
            smoothed data from series with same indices as series
        '''
        if spline_window is None:
            spline_window = self.window
        spline_list = []
        # def central_val(array_like):
        #     return array_like[len(array_like) // 2]
        for day_str, day_group in series.groupby(series.index.date):
            # resampled = day_group.replace([-np.inf, np.inf], np.nan).dropna().\
            #             resample(str(spline_window) + 'T').median()
            # resampled = day_group.replace([-np.inf, np.inf], np.nan).dropna().\
            #             resample(str(spline_window) + 'T').apply(central_val)
            resampled = day_group.replace([-np.inf, np.inf], np.nan).dropna().\
                        resample(str(spline_window) + 'T').mean()
            xs = np.arange(0, len(resampled))
            ys = resampled.values
            spline = interpolate.interp1d(xs, ys, kind=kind)
            xnew = np.linspace(xs[0], xs[-1], len(day_group))
            y_pred = spline(xnew)
            spline_ser = pd.Series(y_pred)
            spline_ser.index = day_group.index
            spline_ser.replace(np.nan, 0, inplace=True)
            spline_list.append(spline_ser.copy())
        spline = pd.concat(spline_list)
        return spline

    def calc_window_line_length_norm_spline(self, series, slices=None, spline_window=None):
        '''Calculate the normalized line length of windows.  Series window line lengths will be
        normalized by the line length of a smoothing spline over that time.

        Arguments
        ---------
        series: pd.Series
            time series data
        slices, optional: np.array
            slices for windows - will be generated based on self.window if None
        spline_window, optional: int
            size of step to take in between points used to calculate splines

        Returns
        -------
        ser_norm_line_length: pd.Series
            time series of normalized line lenghts for each window - window is labeled at central index
        '''
        if slices is None:
            slices = self.generate_window_slices(self.data)

        spline_ser = self.calc_spline(series, spline_window=spline_window)

        midpoints = np.apply_along_axis(self.get_midval, 1, slices)
        ser_norm_line_length = pd.Series(np.nan, index=series.index)

        for mid, s in zip(midpoints, slices):
            ser_line_length = self.calc_window_line_length(series.values[s])
            spline_line_length = self.calc_window_line_length(spline_ser.values[s])
            ser_norm_line_length.iloc[mid] = ser_line_length / spline_line_length

        return ser_norm_line_length

    def calc_window_spline_rmse(self, series, slices=None):
        if slices is None:
            slices = self.generate_window_slices(self.data)

        spline_ser = self.calc_spline(series)

        midpoints = np.apply_along_axis(self.get_midval, 1, slices)
        ser_rmse = pd.Series(np.nan, index=series.index)

        for mid, s in zip(midpoints, slices):
            rmse = np.sqrt(np.mean(np.square(series.values[s] - spline_ser.values[s])))
            ser_rmse[mid] = rmse

        return ser_rmse





if __name__ == '__main__':
    main()


