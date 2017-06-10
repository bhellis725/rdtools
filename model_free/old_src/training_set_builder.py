import os
import clearsky_detect_model_free
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_snl():
    filename = os.path.expanduser('~/data_sets/snl_raw_data/1429_1405/raw_1405_weather_for_1429.csv')
    cols = ['Global_Wm2', 'Date-Time']
    data = pd.read_csv(filename, parse_dates=['Date-Time'], usecols=cols, index_col=['Date-Time'])
    data.index = data.index.tz_localize('Etc/GMT+7')
    data = data.reindex(pd.date_range(start=data.index[0], end=data.index[-1], freq='1min')).fillna(0)
    data = pd.Series(data['Global_Wm2'], index=data.index)
    data[data < 50] = 0
    return data


def main():
    snl_data = load_snl()
    sample = snl_data[(snl_data.index >= '2016-07-01') & (snl_data.index < '2016-07-15')]
    is_clear = clearsky_detect_model_free.model_free_detect_meanval(sample)

    picker = PointPicker(sample)
    picker.plot(is_clear)

    correct_points = picker.indices
    for p in correct_points:
        is_clear[p] = False

    fig, ax = plt.subplots()
    ax.plot(sample.index, sample)
    ax.scatter(is_clear[is_clear].index, sample[is_clear], edgecolor='green', facecolor='none')
    plt.show()


class PointPicker(object):
    '''
    https://stackoverflow.com/questions/22052532/matplotlib-python-clickable-points
    '''

    def __init__(self, data):
        self.data = data.copy()
        self.indices = []
        self.last_clicks = []
        self.xy = []

    def on_pick(self, event):
        artist = event.artist
        x, y = artist.get_xdata(), artist.get_ydata()
        ind = event.ind
        print('Data point ', len(self.last_clicks) + 1, ': ', self.data.iloc[[x[ind[0]]]])
        self.last_clicks.append(self.data.iloc[[x[ind[0]]]].index)

        self.xy.append([x[ind[0]], y[ind[0]]])
        self.points.set_offsets(self.xy)
        self.ax.draw_artist(self.points)
        self.fig.canvas.blit(self.ax.bbox)
        if len(self.last_clicks) == 2:
            start = self.last_clicks[0]
            while start <= self.last_clicks[-1]:
                self.indices.append(start)
                start += pd.Timedelta('1min')
            self.last_clicks = []
            print('===' * 20)

    def plot(self, is_clear=None):
        self.fig, self.ax = plt.subplots()
        if is_clear is not None:
            x = np.where(is_clear)
            y = self.data[is_clear]
            self.ax.scatter(x, y, edgecolor='green', facecolor='none')
        self.ax.plot(np.arange(len(self.data)), self.data.values, picker=10)
        ticks = list(pd.date_range(start=self.data.index[0],
                     end=self.data.index[-1] + pd.Timedelta('1min'), freq='2H'))
        ticks = [str(x)[:-12] for x in ticks]
        tickvals = list(np.arange(0, len(self.data), 120))
        self.points = self.ax.scatter([], [], color='red', picker=20)
        self.ax.set_xticks(tickvals)
        self.ax.set_xticklabels(ticks, rotation=45, fontsize='small')
        self.fig.canvas.callbacks.connect('pick_event', self.on_pick)
        plt.show()


if __name__ == '__main__':
    main()
