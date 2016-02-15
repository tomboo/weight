# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATAFILE = 'data.csv'
WEIGHT = 'Weight'
AVGWEIGHT = 'Avg. Weight'
DELTA = 'Delta'


def read():
    df = pd.read_csv(DATAFILE, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    df = df.asfreq('D')             # fill missing values
    df.interpolate(inplace=True)    # linear interpolation

    # Exponentially weighted moving average (ewma)
    df[AVGWEIGHT] = pd.ewma(df[WEIGHT], span=20)

    return df


def compute_stats(df):
    print(df.tail())
    print(df.describe())

    df[DELTA] = df[WEIGHT] - df[AVGWEIGHT]

    prev = df[DELTA].shift(1)
    print(prev.head())
    print(prev.tail())
    # crossover = prev == np.nan or (df[DELTA] >= 0 and prev <= 0) or (df[DELTA] <= 0 and prev >= 0)

    print(df.tail(20))


def plot(df):
    dt = df.index.to_pydatetime()
    x = mdates.date2num(dt)
    fit = np.polyfit(x, df[AVGWEIGHT], 1)
    fit_fn = np.poly1d(fit)
    # fit_fn is now a function which takes in x and returns as estimate for y

    xx = [x[0], x[-1]]

    df.plot(grid=True)
    ax = plt.gca()
    ax.plot(xx, fit_fn(xx))

    plt.plot(xx, fit_fn(xx))
    plt.show()


def plot_all(df):
    plot(df)                # max
    plot(df['2015'])        # last year
    plot(df['2016-01'])     # last month
    plot(df['2016'])        # this year
    plot(df['2016-02'])     # this month

    date_max = df.index.max()
    plot(df[date_max - pd.DateOffset(weeks=1):])        # previous week
    plot(df[date_max - pd.DateOffset(months=1):])       # previous month
    plot(df[date_max - 3 * pd.DateOffset(months=1):])   # previous quarter
    plot(df[date_max - pd.DateOffset(years=1):])        # previous year


def main():
    df = read()

    # compute_stats(df)
    # plot_all(df)
    plot(df['2016-02'])     # this month

if __name__ == '__main__':
    main()
