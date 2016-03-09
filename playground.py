# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# import datetime

DATAFILE = 'data.csv'
WEIGHT = 'Weight'
AVGWEIGHT = 'Avg. Weight'


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


def main():
    df = pd.read_csv(DATAFILE, index_col=0, parse_dates=True)
    # Exponentially weighted moving average (ewma)
    df[AVGWEIGHT] = pd.ewma(df[WEIGHT], span=20)
    print(df.tail())
    print(df.describe())
    plot(df)                # max
    plot(df['2015'])        # last year
    plot(df['2016'])        # this year
    plot(df['2016-01'])     # last month
    plot(df['2016-02'])     # this month

if __name__ == '__main__':
    main()
