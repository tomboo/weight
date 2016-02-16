# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 22:10:18 2016

@author: Tom
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DATAFILE = 'data.csv'
WEIGHT = 'Weight'
AVGWEIGHT = 'Avg. Weight'
TREND = 'Trend'
DELTA = 'Delta'
SPAN = 20


def read():
    df = pd.read_csv(DATAFILE, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    df = df.asfreq('D')             # fill missing values
    df.interpolate(inplace=True)    # linear interpolation

    # Exponentially weighted moving average (ewma)
    df[AVGWEIGHT] = pd.ewma(df[WEIGHT], span=SPAN)

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


def plot(df, title=''):
    df = df.copy()
    dt = df.index.to_pydatetime()
    x = mdates.date2num(dt)
    z = np.polyfit(x, df[AVGWEIGHT], 1)
    p = np.poly1d(z)
    df[TREND] = p(x)

    df.plot(grid=True)
    plt.title(title)
    plt.xlabel('')
    plt.show()

    # Stats
    print('Slope: {:.4f}'.format(z[0]))
    print('Rate (per week): {:.2f}'.format(z[0] * 7))
    print('Rate (per month): {:.2f}'.format(z[0] * 30))
    print('Rate (per year): {:.2f}'.format(z[0] * 365))


def plot_all(df):
    plot(df, title='Max')
    plot(df['2015'], title='Last Year')
    plot(df['2016-01'], title='Last Month')
    plot(df['2016'], title='This Year')
    plot(df['2016-02'], title='This Month')

    date_max = df.index.max()
    plot(df[date_max - pd.DateOffset(weeks=1):], title='Previous Week')
    plot(df[date_max - pd.DateOffset(months=1):], title='Previous Month')
    # plot(df[date_max - 3 * pd.DateOffset(months=1):], title='Previous Quarter')
    plot(df[date_max - pd.DateOffset(years=1):], title='Previous Year')


def main():
    df = read()

    # compute_stats(df)
    plot_all(df)
    # plot(df['2016-02'], title='This Month')     # this month

if __name__ == '__main__':
    main()
