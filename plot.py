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
RUNS = 'Run Length'
STREAK = 'Streak'

SPAN = 20


def read():
    df = pd.read_csv(DATAFILE, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    df = df.asfreq('D')             # fill missing values
    df.interpolate(inplace=True)    # linear interpolation

    # Exponentially weighted moving average (ewma)
    df[AVGWEIGHT] = pd.ewma(df[WEIGHT], span=SPAN)

    return df


def runlengths(df):
    '''
    Returns a list of run lengths.
    '''
    
    # print('\n', 'runlengths:')

    list = []
    row_count = len(df.index)
    if (row_count == 0):
        return list         # empty dataframe

    res_previous = 0        # result (or outcome) for previous row
    run_previous = 0        # run length for previous row
    row_num = 0

    for index, row in df.iterrows():
        res_current = -1 if row[WEIGHT] < row[AVGWEIGHT] else 1

        if row_num == 0:    # first row
            run_current = 1
        else:
            if res_current == res_previous:
                run_current += 1
                list.append(run_previous * res_previous)
            else:           # crossover
                run_current = 1
                list.append(run_previous * res_previous)

        if row_num == row_count - 1:    # last row
            list.append(run_current * res_current)

        # print(row_num, res_current, run_current)

        res_previous = res_current
        run_previous = run_current
        row_num += 1

    return list


def compute_stats(df):
    print(df.tail())
    print(df.describe())

    df[DELTA] = df[WEIGHT] - df[AVGWEIGHT]

    prev = df[DELTA].shift(1)
    print(prev.head())
    print(prev.tail())
    # crossover = prev == np.nan or (df[DELTA] >= 0 and prev <= 0) or (df[DELTA] <= 0 and prev >= 0)

    print(df.tail(10))


def plot(df, title=''):
    df = df.copy()

    # Compute trend
    dt = df.index.to_pydatetime()
    x = mdates.date2num(dt)
    z = np.polyfit(x, df[AVGWEIGHT], 1)
    p = np.poly1d(z)
    df[TREND] = p(x)

    # TODO: Plot selected columns
    df.plot(grid=True)
    plt.title(title)
    plt.xlabel('')
    plt.show()

    # Today
    df[DELTA] = df[WEIGHT] - df[AVGWEIGHT]
    df[RUNS] = runlengths(df)

    list = []
    for index, row in df.iterrows():
        wl = 'W' if row[RUNS] < 0 else 'L'
        list.append(wl + str(int(np.abs(row[RUNS]))))
    df[STREAK] = list

    print(df.tail(10))

    # Stats
    print()
    print('Slope: {:.4f}'.format(z[0]))
    print('Weight Change Rate (per week): {:.2f}'.format(z[0] * 7))
    print('Weight Change Rate (per month): {:.2f}'.format(z[0] * 30))
    print('Weight Change Rate (per year): {:.2f}'.format(z[0] * 365))
    print('\n', df.describe())


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
    # plot_all(df)
    # plot(df['2016-02'], title='This Month')     # this month
    plot(df[df.index.max() - pd.DateOffset(months=1):], title='Previous Month')

if __name__ == '__main__':
    main()
