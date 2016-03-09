# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as ss
import datetime as dt
import pandas as pd
# import matplotlib.pylab as plb

DATAFILE = 'data.csv'
WEIGHT = 'Weight'
AVGWEIGHT = 'Avg. Weight'
DELTA = 'Delta'


def section_1_1():
    '''
    Making and Plotting Some Data.
    '''

    # Plotting a Sine Curve
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.show()

    # Making a Histogram with Normal Data
    g = np.random.randn(1000)
    plt.hist(g, bins=10)
    plt.show()


def section_1_2():
    '''
    Some Basic Statistics with Python.
    '''

    # Normal distributions and the CDF
    def cdf(x):
        '''
        Cumulative distribution function (cdf)

        The cdf will give us the area under our normal curve
        to the left of whatever point we put in
        '''

        # When we say loc = 18, scale = 0.09 this is giving us
        # the mean and standard deviation, respectively,
        # for the normal distribution we're creating
        return ss.norm.cdf(x, loc=18, scale=0.09)

    print(cdf(17.91))   # 15.87%

    # Standardized Scores
    def standardize(x, mean, stdev):
        return (x - mean) / stdev

    print(standardize(60, 40, 10))


def polyfit():
    x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
    y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    z = np.polyfit(x, y, 1)
    print(y)
    print(z)

    # It is convenient to use poly1d objects for dealing with polynomials
    p = np.poly1d(z)
    print(p(0.5))

    p3 = np.poly1d(np.polyfit(x, y, 3))
    p30 = np.poly1d(np.polyfit(x, y, 30))
    xp = np.linspace(-2, 6, 100)
    plt.plot(x, y, '.', xp, p3(xp), '-', xp, p30(xp), '--')
    plt.ylim(-2, 2)
    plt.show()


def plot_timeseries():
    df = pd.DataFrame(columns=('Time', 'Sales'))
    start_date = dt.datetime(2015, 7, 1)
    end_date = dt.datetime(2015, 7, 10)
    daterange = pd.date_range(start_date, end_date)
    for single_date in daterange:
        row = dict(zip(['Time', 'Sales'],
                       [single_date, int(50 * np.random.rand(1))]))
        row_s = pd.Series(row)
        row_s.name = single_date.strftime('%b %d')
        df = df.append(row_s)
    print(df)
    df.ix['Jul 01':'Jul 10', ['Time', 'Sales']].plot()
    z = np.polyfit(range(0, 10), df.as_matrix(['Sales']).flatten(), 1)
    p = np.poly1d(z)
    print('\n', df.as_matrix(['Sales']))
    print('\n', p(df.as_matrix(['Sales'])))
    plt.plot(df.as_matrix(['Sales']), p(df.as_matrix(['Sales'])), 'm-')
    plt.ylim(0, 50)
    plt.xlabel('Sales Date')
    plt.ylabel('Sale Value')
    plt.title('Plotting Time')
    plt.legend(['Sales', 'Trend'])
    plt.show()


def plot_trendline():
    df = pd.read_csv(DATAFILE, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    df = df.asfreq('D')             # fill missing values
    df.interpolate(inplace=True)    # linear interpolation
    df[AVGWEIGHT] = pd.ewma(df[WEIGHT], span=20)
    df = df['2016-02']
    print(df)

    dt = df.index.to_pydatetime()
    x = mdates.date2num(dt)
    z = np.polyfit(x, df[AVGWEIGHT], 1)
    p = np.poly1d(z)
    df['Trend'] = p(x)
    df.plot()


if __name__ == '__main__':
    plot_trendline()
