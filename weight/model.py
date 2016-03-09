# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 04:38:20 2016

@author: Tom
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# TODO: need a better way to reference datafiles
DATAFILE = '~/projects/weight/data/data.csv'
SPAN = 20

# Columns
DATE = 'Date'
WEIGHT = 'Weight'
AVERAGE = 'Average'     # exponentially weighted moving average
TREND = 'Trend'
DELTA = 'Delta'
RUN = 'Run'       # run length
STREAK = 'Streak'


class Model:
    def __init__(self, raw=False):
        self.df = self.read()
        if not raw:
            # TODO: Truncate table
            self.df = self.average(self.df)

    def read(self):
        df = pd.read_csv(DATAFILE, index_col=0, parse_dates=True)
        df.sort_index(inplace=True)
        print('read (rows = {})'.format(len(df)))
        return df

    def average(self, df):
        '''
        Compute exponentially weighted moving average.
        Missing values are filled in using linear interpolation.
        '''
        df = df.asfreq('D')             # fill missing values
        df.interpolate(inplace=True)    # linear interpolation

        # Exponentially weighted moving average (ewma)
        df[AVERAGE] = pd.ewma(df[WEIGHT], span=SPAN)
        print('average (rows = {})'.format(len(df)))
        return df

    def write(self):
        # TODO: Write raw file. Filter out computed rows and columns.
        assert AVERAGE not in self.df.columns

        self.df.to_csv(DATAFILE)
        print('write (rows = {})'.format(len(self.df)))

    def update(self, date, weight):
        '''
        Insert new row or update existing row.
        '''
        self.df.loc[date, WEIGHT] = weight
        self.df.sort_index(inplace=True)

    def enddate(self):
        return self.df.index.max()

    def select(self, startdate):
        '''
        Select a range of dates.
        Compute trend.
        Returns copy.
        '''
        df = self.df[startdate:].copy()
        df = self.trend(df)
        return df

    def trend(self, df):
        '''
        Compute trend
        '''
        dt = df.index.to_pydatetime()
        x = mdates.date2num(dt)
        z = np.polyfit(x, df[AVERAGE], 1)
        p = np.poly1d(z)

        print('slope: {:.4f}'.format(z[0]))
        print('rate:',
              '{:.2f} lbs/wk;'.format(z[0] * 7),
              '{:.2f} lbs/mo;'.format(z[0] * 30),
              '{:.2f} lbs/yr'.format(z[0] * 365))

        # Add trend column
        df[TREND] = p(x)
        return df

    def delta(self, df):
        df[DELTA] = df[WEIGHT] - df[AVERAGE]
        return df

    # TODO: Rewrite as a general purpose (class) method
    def runlength(self, df):
        '''
        Returns a list of run lengths.
        '''
        list = []
        row_count = len(df.index)
        if (row_count == 0):
            return list         # empty dataframe

        res_previous = 0        # result (or outcome) for previous row
        run_previous = 0        # run length for previous row
        row_num = 0

        for index, row in df.iterrows():
            res_current = -1 if row[WEIGHT] < row[AVERAGE] else 1

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

        df[RUN] = list
        return df

    def streak(self, df):
        '''
        Compute win-loss streaks
        '''
        list = []

        # TODO: This can probably be done without iterating
        for index, row in df.iterrows():
            wl = 'W' if row[RUN] < 0 else 'L'
            list.append(wl + str(int(np.abs(row[RUN]))))
        df[STREAK] = list
        return df

    def calculate(self, df):
        # df = self.average(df)
        # df = self.trend(df)
        df = self.delta(df)
        df = self.runlength(df)
        df = self.streak(df)
        return df

    def plot(self, startdate, title=''):
        df = self.select(startdate)
        cols = [WEIGHT, AVERAGE, TREND]
        df[cols].plot(grid=True)
        plt.title(title)
        plt.show()


def main():
    model = Model()

    enddate = model.enddate()
    prev_week = enddate - pd.DateOffset(weeks=1)
    prev_month = enddate - pd.DateOffset(months=1)
    prev_quarter = enddate - 3 * pd.DateOffset(months=1)
    prev_year = enddate - pd.DateOffset(years=1)

    # TODO:
    # last_month = '2016-02'
    # last_year = '2015'
    # this_month = '2016-03'
    # this_year = '2016'

    model.plot(prev_week, 'Previous Week')
    model.plot(prev_month, 'Previous Month')
    model.plot(prev_quarter, 'Previous Quarter')
    model.plot(prev_year, 'Previous Year')

    df = model.select(prev_year)
    df = model.calculate(df)
    print('\n', df.tail(10))
    print('\n', df.describe())

    current = df.loc[enddate, STREAK]
    longestW = df[RUN].min()
    longestL = df[RUN].max()

    print()
    print('Current streak: {}'.format(current))
    print('Longest weight-loss streak: {}'.format(longestW))
    print('Longest weight-gain streak: {}'.format(longestL))
    # TODO: Display start and end dates for longest streaks
    # TODO: Contributions in the last year (overall win-loss record)

    # Plot delta
    # df.hist(column=DELTA)
    df[DELTA].plot(grid=True)
    plt.show()

    # shift frame such that date start with Sunday
    a = -df[DELTA].values
    a = a[-70:]
    a = a.reshape(7, -1, order='F')
    plt.imshow(a, cmap=plt.cm.RdYlGn, interpolation='nearest')
    plt.colorbar()
    plt.title(DELTA)
    plt.show()

    # Plot run length
    # df.hist(column=RUN)
    df[RUN].plot(grid=True)
    plt.show()

    a = df[RUN].values
    a = a[-70:]
    a = a.reshape(7, -1, order='F')
    plt.imshow(a, interpolation='nearest')
    plt.colorbar()
    plt.title(RUN)
    plt.show()

if __name__ == '__main__':
    main()
