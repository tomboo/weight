# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 06:01:19 2016

@author: Tom
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import model


def view(df):
    # Plot delta
    # df.hist(column=DELTA)
    df[model.DELTA].plot(grid=True)
    plt.show()

    df.ix[df[model.DELTA] > 0, model.DELTA] = 0
    df[model.DELTA].plot(grid=True)
    plt.show()

    # TODO: Shift frame such that date start with Sunday
    startdate = df.index[0]
    enddate = df.index[len(df.index) - 1]
    print(startdate, enddate)
    print(startdate.weekday())
    # weekday(): The day of the week with Monday=0, Sunday=6

    a = df[model.DELTA].values
    a = a[-364:]
    a = a.reshape(7, -1, order='F')

    # http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(a, cmap=plt.cm.afmhot, interpolation='nearest')
    plt.colorbar(orientation='horizontal')
    plt.title(model.DELTA)
    plt.show()

if __name__ == '__main__':
    m = model.Model()
    enddate = m.enddate()
    startdate = enddate - pd.DateOffset(years=1)
    df = m.select(startdate)
    view(df)
