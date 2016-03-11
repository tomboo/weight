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

    # shift frame such that date start with Sunday
    startdate = df.index.min()
    print(startdate)
    print(startdate.weekday())

    a = -df[model.DELTA].values
    a = a[-70:]
    a = a.reshape(7, -1, order='F')
    plt.imshow(a, cmap=plt.cm.RdYlGn, interpolation='nearest')
    plt.colorbar()
    plt.title(model.DELTA)
    plt.show()

if __name__ == '__main__':
    connect = model.Model()
    enddate = connect.enddate()
    prev_quarter = enddate - 3 * pd.DateOffset(months=1)
    df = connect.select(prev_quarter)
    df = connect.calculate(df)
    view(df)
