# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 22:10:18 2016

@author: Tom
"""

import pandas as pd
from datetime import datetime

DATAFILE = 'data.csv'
DATE = 'Date'
WEIGHT = 'Weight'


def insert():
    # Read file
    df = pd.read_csv(DATAFILE, index_col=0, parse_dates=True)
    print('\n', df.tail())

    # Enter date field. Default is today's date.
    today = str(datetime.now().date())
    date = input('Enter {0} ({1}): '.format(DATE, today))
    if date == '':
        date = today
    d = pd.Timestamp(date)

    # Enter weight field.
    weight = input('Enter {0}: '.format(WEIGHT))
    w = float(weight)

    # Update table
    df.loc[d, WEIGHT] = w
    df.sort_index(inplace=True)
    print('\n', df.tail())

    # Write file
    confirm = input('Write file [y/n]: ')
    if confirm == 'y':
        df.to_csv(DATAFILE)


if __name__ == '__main__':
    insert()
