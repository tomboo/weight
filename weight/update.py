# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 22:10:18 2016

@author: Tom
"""

from model import Model
from datetime import datetime
import pandas as pd

# TODO: Use column names from model
DATE = 'Date'
WEIGHT = 'Weight'


def update():
    '''
    Prompt user to insert a new record or update an existing record
    to data file.
    '''
    # Read file
    m = Model(compute=False)
    print('\n', m.df.tail())

    # Enter date. Default is today.
    today = str(datetime.now().date())
    date = input('Enter {0} ({1}): '.format(DATE, today))
    if date == '':
        date = today
    d = pd.Timestamp(date)

    # Enter weight.
    weight = input('Enter {0}: '.format(WEIGHT))
    w = float(weight)

    # Update table
    m.update(d, w)
    print('\n', m.df.tail())

    # Write file
    confirm = input('Write file (y/[n])?: ')
    if confirm == 'y':
        m.write()


if __name__ == '__main__':
    update()
