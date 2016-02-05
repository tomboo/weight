# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd

def main():
    x = pd.read_csv('data.csv')
    print(x.head())
    x.plot()

if __name__ == '__main__':
    main()