#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 15:14:20 2020

@author: mattkelsey
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pd.set_option("max_columns",50)
pd.set_option("max_rows",200)
json_logins = pd.read_json('logins.json')
json_logins.set_index('login_time',inplace=True)
json_logins['login_time'] = json_logins.index
json_logins = json_logins.resample('15T').count()
json_logins['hour'] = json_logins.index.hour
json_logins['minute'] = json_logins.index.minute
json_logins['hr_str'] = json_logins['hour'].astype(str)
json_logins['min_str'] = json_logins['minute'].astype(str)
json_logins['hour_min'] = json_logins['hr_str'] + json_logins['min_str']
json_logins['hour_min'] = json_logins['hour_min'].astype(int)
json_logins['time'] = json_logins.index
json_15m_buckets = json_logins.groupby('hour_min').login_time.sum()
#json_15m_buckets = pd.DataFrame(json_logins.groupby('hour_min').login_time.sum(), index=json_logins.index)
# json_15m_buckets = json_logins.groupby(['hour_min','time']).login_time.sum()

# Plot all login count for all days
plt.plot(json_logins.login_time)
plt.xticks(rotation=60)
plt.show()

# Plot login count for single day
plt.plot(json_logins.login_time['1970-01-02'])
plt.xticks(rotation=60)
plt.show()

# Plot login count for each unique 15 minute bucket
plt.plot(json_15m_buckets)
plt.xticks(rotation=60)
plt.xticks(np.arange(0, 2345, step = 200))
plt.show()

