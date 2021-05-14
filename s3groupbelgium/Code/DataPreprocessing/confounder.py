#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 21:32:24 2021

@author: linguohao
"""

import pandas as pd
from functools import reduce

root_dir = '../../Data/PublicHealth/'
hw_dir = '../../Data/HeatWaves/'


def DataPrePro(filename, var):
    df = pd.read_csv(filename)
    if 'SEX' in df.columns:
        df.query('SEX == "ALL"', inplace=True)
    df.rename(columns={'VALUE': var}, inplace=True)
    df = df[['COUNTRY', 'YEAR', var]]
    return df

#import death datasets
death_all = DataPrePro(filename = root_dir+'death_all.csv', var = 'death_all')
death_external = DataPrePro(filename = root_dir+'death_external.csv', var = 'death_external')
death_diabetes_65 = DataPrePro(filename = root_dir+'death_diabetes_65+.csv', var = 'death_diabetes_65')
death_cerebrovascular_65 = DataPrePro(filename = root_dir+'death_cerebrovascular_65+.csv', var = 'death_cerebrovascular_65')
death_respiratory_65 = DataPrePro(filename = root_dir+'death_respiratory_65+.csv', var = 'death_respiratory_65')

 
# define outcome: internal death
death_internal = pd.merge(left = death_all, right = death_external, how = 'inner',
                          left_on=['COUNTRY', 'YEAR'], right_on=['COUNTRY', 'YEAR'])
death_internal.dropna(subset=['COUNTRY'], inplace=True) #drop missing value
death_internal.query('YEAR >= 1986', inplace=True) #filter data >=1986
death_internal['death_internal'] = death_internal['death_all'] - death_internal['death_external']

hot_waves = pd.read_csv(hw_dir+'hot_waves.csv')
codes = pd.Series.tolist(hot_waves['alpha_3_code']) #list of country code in hot_waves data
death_internal[death_internal.COUNTRY.isin(codes)] # filter country only in hot_waves data
death_internal['COUNTRY'].value_counts().reset_index() # frequncey of country


# import confounder datasets
elderly_rate = DataPrePro(filename = root_dir+'elderly_rate.csv', var = 'elderly_rate')
life_expectancy = DataPrePro(filename = root_dir+'life_expectancy.csv', var = 'life_expectancy')
hospital_beds = DataPrePro(filename = root_dir+'hospital_beds.csv', var = 'hospital_beds')
gni = DataPrePro(filename = root_dir+'gni_per_capita.csv', var = 'gni')


# combine dataset
data_combined = [death_internal, death_diabetes_65, death_cerebrovascular_65, death_respiratory_65, 
                 elderly_rate, life_expectancy, hospital_beds, gni]
final_data = reduce(lambda left, right: pd.merge(left, right, on=['COUNTRY','YEAR'], 
                                                 how='left'), data_combined)
final_data.to_csv(root_dir+'final_data.csv',index=False)
