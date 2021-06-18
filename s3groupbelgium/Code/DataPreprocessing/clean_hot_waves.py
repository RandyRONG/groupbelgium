import os
import json
import pandas as pd
import numpy as np
import reverse_geocoder as rg
import netCDF4
from netCDF4 import Dataset
from tqdm import tqdm

if __name__ == '__main__':
    root_dir = '../../Data/HeatWaves/'
    nc_path = root_dir+'eu_ht.nc'
    country_code_path = root_dir+"country_code.csv"
    out_json_path = root_dir+'hot_waves.json'
    out_df_path = root_dir+'hot_waves.csv'
    df_country_code = pd.read_csv(country_code_path)
    country_names = list(df_country_code['English short name lower case'])
    alpha_2_codes = list(df_country_code['Alpha-2 code'])
    alpha_3_codes = list(df_country_code['Alpha-3 code'])
    country_name_dict = {}
    for idx,alpha_2_code in enumerate(alpha_2_codes):
        country_name_dict[alpha_2_code] = {}
        country_name_dict[alpha_2_code]['country_name'] = country_names[idx]
        country_name_dict[alpha_2_code]['alpha_3_code'] = alpha_3_codes[idx]
    nc_obj=Dataset(nc_path)
    lat_list=(nc_obj.variables['lat'][:])
    lon_list=(nc_obj.variables['lon'][:])
    time_list=range(1986,2021)
    HWD_EU_health=(nc_obj.variables['HWD_EU_health'][:])

    coordinates_list = []
    country_ll_dict = {}
    for lat_idx,lat_ in (enumerate((lat_list))):
        for lon_idx,lon_ in (enumerate((lon_list))):
            coordinates_list.append((lat_,lon_))
    results = rg.search(coordinates_list)

    for lat_idx,lat_ in (enumerate((lat_list))):
        for lon_idx,lon_ in (enumerate((lon_list))):
            coordinates = (round(lat_,1),round(lon_,1))
            country = results[lat_idx*len(lon_list)+lon_idx]['cc']
            country_ll_dict[coordinates] = country

    record_dict = {}
    for time_idx,time in enumerate(tqdm(time_list)):
        record_dict[time] = {}
        for lat_idx,lat_ in enumerate(lat_list):
            for lon_idx,lon_ in enumerate(lon_list):
                coordinates = (round(lat_,1),round(lon_,1))
                country = country_ll_dict[coordinates]
                if country not in record_dict[time].keys():
                    record_dict[time][country] = []
                heat_waves_times = HWD_EU_health[time_idx][lat_idx][lon_idx]
                if heat_waves_times  is not np.ma.masked:
                    record_dict[time][country].append(np.float(heat_waves_times))

    out_json_dict = json.dumps(record_dict,indent=4)
    f3 = open(out_json_path, 'w')
    f3.write(out_json_dict)
    f3.close()

    for time in record_dict.keys():
        for country in record_dict[time]:
            try:
                record_dict[time][country] = max(record_dict[time][country])
            except:
                continue

    record_df = pd.DataFrame(record_dict)

    country_code_list = list(record_df.index)
    country_name_list = []
    alpha_3_code_list = []

    for coun in country_code_list:
        try:
            country_name_list.append(country_name_dict[coun]['country_name'])
            alpha_3_code_list.append(country_name_dict[coun]['alpha_3_code'])
        except:
            print (coun)


    record_df.insert(0, 'alpha_3_code', alpha_3_code_list)
    record_df.insert(0, 'country_name', country_name_list)

    record_df = record_df.drop(['LB'],axis=0)

    record_df.to_csv(out_df_path)
