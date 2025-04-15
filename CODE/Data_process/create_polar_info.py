import numpy as np
import pandas as pd
import xarray as xr
import glob
from datetime import datetime, timedelta
import logging
from functools import partial
import multiprocessing

# =========== Utilities ===========
def convert_to_dobj(date):
    """
    Convert a date (string/int) into a datetime object.
    Two types of string format are supported: 20051218 or 2005-12-18.
    """
    if isinstance(date, int) or isinstance(date, float):
        date = str(int(date))
        
    if isinstance(date, str):
        if len(date)>8:
            dateobj = datetime.strptime(date, '%Y-%m-%d')
        else:
            dateobj = datetime.strptime(date, '%Y%m%d')
    else:
        dateobj = date
    return dateobj

def Get_upstream_mean_wind(date:str, level:float, lon_range:tuple, lat_range:tuple):
    """
    Calculate the mean wind for a specified region and level.
    """
    # Special attention to ERA5 latitute
    lat_range = lat_range if lat_range[0]>lat_range[1] and \
                             lat_range[0] is not None and lat_range[1] is not None \
                             else lat_range[::-1]  # for ERA5 data
    # ERA5 data paths
    upath = f'ERA5/PRS/day/u/{date[:4]}/ERA5_PRS_u_{date[:6]}_r1440x721_day.nc'  # replace with ERA5 daily data
    vpath = f'ERA5/PRS/day/v/{date[:4]}/ERA5_PRS_v_{date[:6]}_r1440x721_day.nc'
    # Extract data
    dobj  = convert_to_dobj(date)
    ds_u  = xr.open_dataset(upath).sel(time=dobj, method='nearest').sel(latitude=slice(*lat_range),
                                                                        longitude=slice(*lon_range), 
                                                                        level=level)
    ds_v  = xr.open_dataset(vpath).sel(time=dobj, method='nearest').sel(latitude=slice(*lat_range),
                                                                        longitude=slice(*lon_range), 
                                                                        level=level)
    umean = ds_u.u.mean(dim=['latitude', 'longitude']).values
    vmean = ds_v.v.mean(dim=['latitude', 'longitude']).values
    wsmean= np.sqrt(ds_u.u**2+ds_v.v**2).mean(dim=['latitude', 'longitude']).values
    return umean.item(), vmean.item(), wsmean.item()

def Get_upstream_mean_IVT(date:str, lon_range:tuple, lat_range:tuple):
    """
    Calculate the mean IVT for a specified region.
    """
    # Special attention to ERA5 latitute
    lat_range = lat_range if lat_range[0]>lat_range[1] and \
                             lat_range[0] is not None and lat_range[1] is not None \
                             else lat_range[::-1]  # for ERA5 data
    # IVT data paths
    dpath = f'/ERA5/IVT/{date[:4]}.nc'  # replace with IVT yearly data
    # Extract data
    dobj  = convert_to_dobj(date)
    ds_ivt= xr.open_dataset(dpath).sel(time=dobj, method='nearest').sel(lat=slice(*lat_range), 
                                                                        lon=slice(*lon_range))
    ivt_xmean = ds_ivt.sum_IVTx.mean(dim=['lat', 'lon']).values
    ivt_ymean = ds_ivt.sum_IVTy.mean(dim=['lat', 'lon']).values
    ivt_mean  = ds_ivt.sum_IVT_total.mean(dim=['lat', 'lon']).values
    return ivt_xmean.item(), ivt_ymean.item(), ivt_mean.item()

def uv2polar_angle(u, v, radians=True):
    """
    Calculate the angle in (1) polar coordinate convention (2) meteorological convention from u- and v-wind.
    """
    # Polar coord.
    # if radians: return np.pi+np.arctan2(v, u)
    # else: return (180+np.arctan2(v, u)*180/np.pi)
    
    # Meteorology
    if radians: return (np.pi+np.arctan2(v, u))*(-1)+np.pi/2
    else: return ((180+np.arctan2(v, u)*180/np.pi))*(-1)+90

def Get_r_theta(date:str, env:str, level=None, lon_range:tuple=(115, 119), lat_range:tuple=(22, 20)):
    # Check
    if (env=='wind')&(level is None): raise ValueError("Please enter a specified level for wind.")
    # Loop dates
    if env == 'wind':
        u, v, total = Get_upstream_mean_wind(date=str(date), level=level, lon_range=lon_range, lat_range=lat_range)
    elif env == 'IVT':
        u, v, total = Get_upstream_mean_IVT(date=str(date), lon_range=lon_range, lat_range=lat_range)
    # Get r and theta
    windspeed, angle = total, uv2polar_angle(u, v)
    return windspeed, angle

def cal_for_all_dates(datelist, func, func_config={}, cores=10):
        """
        Call the calculation methods (for single day) and return results for a range of days.
        *Can handle single/multiple outputs from func.
        
        :param datelist: List of dates for iterating calculation
        :type  datelist: list
        :param func: Funciton(method) to call
        :type  func: function
        :param func_config: Parameters for func
        :type  func_config: dict, optional, default={}
        
        :return: Calculation result for each day
        :rtype : tuple or list
        """
        # Create a partial function that pre-binds the config to the func
        func_with_config = partial(func, **func_config)
        with multiprocessing.Pool(processes=cores) as pool:
            results = pool.map(func_with_config, datelist)  # func result * number of processes
            
        # Create nested list to handle various amount of outputs
        output_num = len(results[0]) if isinstance(results[0], tuple) else 1  # check multi-/single output
        nest_list  = [[] for _ in range(output_num)]        # nested list handling func output
        # Store outputs in individual lists
        for output in results:                              # output: output for single call of func
            if output_num > 1:
                for i, val in enumerate(output):
                    nest_list[i].append(val)
            else:
                nest_list[0].append(output)
        return tuple(nest_list) if output_num > 1 else nest_list[0]

# =========== Execution ===========
wtab_all   = pd.read_csv('../../TemporaryData/weather_table_self/all_withlv.csv')
# Initiate dict for weather table
datelist   = wtab_all['yyyymmdd']
polar_dict = {'yyyymmdd':datelist}
# Wind
for lev in [1000, 925, 850, 700, 500, 300, 200]:
    polar_dict[f"wind{lev}_r"], polar_dict[f"wind{lev}_theta"] = cal_for_all_dates(datelist=datelist, func=Get_r_theta, func_config={'env':'wind', 'level':lev})
    print(f"Current progress: Wind {lev}hPa")
    print(polar_dict[f"wind{lev}_r"][-3:], polar_dict[f"wind{lev}_theta"][-3:])
# IVT
polar_dict["IVT_r"], polar_dict["IVT_theta"] = cal_for_all_dates(datelist=datelist, func=Get_r_theta, func_config={'env':'IVT'})
print(f"Current progress: IVT")
print(polar_dict[f"IVT_r"][-3:], polar_dict[f"IVT_theta"][-3:])
# Store and save
df_polar = pd.DataFrame(polar_dict)
df_polar.to_csv('../../TemporaryData/weather_table_self/all_polar.csv', index=False)