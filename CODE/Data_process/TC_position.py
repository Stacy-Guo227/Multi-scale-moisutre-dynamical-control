import numpy as np
import pandas as pd
import xarray as xr
import glob
from datetime import datetime, timedelta
import logging
from functools import partial
import multiprocessing

ds_ibtracs = xr.open_dataset('/../../TemporaryData/IBTrACS.since1980.v04r00.nc')
wtab_all   = pd.read_csv('../../TemporaryData/weather_table_self/all_withlv.csv')
class StormDataset():
    def __init__(self, year_list, month_list, nature_list):
        # Masks
        self._time_mask   = ((ds_ibtracs.time.dt.year.isin(year_list))&(ds_ibtracs.time.dt.month.isin(month_list)))   # storm's lifetime in given time range
        self._nature_mask = ds_ibtracs.nature.isin(nature_list)  # storm's type
        self._masked_time = ds_ibtracs.time.where(self._time_mask)
        self._storm_lat   = ds_ibtracs.lat.where(self._time_mask&self._nature_mask, np.nan)
        self._storm_lon   = ds_ibtracs.lon.where(self._time_mask&self._nature_mask, np.nan)
        # Create data
        self.storm_dict = dict(zip(wtab_all['yyyymmdd'].to_list(), [{'lat':[], 'lon':[]} for i in range(len(wtab_all))])) # yyyymmdd is type(int)
        lat_result, lon_result = self._cal_for_all_dates(datelist=wtab_all['yyyymmdd'].to_list(), func=self._extract_tc_pos_date)
        for idx, dd in enumerate(self.storm_dict.keys()):
            self.storm_dict[dd]['lat'] = self.storm_dict[dd]['lat'] + lat_result[idx]
            self.storm_dict[dd]['lon'] = self.storm_dict[dd]['lon'] + lon_result[idx]
        ## Add description after calculation
        self.storm_dict['Description'] = ("Print to get better readability. \n"
                                          f"Storing lat/lon of storms in lists on each day in years {year_list} & months {month_list}\n"
                                          f"Storm type: {nature_list}\n"
                                          "Notice: Longitude can be either (0, 360) or (-180, 180).\n"
                                          "Source: IBTrACS")
        
    def _extract_tc_pos_date(self, date):
        """
        Extract TC positions of assigned date (the date should be cover in the given year_list and month_list).
        :return: position latitude and position longitude
        :rtype : list and list, empty lists if no TCs
        """
        if isinstance(date, int):
            dd_str = str(date)
            dd_str = f"{dd_str[:4]}-{dd_str[4:6]}-{dd_str[6:]}"   # 'yyyy-mm-dd'
        elif isinstance(date, str):
            dd_str = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        mask_dd     = self._masked_time.dt.floor('D')==np.datetime64(dd_str)
        lat_dd      = self._storm_lat.where(mask_dd)
        lon_dd      = self._storm_lon.where(mask_dd)
        lat_date = lat_dd.mean(dim='date_time', skipna=True)[~np.isnan(lat_dd.mean(dim='date_time', skipna=True))].data.tolist()
        lon_date = lon_dd.mean(dim='date_time', skipna=True)[~np.isnan(lon_dd.mean(dim='date_time', skipna=True))].data.tolist()
        return lat_date, lon_date
    
    def _cal_for_all_dates(self, datelist, func, func_config={}, cores=10):
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
        
sd = StormDataset(year_list=np.arange(2001, 2020), month_list=np.arange(4, 10), nature_list=[b'TS'])
# np.save('/data/ch995334/SSWF_proj/TemporaryData/sum_storm_dict_TS.npy', sd.storm_dict)