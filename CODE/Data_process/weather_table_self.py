import numpy as np
import pandas as pd
import xarray as xr
import glob
from datetime import datetime
import logging

class weather_table():
    def __init__(self, year_list:list, month_list:list, 
                 special_start_date:str=False):
        self.YEARS     = year_list
        self.MONTHS    = month_list
        self._ORGTABLE = pd.read_csv(f'../../TemporaryData/weather_event_2000_2019.csv')  # this table is available upon request
        self.ORGLABEL  = list(self._ORGTABLE.keys())
        
        self._wtab_specific_date(special_start_date)
    
    def _wtab_specific_date(self, special_start_date):
        """
        Return weather table (as a DataFrame) according to assigned dates.
        """
        # Convert date strings to datetime
        self._ORGTABLE['yyyymmdd'] = pd.to_datetime(self._ORGTABLE['yyyymmdd'], format='%Y%m%d')
        # Check if special start date is assigned
        if special_start_date:
            special_start_date_dobj    = datetime.strptime(special_start_date, "%Y%m%d")
            table_temp                 = self._ORGTABLE[self._ORGTABLE['yyyymmdd']>=special_start_date_dobj]
        else:
            table_temp                 = self._ORGTABLE
        # Table and date list within requested date range
        self.SPECDATE_TABLE            = table_temp[(table_temp['yyyymmdd'].dt.year.isin(self.YEARS)) &
                                                    (table_temp['yyyymmdd'].dt.month.isin(self.MONTHS))
                                                    ]
        self.SPEC_DATELIST             = self.SPECDATE_TABLE['yyyymmdd'].dt.strftime('%Y%m%d').to_list()
    
    def wtab_wtype(self, 
                   wtype_condition_true:list=[], wtype_condition_false:list=[], 
                   return_value:str='both'):
        """
        Return weather table (as a DataFrame) with assigned weather types.
        """
        
        table_temp = self.SPECDATE_TABLE
        didx_temp  = np.arange(len(self.SPEC_DATELIST))
        
        # Weather types to be remained
        if len(wtype_condition_true)>0:
            for i in range(len(wtype_condition_true)):
                didx_temp  = didx_temp[table_temp[wtype_condition_true[i]]==1]
                table_temp = table_temp[table_temp[wtype_condition_true[i]]==1]
        
        # Weather types to be filtered out
        if len(wtype_condition_false)>0:
            for i in range(len(wtype_condition_false)):
                didx_temp  = didx_temp[table_temp[wtype_condition_false[i]]==0]
                table_temp = table_temp[table_temp[wtype_condition_false[i]]==0]
        
        # !! Special situation
        if (len(wtype_condition_true)+len(wtype_condition_false))<1:
            logging.warning("Current SPECWTYPE_TABLE/DAYS/DIDX is identical to SPECDATE_TABLE/DAYS/DIDX, since no weather-type-condition is assigned.")
        
        self.SPECWTYPE_TABLE = table_temp
        self.SPECWTYPE_DAYS  = len(self.SPECWTYPE_TABLE)
        self.SPECWTYPE_DIDX  = didx_temp
        
        # User-selected returns
        if return_value == 'table':
            return self.SPECWTYPE_TABLE
        elif return_value == 'num of days':
            return self.SPECWTYPE_DAYS
        elif return_value == 'didx':
            return self.SPECWTYPE_DIDX
        elif return_value == 'all':
            return self.SPECWTYPE_TABLE, self.SPECWTYPE_DAYS, self.SPECWTYPE_DIDX
        else:
            raise ValueError("Unrecognized return_value. Please choose from: 'table', 'num of days', 'didx', 'all'.")
            
    def get_cwb_precip_table(self, date:str, accumulate_daily=False):
        # File list
        cwb_fpath = f'CWA/{date[:4]}/{date[4:6]}/{date}'     # replace with cwa data
        cwb_flist = sorted(glob.glob(f'{cwb_fpath}/{date}*'))
        # Define column widths based on README
        col_widths = [7, 10, 9, 7, 7, 7, 7, 7, 7, 7, 7, 3]
        # Define the column names
        col_names = ['stno', 'lon', 'lat', 'elv', 'PS', 'T', 'RH',
                     'WS', 'WD', 'PP', 'odSS01', 'ojits']
        # Use read_fwf to read the fixed-width formatted file
        for hh in range(len(cwb_flist)):
            data = pd.read_fwf(cwb_flist[hh], widths=col_widths, names=col_names)
            cwb_stno, cwb_lon, cwb_lat, cwb_pcp = data[['stno', 'lon', 'lat', 'PP']].values.T
            self.cwb_LON = cwb_lon
            self.cwb_LAT = cwb_lat
            
            if accumulate_daily:                              # target_pcp is daily precip.
                if hh<1:
                    target_pcp = cwb_pcp
                else:
                    target_pcp = np.add(target_pcp, cwb_pcp)
                
            else:                                            # target_pcp is hourly precip.
                if hh<1:
                    target_pcp= cwb_pcp[..., np.newaxis]
                else:
                    target_pcp= np.concatenate((target_pcp, cwb_pcp[..., np.newaxis]), axis=1)
        target_pcp    = np.where(target_pcp>=0., target_pcp, np.nan)
        if accumulate_daily:
            pcp_table = pd.DataFrame({'stn_lon':cwb_lon, 'stn_lat':cwb_lat, 'precip':target_pcp})
            return pcp_table
        else:
            stno_table = pd.DataFrame({'stno':cwb_stno, 'stn_lon':cwb_lon, 'stn_lat':cwb_lat})
            #raise ValueError("Table for hourly precipitation isn't built yet.")
            return stno_table, target_pcp
                

    def cwb_metrics_dprecip(self, 
                            select_region:tuple=(None, None, None, None),
                            Accu=False,
                            Mean=False,
                            Top10_mean=False,
                            Max=False,
                            Std=False):
        # Create storage
        pcp_table   = {'yyyymmdd':self.SPEC_DATELIST}
        # Get metrics for all selected days
        for idx, dd in enumerate(self.SPEC_DATELIST):
            # Daily preicp.
            dpcp_table = self.get_cwb_precip_table(dd, accumulate_daily=True)
            # Select region
            lon_min, lon_max, lat_min, lat_max = select_region
            dpcp_table_reg = dpcp_table.copy()
            if lon_min is not None:
                dpcp_table_reg = dpcp_table_reg[dpcp_table_reg['stn_lon'] >= lon_min]
            if lon_max is not None:
                dpcp_table_reg = dpcp_table_reg[dpcp_table_reg['stn_lon'] < lon_max]
            if lat_min is not None:
                dpcp_table_reg = dpcp_table_reg[dpcp_table_reg['stn_lat'] >= lat_min]
            if lat_max is not None:
                dpcp_table_reg = dpcp_table_reg[dpcp_table_reg['stn_lat'] < lat_max]
            # Checking messages
            print(f"Original length of precip. table: {len(dpcp_table)}")
            print(f"Region-filtered length of precip. table: {len(dpcp_table_reg)}")
            
            dpcp_reg       = np.asarray(dpcp_table_reg['precip'])
            # Calculate metrics=True
            if idx<1:
                if Accu:
                    pcp_table['accu'] = [np.nansum(dpcp_reg)]
                if Mean:
                    pcp_table['mean'] = [np.nanmean(dpcp_reg)]
                if Top10_mean:
                    top10_dprecip = np.sort(dpcp_reg[~np.isnan(dpcp_reg)])[-10:]
                    pcp_table['top10_mean'] = [np.nanmean(top10_dprecip)]
                if Max:
                    pcp_table['max'] = [np.nanmax(dpcp_reg)]
                if Std:
                    pcp_table['std'] = [np.nanstd(dpcp_reg)]
                
            else:
                if Accu:
                    pcp_table['accu'].append(np.nansum(dpcp_reg))
                if Mean:
                    pcp_table['mean'].append(np.nanmean(dpcp_reg))
                if Top10_mean:
                    top10_dprecip = np.sort(dpcp_reg[~np.isnan(dpcp_reg)])[-10:]
                    pcp_table['top10_mean'].append(np.nanmean(top10_dprecip))
                if Max:
                    pcp_table['max'].append(np.nanmax(dpcp_reg))
                if Std:
                    pcp_table['std'].append(np.nanstd(dpcp_reg))
        return pd.DataFrame(pcp_table)
    
    def cwb_regmean_hpcp(self, select_region:tuple=(None, None, None, None), print_check:bool=False):
        # Create storage
        pcp_table   = pd.DataFrame({'yyyymmdd':self.SPEC_DATELIST})
        hourly_columns = [f"{hour:02d}" for hour in range(24)]
        pcp_table[hourly_columns] = np.nan                     # Initialize with NaN
        # Get metrics for all selected days
        for idx, dd in enumerate(self.SPEC_DATELIST):
            # Hourly preicp.
            stno_table, pcp_array = self.get_cwb_precip_table(dd, accumulate_daily=False)  # stno_table: [[stno, stn_lon, stn_lat]], pcp_array: (stn, hh)
            # Select region
            lon_min, lon_max, lat_min, lat_max = select_region
            stno_table_reg = stno_table.copy()
            if lon_min is not None:
                stno_table_reg = stno_table_reg[stno_table_reg['stn_lon'] >= lon_min]
            if lon_max is not None:
                stno_table_reg = stno_table_reg[stno_table_reg['stn_lon'] < lon_max]
            if lat_min is not None:
                stno_table_reg = stno_table_reg[stno_table_reg['stn_lat'] >= lat_min]
            if lat_max is not None:
                stno_table_reg = stno_table_reg[stno_table_reg['stn_lat'] < lat_max]
            reg_stn_idx = stno_table_reg.index.to_numpy()
            # Checking messages
            if print_check:
                print(f"Original length of precip. table: {len(stno_table)}")
                print(f"Region-filtered length of precip. table: {len(stno_table_reg)}")
                print(f"Row indices: {reg_stn_idx}")
            # Average regional hourly precip.
            reg_hpcp = np.nanmean(pcp_array[reg_stn_idx, ...], axis=0)
            pcp_table.loc[idx, hourly_columns] = reg_hpcp
            # for hh in range(24):
            #     pcp_table[f"{hh:02d}"] = reg_hpcp[hh]
            print(pd.DataFrame(pcp_table))
        return pd.DataFrame(pcp_table)

# Instance testing dataset
wtab_test = weather_table(year_list=np.arange(2014, 2020).tolist(), month_list=np.arange(4, 10).tolist(), 
                          special_start_date='20140731') 
# Instance summer(whole) dataset
wtab_all  = weather_table(year_list=np.arange(2001, 2020).tolist(), month_list=np.arange(4, 10).tolist()) 

# Add latent vectors to weather table
df_wtab   = wtab_all.SPECDATE_TABLE
lv_dict   = np.load('fix/latent/vector/file.npy', allow_pickle=True).item()
lv_all    = lv_dict['ERA5(all)']
df_wtab['ERA5_all_lv0'] = lv_all[:, 0]
df_wtab['ERA5_all_lv1'] = lv_all[:, 1]
df_wtab.to_csv('../../TemporaryData/weather_table_self/all_withlv.csv', index=False)

# Precipitation tables
swland_pcp_all = wtab_all.cwb_metrics_dprecip(select_region=(120, 121, 21.9, 23.5), 
                                              Accu=True,
                                              Mean=True, 
                                              Top10_mean=True, 
                                              Max=True, 
                                              Std=True)
# swland_pcp_all.to_csv('../../TemporaryData/weather_table_self/swland_pcp_all.csv', index=False)
taiwan_pcp_all = wtab_all.cwb_metrics_dprecip(Accu=True,
                                              Mean=True, 
                                              Top10_mean=True, 
                                              Max=True, 
                                              Std=True)
# taiwan_pcp_all.to_csv('../../TemporaryData/weather_table_self/taiwan_pcp_all.csv', index=False)
swland_hpcp_all = wtab_all.cwb_regmean_hpcp(select_region=(120, 121, 21.9, 23.5))
# swland_hpcp_all.to_csv('../../TemporaryData/weather_table_self/swland_hpcp_all.csv', index=False)