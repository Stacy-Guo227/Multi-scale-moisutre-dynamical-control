import numpy as np
import pandas as pd
import xarray as xr
import glob
from datetime import datetime, timedelta
import logging
from functools import partial
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
import seaborn.colors.xkcd_rgb as c
from matplotlib.gridspec import GridSpec
import importlib
import polar_util_draw as util_draw
importlib.reload(util_draw)

# Utilities
dlist_swsim = ['20140810', '20140813',
               '20150522', '20150524', '20150719', '20150720', '20150809', '20150828', '20150829', 
               '20160610', '20160611', '20160711', '20160712', '20160902',
               '20170602', '20170614', '20170615', '20170731',
               '20180617', '20180618', '20180619', '20180620', '20180701', '20180702', '20180815', '20180828',
               '20190611', '20190710', '20190810', '20190815']
dlist_atsim32 = ['20050702', '20050712', '20050723',
                 '20060623', '20060718', '20060721',
                 '20070830', 
                 '20080715',
                 '20090707', '20090817', '20090827',
                 '20100629', '20100630', '20100802', '20100803', '20100912',
                 '20110702', '20110723', '20110802', '20110816', '20110821',
                 '20120819',
                 '20130630', '20130703', '20130705', '20130723', '20130807', 
                 '20140703', '20140711', '20140714', '20140825', 
                 '20150613']

def Read_weather_table(data_label:str):
    return pd.read_csv(f'../../TemporaryData/weather_table_self/{data_label}_withlv.csv')

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

class Data_for_plot():
    def __init__(self, date, lat_range, lon_range):
        if not isinstance(date, str): self.DATE = str(date)
        else: self.DATE = date
        self.LAT  = lat_range
        self.LON  = lon_range
        self._ERA5_TPATH = f'ERA5/PRS/day/t/{self.DATE[:4]}/ERA5_PRS_t_{self.DATE[:6]}_r1440x721_day.nc'  # replace with ERA5 daily data
        self._ERA5_UPATH = f'ERA5/PRS/day/u/{self.DATE[:4]}/ERA5_PRS_u_{self.DATE[:6]}_r1440x721_day.nc' 
        self._ERA5_VPATH = f'ERA5/PRS/day/v/{self.DATE[:4]}/ERA5_PRS_v_{self.DATE[:6]}_r1440x721_day.nc' 
        self._ERA5_QPATH = f'ERA5/PRS/day/q/{self.DATE[:4]}/ERA5_PRS_q_{self.DATE[:6]}_r1440x721_day.nc'
        self._IVTPATH    = f'ERA5/IVT/{self.DATE[:4]}.nc'
        self._CWVPATH    = f'ERA5/CWV/1000_700/{self.DATE[:4]}.nc'
        
        self._convert_to_dobj()
    def _convert_to_dobj(self):
        if isinstance(self.DATE, str):
            if len(self.DATE)>8:
                self.DATEOBJ = datetime.strptime(self.DATE, '%Y-%m-%d')
            else:
                self.DATEOBJ = datetime.strptime(self.DATE, '%Y%m%d')
        else:
            self.DATEOBJ = self.DATE
        
    def _get_wind_general(self):
        ds_u = xr.open_dataset(self._ERA5_UPATH)
        ds_v = xr.open_dataset(self._ERA5_VPATH)
        self.u_general = ds_u.u.sel(time=self.DATEOBJ, method='nearest').sel(longitude=slice(*self.LON), 
                                                                             latitude=slice(*self.LAT))
        self.v_general = ds_v.v.sel(time=self.DATEOBJ, method='nearest').sel(longitude=slice(*self.LON), 
                                                                             latitude=slice(*self.LAT))
    
    def get_windprof(self, lat, lon):
        if lat > max(self.LAT) or (lat < min(self.LAT)): raise ValueError(f'Profile lat. not in current data lat range.\nProfile lat.: {lat}, current data lat range: {self.LAT}')
        if lon > max(self.LON) or (lon < min(self.LON)): raise ValueError(f'Profile lon. not in current data lon range.\nProfile lon.: {lon}, current data lon range: {self.LON}')
        self._get_wind_general()
        uprof   = self.u_general.sel(latitude=lat, longitude=lon)
        vprof   = self.v_general.sel(latitude=lat, longitude=lon)
        # print(uprof.longitude, uprof.latitude)
        return uprof, vprof
    
    def get_qprof(self, lat, lon):
        if lat > max(self.LAT) or (lat < min(self.LAT)): raise ValueError(f'Profile lat. not in current data lat range.\nProfile lat.: {lat}, current data lat range: {self.LAT}')
        if lon > max(self.LON) or (lon < min(self.LON)): raise ValueError(f'Profile lon. not in current data lon range.\nProfile lon.: {lon}, current data lon range: {self.LON}')
        ds_q = xr.open_dataset(self._ERA5_QPATH)
        da_q = ds_q.q.sel(time=self.DATEOBJ, method='nearest').sel(latitude=lat, longitude=lon)
        # print(da_q.longitude, da_q.latitude)
        return da_q
    
    def get_Tprof(self, lat, lon):
        if lat > max(self.LAT) or (lat < min(self.LAT)): raise ValueError(f'Profile lat. not in current data lat range.\nProfile lat.: {lat}, current data lat range: {self.LAT}')
        if lon > max(self.LON) or (lon < min(self.LON)): raise ValueError(f'Profile lon. not in current data lon range.\nProfile lon.: {lon}, current data lon range: {self.LON}')
        ds_T = xr.open_dataset(self._ERA5_TPATH)
        da_T = ds_T.t.sel(time=self.DATEOBJ, method='nearest').sel(latitude=lat, longitude=lon)
        # print(da_T.longitude.data, da_T.latitude.data)
        return da_T
    
# Store all upstream profiles
def concat_prof(date:str, var:str, lat_range:tuple, lon_range:tuple, upstream_point:tuple):
    DFP = Data_for_plot(date=date, lat_range=lat_range, lon_range=lon_range)
    prof = eval(f'DFP.get_{var}prof(lat=upstream_point[1], lon=upstream_point[0])')
    return prof

wtab_all = Read_weather_table(data_label='all')
upstream_uprof, upstream_vprof = cal_for_all_dates(wtab_all['yyyymmdd'], concat_prof, 
                                                   func_config={'var':'wind', 'lat_range':(26, 20), 'lon_range':(115, 119), 'upstream_point':(118.5, 21.5)})
upstream_qprof = cal_for_all_dates(wtab_all['yyyymmdd'], concat_prof, 
                                   func_config={'var':'q', 'lat_range':(26, 20), 'lon_range':(115, 119), 'upstream_point':(118.5, 21.5)})
upstream_Tprof = cal_for_all_dates(wtab_all['yyyymmdd'], concat_prof, 
                                   func_config={'var':'T', 'lat_range':(26, 20), 'lon_range':(115, 119), 'upstream_point':(118.5, 21.5)})

for i in range(1, len(upstream_qprof)):
    if i == 1:
        ds_up_uprof = xr.concat([upstream_uprof[0], upstream_uprof[1]], 'time')
        ds_up_vprof = xr.concat([upstream_vprof[0], upstream_vprof[1]], 'time')
        ds_up_qprof = xr.concat([upstream_qprof[0], upstream_qprof[1]], 'time')
        ds_up_Tprof = xr.concat([upstream_Tprof[0], upstream_Tprof[1]], 'time')
    else:
        ds_up_uprof = xr.concat([ds_up_uprof, upstream_uprof[i]], 'time')
        ds_up_vprof = xr.concat([ds_up_vprof, upstream_vprof[i]], 'time')
        ds_up_qprof = xr.concat([ds_up_qprof, upstream_qprof[i]], 'time')
        ds_up_Tprof = xr.concat([ds_up_Tprof, upstream_Tprof[i]], 'time')
        if i%10 == 0:
            print(i)
            
# Qv box plot
def Plot_upstream_box(level:float, plot_upstream=False, plot_vvmsim=False, box_xlim=(0.004, 0.018), figsize=(4, .6), figname=False):
    # Figure initialize
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    # upstream samples
    if plot_upstream:
        wtab_6reg = wtab_all[(wtab_all['ERA5_all_lv0']>=-4)&\
                              (wtab_all['ERA5_all_lv0']<2)&\
                              (wtab_all['ERA5_all_lv1']>=-4)&\
                              (wtab_all['ERA5_all_lv1']<=0)]
        df_up_polar = pd.read_csv('../../TemporaryData/weather_table_self/all_polar.csv')
        temp   = df_up_polar['IVT_theta'].where(df_up_polar['IVT_theta']>=0, df_up_polar['IVT_theta']+np.pi*2)
        swcond = (df_up_polar['IVT_r']>=250)&(temp>=200*np.pi/180)&(temp<=280*np.pi/180)
        medianprops = dict(linestyle='-', linewidth=1.3, color='k')
        boxprops    = dict(linestyle='-', linewidth=1.3, color='k')
        whiskerprops= dict(linestyle='-', linewidth=1.3, color='k')
        capprops    = dict(linestyle='-', linewidth=1.3, color='k')
        if plot_upstream == 'sw':
            matching_days = df_up_polar.loc[swcond, 'yyyymmdd']
            dlist_sw = wtab_6reg[wtab_6reg['yyyymmdd'].isin(matching_days)]['yyyymmdd']
            dlist_sw = pd.to_datetime(dlist_sw, format="%Y%m%d").to_list()
            qlev_sw  = ds_up_qprof.sel(time=dlist_sw, level=level, method='nearest')
            ax1.boxplot(qlev_sw, vert=False, sym='k+', widths=.7, 
                        medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)
        elif plot_upstream == 'at':
            matching_days = df_up_polar.loc[~swcond, 'yyyymmdd']
            dlist_at = wtab_6reg[wtab_6reg['yyyymmdd'].isin(matching_days)]['yyyymmdd']
            dlist_at = pd.to_datetime(dlist_at, format="%Y%m%d").to_list()
            qlev_at  = ds_up_qprof.sel(time=dlist_at, level=level, method='nearest')
            ax1.boxplot(qlev_at, vert=False, sym='k+', widths=.7, 
                        medianprops=medianprops, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops)
    # vvm sim. samples
    if plot_vvmsim:
        if plot_vvmsim == 'sw':
            dlist_vvm_sw = pd.to_datetime(dlist_swsim, format="%Y%m%d").to_list()
            qlev_vvm_sw  = ds_up_qprof.sel(time=dlist_vvm_sw, level=level, method='nearest')
            ax1.scatter(qlev_vvm_sw, np.ones(len(dlist_vvm_sw)), s=35, color=c['bluish'], alpha=0.35, zorder=10)
        if plot_vvmsim == 'at':
            dlist_vvm_at = pd.to_datetime(dlist_atsim32, format="%Y%m%d").to_list()
            qlev_vvm_at  = ds_up_qprof.sel(time=dlist_vvm_at, level=level, method='nearest')
            ax1.scatter(qlev_vvm_at, np.ones(len(dlist_vvm_at)), s=35, color=c['burnt sienna'], alpha=0.35, zorder=10)
    
    ax1.grid(linestyle=':', linewidth=0.5, color='grey', axis='x')
    ax1.set_xlim(box_xlim)
    ax1.set_xticks(np.arange(box_xlim[0], box_xlim[1]+0.001, 0.002))
    ax1.set_xticklabels((np.arange(box_xlim[0], box_xlim[1]+0.001, 0.002)*1000).astype(int))
    ax1.set(yticklabels=[])  # remove the tick labels
    ax1.tick_params(left=False)  # remove the ticks
    ax1.set_title(f"qv (g/kg) @ {int(level)}hPa", loc='left', fontsize=10)
    if figname:
        plt.savefig(f'../../Figure/{figname}_{plot_upstream}.png', facecolor='w', bbox_inches='tight', dpi=400)
    plt.show()
Plot_upstream_box(level=925, plot_upstream='at', plot_vvmsim='at', box_xlim=(0.008, 0.02), figname='Fig5abottom')
Plot_upstream_box(level=925, plot_upstream='sw', plot_vvmsim='sw', box_xlim=(0.008, 0.02), figname='Fig5dbottom')
for lev in [1000, 925, 850, 700]:
    for sim in ['at', 'sw']:
        Plot_upstream_box(level=lev, plot_upstream=sim, plot_vvmsim=sim, box_xlim=(0, 0.024), figsize=(6, .6), figname=f'FigS8_{int(lev)}')
        
# Hourly mean SW-land pcp.
wtab_all      = Read_weather_table(data_label='all')
cond_lv0_all  = (wtab_all['ERA5_all_lv0']>=-4)&(wtab_all['ERA5_all_lv0']<2)
cond_lv1_all  = (wtab_all['ERA5_all_lv1']>=-4)&(wtab_all['ERA5_all_lv1']<0)
# My definition -------------------
updict_6reg_all = pd.read_csv('../../TemporaryData/weather_table_self/all_polar.csv')
wtab_all_6reg = wtab_all[cond_lv0_all&cond_lv1_all].copy()
wtab_all_6reg['IVT dir. (rad.)'] = updict_6reg_all['IVT_theta'].where(updict_6reg_all['IVT_theta']>=0, updict_6reg_all['IVT_theta']+np.pi*2)
mysswf_cond    = (wtab_all_6reg['IVT']>=250)&(wtab_all_6reg['IVT dir. (rad.)']>=200*np.pi/180)&(wtab_all_6reg['IVT dir. (rad.)']<=280*np.pi/180)
dlist_6reg_sswf= wtab_all_6reg[mysswf_cond]['yyyymmdd'].to_list()
dlist_6reg_nosswf= wtab_all_6reg[~mysswf_cond]['yyyymmdd'].to_list()
dobj_6reg_sswf = pd.to_datetime(dlist_6reg_sswf, format='%Y%m%d')
dobj_6reg_nosswf = pd.to_datetime(dlist_6reg_nosswf, format='%Y%m%d')
swhpcp_all = pd.read_csv('../../TemporaryData/weather_table_self/swland_hpcp_all.csv')
swhpcp_all['yyyymmdd'] = pd.to_datetime(swhpcp_all['yyyymmdd'], format='%Y%m%d')
swhpcp_6reg_sswf = swhpcp_all[swhpcp_all['yyyymmdd'].isin(dobj_6reg_sswf)]
swhpcp_6reg_nosswf = swhpcp_all[swhpcp_all['yyyymmdd'].isin(dobj_6reg_nosswf)]
hpcp_mean_6reg_sswf = swhpcp_6reg_sswf.drop(columns=['yyyymmdd']).mean(axis=0).values
hpcp_mean_6reg_nosswf = swhpcp_6reg_nosswf.drop(columns=['yyyymmdd']).mean(axis=0).values

def Plot_diurnal(figname=False):
    fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True, sharey=True, gridspec_kw={'hspace':0.15})
    # individual
    for dd in swhpcp_6reg_sswf['yyyymmdd']:
        ax[1].plot(np.arange(24), np.squeeze(swhpcp_6reg_sswf[swhpcp_6reg_sswf['yyyymmdd']==dd].drop(columns=['yyyymmdd']).to_numpy()), color=c['faded blue'], linewidth=0.2, alpha=0.2)
    for dd in swhpcp_6reg_nosswf['yyyymmdd']:
        ax[0].plot(np.arange(24), np.squeeze(swhpcp_6reg_nosswf[swhpcp_6reg_nosswf['yyyymmdd']==dd].drop(columns=['yyyymmdd']).to_numpy()), color=c['dull orange'], linewidth=0.2, alpha=0.2)
    # mean
    ax[1].plot(np.arange(24), hpcp_mean_6reg_sswf, color=c['bluish'], linewidth=5)
    ax[0].plot(np.arange(24), hpcp_mean_6reg_nosswf, color=c['burnt sienna'], linewidth=5)
    # settings
    ax[0].grid(linestyle=':', linewidth=0.5, color='grey')
    ax[1].grid(linestyle=':', linewidth=0.5, color='grey')
    ax[0].set_xlim(0, 23)
    ax[0].set_xticks(np.arange(1, 24, 2))
    ax[1].set_xticklabels([f"{i:02d}" for i in range(1, 24, 2)], fontsize=12)
    ax[1].set_xlabel('LST', fontsize=12)
    ax[0].set_ylim(0, 5)
    ax[0].set_yticks(np.arange(6))
    ax[0].set_yticklabels([f"{int(i)}" for i in range(6)], fontsize=12)
    ax[0].set_ylabel('mm/hr', fontsize=12)
    ax[1].set_yticks(np.arange(6))
    ax[1].set_yticklabels([f"{int(i)}" for i in range(6)], fontsize=12)
    ax[1].set_ylabel('mm/hr', fontsize=12)
    ax[0].set_title(f'Weakly-forced ({len(dlist_6reg_nosswf)} Days)', fontsize=14)
    ax[1].set_title(f'Strongly-forced ({len(dlist_6reg_sswf)} Days)', fontsize=14)
    plt.savefig(f'../../Figure/{figname}.png', facecolor='w', bbox_inches='tight', dpi=400)
Plot_diurnal(figname='FigS8')