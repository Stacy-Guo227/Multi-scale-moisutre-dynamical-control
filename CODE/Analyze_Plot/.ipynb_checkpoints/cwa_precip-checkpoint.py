import numpy as np
import pandas as pd
import xarray as xr
import glob
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
import seaborn.colors.xkcd_rgb as c
from matplotlib.gridspec import GridSpec

# Utilities
def Read_weather_table(data_label:str):
    return pd.read_csv(f'../../TemporaryData/weather_table_self/{data_label}_withlv.csv')

class weather_table():
    def __init__(self, year_list:list, month_list:list, 
                 special_start_date:str=False, special_end_date:str=False, 
                 lat_range=(None, None), lon_range=(None, None)):
        self.YEARS     = year_list
        self.MONTHS    = month_list
        self.STARTDATE = self._convert_to_dobj(special_start_date) if special_start_date else None
        self.ENDDATE   = self._convert_to_dobj(special_end_date) if special_end_date else None
        self.LATRANGE  = lat_range if lat_range[0]>lat_range[1] and \
                         lat_range[0] is not None and lat_range[1] is not None \
                         else lat_range[::-1]  # for ERA5 data
        self.LONRANGE  = lon_range
        
        self._create_date_list()
    
    def _convert_to_dobj(self, date):
        """
        Convert a date string into a datetime object.
        Two types of string format are supported: 20051218 or 2005-12-18.
        """
        if isinstance(date, str):
            if len(date)>8:
                dateobj = datetime.strptime(date, '%Y-%m-%d')
            else:
                dateobj = datetime.strptime(date, '%Y%m%d')
        else:
            dateobj = date
        return dateobj
        
    def _create_date_list(self):
        """
        Create date list (of strings and datetime objects) in specific months for specific year range.
        Supports discrete month choices, designated start date and end date.
        !!Future adjustment!!: discrete year choices
        """
        start_date = self.STARTDATE if self.STARTDATE is not None else datetime(self.YEARS[0], 1, 1)
        end_date   = self.ENDDATE if self.ENDDATE is not None else datetime(self.YEARS[-1], 12, 31)
        # All dates in the year range (datetime objects and strings)
        self._dlist= [start_date+timedelta(days=i) for i in range((end_date-start_date).days+1)]
        self.DLIST = [(start_date+timedelta(days=i)).strftime("%Y%m%d") for i in range((end_date-start_date).days+1)]
        # Addtionally select months
        self._dlist_month= [dobj for dobj in self._dlist if dobj.month in self.MONTHS]
        self.DLIST_Month = [dobj.strftime("%Y%m%d") for dobj in self._dlist if dobj.month in self.MONTHS]
    
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
    
    def get_cwb_precip_table(self, date:str, accumulate_daily=False):
        # File list
        cwb_fpath = f'CWA/{date[:4]}/{date[4:6]}/{date}'  # CWA data has to be downloaded
        cwb_flist = sorted(glob.glob(f'{cwb_fpath}/{date}*'))
        # Define column widths based on README
        col_widths = [7, 10, 9, 7, 7, 7, 7, 7, 7, 7, 7, 3]
        # Define the column names
        col_names = ['stno', 'lon', 'lat', 'elv', 'PS', 'T', 'RH',
                     'WS', 'WD', 'PP', 'odSS01', 'ojits']
        # Use read_fwf to read the fixed-width formatted file
        for hh in range(len(cwb_flist)):
            data = pd.read_fwf(cwb_flist[hh], widths=col_widths, names=col_names)
            cwb_lon, cwb_lat, cwb_pcp = data[['lon', 'lat', 'PP']].values.T
            self.cwb_LON = cwb_lon
            self.cwb_LAT = cwb_lat
            
            if accumulate_daily:                              # target_pcp is daily precip.
                if hh<1:
                    target_pcp = cwb_pcp
                else:
                    target_pcp = np.add(target_pcp, cwb_pcp)
                
            else:                                            # target_pcp is hourly precip.
                if hh<1:
                    target_pcp= cwb_pcp[np.newaxis, ...]
                else:
                    target_pcp= np.concatenate((target_pcp, cwb_pcp[np.newaxis, ...]), axis=0)
        target_pcp    = np.where(target_pcp>=0., target_pcp, np.nan)
        if accumulate_daily:
            pcp_table = pd.DataFrame({'stn_lon':cwb_lon, 'stn_lat':cwb_lat, 'precip':target_pcp})
        else:
            raise ValueError("Table for hourly precipitation isn't built yet.")
        return pcp_table
    
    def get_binned_dprecip_day(self, date, 
                               select_region:tuple=(None, None, None, None)):
        """
        Binned the daily precip. for all CWA stations in the selected region. Return counts in those bins.
        """
        # Set PDF bins
        bounds      = np.array([1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300])  # 16
        bin_array   = np.zeros(bounds.shape[0]+1)  # 17
        # Get daily precip.
        dpcp_table = self.get_cwb_precip_table(date, accumulate_daily=True)
        # Select region
        lon_min, lon_max, lat_min, lat_max = select_region
        dpcp_table_reg = dpcp_table[(dpcp_table['stn_lon']>=lon_min)&
                                    (dpcp_table['stn_lon']< lon_max)&
                                    (dpcp_table['stn_lat']>=lat_min)&
                                    (dpcp_table['stn_lat']< lat_max)]
        dpcp_reg       = np.asarray(dpcp_table_reg['precip'])
        # Loop for all bins
        for i in range(bin_array.shape[0]):        # i = 0 ~ 16
            if i < 1:                              # i = 0
                cond         = (dpcp_reg>=0)&(dpcp_reg<bounds[i])
            elif i > bin_array.shape[0]-2:         # i = 16
                cond         = (dpcp_reg>=bounds[i-1])
            else:                                  # i = 1~15
                cond         = (dpcp_reg>=bounds[i-1])&(dpcp_reg<bounds[i])
            bin_array[i] = dpcp_reg[cond].shape[0]
        # Check calculation
        # print("Total number of stations:   ", dpcp_reg.shape[0])
        # print("Total number of >=0 precip.:", bin_array.sum())
        # print("Total number of <0  precip.:", np.isnan(dpcp_reg).sum())
        return bin_array
    
    def get_binned_dprecip_drange(self, datelist, 
                                  select_region:tuple=(None, None, None, None), 
                                  probability=False):
        """
        Get the binned counts of precip. in a range of dates. Return PDF if `probability` is set to True.
        """
        func_config = {'select_region':select_region}
        count_dates = self._cal_for_all_dates(datelist=datelist, 
                                              func=self.get_binned_dprecip_day, 
                                              func_config=func_config)
        count_dates = sum(count_dates)   # sum the result from different days
        if probability:
            return count_dates/count_dates.sum()
        else:
            return count_dates
                

wtab_module_test = weather_table(year_list=np.arange(2014, 2020).tolist(), month_list=np.arange(4, 10).tolist(),
                                 special_start_date='20140731', 
                                 lat_range=(22, 20), lon_range=(115, 119))
wtab_module_train= weather_table(year_list=np.arange(2001, 2020).tolist(), month_list=np.arange(4, 10).tolist(),
                                 special_end_date='20110917',
                                 lat_range=(22, 20), lon_range=(115, 119))
wtab_module_all  = weather_table(year_list=np.arange(2001, 2020).tolist(), month_list=np.arange(4, 10).tolist(), 
                                 lat_range=(22, 20), lon_range=(115, 119))

class Plot_analysis():
    def __init__(self):
        self.proj = ccrs.PlateCarree()
        self.ds_topo = xr.open_dataset('TOPO.nc')  # TaiwanVVM topography, available upon request
        
    
    def Axe_map(self, fig, gs, 
                xlim_, ylim_, **grid_info):
        # Set map extent
        axe  = fig.add_subplot(gs, projection=self.proj)
        axe.set_extent([xlim_[0], xlim_[-1], ylim_[0], ylim_[-1]], crs=self.proj)
        # Set additional grid information
        if len(grid_info)>0:
            if grid_info['xloc_'] is not None:
                axe.set_xticks(grid_info['xloc_'], crs=self.proj)
                axe.set_xticklabels(['' for i in range(len(grid_info['xloc_']))])  # default: no tick labels
            if grid_info['yloc_'] is not None:
                axe.set_yticks(grid_info['yloc_'], crs=self.proj)
                axe.set_yticklabels(['' for i in range(len(grid_info['yloc_']))])
            gl = axe.gridlines(xlocs=grid_info['xloc_'], ylocs=grid_info['yloc_'], 
                               draw_labels=False)
        return axe
    
    def Plot_cartopy_map(self, axe):
        axe.add_feature(cfeature.LAND,color='grey',alpha=0.1)
        axe.coastlines(resolution='50m', color='black', linewidth=1)
        return axe
    
    def Plot_vvm_map(self, axe, color, linewidth):
        axe.contour(self.ds_topo.lon, self.ds_topo.lat, self.ds_topo.TOPO, 
                    levels=np.array([-1e-3, 1e-3]), 
                    colors=color, linewidths=linewidth)
    
    def Plot_vvm_topo(self, axe, color, linewidth=None):
        topo_bounds= np.arange(0, 3500.1, 500)
        alpha_list = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        cmap_topo  = colors.ListedColormap([(0, 0, 0, i) for i in alpha_list])
        norm_      = colors.BoundaryNorm(topo_bounds, cmap_topo.N, extend='max')
        imtopoh    = axe.contourf(self.ds_topo.lon, self.ds_topo.lat, self.ds_topo.TOPO*1e2, 
                                  levels=topo_bounds, 
                                  cmap=cmap_topo, norm=norm_, extend='max', antialiased=1)
        if linewidth is not None:
            axe.contour(self.ds_topo.lon, self.ds_topo.lat, self.ds_topo.TOPO*1e2, levels=np.array([499.99, 500.01]), 
                        colors=color, linewidths=linewidth)
        else:
            pass
        return imtopoh
        
    def Plot_twcwb_pcp(self, axe, 
                       lon, lat, precip,
                       s, alpha, precip_bounds=[1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300], edgecolors=None):
        bounds= np.array(precip_bounds)
        cmap  = cmaps.WhiteBlueGreenYellowRed
        norm  = colors.BoundaryNorm(bounds, cmap.N, extend='both')
        im    = axe.scatter(lon, lat, c=precip, s=s, 
                            cmap=cmap, norm=norm, alpha=alpha, edgecolors=edgecolors)
        return im
    
plottools = Plot_analysis()

# Plot daily precip.
def Plot_cwa_dpcp(date:str, regional_rect=False, figtitle=False, savefig=False, figname=False):
    # Data initialize
    pcp_table = wtab_module_test.get_cwb_precip_table(date, accumulate_daily=True)
    bounds    = np.array([1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300])
    # Figure initialize
    fig = plt.figure(figsize=(4, 8))
    gs  = GridSpec(1, 1, figure=fig)
    ## Plot:
    ax1   = plottools.Axe_map(fig, gs[0], xlim_=[119.95, 122.05], ylim_=[21.85, 25.5], 
                              xloc_=np.arange(120, 122.1, 1), yloc_=np.arange(22, 25.1, 1))
    plottools.Plot_vvm_map(ax1, c['dark grey'], 1.5)  # coastline
    imcwb = plottools.Plot_twcwb_pcp(ax1, pcp_table['stn_lon'], pcp_table['stn_lat'], pcp_table['precip'], s=70, alpha=0.8, precip_bounds=bounds)
    cax  = fig.add_axes([ax1.get_position().x1+0.28, ax1.get_position().y0, 0.03, ax1.get_position().height])
    cbar = fig.colorbar(imcwb, orientation='vertical', cax=cax)
    cbar.solids.set(alpha=1)  # Default: set cbar to full color (w/out tranparency)
    cbar.set_ticks(ticks=bounds, labels=[int(i) for i in bounds])
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('mm/day', fontsize=16)
    cbar.outline.set_linewidth(1.5)
    
    topo_bounds= np.arange(0, 3500.1, 500)
    imtopoh    = plottools.Plot_vvm_topo(ax1, c['dark grey'], 0.5)
    # cax  = fig.add_axes([ax1.get_position().x0,ax1.get_position().y0-0.06, ax1.get_position().width, 0.015])
    # cbar = fig.colorbar(imtopoh, orientation='horizontal', cax=cax, extend='max')
    cax  = fig.add_axes([ax1.get_position().x1+0.04, ax1.get_position().y0, 0.03, ax1.get_position().height])
    cbar = fig.colorbar(imtopoh, orientation='vertical', cax=cax)
    cbar.set_ticks(ticks=topo_bounds, labels=[f"{i/1e3:.1f}" for i in topo_bounds])
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('Topography Height (km)', fontsize=16)
    cbar.outline.set_linewidth(1.5)
    
    ax1.set_xticklabels([f'{int(i)}E' for i in np.arange(120, 122.1, 1)], fontsize=16)
    ax1.set_yticklabels([f'{int(i)}N' for i in np.arange(22, 25.1, 1)], fontsize=16)
    if regional_rect:
        rect = patches.Rectangle((regional_rect[0], regional_rect[2]), regional_rect[1]-regional_rect[0], regional_rect[3]-regional_rect[2], 
                                 fc="none", ec=c['black'], linewidth=2, zorder=50)
        ax1.add_artist(rect)
    if figtitle:
        ax1.set_title(f'{figtitle}', loc='left', fontsize=20)
    else:
        ax1.set_title('CWA precipitation', fontsize=16)
        fig.suptitle(x=0.5, y=0.91, t=date, fontsize=18, fontweight='bold')
    if savefig:
        if figname:
            plt.savefig(f'../../Figure/{figname}_{date}.png', facecolor='w', bbox_inches='tight', dpi=400)
        else:
            plt.savefig(f'../../Figure/{date}.png', facecolor='w', bbox_inches='tight', dpi=400)
    else:
        plt.show()
Plot_cwa_dpcp('20180828', regional_rect=(120, 121, 21.9, 23.5), figtitle='(c) Daily Precip.', savefig=True, figname='Fig3c')
Plot_cwa_dpcp('20140714', figtitle='(a) Weak - 20140714', savefig=True, figname='Fig6a')
Plot_cwa_dpcp('20120819', figtitle='(b) Weak - 20120819', savefig=True, figname='Fig6b')
Plot_cwa_dpcp('20150828', regional_rect=(120, 121, 21.9, 23.5), figtitle='(c) Strong - 20150828', savefig=True, figname='Fig6c')
Plot_cwa_dpcp('20180828', regional_rect=(120, 121, 21.9, 23.5), figtitle='(d) Strong - 20180828', savefig=True, figname='Fig6d')
Plot_cwa_dpcp('20190815', regional_rect=(120, 121, 21.9, 23.5), figtitle='(e) Strong - 20190815', savefig=True, figname='Fig6e')
Plot_cwa_dpcp('20160610', regional_rect=(120, 121, 21.9, 23.5), figtitle='(a) Strong - 20160610', savefig=True, figname='FigS6a')
Plot_cwa_dpcp('20140810', regional_rect=(120, 121, 21.9, 23.5), figtitle='(b) Strong - 20140810', savefig=True, figname='FigS6b')
Plot_cwa_dpcp('20170602', regional_rect=(120, 121, 21.9, 23.5), figtitle='(c) Strong - 20170602', savefig=True, figname='FigS6c')

# Plot composite
def Cal_cwa_dpcp_comp(datelist:list, pcp_filter_min=0.):
    for i, dd in enumerate(datelist):
        # Data initialize
        if i < 1:
            pcp_table = wtab_module_all.get_cwb_precip_table(str(dd), accumulate_daily=True)
            pcp_comp  = pcp_table['precip'].values[:, np.newaxis]
        else:
            temp = wtab_module_all.get_cwb_precip_table(str(dd), accumulate_daily=True)['precip'].values
            pcp_comp  = np.concatenate((pcp_comp, temp[:, np.newaxis]), axis=1)
    print(pcp_comp.shape)
    daily_max = np.nanmax(pcp_comp, axis=0)
    mask      = daily_max>pcp_filter_min
    print(pcp_comp[..., mask].shape)
    mean_pcp  = np.nanmean(pcp_comp[..., mask], axis=1)
    return pcp_table['stn_lon'], pcp_table['stn_lat'], mean_pcp

# Strongly-forced composite
wtab_all      = Read_weather_table(data_label='all')
cond_lv0_all  = (wtab_all['ERA5_all_lv0']>=-4)&(wtab_all['ERA5_all_lv0']<2)
cond_lv1_all  = (wtab_all['ERA5_all_lv1']>=-4)&(wtab_all['ERA5_all_lv1']<0)
# My definition -------------------
updict_6reg_all = pd.read_csv('../../TemporaryData/weather_table_self/all_polar.csv')
wtab_all_6reg = wtab_all[cond_lv0_all&cond_lv1_all].copy()
wtab_all_6reg['IVT dir. (rad.)'] = updict_6reg_all['IVT_theta'].where(updict_6reg_all['IVT_theta']>=0, updict_6reg_all['IVT_theta']+np.pi*2)
mysswf_cond    = (wtab_all_6reg['IVT']>=250)&(wtab_all_6reg['IVT dir. (rad.)']>=200*np.pi/180)&(wtab_all_6reg['IVT dir. (rad.)']<=280*np.pi/180)
dlist_6reg_sswf  = wtab_all_6reg[mysswf_cond]['yyyymmdd'].to_list()
dlist_6reg_nosswf= wtab_all_6reg[~mysswf_cond]['yyyymmdd'].to_list()
stn_lon, stn_lat, mean_pcp_6reg_sswf   = Cal_cwa_dpcp_comp(datelist=dlist_6reg_sswf)
stn_lon, stn_lat, mean_pcp_6reg_nosswf = Cal_cwa_dpcp_comp(datelist=dlist_6reg_nosswf)

def Plot_cwb_comp(rainfall_label:str, precip_array, figtitle=False, figname=False):
    if rainfall_label=='strong':
        bounds    = np.array([2, 5, 10, 15, 25, 40, 50, 70, 90])
        alpha     = 0.7
    elif rainfall_label=='weak':
        bounds    = np.array([2, 5, 10, 15, 25])
        alpha     = 0.75
    # Figure initialize
    fig = plt.figure(figsize=(4, 8))
    gs  = GridSpec(1, 1, figure=fig)
    ## Plot:
    ax1   = plottools.Axe_map(fig, gs[0], xlim_=[119.95, 122.05], ylim_=[21.85, 25.5], 
                              xloc_=np.arange(120, 122.1, 1), yloc_=np.arange(22, 25.1, 1))
    plottools.Plot_vvm_map(ax1, c['dark grey'], 1.5)  # coastline
    temp = np.argsort(precip_array)
    imcwb = plottools.Plot_twcwb_pcp(ax1, stn_lon[temp], stn_lat[temp], precip_array[temp], s=75, alpha=alpha, precip_bounds=bounds)
    cax  = fig.add_axes([ax1.get_position().x1+0.06, ax1.get_position().y0, 0.03, ax1.get_position().height])
    cbar = fig.colorbar(imcwb, orientation='vertical', cax=cax)
    cbar.solids.set(alpha=1)  # Default: set cbar to full color (w/out tranparency)
    cbar.set_ticks(ticks=bounds, labels=[int(i) for i in bounds])
    cbar.set_label('mm/day', fontsize=16)
    cbar.ax.tick_params(labelsize=13)
    cbar.outline.set_linewidth(1.5)

    topo_bounds= np.arange(0, 3500.1, 500)
    imtopoh    = plottools.Plot_vvm_topo(ax1, c['dark grey'], 0.5)
    cax  = fig.add_axes([ax1.get_position().x0,ax1.get_position().y0-0.06, ax1.get_position().width, 0.015])
    cbar = fig.colorbar(imtopoh, orientation='horizontal', cax=cax, extend='max')
    cbar.set_ticks(ticks=topo_bounds, labels=[f"{i/1e3:.1f}" for i in topo_bounds])
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('Topography Height (km)', fontsize=16)
    cbar.outline.set_linewidth(1.5)

    ax1.set_xticklabels([f'{int(i)}E' for i in np.arange(120, 122.1, 1)], fontsize=16)
    ax1.set_yticklabels([f'{int(i)}N' for i in np.arange(22, 25.1, 1)], fontsize=16)
    if figtitle:
        ax1.set_title(f'{figtitle}', loc='left', fontsize=20)
    else:
        ax1.set_title('CWA precipitation', fontsize=16)
    if figname:
        plt.savefig(f'../../Figure/{figname}.png', facecolor='w', bbox_inches='tight', dpi=400)
    else:
        plt.show()
Plot_cwb_comp(rainfall_label='strong', precip_array=mean_pcp_6reg_sswf, figtitle='(e) Obs. (strong)', figname='Fig5e')
Plot_cwb_comp(rainfall_label='weak', precip_array=mean_pcp_6reg_nosswf, figtitle='(b) Obs. (weak)', figname='Fig5b')