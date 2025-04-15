import numpy as np
import pandas as pd
import xarray as xr
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

# Utilities
class PlotTools_cartopy():
    def __init__(self):
        self.proj = ccrs.PlateCarree()
    
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
    
plottools_ctpy = PlotTools_cartopy()

def Plot_scatter(axe, x_, y_, c_=None, color_=None, 
                 s_=None, alpha_=None, label_=None, 
                 cmap_=None, bounds_=None, norm_=None, extend_='neither'):
    """
    Create scatter plot with some personal settings.
    """
    if bounds_ is not None:
        norm_    = colors.BoundaryNorm(bounds_, cmap_.N, extend=extend_)
    im = axe.scatter(x_, y_, c=c_, color=color_, s=s_, alpha=alpha_, 
                     label=label_, cmap=cmap_, norm=norm_)
    return im

def draw_cat_bound(axe):
    axe.hlines(y=2, xmin=-6, xmax=-3.89, linewidths=5, colors='k')
    axe.hlines(y=0, xmin=-4.08, xmax=6, linewidths=5, colors='k')
    axe.vlines(x=-4, ymin=-0.12, ymax=2.12, linewidths=5, colors='k')
    axe.vlines(x=-2, ymin=-6, ymax=0, linewidths=5, colors='k')
    
# AAM summer mean precip. map
class AAM_mean_precip_map():
    def __init__(self, year_list:list, month_list:list, 
                 special_start_date:str=False, special_end_date:str=False):
        self.YEARS     = year_list
        self.MONTHS    = month_list
        self.STARTDATE = self._convert_to_dobj(special_start_date) if special_start_date else None
        self.ENDDATE   = self._convert_to_dobj(special_end_date) if special_end_date else None
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
        # Addtionally, extract dates in selected months
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
    
    def get_AAM_precip(self, date:str):
        # Load gpm file
        gpm_date = xr.open_dataset(f"/GPM/AsianMonsoon/{date[:4]}/{date[4:6]}/{date}.nc")  # replace with IMERG daily-accumulated precipitation
        # Masked precip
        gpm_aam  = gpm_date.daily_precip.sel(lon=slice(66, 156), lat=slice(-15, 30))
        return gpm_aam
    
    def cal_mean_pcp(self, datelist:list=[]):
        # Initiate dict with datelist entry
        datelist   = datelist if len(datelist)>0 else self.DLIST_Month
        # Average pcp
        daily_pcps = self._cal_for_all_dates(datelist=datelist,
                                             func=self.get_AAM_precip,
                                             cores=10)
        precip_stack= xr.concat(daily_pcps, dim='time')
        final_mean  = precip_stack.mean(dim='time', skipna=True)
        return precip_stack, final_mean
    
aam_pcp = AAM_mean_precip_map(year_list=np.arange(2001, 2020).tolist(), month_list=np.arange(4, 10).tolist())
aam_pcp_stack, aam_pcp_mean = aam_pcp.cal_mean_pcp()

def Plot_AAM_precip_map(figtitle=False, figname=False):
    fig = plt.figure(figsize=(16, 10))
    gs  = GridSpec(1, 1, figure=fig)
    ax  = plottools_ctpy.Axe_map(fig, gs[0], xlim_=[65.95, 155.05], ylim_=[-15.05, 30], 
                                 xloc_=np.arange(90, 160.1, 30), yloc_=np.arange(-10, 30.1, 10))
    plottools_ctpy.Plot_cartopy_map(ax)
    
    # Prepare topography
    global_z = xr.open_dataset('../../TemporaryData/ERA5Land_surf_geopotential.nc')
    AAM_z    = global_z.z.sel(longitude=slice(66, 155), latitude=slice(30, -15)).isel(time=0)
    AAM_topo = AAM_z/9.8
    # Prepare mask
    gpm_temp   = xr.open_dataset("GPM/AsianMonsoon/2001/07/20010701.nc")   # replace with single IMERG file
    mask_empty = np.zeros(gpm_temp.daily_precip.shape)
    # Mask of India Ghats
    mask_india = mask_empty.copy()
    mask_india[387:411, 126:141] = 1
    mask_india[375:387, 127:143] = 1
    mask_india[363:375, 130:146] = 1
    mask_india[351:363, 133:151] = 1
    mask_india[339:351, 140:157] = 1
    mask_india[327:339, 143:161] = 1
    mask_india[315:327, 148:166] = 1
    mask_india[303:315, 155:173] = 1
    mask_india[280:303, 162:176] = 1
    # Mask of BoB
    mask_bob   = mask_empty.copy()
    mask_bob[423:435, 310:331] = 1
    mask_bob[411:423, 313:334] = 1
    mask_bob[399:411, 318:339] = 1
    mask_bob[387:399, 330:351] = 1
    mask_bob[375:387, 336:357] = 1
    mask_bob[360:375, 335:356] = 1
    # Mask of Luzon
    mask_luzon = mask_empty.copy()
    mask_luzon[375:387, 598:619] = 1
    mask_luzon[363:375, 595:616] = 1
    mask_luzon[339:363, 590:611] = 1
    # Mask of Sumatra
    mask_smt   = mask_empty.copy()
    mask_smt[245:257, 340:356] = 1
    mask_smt[239:245, 347:363] = 1
    mask_smt[230:239, 355:371] = 1
    mask_smt[222:230, 360:376] = 1
    mask_smt[212:222, 370:387] = 1
    mask_smt[202:212, 373:390] = 1
    mask_smt[196:202, 380:399] = 1
    mask_smt[186:196, 382:404] = 1
    mask_smt[176:186, 386:408] = 1
    mask_smt[166:176, 395:416] = 1
    mask_total = mask_india+mask_bob+mask_luzon+mask_smt
    
    # Mask hatch
    none_cmap = colors.ListedColormap(['none'])
    ax.pcolor(gpm_temp.lon, gpm_temp.lat, np.where(mask_total<1, np.nan, mask_total), shading='nearest', hatch='xxxx', 
              cmap=none_cmap, edgecolor=c['rich blue'], lw=0, zorder=20)
    # Precipitation
    bounds = [1]+np.arange(2.5, 25.1, 2.5).tolist()
    cmap   = cmaps.WhiteBlueGreenYellowRed.copy()
    cmap.set_under('w')
    norm   = colors.BoundaryNorm(bounds, cmap.N, extend='both')
    im_aampcp = ax.pcolormesh(aam_pcp_mean.lon, aam_pcp_mean.lat, aam_pcp_mean, cmap=cmap, norm=norm, alpha=0.85)
    cax1  = fig.add_axes([ax.get_position().x0+ax.get_position().width/2+0.01, ax.get_position().y0-0.06, ax.get_position().width/2-0.01, 0.015])
    cbar1 = fig.colorbar(im_aampcp, orientation='horizontal', cax=cax1)
    cbar1.solids.set(alpha=1)
    cbar1.set_ticks(ticks=bounds[::2], labels=[int(i) for i in bounds[::2]])
    cbar1.ax.tick_params(labelsize=14)
    cbar1.set_label('Precip. (mm/day)', fontsize=14)
    cbar1.outline.set_linewidth(1.5)
    # Topography
    topo_bounds= np.arange(0, 5000.1, 500)
    alpha_list = [0]+np.linspace(0.35, 1, 10).tolist()
    cmap_topo  = colors.ListedColormap([(92/256, 73/256, 19/256, i) for i in alpha_list])
    norm_      = colors.BoundaryNorm(topo_bounds, cmap_topo.N, extend='max')
    imtopoh    = ax.contourf(AAM_topo.longitude, AAM_topo.latitude, AAM_topo, 
                             levels=topo_bounds, 
                             cmap=cmap_topo, norm=norm_, extend='max', antialiased=1)
    cstopoh    = ax.contour(AAM_topo.longitude, AAM_topo.latitude, AAM_topo, levels=[500], colors=[(102/256, 68/256, 25/256)], linewidths=0.8)
    cax2  = fig.add_axes([ax.get_position().x0, ax.get_position().y0-0.06, ax.get_position().width/2-0.01, 0.015])
    cbar2 = fig.colorbar(imtopoh, orientation='horizontal', cax=cax2)
    cbar2.set_ticks(ticks=topo_bounds[::2], labels=[int(i/1e3) for i in topo_bounds[::2]])
    cbar2.ax.tick_params(labelsize=14)
    cbar2.set_label('Topo. Height (km)', fontsize=14)
    cbar2.outline.set_linewidth(1.5)
    # SW Taiwan
    rect = patches.Rectangle((120, 21.9), 1, 1.6, fc="none", ec=c['black'], linewidth=1.5, zorder=50)
    ax.add_artist(rect)
    # Texts
    ax.text(x=69.5, y=18.5, s='(b)', fontsize=16, fontweight='bold')
    ax.text(x=87.5, y=22.5, s='(c)', fontsize=16, fontweight='bold')
    ax.text(x=116, y=17, s='(d)', fontsize=16, fontweight='bold')
    ax.text(x=91, y=4, s='(e)', fontsize=16, fontweight='bold')
    ax.text(x=122.5, y=20, s='Southwestern\nTaiwan', fontsize=16, fontweight='bold')
    # Other settings
    ax.set_title(f'{figtitle}', loc='left', fontsize=16)
    ax.set_xticklabels([f'{int(i)}E' for i in np.arange(90, 160.1, 30)], fontsize=14)
    ax.set_yticklabels([f'{int(i)}N' if i>=0 else f'{int(i*(-1))}S' for i in np.arange(-10, 30.1, 10)], fontsize=14)
    plt.savefig(f'../../Figure/{figname}.png', facecolor='w', bbox_inches='tight', dpi=400)
Plot_AAM_precip_map(figtitle='(a) Daily Precipitation (Apr.-Sep. Average, 2001-2019)', figname='Fig7a')

# Latent space projection
class precip_table():
    def __init__(self, mask_region_label:str, 
                 year_list:list, month_list:list, 
                 special_start_date:str=False, special_end_date:str=False):
        # Input arguments
        self.REGIONLIST= ['india', 'bob', 'luzon', 'sumatra']
        if mask_region_label in self.REGIONLIST:
            self.REGION    = mask_region_label  
            self.YEARS     = year_list
            self.MONTHS    = month_list
            self.STARTDATE = self._convert_to_dobj(special_start_date) if special_start_date else None
            self.ENDDATE   = self._convert_to_dobj(special_end_date) if special_end_date else None
            self._create_date_list()
            # Mask template
            ds_gpm           = xr.open_dataset("/data/ch995334/DATA/GPM/AsianMonsoon/2001/07/20010701.nc")
            self._mask_empty = np.zeros(ds_gpm.daily_precip.shape)
            self.mask_reg    = self._self_defined_regional_mask()
        else: raise ValueError(f"Inavailable region. Please choose from {self.REGIONLIST}")
        
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
        # Addtionally, extract dates in selected months
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
            
    def _self_defined_regional_mask(self):
        mask_reg = self._mask_empty.copy()   # Initialize
        if self.REGION == 'india':
            mask_reg[387:411, 126:141] = 1
            mask_reg[375:387, 127:143] = 1
            mask_reg[363:375, 130:146] = 1
            mask_reg[351:363, 133:151] = 1
            mask_reg[339:351, 140:157] = 1
            mask_reg[327:339, 143:161] = 1
            mask_reg[315:327, 148:166] = 1
            mask_reg[303:315, 155:173] = 1
            mask_reg[280:303, 162:176] = 1
        elif self.REGION == 'bob':
            mask_reg[423:435, 310:331] = 1
            mask_reg[411:423, 313:334] = 1
            mask_reg[399:411, 318:339] = 1
            mask_reg[387:399, 330:351] = 1
            mask_reg[375:387, 336:357] = 1
            mask_reg[360:375, 335:356] = 1
        elif self.REGION == 'luzon':
            mask_reg[375:387, 598:619] = 1
            mask_reg[363:375, 595:616] = 1
            mask_reg[339:363, 590:611] = 1
        elif self.REGION == 'sumatra':
            mask_reg[245:257, 340:356] = 1
            mask_reg[239:245, 347:363] = 1
            mask_reg[230:239, 355:371] = 1
            mask_reg[222:230, 360:376] = 1
            mask_reg[212:222, 370:387] = 1
            mask_reg[202:212, 373:390] = 1
            mask_reg[196:202, 380:399] = 1
            mask_reg[186:196, 382:404] = 1
            mask_reg[176:186, 386:408] = 1
            mask_reg[166:176, 395:416] = 1
            
        return mask_reg    
    
    def get_masked_precip(self, date:str):
        # Load gpm file
        gpm_date = xr.open_dataset(f"/data/ch995334/DATA/GPM/AsianMonsoon/{date[:4]}/{date[4:6]}/{date}.nc")
        # Masked precip
        gpm_mask = gpm_date.daily_precip.where(self.mask_reg)
        return gpm_mask
    
    def cal_mean_dpcp(self, date:str):
        gpm_mask = self.get_masked_precip(date)
        return np.nanmean(gpm_mask)
    
    def cal_top10mean_dpcp(self, date:str):
        gpm_mask = self.get_masked_precip(date).values
        top10    = np.sort(gpm_mask[~np.isnan(gpm_mask)])[-10:]
        return np.nanmean(top10)
    
    def cal_max_dpcp(self, date:str):
        gpm_mask = self.get_masked_precip(date)
        return np.nanmax(gpm_mask)
    
    def cal_std_dpcp(self, date:str):
        gpm_mask = self.get_masked_precip(date)
        return np.nanstd(gpm_mask)
    
    def getdf_metrics_maskpcp(self, datelist:list=[], Mean=False, Top10_mean=False, Max=False, Std=False):        
        # Initiate dict with datelist entry
        datelist = datelist if len(datelist)>0 else self.DLIST_Month
        pcp_table= {'yyyymmdd':datelist}
        # Add other entries
        if Mean:
            mean = self._cal_for_all_dates(datelist=datelist, func=self.cal_mean_dpcp)
            pcp_table['mean'] = mean
        if Top10_mean:
            top10= self._cal_for_all_dates(datelist=datelist, func=self.cal_top10mean_dpcp)
            pcp_table['top10_mean'] = top10
        if Max:
            _max = self._cal_for_all_dates(datelist=datelist, func=self.cal_max_dpcp)
            pcp_table['max']  = _max
        if Std:
            std  = self._cal_for_all_dates(datelist=datelist, func=self.cal_std_dpcp)
            pcp_table['std']  = std
        return pd.DataFrame(pcp_table)
# Save masked precip. metrics
# India
ptable_india_all  = precip_table(mask_region_label='india', 
                                 year_list=np.arange(2001, 2020).tolist(), month_list=np.arange(4, 10).tolist())
df_india_all      = ptable_india_all.getdf_metrics_maskpcp(Mean=True, Top10_mean=True, Max=True, Std=True)
# df_india_all.to_csv('../TemporaryData/weather_table_self/india_pcp_all.csv', index=False)
df_india_all_wattr= df_india_all.copy()
df_india_all_wattr.attrs["mask_reg_array"] = ptable_india_all.mask_reg
# df_india_all_wattr.to_pickle("../TemporaryData/weather_table_self/india_pcp_all_wattr.pkl")
# BoB
ptable_bob_all  = precip_table(mask_region_label='bob', 
                               year_list=np.arange(2001, 2020).tolist(), month_list=np.arange(4, 10).tolist())
df_bob_all      = ptable_bob_all.getdf_metrics_maskpcp(Mean=True, Top10_mean=True, Max=True, Std=True)
# df_bob_all.to_csv('../TemporaryData/weather_table_self/bob_pcp_all.csv', index=False)
df_bob_all_wattr= df_bob_all.copy()
df_bob_all_wattr.attrs["mask_reg_array"] = ptable_bob_all.mask_reg
# df_bob_all_wattr.to_pickle("../TemporaryData/weather_table_self/bob_pcp_all_wattr.pkl")
# Luzon
ptable_luzon_all  = precip_table(mask_region_label='luzon', 
                                 year_list=np.arange(2001, 2020).tolist(), month_list=np.arange(4, 10).tolist())
df_luzon_all      = ptable_luzon_all.getdf_metrics_maskpcp(Mean=True, Top10_mean=True, Max=True, Std=True)
# df_luzon_all.to_csv('../TemporaryData/weather_table_self/luzon_pcp_all.csv', index=False)
df_luzon_all_wattr= df_luzon_all.copy()
df_luzon_all_wattr.attrs["mask_reg_array"] = ptable_luzon_all.mask_reg
# df_luzon_all_wattr.to_pickle("../TemporaryData/weather_table_self/luzon_pcp_all_wattr.pkl")
# Sumatra
ptable_sumatra_all  = precip_table(mask_region_label='sumatra', 
                                   year_list=np.arange(2001, 2020).tolist(), month_list=np.arange(4, 10).tolist())
df_sumatra_all      = ptable_sumatra_all.getdf_metrics_maskpcp(Mean=True, Top10_mean=True, Max=True, Std=True)
# df_sumatra_all.to_csv('../TemporaryData/weather_table_self/sumatra_pcp_all.csv', index=False)
df_sumatra_all_wattr= df_sumatra_all.copy()
df_sumatra_all_wattr.attrs["mask_reg_array"] = ptable_sumatra_all.mask_reg
# df_sumatra_all_wattr.to_pickle("../TemporaryData/weather_table_self/sumatra_pcp_all_wattr.pkl")

# Read saved masks
pcp_india_all   = pd.read_pickle('../../TemporaryData/weather_table_self/india_pcp_all_wattr.pkl')
pcp_bob_all     = pd.read_pickle('../../TemporaryData/weather_table_self/bob_pcp_all_wattr.pkl')
pcp_luzon_all   = pd.read_pickle('../../TemporaryData/weather_table_self/luzon_pcp_all_wattr.pkl')
pcp_sumatra_all = pd.read_pickle('../../TemporaryData/weather_table_self/sumatra_pcp_all_wattr.pkl')
wtab_all        = pd.read_csv(f"../../TemporaryData/weather_table_self/all_withlv.csv")
# Plot
def Plot_lv_regional_pcp(data_label:str, figtitle_dict:dict, value_info:str, figname=False):
    fig, ax = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True, gridspec_kw={'wspace':0.08})
    # Value levels and colormaps
    if value_info == 'std':
        bounds_  = np.arange(0, 50.1, 5)
        cmap_    = cmaps.MPL_RdYlBu_r
    else:
        bounds_  = np.array([1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130])
        cmap_    = cmaps.WhiteBlueGreenYellowRed.copy()
        cmap_.set_under((0.8, 0.8, 0.8, 0.7))
    # Dataframes (precip. and weather table)
    pcp_india    = eval(f"pcp_india_{data_label}")
    pcp_bob      = eval(f"pcp_bob_{data_label}")
    pcp_luzon    = eval(f"pcp_luzon_{data_label}")
    pcp_sumatra  = eval(f"pcp_sumatra_{data_label}")
    wtab_used    = eval(f"wtab_{data_label}")
    # Scatter plot
    """
    Addiontal adjustment on the indexing, in order to lay the larger values on lop.
    Specialization has been made to take care of the DataFrame data type (such as .values and iloc)
    """
    temp1        = np.argsort(pcp_india[value_info].values)
    imindia      = Plot_scatter(ax[0], x_=wtab_used[f"ERA5_{data_label}_lv0"].iloc[temp1], y_=wtab_used[f"ERA5_{data_label}_lv1"].iloc[temp1], 
                                c_=pcp_india[value_info].iloc[temp1], bounds_=bounds_, cmap_=cmap_, s_=100, alpha_=0.75, extend_='both') # changed from s=180
    
    temp2        = np.argsort(pcp_bob[value_info].values)
    imindia      = Plot_scatter(ax[1], x_=wtab_used[f"ERA5_{data_label}_lv0"].iloc[temp2], y_=wtab_used[f"ERA5_{data_label}_lv1"].iloc[temp2], 
                                c_=pcp_bob[value_info].iloc[temp2], bounds_=bounds_, cmap_=cmap_, s_=100, alpha_=0.75, extend_='both')
    
    temp3        = np.argsort(pcp_luzon[value_info].values)
    imluzon      = Plot_scatter(ax[2], x_=wtab_used[f"ERA5_{data_label}_lv0"].iloc[temp3], y_=wtab_used[f"ERA5_{data_label}_lv1"].iloc[temp3], 
                                c_=pcp_luzon[value_info].iloc[temp3], bounds_=bounds_, cmap_=cmap_, s_=100, alpha_=0.75, extend_='both')
    
    temp4        = np.argsort(pcp_sumatra[value_info].values)
    imsumatra    = Plot_scatter(ax[3], x_=wtab_used[f"ERA5_{data_label}_lv0"].iloc[temp4], y_=wtab_used[f"ERA5_{data_label}_lv1"].iloc[temp4], 
                                c_=pcp_sumatra[value_info].iloc[temp4], bounds_=bounds_, cmap_=cmap_, s_=100, alpha_=0.75, extend_='both')
    # Colorbar
    cax          = fig.add_axes([ax[-1].get_position().x1+0.01, ax[-1].get_position().y0, 0.008, ax[-1].get_position().height])
    cbar         = fig.colorbar(imsumatra, orientation='vertical', cax=cax)
    cbar.solids.set(alpha=1)
    cbar.set_ticks(ticks=bounds_, labels=bounds_.astype(int))
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(f'Precip. {value_info.title()} (mm/day)', fontsize=12)
    cbar.outline.set_linewidth(1.5)
    # Fig range and boundaries
    for Axe in ax.flatten():
        Axe.grid(linestyle=':', linewidth=1.2, color='k')     # regime boundary
        draw_cat_bound(Axe)                                   # category boundary
        Axe.set_aspect('equal', 'box')
        Axe.set_xlim(-6, 6)
        Axe.set_xticks(np.arange(-6, 6.1, 2))
        Axe.set_xticklabels([f'{int(ii)}' for ii in np.arange(-6, 6.1, 2)], fontsize=14)
        Axe.set_ylim(-6, 6)
        Axe.set_yticks(np.arange(-6, 6.1, 2))
        Axe.set_yticklabels([f'{int(ii)}' for ii in np.arange(-6, 6.1, 2)], fontsize=14)
        
    ax[0].set_xlabel('Latent Dim. 0', fontsize=14)
    ax[0].set_ylabel('Latent Dim. 1', fontsize=14)
    ax[0].set_title(f"(b) {figtitle_dict['title1']}", fontsize=16, loc='left')
    ax[1].set_title(f"(c) {figtitle_dict['title2']}", fontsize=16, loc='left')
    ax[2].set_title(f"(d) {figtitle_dict['title3']}", fontsize=16, loc='left')
    ax[3].set_title(f"(e) {figtitle_dict['title4']}", fontsize=16, loc='left')
    # Savefig or just show
    if figname:
        plt.savefig(f"../../Figure/{figname}.png", bbox_inches='tight', facecolor='w', dpi=400)
    else:
        plt.show()
Plot_lv_regional_pcp(data_label='all', 
                     figtitle_dict={'title1':'India (Western Ghats)', 'title2':'Eastern Coast of BoB', 'title3':'Western Luzon', 'title4': 'Eastern Sumatra'}, 
                     value_info='mean', figname='Fig7b')