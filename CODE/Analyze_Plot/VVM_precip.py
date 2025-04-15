import numpy as np
import xarray as xr
import glob
import pandas as pd
from datetime import *
from vvmtools.analyze import DataRetriever
# --- self define modules --- #
import importlib
import PlotTools
importlib.reload(PlotTools)import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
import seaborn.colors.xkcd_rgb as c
from matplotlib.gridspec import GridSpec

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
wtab_all = pd.read_csv('../../TemporaryData/weather_table_self/all_withlv.csv')
plottools= PlotTools.PlotTools_vvm()

def cal_vvm_dpcp(case:str, date:str, swsuffix:str='', domain_range:tuple=(None, None, None, None), land_only:bool=True):
    # Check
    print(f"Calculating daily precipitation. The Land-only option is now: {land_only}.")
    # Calculation
    j1, j2, i1, i2 = domain_range
    if case == 'sw':
        vvmtools_date  = DataRetriever(f"../TemporaryData/taiwanvvm_sw/case_sw_{date}{swsuffix}")  # data available upon request
    elif case == 'at':
        vvmtools_date  = DataRetriever(f"../TemporaryData/taiwanvvm_at/tpe{date}nor")
    else:
        raise ValueError("Argument `case` should be either 'sw' or 'at'.")
    arr_pcp_all    = vvmtools_date.get_var_parallel(var='sprec', time_steps=np.arange(145))
    arr_day_pcp    = np.sum(arr_pcp_all, axis=0)*3600/6
    if land_only:
        arr_day_pcp = np.where(plottools.ds_topo.topo>0, arr_day_pcp, np.nan)
    return arr_day_pcp[j1:j2, i1:i2]

# Daily Precip.
def Plot_vvm_dpcp(case:str, date:str, swsuffix:str='', regional_rect=False, land_only:bool=False, figtitle=False, figname=False):
    # Data initialize
    arr_day_pcp = cal_vvm_dpcp(case=case, date=date, swsuffix=swsuffix, land_only=land_only)
    # Figure initialize
    fig = plt.figure(figsize=(4, 8))
    gs  = GridSpec(1, 1, figure=fig)
    bounds = np.array([1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300])
    # ----------------------------------------------------------------------------------------
    # Plot
    ax1   = plottools.Axe_map(fig, gs[0], xlim_=[119.95, 122.05], ylim_=[21.85, 25.5],
                                    xloc_=np.arange(120, 122.1, 1), yloc_=np.arange(22, 25.1, 1))
    plottools.Plot_vvm_map(ax1, c['dark grey'], 1.5)  # coastline
    ## precip.
    imvvm = plottools.Plot_vvm_pcp(ax1, arr_day_pcp, transform_option='map')
    cax  = fig.add_axes([ax1.get_position().x1+0.28, ax1.get_position().y0, 0.03, ax1.get_position().height])
    cbar = fig.colorbar(imvvm, orientation='vertical', cax=cax)
    cbar.solids.set(alpha=1)  # Default: set cbar to full color (w/out tranparency)
    cbar.set_ticks(ticks=bounds, labels=[int(i) for i in bounds])
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label('mm/day', fontsize=16)
    cbar.outline.set_linewidth(1.5)
    ## topo
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
        ax1.set_title('VVM precipitation', fontsize=16)
        fig.suptitle(x=0.5, y=0.91, t=date, fontsize=18, fontweight='bold')
    if figname:
        if land_only:
            plt.savefig(f'../../Figure/{figname}_{date}_landonly.png', facecolor='w', bbox_inches='tight', dpi=400)
        else:
            plt.savefig(f'../../Figure/{figname}_{date}.png', facecolor='w', bbox_inches='tight', dpi=400)
    else:
        plt.show()
Plot_vvm_dpcp(case='at', date=20140714, land_only=True, figname='Fig6avvm')
Plot_vvm_dpcp(case='at', date=20120819, land_only=True, figname='Fig6bvvm')
Plot_vvm_dpcp(case='sw', date=20150828, land_only=True, regional_rect=(120, 121, 21.9, 23.5), figname='Fig6cvvm')
Plot_vvm_dpcp(case='sw', date=20180828, land_only=True, regional_rect=(120, 121, 21.9, 23.5), figname='Fig6dvvm')
Plot_vvm_dpcp(case='sw', date=20190815, land_only=True, regional_rect=(120, 121, 21.9, 23.5), figname='Fig6evvm')
Plot_vvm_dpcp(case='sw', date=20160610, swsuffix='_sim6', land_only=True, regional_rect=(120, 121, 21.9, 23.5), figname='FigS6avvm')
Plot_vvm_dpcp(case='sw', date=20140810, land_only=True, regional_rect=(120, 121, 21.9, 23.5), figname='FigS6bvvm')
Plot_vvm_dpcp(case='sw', date=20170602, land_only=True, regional_rect=(120, 121, 21.9, 23.5), figname='FigS6cvvm')

# Composite
def Cal_vvm_dpcp_comp(case:str, datelist:list, land_only:bool=False):
    # Check for available simulation
    if case not in ('sw', 'at'):
        raise ValueError("Argument `case` should be either 'sw' or 'at'.")
    # Calculation
    counter = 1
    for i, dd in enumerate(datelist):
        if dd == '20180815':
            dd = '20140815'  # rename date (this one is a typo)
        # Data initialize
        if i < 1:
            arr_day_pcp = cal_vvm_dpcp(case=case, date=dd, land_only=land_only)[..., np.newaxis]
        else:
            try:
                temp    = cal_vvm_dpcp(case=case, date=dd, land_only=land_only)[..., np.newaxis]
            except:
                temp    = cal_vvm_dpcp(case=case, date=dd, swsuffix=f'_sim{counter}', land_only=land_only)[..., np.newaxis]
                counter+= 1
            arr_day_pcp = np.concatenate((arr_day_pcp, temp), axis=-1)
    mean_pcp = np.nanmean(arr_day_pcp, axis=-1)
    return mean_pcp

def Plot_vvm_dpcp_comp(case:str, mean_pcp:np.ndarray, land_only:bool=False, figtitle=False, figname=False):
    # Data initialize
    if land_only:
        mean_pcp = np.where(plottools.ds_topo.topo>0, mean_pcp, np.nan)
    # Figure initialize
    fig = plt.figure(figsize=(4, 8))
    gs  = GridSpec(1, 1, figure=fig)
    if case == 'sw':
        bounds    = np.array([2, 5, 10, 15, 25, 40, 50, 70, 90])
    elif case == 'at':
        bounds    = np.array([2, 5, 10, 15, 25])
    else:
        raise ValueError("Argument `case` should be either 'sw' or 'at'.")
    # ----------------------------------------------------------------------------------------
    # Plot
    ax1   = plottools.Axe_map(fig, gs[0], xlim_=[119.95, 122.05], ylim_=[21.85, 25.5],
                                    xloc_=np.arange(120, 122.1, 1), yloc_=np.arange(22, 25.1, 1))
    plottools.Plot_vvm_map(ax1, c['dark grey'], 1.5)  # coastline
    ## precip.
    imvvm = plottools.Plot_vvm_pcp(ax1, mean_pcp, bounds=bounds, transform_option='map')
    cax  = fig.add_axes([ax1.get_position().x1+0.06, ax1.get_position().y0, 0.03, ax1.get_position().height])
    cbar = fig.colorbar(imvvm, orientation='vertical', cax=cax)
    cbar.solids.set(alpha=1)  # Default: set cbar to full color (w/out tranparency)
    cbar.set_label('mm/day', fontsize=16)
    cbar.ax.tick_params(labelsize=13)
    cbar.outline.set_linewidth(1.5)
    ## topo
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
        ax1.set_title('VVM precipitation', fontsize=16)
    if figname:
        if land_only:
            plt.savefig(f'../../Figure/{figname}_landonly.png', facecolor='w', bbox_inches='tight', dpi=400)
        else:
            plt.savefig(f'../../Figure/{figname}.png', facecolor='w', bbox_inches='tight', dpi=400)
    else:
        plt.show()
        
# Weakly-forced cases
meanpcp_atsim_6reg = Cal_vvm_dpcp_comp(case='at', datelist=dlist_atsim32)
Plot_vvm_dpcp_comp(case='at', mean_pcp=meanpcp_atsim_6reg, land_only=True, figtitle='(c) TaiwanVVM', figname='FigS5cvvm')
# Strongly-forced cases
meanpcp_swsim_6reg = Cal_vvm_dpcp_comp(case='sw', datelist=dlist_swsim)
Plot_vvm_dpcp_comp(case='sw', mean_pcp=meanpcp_swsim_6reg, land_only=True, figtitle='(f) TaiwanVVM', figname='FigS5fvvm')