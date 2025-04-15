import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
import seaborn.colors.xkcd_rgb as c
from matplotlib.gridspec import GridSpec
# --- self define modules --- #
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

# Precip. polar plot
def Plot_precip_polar(data_label:str, type_:str, figtitle:str, dotsize_ratio=1, figname=False):
    # Load data
    df_up_polar = pd.read_csv(f'/data/ch995334/SSWF_proj/TemporaryData/weather_table_self/{data_label}_polar.csv')
    df_up_polar['yyyymmdd'] = pd.to_datetime(df_up_polar['yyyymmdd'], format="%Y%m%d")
    wtab_used   = pd.read_csv(f"../../TemporaryData/weather_table_self/{data_label}_withlv.csv")
    swpcp_used  = pd.read_csv(f"../../TemporaryData/weather_table_self/swland_pcp_{data_label}.csv")
    twpcp_used  = pd.read_csv(f"../../TemporaryData/weather_table_self/taiwan_pcp_{data_label}.csv")
    swpcp_ratio = ((swpcp_used['accu'])/(twpcp_used['accu']))
    # 6 regime
    cond0       = (wtab_used['ERA5_all_lv0']>=-4)&(wtab_used['ERA5_all_lv0']<2)
    cond1       = (wtab_used['ERA5_all_lv1']>=-2)&(wtab_used['ERA5_all_lv1']<2)
    # Type settings
    if type_ == 'fraction':
        projcolors= np.where(np.isnan(swpcp_ratio[cond0&cond1]), 0, swpcp_ratio[cond0&cond1])
        bounds_  = np.arange(0.35, 0.91, 0.025)
        color    = ['w', '#048DC3', '#02723E', '#DCB345', '#C27425', '#A92009']
        nodes    = np.linspace(0,1,len(color))
        cmap_    = colors.LinearSegmentedColormap.from_list("cmap", list(zip(nodes, color)))
        cmap_.set_under((0.8, 0.8, 0.8, 0.7))
        size_bounds = [.4, .6, .7, .8]                               # Define bounds for size categories
        tick_bounds = bounds_[2::4]
        tick_labels = [f"{i:.1f}" for i in bounds_[2::4]]
        cbar_label  = ''
    elif type_ == 'mean':
        projcolors= swpcp_used[cond0&cond1]['mean'].values
        bounds_  = np.array([1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300])
        cmap_    = cmaps.WhiteBlueGreenYellowRed.copy()
        cmap_.set_under((0.8, 0.8, 0.8, 0.7))
        size_bounds = [15, 40, 100, 150]                               # Define bounds for size categories
        tick_bounds = bounds_
        tick_labels = bounds_
        cbar_label  = '(mm/day)'
    # Size settings
    dot_sizes = np.array([80, 100, 130, 160, 190])*dotsize_ratio  # Sizes corresponding to the bounds# Map color levels to dot sizes
    sizes = np.zeros_like(projcolors, dtype=int)
    for i, (low, high) in enumerate(zip(size_bounds[:-1], size_bounds[1:])):
        sizes[(projcolors >= low) & (projcolors < high)] = dot_sizes[i+1]
    sizes[projcolors <  size_bounds[0]] = dot_sizes[0]    # For the lowest category
    sizes[projcolors >= size_bounds[-1]] = dot_sizes[-1]  # For the highest category
    # Figure
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection':'polar'})
    ax.grid(linestyle=':',linewidth=1, color='grey')
    util_draw.draw_met_polar(ax)
    if type_ == 'mean':
        util_draw.draw_sswf_sector(axe=ax)
    norm_    = colors.BoundaryNorm(bounds_, cmap_.N, extend='both')
    temp1    = np.argsort(projcolors)
    im = ax.scatter(df_up_polar[cond0&cond1]['IVT_theta'].iloc[temp1], df_up_polar[cond0&cond1]['IVT_r'].iloc[temp1], s=sizes[temp1], c=projcolors[temp1], alpha=0.65, cmap=cmap_, norm=norm_)
    util_draw.draw_met_polar_rticks(axe=ax, rmin=0, rmax=800, 
                                    ticks=np.arange(200, 800.1, 200), ticklabels=[f"{int(i)}" for i in np.arange(200, 800.1, 200)], tick_fs=13)
    cax          = fig.add_axes([ax.get_position().x0, ax.get_position().y0-0.07, ax.get_position().width, 0.02])
    cbar         = fig.colorbar(im, orientation='horizontal', cax=cax)
    cbar.solids.set(alpha=1)
    cbar.set_ticks(ticks=tick_bounds, labels=tick_labels)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label(f'{cbar_label}', fontsize=13)
    cbar.outline.set_linewidth(1.5)
    ax.set_title(f'{figtitle}', loc='left', fontsize=18)
    if figname:
        plt.savefig(f'../../Figure/{figname}.png', facecolor='w', bbox_inches='tight', dpi=400)
Plot_precip_polar(data_label='all', type_='fraction', figtitle='(c) Precip. Fraction', dotsize_ratio=0.85, figname='Fig4c')
Plot_precip_polar(data_label='all', type_='mean', figtitle='(d) Precip. Mean', dotsize_ratio=0.85, figname='Fig4d')

# VVM simulation dates
def Plot_vvmsim_polar(sim:str, figtitle:str, figname=False):
    # Load data
    df_up_polar = pd.read_csv('../../TemporaryData/weather_table_self/all_polar.csv')
    df_up_polar['yyyymmdd'] = pd.to_datetime(df_up_polar['yyyymmdd'], format="%Y%m%d")
    # Figure
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection':'polar'})
    ax.grid(linestyle=':',linewidth=1, color='grey')
    util_draw.draw_met_polar(ax)
    util_draw.draw_sswf_sector(axe=ax)
    if sim == 'sw':
        dlist = pd.to_datetime(dlist_swsim, format="%Y%m%d")
        dcolor= c['bluish']
    elif sim == 'at':
        dlist = pd.to_datetime(dlist_atsim32, format="%Y%m%d")
        dcolor=c['burnt sienna']
    df_case = df_up_polar[df_up_polar['yyyymmdd'].isin(dlist)]
    ax.scatter(df_case['IVT_theta'], df_case['IVT_r'], s=150, color=dcolor, alpha=0.7)
    util_draw.draw_met_polar_rticks(axe=ax, rmin=0, rmax=800, 
                                    ticks=np.arange(200, 800.1, 200), ticklabels=[f"{int(i)}" for i in np.arange(200, 800.1, 200)], tick_fs=13)
    ax.set_title(f'{figtitle}', loc='left', fontsize=20)
    if figname:
        plt.savefig(f'../../Figure/{figname}_vvm_{sim}.png', facecolor='w', bbox_inches='tight', dpi=400)
    plt.show()
Plot_vvmsim_polar(sim='at', figtitle='(a) Upstream IVT (weakly-forced)', figname='Fig5atop')
Plot_vvmsim_polar(sim='sw', figtitle='(d) Upstream IVT (strongly-forced)', figname='Fig5dtop')

# Wind speed control
def Plot_IVTdir_polar_precip(IVT_dir_degree:float, tolerance_degree:float, figname:str=False):
    # Load data
    df_up_polar = pd.read_csv('../../TemporaryData/weather_table_self/all_polar.csv')
    df_up_polar['yyyymmdd'] = pd.to_datetime(df_up_polar['yyyymmdd'], format="%Y%m%d")
    temp   = df_up_polar['IVT_theta'].where(df_up_polar['IVT_theta']>=0, df_up_polar['IVT_theta']+np.pi*2)
    # Specified IVT direction range
    if (IVT_dir_degree-tolerance_degree) < 200:
        raise ValueError(f"Min. IVT direction should be >= 200 (degree). Current input: {IVT_dir_degree-tolerance_degree}")
    elif (IVT_dir_degree+tolerance_degree) > 280:
        raise ValueError(f"Max. IVT direction should be <= 280 (degree). Current input: {IVT_dir_degree+tolerance_degree}")
    else:
        dir_cond = (temp>=(IVT_dir_degree-tolerance_degree)*np.pi/180)&(temp<=(IVT_dir_degree+tolerance_degree)*np.pi/180)
    # Extract swpcp
    wtab_all  = pd.read_csv('../../TemporaryData/weather_table_self/all_withlv.csv')
    wtab_6reg = wtab_all[(wtab_all['ERA5_all_lv0']>=-4)&\
                         (wtab_all['ERA5_all_lv0']<2)&\
                         (wtab_all['ERA5_all_lv1']>=-4)&\
                         (wtab_all['ERA5_all_lv1']<=0)]
    wtab_temp = wtab_6reg.copy()
    matching_days = df_up_polar.loc[dir_cond, 'yyyymmdd']
    wtab_temp['yyyymmdd'] = pd.to_datetime(wtab_6reg['yyyymmdd'], format="%Y%m%d")
    dlist_dir     = wtab_temp[wtab_temp['yyyymmdd'].isin(matching_days)]['yyyymmdd']
    dlist_dir     = pd.to_datetime(dlist_dir, format="%Y%m%d").to_list()
    swpcp_all = pd.read_csv('../../TemporaryData/weather_table_self/swland_pcp_all.csv')
    swpcp_temp= swpcp_all.copy()
    swpcp_temp['yyyymmdd'] = pd.to_datetime(swpcp_all['yyyymmdd'], format="%Y%m%d")
    swpcp_dir     = swpcp_all[swpcp_temp['yyyymmdd'].isin(dlist_dir)]['mean']
    # Figure
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection':'polar'})
    ax.grid(linestyle=':',linewidth=1, color='grey')
    util_draw.draw_met_polar(ax)
    util_draw.draw_sswf_sector(axe=ax)
    bounds  = np.array([1, 2, 6, 10, 15, 20, 30, 40, 50, 70, 90, 110, 130, 150, 200, 300])
    cmap    = cmaps.WhiteBlueGreenYellowRed.copy()
    cmap.set_under((0.8, 0.8, 0.8, 0.7))
    norm    = colors.BoundaryNorm(bounds, cmap.N, extend='both')
    norm    = colors.BoundaryNorm(bounds, cmap.N, extend='both')
    df_case = df_up_polar[df_up_polar['yyyymmdd'].isin(dlist_dir)]
    pcp_rank= np.argsort(swpcp_dir).values
    im      = ax.scatter(df_case['IVT_theta'].iloc[pcp_rank], df_case['IVT_r'].iloc[pcp_rank], s=150, c=swpcp_dir.iloc[pcp_rank], alpha=0.7, cmap=cmap, norm=norm, zorder=10)
    cax     = fig.add_axes([ax.get_position().x1+0.065, ax.get_position().y0, 0.025, ax.get_position().height])
    cbar    = fig.colorbar(im, orientation='vertical', cax=cax)
    cbar.solids.set(alpha=1)
    cbar.set_ticks(ticks=bounds, labels=bounds.astype(int))
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label(f'SW-Taiwan mean precip. (mm/day)', fontsize=13)
    cbar.outline.set_linewidth(1.5)
    util_draw.draw_met_polar_rticks(axe=ax, rmin=0, rmax=800, 
                                    ticks=np.arange(200, 800.1, 200), ticklabels=[f"{int(i)}" for i in np.arange(200, 800.1, 200)], tick_fs=13)
    ax.set_title('(a) Fixed IVT Direction', fontsize=16, loc='left')
    if figname:
        plt.savefig(f'../../Figure/{figname}_{int(IVT_dir_degree)}_swpcp.png', facecolor='w', bbox_inches='tight', dpi=400)
    plt.show()
Plot_IVTdir_polar_precip(IVT_dir_degree=230, tolerance_degree=2.5, figname='FigS9a')