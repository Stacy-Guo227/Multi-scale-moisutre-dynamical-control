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

# Utilities
AAM_lon_range, AAM_lat_range = (66, 154), (30, -15)   # research domain
TWs_lon_range, TWs_lat_range = (115, 124), (26, 20)   # Taiwan surrounding
AAM_xloc, AAM_yloc = np.arange(90, 160.1, 30), np.arange(-10, 30.1, 10)
TWs_xloc, TWs_yloc = np.arange(115, 124.1, 2), np.arange(20, 26.1, 2)

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

def Get_regional_IVT(date:str, lon_range:tuple, lat_range:tuple):
    """
    Extract IVT for a specified region.
    """
    # Special attention to ERA5 latitute
    lat_range = lat_range if lat_range[0]>lat_range[1] and \
                             lat_range[0] is not None and lat_range[1] is not None \
                             else lat_range[::-1]  # for ERA5 data
    # IVT data paths
    dpath = f'ERA5/IVT/{date[:4]}.nc' # should replace with the yearly IVT files and change below accordingly to self-named variables
    # Extract data
    dobj  = convert_to_dobj(date)
    ds_ivt= xr.open_dataset(dpath).sel(time=dobj, method='nearest').sel(lat=slice(*lat_range), 
                                                                        lon=slice(*lon_range))
    return ds_ivt.sum_IVTx, ds_ivt.sum_IVTy, ds_ivt.sum_IVT_total

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

class PlotTools_vvm():
    def __init__(self):
        self.proj = ccrs.PlateCarree()
        self.ds_topo = xr.open_dataset('TOPO.nc')   # TaiwanVVM topography, available upon request

    def Plot_vvm_topo(self, axe, color, linewidth=None):
        topo_bounds= np.arange(0, 3500.1, 500)
        alpha_list = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        cmap_topo  = colors.ListedColormap([(0, 0, 0, i) for i in alpha_list])
        norm_      = colors.BoundaryNorm(topo_bounds, cmap_topo.N, extend='max')
        imtopoh    = axe.contourf(self.ds_topo.lon, self.ds_topo.lat, self.ds_topo.topo*1e2, 
                                  levels=topo_bounds, 
                                  cmap=cmap_topo, norm=norm_, extend='max', antialiased=1)
        if linewidth is not None:
            axe.contour(self.ds_topo.lon, self.ds_topo.lat, self.ds_topo.topo*1e2, levels=np.array([499.99, 500.01]), 
                        colors=color, linewidths=linewidth)
        else:
            pass
plottools_vvm = PlotTools_vvm()

# Plot IVT maps in different regions
def Plot_regional_IVT(date, lon_range:tuple, lat_range:tuple, 
                      xloc, yloc, figsize, 
                      quiver:bool=False, quiver_scale=None, quiver_int=None,
                      upstream_rect:bool=False, upstream_dot:tuple=False, twtopo:bool=False, 
                      figtitle=False, figname=False, savefig=False):
    # Data initialization
    if isinstance(date, str): date = date
    else: date = str(date)
    reg_IVTx, reg_IVTy, reg_IVT = Get_regional_IVT(date=date, lon_range=lon_range, lat_range=lat_range)
    # Figure initialization
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(1, 1, figure=fig)
    ## AAM
    ax1 = plottools_ctpy.Axe_map(fig, gs[0], xlim_=[lon_range[0]-0.01, lon_range[1]+0.01], ylim_=[lat_range[1]-0.01, lat_range[0]+0.01], 
                                 xloc_=xloc, yloc_=yloc)
    plottools_ctpy.Plot_cartopy_map(ax1) # draw map
    ax1.set_xticks(xloc)
    ax1.set_xticklabels([f"{int(i)}E" for i in xloc], fontsize=12)
    ax1.set_yticks(yloc)
    ax1.set_yticklabels([f"{int(i)}N" if i>=0 else f"{int(i*(-1))}S" for i in yloc], fontsize=12)
    
    im   = ax1.contourf(reg_IVT.lon, reg_IVT.lat, reg_IVT, cmap=cmaps.MPL_Blues, levels=np.arange(0, 800.1, 100), extend='max')
    if quiver: ax1.quiver(reg_IVT.lon[::quiver_int], reg_IVT.lat[::quiver_int], reg_IVTx[::quiver_int, ::quiver_int], reg_IVTy[::quiver_int, ::quiver_int], scale=quiver_scale)
    if figtitle:
        ax1.set_title(figtitle, fontsize=16, loc='left')
    else:
        ax1.set_title(date, fontsize=16, loc='left')
    ## cbar
    cax  = fig.add_axes([ax1.get_position().x1+0.01, ax1.get_position().y0, 0.01, ax1.get_position().height])
    cbar = fig.colorbar(im, orientation='vertical', cax=cax)
    cbar.solids.set(alpha=1)  # Default: set cbar to full color (w/out tranparency)
    cbar.set_ticks(ticks=np.arange(0, 800.1, 100), labels=[int(i) for i in np.arange(0, 800.1, 100)])
    cbar.set_label('(kg/m/s)', fontsize=12)
    cbar.outline.set_linewidth(1.5)
    ## Upstream
    if upstream_rect:
        rect = patches.Rectangle((115, 20), 4, 2, fc="none", ec=c['black'], linewidth=4, zorder=50)
        ax1.add_artist(rect)
    if upstream_dot:
        ax1.scatter(x=upstream_dot[0], y=upstream_dot[1], s=100, c=c['cherry'], ec='k')
    ## Taiwan TOPO
    if twtopo:
        plottools_vvm.Plot_vvm_topo(ax1, c['dark grey'])
    ## Save fig or just show
    if savefig:
        if figname:
            plt.savefig(f'../../Figure/{figname}_{date}.png', facecolor='w', bbox_inches='tight', dpi=400)
        else:
            plt.savefig(f'../../Figure/{date}.png', facecolor='w', bbox_inches='tight', dpi=400)
    plt.show()
    plt.close()
    
Plot_regional_IVT(date='20180828', lon_range=AAM_lon_range, lat_range=AAM_lat_range, 
                  xloc=AAM_xloc, yloc=AAM_yloc, 
                  figsize=(10, 4), upstream_rect=True, 
                  figtitle='(a) Asian-Australian monsoon domain', figname='Fig3a', savefig=True)
Plot_regional_IVT(date='20180828', lon_range=TWs_lon_range, lat_range=TWs_lat_range, 
                  xloc=TWs_xloc, yloc=TWs_yloc, 
                  figsize=(10, 4), quiver=True, quiver_scale=8000, quiver_int=3, 
                  upstream_rect=True, upstream_dot=(118.5, 21.5), twtopo=True, 
                  figtitle='(b) Regional zoom-in', figname='Fig3b', savefig=True)

def Plot_raw_norm_IVT(date, figname=False, savefig=False):
    # Data initialization
    if isinstance(date, str): date = date
    else: date = str(date)
    large_IVTx, large_IVTy, large_IVT = Get_regional_IVT(date=date, lon_range=(66, 154), lat_range=(40, -15))
    # Figure initialization
    fig = plt.figure(figsize=(23, 10))
    gs  = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1], wspace=0.1)
    ## AAM raw
    ax1 = plottools_ctpy.Axe_map(fig, gs[0], xlim_=[65.99, 154.01], ylim_=[-15.01, 30.01], 
                                 xloc_=np.arange(90, 160.1, 30), yloc_=np.arange(-10, 30.1, 10))
    plottools_ctpy.Plot_cartopy_map(ax1) # draw map
    ax1.set_xticks(np.arange(90, 160.1, 30))
    ax1.set_xticklabels([f"{int(i)}E" for i in np.arange(90, 160.1, 30)], fontsize=12)
    ax1.set_yticks(np.arange(-10, 30.1, 10))
    ax1.set_yticklabels([f"{int(i)}N" if i>=0 else f"{int(i*(-1))}S" for i in np.arange(-10, 30.1, 10)], fontsize=12)
    
    imraw = ax1.contourf(large_IVT.lon, large_IVT.lat, large_IVT, cmap=cmaps.MPL_Blues, levels=np.arange(0, 800.1, 100), extend='max')
    ax1.quiver(large_IVT.lon[::6], large_IVT.lat[::6], large_IVTx[::6, ::6], large_IVTy[::6, ::6], scale=2e4)
    ax1.set_title(f"{date} Raw. IVT", fontsize=16)
    ## cbar (raw)
    cax  = fig.add_axes([ax1.get_position().x1+0.008, ax1.get_position().y0, 0.007, ax1.get_position().height])
    cbar = fig.colorbar(imraw, orientation='vertical', cax=cax)
    cbar.set_ticks(ticks=np.arange(0, 800.1, 100), labels=[int(i) for i in np.arange(0, 800.1, 100)])
    cbar.outline.set_linewidth(1.5)
    ## AAM norm
    ax2 = plottools_ctpy.Axe_map(fig, gs[1], xlim_=[65.99, 154.01], ylim_=[-15.01, 30.01], 
                                 xloc_=np.arange(90, 160.1, 30), yloc_=np.arange(-10, 30.1, 10))
    plottools_ctpy.Plot_cartopy_map(ax2) # draw map
    ax2.set_xticks(np.arange(90, 160.1, 30))
    ax2.set_xticklabels([f"{int(i)}E" for i in np.arange(90, 160.1, 30)], fontsize=12)
    
    imnorm = ax2.contourf(large_IVT.lon[::5], large_IVT.lat[::5], ((large_IVT-large_IVT.min())/(large_IVT.max()-large_IVT.min()))[::5, ::5], 
                          cmap=cmaps.MPL_Blues, levels=np.arange(0, 1.1, 0.1), extend='max')
    ax2.set_title('MinMax Norm.', fontsize=16)
    ## cbar (norm)
    cax  = fig.add_axes([ax2.get_position().x1+0.008, ax2.get_position().y0, 0.007, ax2.get_position().height])
    cbar = fig.colorbar(imnorm, orientation='vertical', cax=cax)
    cbar.set_ticks(ticks=np.arange(0, 1.1, 0.1), labels=[f"{i:3.1f}" for i in np.arange(0, 1.1, 0.1)])
    cbar.outline.set_linewidth(1.5)
    
    if savefig:
        if figname:
            plt.savefig(f'../../Figure/{figname}_{date}.png', facecolor='w', bbox_inches='tight', dpi=400)
        else:
            plt.savefig(f'../../Figure/{date}.png', facecolor='w', bbox_inches='tight', dpi=400)
    plt.show()
Plot_raw_norm_IVT(date='20010704', figname='FigS4a', savefig=True)
Plot_raw_norm_IVT(date='20080819', figname='FigS4b', savefig=True)
Plot_raw_norm_IVT(date='20140807', figname='FigS5a', savefig=True)
Plot_raw_norm_IVT(date='20150927', figname='FigS5b', savefig=True)