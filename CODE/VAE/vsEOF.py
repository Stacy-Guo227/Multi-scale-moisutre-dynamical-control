import numpy as np
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import seaborn.colors.xkcd_rgb as c
import cmaps
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from ML_model_data import *

import tensorflow as tf
from tensorflow.keras import layers, Input, backend, Model, losses, optimizers, models

encoder, decoder = Load_ML()
# Original IVT
era5_IVT_all = xr.open_dataset('../../TemporaryData/IVT_1000_700_2001_2019.nc')
# Normalized IVT (training & testing)
era5_normIVT_ssn_train= Load_Dataset()['x_train']
era5_normIVT_ssn_test = Load_Dataset()['x_test']
# Latent vectors
era5_lvIVT_ssn_train= Load_latent_vectors('ERA5(train)')
era5_lvIVT_ssn_test   = Load_latent_vectors('ERA5(test)')
# Reconstruction
era5_reconIVT_ssn_train= Load_reconstruction(era5_lvIVT_ssn_train)
era5_reconIVT_ssn_test= Load_reconstruction(era5_lvIVT_ssn_test)

# EOF
era5_train_reshape = era5_normIVT_ssn_train.reshape(era5_normIVT_ssn_train.shape[0], 
                                                    era5_normIVT_ssn_train.shape[1]*era5_normIVT_ssn_train.shape[2])
era5_test_reshape = era5_normIVT_ssn_test.reshape(era5_normIVT_ssn_test.shape[0], 
                                                  era5_normIVT_ssn_test.shape[1]*era5_normIVT_ssn_test.shape[2])
# standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(era5_train_reshape)
era5_train_std = sc.transform(era5_train_reshape)
era5_test_std  = sc.transform(era5_test_reshape)
# fit with training set, transform testing set
from sklearn.decomposition import PCA
n_components    = 2
pca             = PCA(n_components=n_components)
era5_train_fit  = pca.fit(era5_train_std)
era5_test_trans = era5_train_fit.transform(era5_test_std)
# Reconstruction with 2 EOF modes
pca_recon_std = pca.inverse_transform(era5_test_trans)
pca_recon     = sc.inverse_transform(pca_recon_std)
pca_recon     = pca_recon.reshape(era5_normIVT_ssn_test.shape[0], 
                                  era5_normIVT_ssn_test.shape[1], era5_normIVT_ssn_test.shape[2])

# Pattern correlation
def weighted_patt_corr(org_array:np.ndarray, pred_array:np.ndarray, lat_info:np.ndarray)->np.ndarray:
    if len(org_array.shape) > 3: org_array = org_array[..., 0]      # check org_array's dimension
    if len(pred_array.shape) > 3: pred_array = pred_array[..., 0]   # check pred_array's dimension
    weighting    = np.cos(lat_info*np.pi/180)                       # geographical weighting
    weighting_2d = np.repeat(weighting[:, np.newaxis], axis=-1, repeats=org_array.shape[2])
    patt_corr    = np.zeros(org_array.shape[0])
    for dd in range(org_array.shape[0]):
        wmean_org, wmean_pred = np.mean(org_array[dd, :]*weighting_2d), np.mean(pred_array[dd, :]*weighting_2d)
        nominator = weighting_2d*(org_array[dd, :]-wmean_org)*(pred_array[dd, :]-wmean_pred)
        wcovar    = nominator.sum()/weighting_2d.sum()
        patt_corr[dd] = wcovar/(np.std(org_array[dd, :])*np.std(pred_array[dd, :]))
    return patt_corr
era5_patcor_pca = weighted_patt_corr(era5_normIVT_ssn_test, pca_recon, era5_IVT_all.latitude.data[::5])
era5_patcor_vae = weighted_patt_corr(era5_normIVT_ssn_test, era5_reconIVT_ssn_test, era5_IVT_all.latitude.data[::5])
era5_patcor_vae_train = weighted_patt_corr(era5_normIVT_ssn_train, era5_reconIVT_ssn_train, era5_IVT_all.latitude.data[::5])

# RMSE
era5_rmse_pca = np.sqrt(np.mean((era5_normIVT_ssn_test[..., 0]-pca_recon)**2, axis=(1, 2)))
era5_rmse_vae = np.sqrt(np.mean((era5_normIVT_ssn_test[..., 0]-era5_reconIVT_ssn_test[..., 0])**2, axis=(1, 2)))
era5_rmse_vae_train = np.sqrt(np.mean((era5_normIVT_ssn_train[..., 0]-era5_reconIVT_ssn_train[..., 0])**2, axis=(1, 2)))

def Plot_metrics_box_onlyERA5(season:str):
    ssn_color = {'sum':c['crimson'], 'win':c['denim']}
    boxprops_eof = dict(linestyle=':', linewidth=4, color=ssn_color[season])
    boxprops_vae = dict(linestyle='-', linewidth=4, color=ssn_color[season])
    medianprops  = dict(linestyle='-', linewidth=4, color='k')
    fig, ax = plt.subplots(1, 2, sharex=True, figsize=(8, 5), gridspec_kw={'wspace':0.25})
    # Pattern Correlation & Explained variance
    ax[0].grid(':', linewidth=0.5, axis='y')
    CE = ax[0].boxplot([era5_patcor_pca], 
                  positions=np.array([0]), labels=['Test'],
                  showfliers=False, boxprops=boxprops_eof, widths=.3, medianprops=medianprops)
    CV = ax[0].boxplot([era5_patcor_vae, era5_patcor_vae_train], 
                  positions=np.array([1, 2]), labels=['Test', 'Train'], 
                  showfliers=False, boxprops=boxprops_vae, widths=.3, medianprops=medianprops)
    CM = ax[0].plot(0, np.mean(era5_patcor_pca**2), 'k*', markersize=15)
    ax[0].plot(1, np.mean(era5_patcor_vae**2), 'k*', markersize=15)
    ax[0].plot(2, np.mean(era5_patcor_vae_train**2), 'k*', markersize=15)

    ax[0].set_xlim([-1, 3])
    ax[0].set_xticklabels(['Test', 'Test', 'Train'], fontsize=14)
    ax[0].set_ylim([0, 1])
    ax[0].set_yticks(np.arange(0, 1.1, 0.1))
    ax[0].set_yticklabels([f'{i:3.1f}' for i in np.arange(0, 1.1, 0.1)], fontsize=12)
    ax[0].legend([CE['boxes'][0], CV['boxes'][0], CM[0]], 
                 ["EOF", "VAE", "Explained Variance"], fontsize=12)
    ax[0].set_title('Pattern Correlation', fontsize=16)

    ax[1].grid(':', linewidth=0.5, axis='y')
    ax[1].boxplot([era5_rmse_pca], 
                  positions=np.array([0]), labels=['Test'], 
                  showfliers=False, boxprops=boxprops_eof, widths=.3, medianprops=medianprops)
    ax[1].boxplot([era5_rmse_vae, era5_rmse_vae_train], 
                  positions=np.array([1, 2]), labels=['Test', 'Train'], 
                  showfliers=False, boxprops=boxprops_vae, widths=.3, medianprops=medianprops)

    ax[1].set_xlim([-1, 3])
    ax[1].set_xticklabels(['Test', 'Test', 'Train', 'Test', 'Test', 'Train'], fontsize=14)
    ax[1].set_ylim([0., 0.2])
    ax[1].set_yticks(np.arange(0, 0.21, 0.025))
    ax[1].set_yticklabels([f'{i:3.3f}' for i in np.arange(0, 0.21, 0.025)], fontsize=12)
    ax[1].set_title(f'RMSE', fontsize=16)
    plt.savefig(f'../../Figure/FigureS11.png', 
                bbox_inches='tight', facecolor='w', dpi=400)
    
Plot_metrics_box_onlyERA5(season='sum')

# Reconstruction comparison
ssn_date_list = Xtest_datelist()
def Plot_2D_map(axe, lon_, lat_,
                xlim_, ylim_, xloc_, yloc_, arr_,
                bound_, cmap_, alpha_=None, extend_='neither', bottom_label_=False, left_label_=False):
    """
    2D map with geographical settings
    """
    axe.set_extent([xlim_[0], xlim_[-1], ylim_[0], ylim_[-1]], crs=ccrs.PlateCarree())
    axe.add_feature(cfeature.LAND,color='grey',alpha=0.1)
    axe.coastlines(resolution='50m', color='black', linewidth=1)
    if bottom_label_:
        gx = axe.gridlines(xlocs=xloc_, crs=ccrs.PlateCarree(), zorder=1)
        gx.bottom_labels = True
    if left_label_:
        gy = axe.gridlines(ylocs=yloc_, crs=ccrs.PlateCarree(), zorder=1)
        gy.left_labels = True
        
    im = axe.contourf(lon_,lat_,arr_,
                      transform=ccrs.PlateCarree(),
                      levels=bound_,extend=extend_,cmap=cmap_, alpha=alpha_)
    return im
def Colorinfo(fig, axe, type_:str, pos_:list,  
              fs_:int, title_=None, title_fs_=None,
              var_=None, cmap_=None, alpha_=1, extend_='neither',
              ticks_=None, labels_=None):
    """
    2 options: legend/colorbar
    """
    if type_=='legend':
        legend = axe.legend(bbox_to_anchor=pos_,
                             title=title_, 
                             fontsize=fs_, title_fontsize=fs_)
        for lh in legend.legendHandles: 
            lh.set_alpha(1)
        return legend
    elif type_=='colorbar':
        cax     = fig.add_axes(pos_)
        cbar    = fig.colorbar(var_, extend=extend_, cax=cax)
        cbar.solids.set(alpha=alpha_)  # Default: set cbar to full color (w/out tranparency)
        cbar.set_ticks(ticks=ticks_, labels=labels_)
        cbar.ax.tick_params(labelsize=fs_)
        cbar.set_label(title_, fontsize=title_fs_)
        return cbar
    
def Fig_title(axe, 
              right_title_='', mid_title_='', left_title_='', title_fs_=None,):
    axe.set_title(right_title_, loc='right', fontsize=title_fs_)
    axe.set_title(mid_title_, fontsize=title_fs_)
    axe.set_title(left_title_, loc='left', fontsize=title_fs_)
    
def Plot_5cases_compar(season:str):
    proj    = ccrs.PlateCarree()
    fig, ax = plt.subplots(5, 3, figsize=(15, 17), sharex=True, sharey=True,
                           subplot_kw={'projection': proj, "aspect": 1},
                           gridspec_kw = {'wspace':0.1, 'hspace':0.06},
                           )
    for i, case_date in enumerate(case_date_list[season]):
        case_idx = [idx for idx, dd in enumerate(ssn_date_list) if dd==case_date][0]
        im_input= Plot_2D_map(axe=ax[i, 0], lon_=era5_IVT_all.longitude[::5], lat_=era5_IVT_all.latitude[::5],
                              xlim_=[65.9, 153.6],
                              ylim_=[-15.1, 30.1], 
                              xloc_=[80, 100, 120, 140], 
                              yloc_=[-20, -10, 0, 10, 20, 30, 35], 
                              arr_ =era5_normIVT_ssn_test[case_idx, ..., 0],
                              bound_=np.arange(0, 0.71, 0.1), 
                              cmap_ = cmaps.MPL_Blues, 
                              extend_='max', bottom_label_=True, left_label_=True)
        im_vae= Plot_2D_map(axe=ax[i, 1], lon_=era5_IVT_all.longitude[::5], lat_=era5_IVT_all.latitude[::5],
                            xlim_=[65.9, 153.6],
                            ylim_=[-15.1, 30.1], 
                            xloc_=[80, 100, 120, 140], 
                            yloc_=[-20, -10, 0, 10, 20, 30, 35], 
                            arr_ =era5_reconIVT_ssn_test[case_idx, ..., 0],
                            bound_=np.arange(0, 0.71, 0.1), 
                            cmap_ = cmaps.MPL_Blues, 
                            extend_='max', bottom_label_=True, left_label_=False)
        im_pca= Plot_2D_map(axe=ax[i, 2], lon_=era5_IVT_all.longitude[::5], lat_=era5_IVT_all.latitude[::5],
                            xlim_=[65.9, 153.6],
                            ylim_=[-15.1, 30.1], 
                            xloc_=[80, 100, 120, 140], 
                            yloc_=[-20, -10, 0, 10, 20, 30, 35], 
                            arr_ =pca_recon[case_idx, ...],
                            bound_=np.arange(0, 0.71, 0.1), 
                            cmap_ = cmaps.MPL_Blues, 
                            extend_='max', bottom_label_=True, left_label_=False)
        cbar    = Colorinfo(fig=fig, axe=ax[i, 2], type_='colorbar', 
                            pos_ =[ax[i, 2].get_position().x1+0.01, 
                                   ax[i, 2].get_position().y0, 
                                   0.01, 
                                   ax[i, 2].get_position().height], 
                            var_ =im_vae, 
                            cmap_=cmaps.MPL_Blues, 
                            extend_='max',
                            ticks_ =np.arange(0, 0.71, 0.1), fs_  =14,
                            labels_=[f'{i:.1f}' for i in np.arange(0, 0.71, 0.1)])

        Fig_title(axe=ax[i, 0], mid_title_=f'Input', right_title_=f'{case_date}', title_fs_=14)
        Fig_title(axe=ax[i, 1], mid_title_=f'VAE Recon.', title_fs_=14)
        Fig_title(axe=ax[i, 2], mid_title_=f'EOF Recon.', title_fs_=14)

        plt.savefig(f'../../Figure/FigS10.png', 
                    bbox_inches='tight', facecolor='w', dpi=400)
        
case_date_list = {'sum':['20150528', '20180402', '20160624', '20160811', '20150715']}
Plot_5cases_compar(season='sum')