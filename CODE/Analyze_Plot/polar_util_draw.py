import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import cmaps
import seaborn.colors.xkcd_rgb as c
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def draw_met_polar(axe, tick_fs=14):
    axe.set_theta_zero_location("N")  # 0 degrees (north) at the top
    axe.set_theta_direction(-1)       # Clockwise direction
    axe.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    axe.set_xticklabels(['N', '', 'E', '', 'S', '', 'W', ''], fontsize=tick_fs)
    
def draw_met_polar_rticks(axe, rmin:float, rmax:float, ticks:list, ticklabels:list, label_pos=135, tick_fs=16):
    axe.set_rmin(rmin)
    axe.set_rmax(rmax)
    axe.set_rticks(ticks)
    axe.set_yticklabels(ticklabels, fontsize=tick_fs)
    axe.set_rlabel_position(label_pos)
    
def draw_sswf_sector(axe, ls='-', lc='gunmetal', lw=6):
    axe.plot(np.array([200, 200])*np.pi/180, [250, 800], linestyle=ls, color=c[lc], linewidth=lw)  # right edge
    axe.plot(np.array([280, 280])*np.pi/180, [250, 800], linestyle=ls, color=c[lc], linewidth=lw)  # left edge
    axe.plot(np.linspace(200, 280, 100)*np.pi/180, np.ones(100)*250, linestyle=ls, color=c[lc], linewidth=lw)  # inner edge
    
# Inset plot settings
def draw_inset(axe, bbox_to_anchor:tuple=(1., 0, .06, .35), tick_fs=10):
    inAxes = inset_axes(axe, width="100%", height="100%", loc='lower right', bbox_to_anchor=bbox_to_anchor, bbox_transform=axe.transAxes)
    inAxes.set_yscale('log')
    inAxes.tick_params(left=False, right=True, labelleft=False, labelright=True, bottom=False, labelbottom=False)
    inAxes.yaxis.set_minor_locator(ticker.NullLocator())  # Turn off automatic log ticks
    inAxes.set_ylim(1050, 150)
    inAxes.set_yticks([1000, 925, 850, 700, 500, 300, 200])
    inAxes.set_yticklabels([1000, 925, 850, 700, 500, 300, 200], fontsize=tick_fs)
    inAxes.grid(linestyle=':', linewidth=0.5, color='grey')
    return inAxes