"""
Misc functions:
    -- Creation of colors and mappings
    -- Creation of directories on Windows
    
@date: Sept 2024
@author: Anthony Coache    
"""
# misc
import os.path
from matplotlib.colors import LinearSegmentedColormap


# define colors
mblue = (0.098, 0.18, 0.357)
mred = (0.902, 0.4157, 0.0196)
mgreen = (0., 0.455, 0.247)
myellow = (0.8, 0.8, 0)
mpurple = (0.5804, 0.2157, 0.9412)
mgray = (0.5012, 0.5012, 0.5012)
mblack = (0., 0., 0.)
mwhite = (1., 1., 1.)

cmap2 = LinearSegmentedColormap.from_list(
    'beamer_cmap', [mwhite, mred])  # mapping for heatmaps with 2 colors

cmap3 = LinearSegmentedColormap.from_list(
    'beamer_cmap', [mred, mwhite, mblue])  # mapping for heatmaps with 3 colors

colors = [mblue, mred, mgreen, myellow, mpurple, mgray]  # individual colors

rainbow = ['magenta', 'darkblue', 'blue',
           'deepskyblue', 'springgreen',
           'gold', 'orange', 'red']  # rainbow colors


def directory(foldername):
    """create a folder if it does not exist

    Parameters
    ----------
    foldername : str
        Name of the folder
    """
    if os.path.exists(foldername):
        return
    else:
        os.mkdir(foldername)
    return
