import matplotlib.pyplot as plt

from utils.display_image import plot_one_band



def display_detection(diff_image,change_map,clean_change_map,ax=None,fig=None,add_tilte=""):
    if ax is None:
        fig,ax=plt.subplots(1,3,figsize=(15,4))

    plot_one_band(diff_image[0, :, :], fig=fig, ax=ax[0],title="diff map abs(date0-date1) for {}".format(add_tilte),vminmax=(0, 0.2))
    plot_one_band(change_map, fig=fig, ax=ax[1],title="change map for {}".format(add_tilte))
    plot_one_band(clean_change_map, fig=fig, ax=ax[2],title="change map with erosion".format(add_tilte))
    plt.show()