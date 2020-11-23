import numpy as np
import mat
from matplotlib import pyplot as plt
from colorspacious import cspace_converter

converter = cspace_converter("sRGB1", "CAM02-UCS")

for cmap_name in plt.colormaps():
    if cmap_name[-2:] == "_r":
        continue

    # Get colour map data
    cmap = plt.get_cmap(cmap_name)

    try:
        cmap_Lab = converter(cmap.colors)
    except AttributeError:
        colors = cmap(np.linspace(0,1,256))[:,:3]
        cmap_Lab = converter(colors)


    # Calculate Delta E 00
    L1, a1, b1 = cmap_Lab[...,0][np.newaxis,:], cmap_Lab[...,1][np.newaxis,:], cmap_Lab[...,2][np.newaxis,:]
    L2, a2, b2 = cmap_Lab[...,0][:,np.newaxis], cmap_Lab[...,1][:,np.newaxis], cmap_Lab[...,2][:,np.newaxis]

    distances_Lab = mat.dE00(L1, a1, b1, L2, a2, b2)
    distances_Lab_regular = distances_Lab[0,-1]
    distances_Lab_JND = distances_Lab/mat.JND

    plt.imshow(distances_Lab_JND, cmap=plt.get_cmap("cividis_r", 15), vmin=0, vmax=15)
    plt.colorbar(label="$\Delta E_{00}$ / JND")
    plt.title(cmap_name)
    plt.savefig(f"cmaps/{cmap_name}.pdf", bbox_inches="tight")
    plt.show()
