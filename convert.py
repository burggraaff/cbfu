import numpy as np
import mat
import fu
from spectacle.linearity import sRGB_generic
from matplotlib import pyplot as plt, patches
from scipy.spatial import distance_matrix

FU_LMS_deficiency = np.einsum("caij,fj->cafi",mat.SLMS, fu.FU_LMS) # axes: deficiency (lms), a, FU number, lms
FU_deficient_XYZ = np.einsum("ij,cafj->cafi", mat.M_lms_to_xyz_e, FU_LMS_deficiency) # axes: deficiency (lms), a, FU number, xyz
FU_deficient_RGB = np.einsum("ij,cafj->cafi", mat.M_xyz_to_rgb, FU_deficient_XYZ) # axes: deficiency (lms), a, FU number, rgb (linear)

FU_deficient_sRGB = sRGB_generic(FU_deficient_RGB, normalization=1)/255. # Gamma-expanded (non-linear) sRGB values. Note these are clipped to 0-255 to accommodate the limited gamut of sRGB.

example_indices = ((0, 0, 0, 1, 1, 2, 2), (-1, 50, 0, 50, 0, 50, 0))
examples_sRGB = FU_deficient_sRGB[example_indices]
examples_labels = ["Regular", "50% L-deficient", "Fully L-deficient", "50% M-deficient", "Fully M-deficient", "50% S-deficient", "Fully S-deficient"]

kwargs = {"width": 0.95, "height": 0.95, "edgecolor": "none"}
fig, ax = plt.subplots(figsize=(9, 3))
for i, (FU_list, label) in enumerate(zip(examples_sRGB[::-1], examples_labels[::-1])):
    print(i, label)
    rectangles = [patches.Rectangle(xy=(j,i), facecolor=rgb, **kwargs) for j, rgb in enumerate(FU_list)]
    for rect in rectangles:
        ax.add_patch(rect)

ax.axis("equal")
ax.set_yticks(np.arange(6.5, 0, -1))
ax.set_yticklabels(examples_labels)
ax.tick_params(axis="y", left=False, pad=0)

ax.set_xticks(np.arange(kwargs["width"]/2, 21, 1))
ax.set_xticklabels(fu.numbers)
ax.set_xlabel("Forel-Ule colour")
ax.tick_params(axis="x", bottom=False, labelbottom=False, labeltop=True)
ax.xaxis.set_label_position("top")
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_horizontalalignment("center")

plt.box()
plt.savefig("FU_example.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Chromaticities
FU_deficient_xy = FU_deficient_XYZ[...,:2] / FU_deficient_XYZ.sum(axis=3)[...,np.newaxis]

# Hue angles
FU_deficient_alpha = np.arctan2(FU_deficient_xy[...,1]-1/3, FU_deficient_xy[...,0]-1/3) % (2*np.pi)
FU_deficient_alpha_deg = np.rad2deg(FU_deficient_alpha)

# Plot hue angle for examples
for i, (alphas, label) in enumerate(zip(FU_deficient_alpha_deg[example_indices], examples_labels)):
    print(i, label)
    plt.plot(fu.numbers, alphas, label=label, lw=3)
plt.xlabel("Forel-Ule colour")
plt.ylabel("Hue angle")
plt.xlim(0.9, 21.1)
plt.legend(loc="best")
plt.show()
plt.close()

def plot_distance_matrices(FU_arrays, saveto="image.pdf", title="", ylabel="XYZ", **kwargs):
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10,5.2))
    for ax, arr, label in zip(axs.ravel()[1:], FU_arrays[example_indices], examples_labels):
        distances = distance_matrix(arr, arr)
        im = ax.imshow(distances, extent=(0, 21, 21, 0), cmap="cividis", **kwargs)
        ax.set_title(f"\n{label}")
        ax.set_xlim(0, 21)
        ax.set_ylim(0, 21)
        ax.set_xticks([0.5, 10.5, 20.5])
        ax.set_xticklabels([1, 11, 21])
        ax.set_yticks([0.5, 10.5, 20.5])
        ax.set_yticklabels([1, 11, 21])
    cax = axs.ravel()[0]
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cax.set_aspect("equal")
    cax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False)
    cax.set_ylabel(f"Euclidean distance ({ylabel})")
    cax.yaxis.set_label_position("left")
    fig.suptitle(title)
    plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()

# Distance matrices - XYZ
plot_distance_matrices(FU_deficient_XYZ, saveto="distance_matrix_XYZ.pdf", vmin=0, vmax=0.9, title="Euclidean distances between Forel-Ule colours", ylabel="XYZ")

# Distance matrices - xy
plot_distance_matrices(FU_deficient_xy, saveto="distance_matrix_xy.pdf", vmin=0, vmax=0.45, title="Euclidean distances between Forel-Ule colours", ylabel="xy")
