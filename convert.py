import numpy as np
import mat
import fu
from spectacle.linearity import sRGB_generic
from matplotlib import pyplot as plt, patches
from scipy.spatial import minkowski_distance
from colorio._tools import plot_flat_gamut
from mpl_toolkits.axes_grid1 import AxesGrid

FU_LMS_deficiency = np.einsum("caij,fj->cafi",mat.SLMS, fu.FU_LMS) # axes: deficiency (lms), a, FU number, lms
FU_deficient_XYZ = np.einsum("ij,cafj->cafi", mat.M_lms_to_xyz_e, FU_LMS_deficiency) # axes: deficiency (lms), a, FU number, xyz

# Following steps are just for the sRGB demonstration image
FU_deficient_XYZ_D65 = np.einsum("ij,cafj->cafi", mat.M_xyz_e_to_xyz_d65, FU_deficient_XYZ) # axes: deficiency (lms), a, FU number, xyz
FU_deficient_RGB = np.einsum("ij,cafj->cafi", mat.M_xyz_to_rgb, FU_deficient_XYZ_D65) # axes: deficiency (lms), a, FU number, rgb (linear)
FU_deficient_sRGB = sRGB_generic(FU_deficient_RGB, normalization=1)/255. # Gamma-expanded (non-linear) sRGB values. Note these are clipped to 0-255 to accommodate the limited gamut of sRGB.

example_indices = ((0, 0, 0, 1, 1, 2, 2), (-1, 50, 0, 50, 0, 50, 0))
examples_sRGB = FU_deficient_sRGB[example_indices]
examples_labels = ["Regular", "50% L-deficient", "Fully L-deficient", "50% M-deficient", "Fully M-deficient", "50% S-deficient", "Fully S-deficient"]

kwargs = {"width": 0.92, "height": 0.92, "edgecolor": "none"}
fig, ax = plt.subplots(figsize=(7, 2.3))
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

# Plot chromaticities on gamut
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(7,4), sharex=True, sharey=True)
axs[0,0].axis("off")
for ax, xy, label in zip(axs.ravel()[1:], FU_deficient_xy[example_indices], examples_labels):
    plt.sca(ax)
    plot_flat_gamut(plot_planckian_locus=False, axes_labels=("", ""))
    ax.scatter(*xy.T, c="k", marker="o", s=4, label="FU colours")
    ax.plot(*xy.T, c="k")
    ax.set_title(label)
for ax in axs[0,1:]:
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
for ax in axs[1]:
    ax.set_xlabel("x")
for ax in np.concatenate((axs[0,2:], axs[1,1:])):
    ax.tick_params(axis="y", left=False, labelleft=False)
axs[0,1].tick_params(axis="y", left=True, labelleft=True)
axs[0,1].set_ylabel("y")
axs[1,0].set_ylabel("y")
fig.suptitle("Forel-Ule colour gamut for various cone deficiencies")
plt.savefig("gamut.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Plot xyY as function of a for several FU colours
FU_deficient_xyY = np.concatenate((FU_deficient_xy, FU_deficient_XYZ[...,1][...,np.newaxis]), axis=3)
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(10,5), sharex=True, sharey=True)
for i, (ax_col, def_xyY, cone) in enumerate(zip(axs.T, FU_deficient_xyY, "LMS")):
    for k, ax in enumerate(ax_col):
        for f in [1, 5, 11, 16, 21]:
            ax.plot(mat.a, def_xyY[:,f-1,k], label=f"FU {f:>2}", lw=3)
        ax.grid(ls="--", color="0.7")
    ax_col[0].set_title(f"{cone} deficiency")
for ax in axs[:,1:].ravel():
    ax.tick_params(axis="y", left=False, labelleft=False)
for ax, ylabel in zip(axs[:,0], "xyY"):
    ax.set_ylabel(ylabel)
axs[0,0].set_xlim(1, 0)
for ax in axs[2]:
    ax.set_xlabel("$a$")
    ax.set_xticks([1, 0.75, 0.5, 0.25, 0])
for ax in axs[:2].ravel():
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
axs[1,2].legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.savefig("xyY.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Plot x vs Y for all FU colours at full deficiency
plt.figure(figsize=(7,2))
plt.plot(*FU_deficient_xyY[0,-1,:,::2].T, "o-", lw=3, label="Regular")
for i, label in enumerate("LMS"):
    plt.plot(*FU_deficient_xyY[i,0,:,::2].T, "o-", lw=3, label=f"{label}-deficient")
plt.xlim(0, 0.55)
plt.ylim(0, 0.55)
plt.xlabel("$x$")
plt.ylabel("$Y$")
plt.grid(ls="--", color="0.7")
plt.title("Forel-Ule $x$ vs $Y$ for various cone deficiencies")
plt.legend(loc="best")
plt.savefig("xY.pdf", bbox_inches="tight")
plt.show()
plt.close()

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

# Calculate distance matrices
def distance_matrix(FU_deficient_array):
    """
    Calculate the distance matrix between all FU colours (21x21 matrix) for each
    kind of cone deficiency.
    """
    distances = minkowski_distance(FU_deficient_array[:,:,:,np.newaxis,:], FU_deficient_array[:,:,np.newaxis,:,:])
    return distances

distances_XYZ = distance_matrix(FU_deficient_XYZ)
distances_xy = distance_matrix(FU_deficient_xy)

# Plot distance matrices
def plot_distance_matrices(FU_distance_matrices, saveto="image.pdf", title="", ylabel="Euclidean distance (XYZ)", **kwargs):
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10,5.2))
    for ax, distances, label in zip(axs.ravel()[1:], FU_distance_matrices[example_indices], examples_labels):
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
    cax.set_ylabel(ylabel)
    cax.yaxis.set_label_position("left")
    fig.suptitle(title)
    plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()

# Distance matrices - XYZ
plot_distance_matrices(distances_XYZ, saveto="distance_matrix_XYZ.pdf", vmin=0, vmax=0.9, title="Euclidean distances between Forel-Ule colours in XYZ", ylabel="Euclidean distance (XYZ)")

# Distance matrices - xy
plot_distance_matrices(distances_xy, saveto="distance_matrix_xy.pdf", vmin=0, vmax=0.45, title="Euclidean distances between Forel-Ule colours in xy", ylabel="Euclidean distance (xy)")

# Matrices to select diagonal and off-diagonal elements
diag = np.eye(21, dtype=bool)
off_diag = ~diag

# Minimum distance with regular vision
distances_XYZ_regular = distances_XYZ[0,-1]
distances_XYZ_regular_min = np.min(distances_XYZ_regular[off_diag])

distances_xy_regular = distances_xy[0,-1]
distances_xy_regular_min = np.min(distances_xy_regular[off_diag])

# Median distance as a function of a
median_distance_XYZ = np.median(distances_XYZ[...,off_diag], axis=2)
median_distance_xy = np.median(distances_xy[...,off_diag], axis=2)

# Minimum distance as a function of a
min_distance_XYZ = np.min(distances_XYZ[...,off_diag], axis=2)
min_distance_xy = np.min(distances_xy[...,off_diag], axis=2)

# Plot distance statistics
def plot_distances(distances, baseline=0, statistic_label="", coordinate_label="", saveto="image.pdf"):
    plt.figure(figsize=(5,3))
    for i, label in enumerate("LMS"):
        plt.plot(mat.a, distances[i], lw=3, label=f"{label}-deficient")
    plt.axhline(baseline, c='k', lw=3, label=f"Baseline ({baseline:.3f})")
    plt.xlim(1, 0)
    plt.xticks([1, 0.75, 0.5, 0.25, 0])
    plt.ylim(ymin=0)
    plt.grid(ls="--", c="0.7")
    plt.xlabel("$a$")
    plt.ylabel(f"{statistic_label} distance in {coordinate_label}")
    plt.title(f"{statistic_label} Euclidean distances between FU colours ({coordinate_label})")
    plt.legend(loc="best")
    plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()

# Plot median distances
plot_distances(median_distance_XYZ, baseline=distances_XYZ_regular_min, statistic_label="Median", coordinate_label="XYZ", saveto="distance_median_XYZ.pdf")
plot_distances(median_distance_xy, baseline=distances_xy_regular_min, statistic_label="Median", coordinate_label="xy", saveto="distance_median_xy.pdf")

# Plot minimum distances
plot_distances(min_distance_XYZ, baseline=distances_XYZ_regular_min, statistic_label="Minimum", coordinate_label="XYZ", saveto="distance_min_XYZ.pdf")
plot_distances(min_distance_xy, baseline=distances_xy_regular_min, statistic_label="Minimum", coordinate_label="xy", saveto="distance_min_xy.pdf")

# Calculate distances relative to regular vision
rel_distances_XYZ = distances_XYZ / distances_XYZ_regular * 100  # %
rel_distances_xy = distances_xy / distances_xy_regular * 100  # %

# Calculate change in distances relative to regular vision
diff_distances_XYZ = rel_distances_XYZ - 100.
diff_distances_xy = rel_distances_xy - 100.

# Calculate distances relative to minimum in regular vision
distances_XYZ_div_min = distances_XYZ / distances_XYZ_regular_min * 100.
distances_xy_div_min = distances_xy / distances_xy_regular_min * 100.
distances_XYZ_div_min[...,diag] = np.nan
distances_xy_div_min[...,diag] = np.nan

# Relative distances matrices
plot_distance_matrices(rel_distances_XYZ, saveto="distance_matrix_XYZ_relative.pdf", vmin=0, vmax=100, title="Relative Euclidean distances between Forel-Ule colours in XYZ", ylabel="Relative\nEuclidean distance (XYZ, %)")

plot_distance_matrices(rel_distances_xy, saveto="distance_matrix_xy_relative.pdf", vmin=0, vmax=100, title="Relative Euclidean distances between Forel-Ule colours in xy", ylabel="Relative\nEuclidean distance (xy, %)")

# Combined absolute and relative distance matrix plot
def plot_distance_matrices_combined(absolute_distances, difference_distances, min_relative_distances, saveto="image.pdf", title="",):
    extreme_indices = ((0, 0, 1, 2), (-1, 0, 0, 0))
    extreme_labels = ["Regular", "L-deficient", "M-deficient", "S-deficient"]
    fig = plt.figure(figsize=(10,7))
    grid = AxesGrid(fig, 111, nrows_ncols=(3,4), axes_pad=0.15, cbar_mode="edge", cbar_location="right", cbar_pad=0.15)
    kwargs = {"extent": (0, 21, 21, 0), "cmap": "cividis"}

    vmax = np.nanmax(absolute_distances[extreme_indices])
    for ax, distances_absolute, label in zip(grid[:4], absolute_distances[extreme_indices], extreme_labels):
        im_abs = ax.imshow(distances_absolute, vmin=0, vmax=vmax, **kwargs)
        ax.set_title(f"\n{label}")
    cbar_abs = grid.cbar_axes[0].colorbar(im_abs)
    cbar_abs.set_label_text("Euclidean distance")

    for ax, distances_diff, label in zip(grid[4:8], difference_distances[extreme_indices], extreme_labels):
        im_diff = ax.imshow(distances_diff, vmin=-50, vmax=0, **kwargs)
    cbar_diff = grid.cbar_axes[1].colorbar(im_diff)
    cbar_diff.set_label_text("$\Delta$ distance (%)")

    for ax, distances_min, label in zip(grid[8:], min_relative_distances[extreme_indices], extreme_labels):
        im_min = ax.imshow(distances_min, vmin=0, vmax=100, **kwargs)
    cbar_min = grid.cbar_axes[2].colorbar(im_min)
    cbar_min.set_label_text("Distance / Minimum (%)")

    for ax in grid:
        ax.set_xlim(0, 21)
        ax.set_ylim(0, 21)
        ax.set_xticks([0.5, 10.5, 20.5])
        ax.set_xticklabels([1, 11, 21])
        ax.set_yticks([0.5, 10.5, 20.5])
        ax.set_yticklabels([1, 11, 21])

    fig.suptitle(title)
    plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()

plot_distance_matrices_combined(distances_XYZ, diff_distances_XYZ, distances_XYZ_div_min, title="Changes in Euclidean distances between Forel-Ule colours in XYZ", saveto="distance_matrix_combined_XYZ.pdf")
plot_distance_matrices_combined(distances_xy, diff_distances_xy, distances_xy_div_min, title="Changes in Euclidean distances between Forel-Ule colours in xy", saveto="distance_matrix_combined_xy.pdf")

# # Plot all distances
# fig, axs = plt.subplots(nrows=21, ncols=21, sharex=True, sharey=True, figsize=(20,20))
# axs = axs[::-1]  # Orientation of diagonal
# for i in range(21):
#     rect = patches.Rectangle(xy=(0,0), facecolor=FU_deficient_sRGB[0,-1,i], width=1, height=1, edgecolor="none")
#     axs[i,i].add_patch(rect)
#     for j in range(i+1, 21):
#         for dist in distances_XYZ[...,i,j]:
#             axs[i,j].plot(mat.a, dist)
# for ax in axs.ravel():
#     ax.tick_params(axis="both", left=False, labelleft=False, bottom=False, labelbottom=False)
# axs[0,0].set_xlim(1, 0)
# plt.savefig("mega.pdf", bbox_inches="tight")
# plt.show()
# plt.close()
# # Diagonal from lower left to upper right
# # Upper left triangle: absolute distances
# # Lower right triangle: relative distances
# # Diagonal: regular colour + label

FU_deficient_lab = mat.XYZ_to_Lab(FU_deficient_XYZ)
