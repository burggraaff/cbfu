import numpy as np
import mat
import fu
from spectacle.linearity import sRGB_generic
from matplotlib import pyplot as plt, patches
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

# Matrices to select diagonal and off-diagonal elements
diag = np.eye(21, dtype=bool)
off_diag = ~diag

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

FU_deficient_lab = mat.XYZ_to_Lab(FU_deficient_XYZ)

# Plot a* vs L* for all FU colours at full deficiency
plt.figure(figsize=(7,2))
plt.plot(*FU_deficient_lab[0,-1,:,1::-1].T, "o-", lw=3, label="Regular")
for i, label in enumerate("LMS"):
    plt.plot(*FU_deficient_lab[i,0,:,1::-1].T, "o-", lw=3, label=f"{label}-deficient")
# plt.xlim(0, 0.55)
# plt.ylim(0, 0.55)
plt.xlabel("$a^*$")
plt.ylabel("$L^*$")
plt.grid(ls="--", color="0.7")
plt.title("Forel-Ule $a^*$ vs $L^*$ for various cone deficiencies")
plt.legend(loc="best")
plt.savefig("aL.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Plot L*, a*, b* as a function of FU at full deficiency
ylabels = ["$L^*$", "$a^*$", "$b^*$"]
fig, axs = plt.subplots(nrows=3, figsize=(4,4), sharex=True)
for k, (ax, ylabel) in enumerate(zip(axs, ylabels)):
    ax.plot(fu.numbers, FU_deficient_lab[0,-1,:,k], "o-", lw=3, label="Regular")
    for i, label in enumerate("LMS"):
        ax.plot(fu.numbers, FU_deficient_lab[i,0,:,k].T, "o-", lw=3, label=f"{label}-deficient")
    ax.set_ylabel(ylabel)
    ax.grid(ls="--", color="0.7")
axs[0].set_xlim(0.9, 21.1)
axs[0].set_xticks([1, 5, 10, 15, 20])
axs[-1].set_xlabel("Forel-Ule color")
axs[0].set_title("Forel-Ule colors in $L^* a^* b^*$ space")
axs[1].legend(loc="upper left", bbox_to_anchor=(1,1.08))
fig.align_labels()
plt.savefig("FU_Lab.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Calculate Delta E 00
L1, a1, b1 = FU_deficient_lab[...,0][:,:,np.newaxis,:], FU_deficient_lab[...,1][:,:,np.newaxis,:], FU_deficient_lab[...,2][:,:,np.newaxis,:]
L2, a2, b2 = FU_deficient_lab[...,0][:,:,:,np.newaxis], FU_deficient_lab[...,1][:,:,:,np.newaxis], FU_deficient_lab[...,2][:,:,:,np.newaxis]

distances_Lab = mat.dE00(L1, a1, b1, L2, a2, b2)
distances_Lab_regular = distances_Lab[0,-1]

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

# Plot distance matrices
plot_distance_matrices(distances_Lab, saveto="distance_matrix_Lab.pdf", vmin=0, title="CIE $L^*a^*b^*$ distances between Forel-Ule colours", ylabel="$\Delta E_{00}$")

# Distance as a function of a
median_distance_Lab = np.median(distances_Lab[...,off_diag], axis=2)
min_distance_Lab = np.min(distances_Lab[...,off_diag], axis=2)

# Combined plot of distance statistics
fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(4,4))
for ax, dist, ylabel in zip(axs, [median_distance_Lab, min_distance_Lab], ["Median", "Minimum"]):
    for i, label in enumerate("LMS"):
        ax.plot(mat.a, dist[i], lw=3, label=f"{label}-deficient")
    ax.axhline(2.3, c='k', lw=3, ls="dotted", label="JND (2.3)")
    ax.set_ylim(ymin=0)
    ax.grid(ls="--", c="0.7")
    ax.set_ylabel(ylabel+" $\Delta E_{00}$")
axs[1].set_xlim(1, 0)
axs[1].set_xticks([1, 0.75, 0.5, 0.25, 0])
axs[1].set_xlabel("Relative cone contribution $a$")
axs[0].set_title("Median/Minimum $\Delta E_{00}$ between FU colors\nwith decreasing $a$")
axs[1].legend(loc="best", ncol=2)
fig.align_labels()
plt.savefig("distance_stats_Lab.pdf", bbox_inches="tight")
plt.show()
plt.close()
