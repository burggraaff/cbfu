import numpy as np
import mat
import fu
from spectacle.linearity import sRGB_generic
from matplotlib import pyplot as plt, patches
from scipy.spatial import distance_matrix
from colorio._tools import plot_flat_gamut

FU_LMS_deficiency = np.einsum("caij,fj->cafi",mat.SLMS, fu.FU_LMS) # axes: deficiency (lms), a, FU number, lms
FU_deficient_XYZ = np.einsum("ij,cafj->cafi", mat.M_lms_to_xyz_e, FU_LMS_deficiency) # axes: deficiency (lms), a, FU number, xyz

# Following steps are just for the sRGB demonstration image
FU_deficient_XYZ_D65 = np.einsum("ij,cafj->cafi", mat.M_xyz_e_to_xyz_d65, FU_deficient_XYZ) # axes: deficiency (lms), a, FU number, xyz
FU_deficient_RGB = np.einsum("ij,cafj->cafi", mat.M_xyz_to_rgb, FU_deficient_XYZ_D65) # axes: deficiency (lms), a, FU number, rgb (linear)
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

# Plot chromaticities on gamut
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10,5.2), sharex=True, sharey=True)
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
plt.figure(figsize=(5,2))
plt.plot(*FU_deficient_xyY[0,-1,:,::2].T, "o-", lw=3, label="Regular")
for i, label in enumerate("LMS"):
    plt.plot(*FU_deficient_xyY[i,0,:,::2].T, "o-", lw=3, label=f"{label}-deficient")
plt.xlim(0, 0.55)
plt.ylim(0, 0.55)
plt.xlabel("$x$")
plt.ylabel("$Y$")
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
