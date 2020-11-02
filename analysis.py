import numpy as np
import mat
import fu
from spectacle.linearity import sRGB_generic
from matplotlib import pyplot as plt, patches, cm
from colorio._tools import plot_flat_gamut

# Applied Optics column widths
col1 = 3.25
col2 = 5.75

FU_LMS_deficiency = np.einsum("caij,fj->cafi",mat.SLMS, fu.FU_LMS) # axes: deficiency (lms), a, FU number, lms
FU_deficient_XYZ = np.einsum("ij,cafj->cafi", mat.M_lms_to_xyz_e, FU_LMS_deficiency) # axes: deficiency (lms), a, FU number, xyz

# Following steps are just for the sRGB demonstration image
FU_deficient_XYZ_D65 = np.einsum("ij,cafj->cafi", mat.M_xyz_e_to_xyz_d65, FU_deficient_XYZ) # axes: deficiency (lms), a, FU number, xyz
FU_deficient_RGB = np.einsum("ij,cafj->cafi", mat.M_xyz_to_rgb, FU_deficient_XYZ_D65) # axes: deficiency (lms), a, FU number, rgb (linear)
FU_deficient_sRGB = sRGB_generic(FU_deficient_RGB, normalization=1)/255. # Gamma-expanded (non-linear) sRGB values. Note these are clipped to 0-255 to accommodate the limited gamut of sRGB.

example_indices = ((0, 0, 0, 1, 1, 2, 2), (-1, 50, 0, 50, 0, 50, 0))
examples_sRGB = FU_deficient_sRGB[example_indices]
examples_labels = ["Regular", "50% L-def.", "Fully L-def.", "50% M-def.", "Fully M-def.", "50% S-def.", "Fully S-def."]

# Color squares plot
kwargs = {"width": 0.9, "height": 0.9, "edgecolor": "none"}
fig, ax = plt.subplots(figsize=(col2, 2))
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
ax.set_xlabel("Forel-Ule color")
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
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(col2,3.5), sharex=True, sharey=True)
axs[0,0].axis("off")
for ax, xy, label in zip(axs.ravel()[1:], FU_deficient_xy[example_indices], examples_labels):
    plt.sca(ax)
    plot_flat_gamut(plot_planckian_locus=False, axes_labels=("", ""))
    ax.scatter(*xy.T, c="k", marker="o", s=4, label="FU colors")
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
fig.suptitle("Forel-Ule color gamut for various cone deficiencies")
plt.savefig("gamut.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Matrices to select diagonal and off-diagonal elements
diag = np.eye(21, dtype=bool)
off_diag = ~diag

FU_deficient_lab = mat.XYZ_to_Lab(FU_deficient_XYZ)

# Plot L*, a*, b* as a function of FU at full deficiency
ylabels = ["L$^*$", "a$^*$", "b$^*$"]
formats = ["^-", "s-", "p-"]
fig, axs = plt.subplots(nrows=3, figsize=(col1,4), sharex=True, gridspec_kw={"hspace": 0.07})
for k, (ax, ylabel) in enumerate(zip(axs, ylabels)):
    ax.plot(fu.numbers, FU_deficient_lab[0,-1,:,k], "o-", lw=3, c='k', label="Regular")
    for i, (label, fmt) in enumerate(zip("LMS", formats)):
        ax.plot(fu.numbers, FU_deficient_lab[i,0,:,k].T, fmt, lw=3, label=f"{label}-def.")
    ax.set_ylabel(ylabel)
    ax.grid(ls="--", color="0.7")
for ax in axs[:-1]:
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
axs[0].set_xlim(0.9, 21.1)
axs[0].set_xticks([1, 5, 10, 15, 20])
axs[-1].set_xlabel("Forel-Ule color")
axs[0].set_title("Forel-Ule colors in CIE L$^*$a$^*$b$^*$ space")
axs[2].legend(ncol=2, loc="center", bbox_to_anchor=(0.5,-0.8))
fig.align_labels()
plt.savefig("FU_Lab.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Calculate Delta E 00
L1, a1, b1 = FU_deficient_lab[...,0][:,:,np.newaxis,:], FU_deficient_lab[...,1][:,:,np.newaxis,:], FU_deficient_lab[...,2][:,:,np.newaxis,:]
L2, a2, b2 = FU_deficient_lab[...,0][:,:,:,np.newaxis], FU_deficient_lab[...,1][:,:,:,np.newaxis], FU_deficient_lab[...,2][:,:,:,np.newaxis]

distances_Lab = mat.dE00(L1, a1, b1, L2, a2, b2)
distances_Lab_regular = distances_Lab[0,-1]
distances_Lab_JND = distances_Lab/mat.JND

extreme_indices = ((0, 0, 1, 2), (-1, 0, 0, 0))
extreme_labels = ["Regular", "L-def.", "M-def.", "S-def."]

# Plot distance matrices
def plot_distance_matrices(FU_distance_matrices, saveto="image.pdf", title="", ylabel="Euclidean distance (XYZ)", nr_samples=None, **kwargs):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(col2,5), gridspec_kw={"wspace": 0.3, "hspace": 0.3})
    for ax, distances, label in zip(axs.ravel(), FU_distance_matrices[extreme_indices], extreme_labels):
        im = ax.imshow(distances, extent=(0, 21, 21, 0), cmap=cm.get_cmap("cividis_r", nr_samples), **kwargs)
        ax.set_title(f"\n{label}")
        ax.set_xlim(0, 21)
        ax.set_ylim(0, 21)
        ax.set_xticks([0.5, 10.5, 20.5])
        ax.set_xticklabels([1, 11, 21])
        ax.set_yticks([0.5, 10.5, 20.5])
        ax.set_yticklabels([1, 11, 21])
    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.83, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cax, orientation="vertical")
    cax.set_ylabel(ylabel)
    fig.suptitle(title)
    plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()

# Plot distance matrices
plot_distance_matrices(distances_Lab, saveto="distance_matrix_Lab.pdf", vmin=0, vmax=50, title="$\Delta E_{00}$ between Forel-Ule colors", ylabel="$\Delta E_{00}$", nr_samples=10)
plot_distance_matrices(distances_Lab_JND, saveto="distance_matrix_Lab_JND.pdf", vmin=0, vmax=20, title="$\Delta E_{00}$ between Forel-Ule colors", ylabel="$\Delta E_{00}$ / JND", nr_samples=8)
plot_distance_matrices(distances_Lab_JND, saveto="distance_matrix_Lab_JND_zoom.pdf", vmin=0, vmax=4, nr_samples=4, title="Confusion matrix for Forel-Ule colors", ylabel="$\Delta E_{00}$ / JND")

# Distance as a function of a
median_distance_Lab = np.median(distances_Lab[...,off_diag], axis=2)
min_distance_Lab = np.min(distances_Lab[...,off_diag], axis=2)

# Number of pairs that are less than a JND apart
nr_under_JND = (np.sum(distances_Lab_JND[...,off_diag] < 1, axis=2))//2

# Combined plot of distance statistics
fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(col1,5))
for ax, dist, ylabel in zip(axs, [median_distance_Lab, min_distance_Lab, nr_under_JND], ["Median $\Delta E_{00}$", "Minimum $\Delta E_{00}$", "Pairs $<$JND"]):
    for i, label in enumerate("LMS"):
        ax.plot(mat.a, dist[i], lw=3, label=f"{label}-def.")
    ax.set_ylim(ymin=0)
    ax.grid(ls="--", c="0.7")
    ax.set_ylabel(ylabel)
for ax in axs[:2]:
    ax.axhline(mat.JND, c='k', lw=3, ls="dotted", label=f"JND")
axs[2].set_yticks([0,2,4,6])
axs[-1].set_xlim(1, 0)
axs[-1].set_xticks([1, 0.75, 0.5, 0.25, 0])
axs[-1].set_xlabel("Relative cone contribution $a$")
axs[0].set_title("Discrimination of FU colors")
axs[1].legend(loc="best", ncol=2)
fig.align_labels()
plt.savefig("distance_stats_Lab.pdf", bbox_inches="tight")
plt.show()
plt.close()

def find_matching_pairs(data, threshold):
    matching_pairs = np.array(np.where(data <= threshold)).T
    matching_pairs = matching_pairs[matching_pairs[:,0] != matching_pairs[:,1]]  # Remove diagonals
    matching_pairs = np.sort(matching_pairs) # Sort horizontally, eg [1, 0] becomes [0, 1]
    matching_pairs = np.unique(matching_pairs, axis=0) # Remove duplicates
    matching_pairs += 1 # From index to FU color
    return matching_pairs

# Find pairs that are < JND
print("Pairs within 1 JND from each other:")
for data, label in zip(distances_Lab_JND[extreme_indices], extreme_labels):
    matching_pairs_1 = find_matching_pairs(data, 1)
    print(f"{label} vision:\n{matching_pairs_1}\n")
print("-----")

# Find pairs that are almost < JND
threshold_close = 2
print(f"Pairs within {threshold_close:.1f} JND from each other:")
for data, label in zip(distances_Lab_JND[extreme_indices], extreme_labels):
    matching_pairs_2 = find_matching_pairs(data, threshold_close)
    print(f"{label} vision:\n{matching_pairs_2}\n")
