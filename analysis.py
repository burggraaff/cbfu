import numpy as np
import mat
import fu
from spectacle.linearity import sRGB_generic
from matplotlib import pyplot as plt, patches, cm
from mpl_toolkits.axes_grid1 import AxesGrid
from colorio._tools import plot_flat_gamut

# PLOS ONE column and maximum width
col = 5.2
maxwidth = 7.5

FU_LMS_deficiency = np.einsum("caij,fj->cafi",mat.SLMS, fu.FU_LMS) # axes: deficiency (lms), a, FU number, lms
FU_deficient_XYZ = np.einsum("ij,cafj->cafi", mat.M_lms_to_xyz_e, FU_LMS_deficiency) # axes: deficiency (lms), a, FU number, xyz

# Following steps are just for the sRGB demonstration image
FU_deficient_XYZ_D65 = np.einsum("ij,cafj->cafi", mat.M_xyz_e_to_xyz_d65, FU_deficient_XYZ) # axes: deficiency (lms), a, FU number, xyz
FU_deficient_RGB = np.einsum("ij,cafj->cafi", mat.M_xyz_to_rgb, FU_deficient_XYZ_D65) # axes: deficiency (lms), a, FU number, rgb (linear)
FU_deficient_sRGB = sRGB_generic(FU_deficient_RGB, normalization=1)/255. # Gamma-expanded (non-linear) sRGB values. Note these are clipped to 0-255 to accommodate the limited gamut of sRGB.

example_indices = ((0, 0, 0, 1, 1, 2, 2), (-1, 50, 0, 50, 0, 50, 0))
examples_sRGB = FU_deficient_sRGB[example_indices]
examples_labels = ["Regular", "Protanomaly", "Protanopia", "Deuteranomaly", "Deuteranopia", "Tritanomaly", "Tritanopia"]

extreme_indices = ((0, 0, 1, 2), (-1, 0, 0, 0))
extreme_labels = ["Regular", "Protan", "Deutan", "Tritan"]

# Chromaticities
FU_deficient_xy = FU_deficient_XYZ[...,:2] / FU_deficient_XYZ.sum(axis=3)[...,np.newaxis]

# Plot chromaticities on gamut
kwargs = {"width": 0.9, "height": 0.9, "edgecolor": "none"}
plt.figure(figsize=(col,4))
plot_flat_gamut(plot_planckian_locus=False, axes_labels=("", ""))
plt.scatter(*FU_deficient_xy[0,-1].T, c="k", marker="o", s=25)
plt.plot(*FU_deficient_xy[0,-1].T, c="k")
rectangles = [patches.Rectangle(xy=(-1,-1), facecolor=rgb, label=j, **kwargs) for j, rgb in enumerate(examples_sRGB[0], start=1)]
for rect in rectangles:
    plt.gca().add_patch(rect)
plt.xlim(-0.05, 0.75)
plt.ylim(-0.05, 0.9)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.title("Forel-Ule scale")
plt.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, -0.15), frameon=False, markerfirst=False, fontsize="large", columnspacing=1.3, borderpad=0, labelspacing=0.1, handletextpad=0.5)
plt.savefig("gamut.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Colour squares plot
fig, ax = plt.subplots(figsize=(col, 2))
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

# Matrices to select diagonal and off-diagonal elements
diag = np.eye(21, dtype=bool)
off_diag = ~diag

FU_deficient_lab = mat.XYZ_to_Lab(FU_deficient_XYZ)

# Plot L*, a*, b* as a function of FU at full deficiency
ylabels = ["L$^*$", "a$^*$", "b$^*$"]
formats = ["^-", "s-", "p-"]
colours = ["#D81B60", "#FFC107", "#1E88E5"]
fig, axs = plt.subplots(nrows=3, figsize=(col,4), sharex=True, gridspec_kw={"hspace": 0.07})
for k, (ax, ylabel) in enumerate(zip(axs, ylabels)):
    ax.plot(fu.numbers, FU_deficient_lab[0,-1,:,k], "o-", lw=3, c="#004D40", label="Regular")
    for i, (label, fmt, c) in enumerate(zip(extreme_labels[1:], formats, colours)):
        ax.plot(fu.numbers, FU_deficient_lab[i,0,:,k].T, fmt, lw=3, label=label, c=c)
    ax.set_ylabel(ylabel)
    ax.grid(ls="--", color="0.7")
for ax in axs[:-1]:
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
axs[0].set_xlim(0.9, 21.1)
axs[0].set_xticks([1, 5, 10, 15, 20])
axs[-1].set_xlabel("Forel-Ule colour")
axs[0].set_title("Forel-Ule colours in CIE Lab space")
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

# Combined distance matrix plot
fig = plt.figure(figsize=(col, 3.2))
grid = AxesGrid(fig, 111, nrows_ncols=(2, 4), axes_pad=0.1, cbar_mode="edge", cbar_location="right", cbar_pad=0.1, cbar_size="13%")
for ax, dist, label in zip(grid.axes_row[0], distances_Lab[extreme_indices], extreme_labels):
    im = ax.imshow(dist, extent=(0, 21, 21, 0), cmap=cm.get_cmap("cividis_r", 10), vmin=0, vmax=50)
    ax.set_title(label)
cbar = ax.cax.colorbar(im)
cbar.cbar_axis.set_ticks(np.arange(0,51,10))
cbar.ax.set_ylabel("$\Delta E_{00}$")

for ax, dist in zip(grid.axes_row[1], distances_Lab_JND[extreme_indices]):
    im = ax.imshow(dist, extent=(0, 21, 21, 0), cmap=cm.get_cmap("cividis_r", 4), vmin=0, vmax=4)
cbar = ax.cax.colorbar(im)
cbar.cbar_axis.set_ticks(np.arange(0,5,1))
cbar.ax.set_ylabel("$\Delta E_{00}$ / JND")

for ax in grid:
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 21)
    ax.set_xticks([0.5, 10.5, 20.5])
    ax.set_xticklabels([1, 11, 21])
    ax.set_yticks([0.5, 10.5, 20.5])
    ax.set_yticklabels([1, 11, 21])

for ax in np.ravel(grid.axes_column[1:]):
    ax.tick_params("y", left=False, labelleft=False)
for ax in grid.axes_row[0]:
    ax.tick_params("x", bottom=False, labelbottom=False)

for ax in grid.axes_column[0]:
    ax.set_ylabel("FU")
for ax in grid.axes_row[1]:
    ax.set_xlabel("FU")

fig.suptitle("Forel-Ule confusion matrix")
plt.savefig("difference_matrix_combined.pdf", bbox_inches="tight")
plt.show()
plt.close()

# Distance as a function of a
median_distance_Lab = np.median(distances_Lab[...,off_diag], axis=2)
min_distance_Lab = np.min(distances_Lab[...,off_diag], axis=2)

# Number of pairs that are less than a JND apart
nr_under_JND = (np.sum(distances_Lab_JND[...,off_diag] < 1, axis=2))//2
nr_under_3_JND = (np.sum(distances_Lab_JND[...,off_diag] < 3, axis=2))//2

# Combined plot of distance statistics
# Make 2x2? Median/Min on left, Pairs on the right
fig, axs = plt.subplots(nrows=4, sharex=True, figsize=(col,6))
for ax, dist, ylabel in zip(axs, [median_distance_Lab, min_distance_Lab, nr_under_3_JND, nr_under_JND], ["Median $\Delta E_{00}$", "Minimum $\Delta E_{00}$", "Pairs $<$ 3 JND", "Pairs $<$JND"]):
    for i, (label, c) in enumerate(zip(extreme_labels[1:], colours)):
        ax.plot(mat.k, dist[i], lw=3, label=label, c=c)
    ax.set_ylim(ymin=0)
    ax.grid(ls="--", c="0.7")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, np.nanmax(dist)*1.05)
    ax.locator_params("y", nbins=5)
for ax in axs[:2]:
    ax.axhline(mat.JND, c="#004D40", lw=3, ls="dotted", label=f"JND")
axs[-1].set_xlim(1, 0)
axs[-1].set_xticks([1, 0.75, 0.5, 0.25, 0])
axs[-1].set_xlabel("Relative cone contribution $k$")
axs[0].set_title("Discriminability of FU colours")
axs[1].legend(loc="best", ncol=2)
fig.align_labels()
plt.savefig("difference_stats_Lab.pdf", bbox_inches="tight")
plt.show()
plt.close()

def find_matching_pairs(data, threshold):
    matching_pairs = np.array(np.where(data <= threshold)).T
    matching_pairs = matching_pairs[matching_pairs[:,0] != matching_pairs[:,1]]  # Remove diagonals
    matching_pairs = np.sort(matching_pairs) # Sort horizontally, eg [1, 0] becomes [0, 1]
    matching_pairs = np.unique(matching_pairs, axis=0) # Remove duplicates
    matching_pairs += 1 # From index to FU colour
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
