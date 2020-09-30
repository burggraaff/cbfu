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

# Plot hue angle for several FU colours as function of deficiency
fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True)
for j, ax in enumerate(axs):
    for FU in (1, 9, 15, 21):
        i = FU-1
        ax.plot(mat.a, FU_deficient_alpha_deg[j,:,i], label=f"FU {FU}", lw=3)
axs[2].set_xlabel("$a$")
axs[1].set_ylabel("Hue angle")
axs[1].legend(loc="best")
axs[0].set_xlim(1, 0)
plt.show()
plt.close()

distances = distance_matrix(fu.FU, fu.FU)
plt.imshow(distances)
plt.colorbar()
plt.show()
plt.close()
