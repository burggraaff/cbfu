import numpy as np
import mat
import fu
from spectacle.linearity import sRGB_generic

FU_LMS_deficiency = np.einsum("caij,fj->cafi",mat.SLMS, fu.FU) # axes: deficiency (lms), a, FU number, lms
FU_deficient_XYZ = np.einsum("ij,cafj->cafi", mat.M_lms_to_xyz, FU_LMS_deficiency) # axes: deficiency (lms), a, FU number, xyz
FU_deficient_RGB = np.einsum("ij,cafj->cafi", mat.M_xyz_to_rgb, FU_deficient_XYZ) # axes: deficiency (lms), a, FU number, rgb (linear)

FU_deficient_sRGB = sRGB_generic(FU_deficient_RGB, normalization=1).astype(np.uint8) # Gamma-expanded (non-linear) sRGB values. Note these are clipped to 0-255 to accommodate the limited gamut of sRGB.

example_indices = ((0, 0, 0, 1, 1, 2, 2), (0, 50, -1, 50, -1, 50, -1))
examples_sRGB = FU_deficient_sRGB[example_indices]
