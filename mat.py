import numpy as np



M_rgb_to_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])

M_xyz_to_lms = np.array([[ 0.4002, 0.7076, -0.0808],
                         [-0.2263, 1.1653,  0.0457],
                         [ 0     , 0     ,  0.9182]])

M_lms_to_xyz = np.linalg.inv(M_xyz_to_lms)

red_RGB = np.array([1,0,0])
blue_RGB = np.array([0,0,1])
white_RGB = np.array([1,1,1])

red_XYZ = M_rgb_to_xyz @ red_RGB
blue_XYZ = M_rgb_to_xyz @ blue_RGB
white_XYZ = M_rgb_to_xyz @ white_RGB

red_LMS = M_xyz_to_lms @ red_XYZ
blue_LMS = M_xyz_to_lms @ blue_XYZ
white_LMS = M_xyz_to_lms @ white_XYZ

a = np.linspace(0, 1, 100)
zeros = np.zeros_like(a)
ones = np.ones_like(a)

# L-weak
q2l = (1 - a) * (blue_LMS[1] - blue_LMS[0]) / (blue_LMS[1] - blue_LMS[2])
q1l = 1 - a - q2l
SL = np.stack([a, q1l, q2l, zeros, ones, zeros, zeros, zeros, ones], axis=1).reshape(-1, 3, 3)


# M-weak
q2m = (1 - a) * (blue_LMS[0] - blue_LMS[1]) / (blue_LMS[0] - blue_LMS[2])
q1m = 1 - a - q2m
SM = np.stack([ones, zeros, zeros, q1m, a, q2m, zeros, zeros, ones], axis=1).reshape(-1, 3, 3)

# S-weak
q2s = (1 - a) * (red_LMS[0] - red_LMS[2]) / (red_LMS[0] - red_LMS[1])
q1s = 1 - a - q2s
SS = np.stack([ones, zeros, zeros, zeros, ones, zeros, q1s, q2s, a], axis=1).reshape(-1, 3, 3)

# LMS-weak combined
SLMS = np.stack([SL, SM, SS]) # Axes: deficiency (lms), a, i, j

# Cone monochromats
S_mono_cone = np.array([[0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1]])
