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

# L-weak
q2l = (1 - a) * (blue_LMS[1] - blue_LMS[0]) / (blue_LMS[1] - blue_LMS[2])
q1l = 1 - a - q2l


# M-weak
q2m = (1 - a) * (blue_LMS[0] - blue_LMS[1]) / (blue_LMS[0] - blue_LMS[2])
q1m = 1 - a - q2m

# S-weak
q2s = (1 - a) * (red_LMS[0] - red_LMS[2]) / (red_LMS[0] - red_LMS[1])
q1s = 1 - a - q2s


# Cone monochromats
S_mono_cone = np.array([[0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1]])
