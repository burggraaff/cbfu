import numpy as np

JND = 2.3


M_rgb_to_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                         [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])

M_xyz_to_rgb = np.linalg.inv(M_rgb_to_xyz)

# https://en.wikipedia.org/wiki/LMS_color_space#Hunt.2C_RLAB
M_xyz_d65_to_lms = np.array([[ 0.4002, 0.7076, -0.0808],
                             [-0.2263, 1.1653,  0.0457],
                             [ 0     , 0     ,  0.9182]])

M_lms_to_xyz_d65 = np.linalg.inv(M_xyz_d65_to_lms)

# https://en.wikipedia.org/wiki/LMS_color_space#Hunt.2C_RLAB
M_xyz_e_to_lms = np.array([[ 0.38971, 0.68898, -0.07868],
                           [-0.22981, 1.18340,  0.04641],
                           [ 0      , 0      ,  1.     ]])

M_lms_to_xyz_e = np.linalg.inv(M_xyz_e_to_lms)

# http://www.brucelindbloom.com/index.html?Eqn_ChromAdapt.html - Bradford
M_xyz_e_to_xyz_d55 = np.array([[ 0.9797934, -0.0133624, -0.0096110],
                               [-0.0216865,  1.0235002, -0.0018137],
                               [-0.0037969,  0.0075580,  0.9177289]])

M_xyz_e_to_xyz_d65 = np.array([[ 0.9531874, -0.0265906,  0.0238731],
                               [-0.0382467,  1.0288406,  0.0094060],
                               [ 0.0026068, -0.0030332,  1.0892565]])

M_xyz_e_to_xyz_d75 = np.array([[ 0.9344831, -0.0355691,  0.0508059],
                               [-0.0490066,  1.0307122,  0.0182943],
                               [ 0.0079472, -0.0119897,  1.2304226]])

red_RGB = np.array([1,0,0])
blue_RGB = np.array([0,0,1])
white_RGB = np.array([1,1,1])

red_XYZ = M_rgb_to_xyz @ red_RGB
blue_XYZ = M_rgb_to_xyz @ blue_RGB
white_XYZ = M_rgb_to_xyz @ white_RGB

red_LMS = M_xyz_d65_to_lms @ red_XYZ
blue_LMS = M_xyz_d65_to_lms @ blue_XYZ
white_LMS = M_xyz_d65_to_lms @ white_XYZ

a = np.linspace(0, 1, 101)
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
# From regular LMS to deficient LMS
SLMS = np.stack([SL, SM, SS]) # Axes: deficiency (lms), a, i, j

# Cone monochromats
S_mono_cone = np.array([[0, 0, 1],
                        [0, 0, 1],
                        [0, 0, 1]])


# Convert CIE XYZ to CIE L*a*b*
# https://en.wikipedia.org/wiki/CIELAB_color_space#Forward_transformation
def XYZ_to_Lab(XYZ, Xn=1., Yn=1., Zn=1.):
    @np.vectorize
    def f(t):
        delta = 6/29
        f = t**(1/3) if t > delta**3 else (t/(3*delta**2) + 4/29)
        return f

    X, Y, Z = XYZ[...,0], XYZ[...,1], XYZ[...,2]

    Lstar = 116 * f(Y/Yn) - 16
    astar = 500 * (f(X/Xn) - f(Y/Yn))
    bstar = 200 * (f(Y/Yn) - f(Z/Zn))

    Lab = np.stack([Lstar, astar, bstar], axis=-1)
    return Lab

# Calculate CIE Delta E00 distance
# doi 10.1002/col.20070
@np.vectorize
def hue(ap, bs):
    return 0. if ap == bs == 0. else np.rad2deg(np.arctan2(bs, ap)) % 360

@np.vectorize
def delta_hue(h1, h2, Cprime1, Cprime2):
    if Cprime1 * Cprime2 == 0:
        dh = 0
    elif np.abs(h2 - h1) <= 180:
        dh = h2 - h1
    elif h2 - h1 > 180:
        dh = h2 - h1 - 360
    else:
        dh = h2 - h1 + 360
    return dh

@np.vectorize
def hprimebar(hprime1, hprime2, Cprime1, Cprime2):
    if Cprime1 * Cprime2 == 0:
        hpb = hprime1 + hprime2
    elif np.abs(hprime1 - hprime2) <= 180:
        hpb = (hprime1 + hprime2) / 2
    elif np.abs(hprime1 - hprime2) > 180 and hprime1 + hprime2 < 360:
        hpb = (hprime1 + hprime2 + 360) / 2
    else:
        hpb = (hprime1 + hprime2 - 360) / 2
    return hpb

@np.vectorize
def dE00(L1, a1, b1, L2, a2, b2, kL=1., kC=1., kH=1.):
    L, a, b = np.array([L1, L2]), np.array([a1, a2]), np.array([b1, b2])

    C = np.sqrt(a**2 + b**2)
    Cbar = C.mean()
    G = 0.5 * (1 - np.sqrt(Cbar**7 / (Cbar**7 + 25**7)))
    aprime = (1 + G) * a
    Cprime = np.sqrt(aprime**2 + b**2)

    h = hue(aprime, b)

    dL = L2 - L1
    dCprime = Cprime[1] - Cprime[0]

    dh = delta_hue(*h, *Cprime)
    dH = 2 * np.sqrt(Cprime[0] * Cprime[1]) * np.sin(np.deg2rad(dh)/2)

    Lprimebar = L.mean()
    Cprimebar = Cprime.mean()

    hbar = hprimebar(*h, *Cprime)

    T = 1 - 0.17*np.cos(np.deg2rad(hbar-30)) + 0.24*np.cos(np.deg2rad(2*hbar)) + 0.32*np.cos(np.deg2rad(3*hbar+6)) - 0.20*np.cos(np.deg2rad(4*hbar-63))

    dtheta = 30 * np.exp(-((hbar-275)/25)**2)

    Rc = 2*np.sqrt(Cprimebar**7 / (Cprimebar**7 + 25**7))

    SL = 1 + (0.015 * (Lprimebar - 50)**2) / np.sqrt(20 + (Lprimebar - 50)**2)
    SC = 1 + 0.045 * Cprimebar
    SH = 1 + 0.015 * Cprimebar * T
    RT = -np.sin(np.deg2rad(2*dtheta)) * Rc

    dE00 = np.sqrt((dL/(kL * SL))**2 + (dCprime/(kC * SC))**2 + (dH/(kH * SH))**2 + RT * (dCprime/(kC*SC)) * (dH/(kH*SH)))
    return dE00