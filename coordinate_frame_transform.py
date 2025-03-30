import numpy as np 

# ---------- Rotation Matrix ----------
def rotation_matrix(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    ctheta, stheta = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    R = np.array([
        [ctheta * cpsi,                     ctheta * spsi,                      -stheta],
        [sphi * stheta * cpsi - cphi * spsi, sphi * stheta * spsi + cphi * cpsi, sphi * ctheta],
        [cphi * stheta * cpsi + sphi * spsi, cphi * stheta * spsi - sphi * cpsi, cphi * ctheta]
    ])
    return R