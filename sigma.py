import numpy as np
from pyscf import cc
from .ccgf import *
from pyscf.gw.gw import get_g

# Self-energy matrix of dimension (nmo, nmo)
def get_sigma(cc, omega):
    eta = 1e-3
    nmo = cc.nmo

    # Build G_0
    g0 = get_g(omega, cc._scf.mo_energy, cc._scf.mo_occ, eta)
    g0 = np.diag(g0)
 
    # Build G
    gf = CCGF(cc, eta=eta)
    ps = qs = range(nmo)
    g = gf.solve_ip(ps, qs, omega) + gf.solve_ea(ps, qs, omega)
    g = g[:,:,0] 
    
    sigma = np.linalg.inv(g0) - np.linalg.inv(g)
    return sigma

# Individual element of the self-energy matrix
def get_sigma_element(cc, omega, p, q):
    sigma = get_sigma(cc, omega)
    return sigma[p,q]
