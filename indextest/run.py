#!/usr/bin/env python

'''
CCSD with k-point sampling
'''

from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto, scf, cc
from pyscf import cc as mol_cc
import kpts_gf
import gf
import numpy as np

nmp = [2, 1, 1]
cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 5
cell.build()

#
#Run old code
#
supcell = super_cell(cell, nmp)
mf = scf.RHF(supcell, exxdiv=None)
mf.diis = None
ehf = mf.kernel()
mycc = cc.RCCSD(mf)
mycc.ip_partition = None
mycc.ea_partition = None
mycc.kernel()
p = mycc.nocc
mos = mycc.nmo
q = mos-p
omegas = [-10.99479191, -11.04387487]
gfunccc = gf.OneParticleGF(mycc)
gfunccc.kernel(p,q,omegas)
print("KRCCSD energy (per unit cell) =", mycc.e_tot)


#
# Running HF and CCSD for single k-point
#
kpts = cell.make_kpts(nmp)
kpts -= kpts[0]
kmf = scf.KRHF(cell, kpts, exxdiv=None)
kmf.kpts = kpts
kmf.diis = None
ehf = kmf.kernel()
mycc = cc.KRCCSD(kmf)
mycc.ip_partition = None
mycc.ea_partition = None
mycc.kernel()
p = mycc.nocc
mos = mycc.nmo
q = mos-p
omegas = np.arange(-10.99479191, -11.04387487,-0.002454148)
#omegas = [-10.99479191, -11.04387487]
gfunccc = kpts_gf.OneParticleGF(mycc)
gfunccc.kernel(kpts,p,q,omegas)
print 'kpts',kpts
print 'p',p
print 'q',q
print("KRCCSD energy (per unit cell) =", mycc.e_tot)


'''
#
# Running HF and CCSD with 2x2x2 k-points
#
kpts = cell.make_kpts([2,2,2])
kmf = scf.KRHF(cell, exxdiv=None)
kmf.kpts = kpts
ehf = kmf.kernel()

mycc = cc.KCCSD(kmf)
mycc.kernel()
print("KRCCSD energy (per unit cell) =", mycc.e_tot)

3.37013732878
'''
