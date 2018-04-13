#!/usr/bin/env python

'''
CCSD with k-point sampling
'''

from pyscf.pbc import gto, scf, cc
from pyscf import cc as mol_cc
import kpts_gf
import gf

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
C 2.52760299659    2.52760299659    2.52760299659
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
mf = scf.RHF(cell, exxdiv=None)
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

