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

nmp = [1, 2, 1]
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
cell.gs = [12, 12, 12]
cell.verbose = 5
cell.precision = 10e-11
cell.build()

#
#Run old code
#
supcell = super_cell(cell, nmp)
mf = scf.RHF(supcell, exxdiv=None)
mf.conv_tol_grad = 10e-10
mf.conv_tol = 10e-10
mf.diis = None
ehf = mf.kernel()
mycc = cc.RCCSD(mf)
mycc.conv_tol_normt = 10e-10
mycc.conv_tol = 10e-10
mycc.ip_partition = None
mycc.ea_partition = None
mycc.kernel()
print 'nocc ',mycc.nocc
print 'nmo ', mycc.nmo
p = [0,1,2,3,4,5,6,7]
q = [0,1,2,3,4,5,6,7]
#p = [8,9,10,11,12,13,14,15]
#q = [8,9,10,11,12,13,14,15]
omegas = [-10.99479191]
#omegas = np.arange(-10.99479191, -11.04387487,-0.002454148)
#omegas = [-10.99479191, -11.04387487]
gfunccc = gf.OneParticleGF(mycc)
end_gfunc = gfunccc.kernel(p,q,omegas)
print 'ip trace for supercell',np.trace(np.asarray(end_gfunc[0]))
print 'ea trace for supercell',np.trace(np.asarray(end_gfunc[1]))
ipmol=np.trace(np.asarray(end_gfunc[0]))
eamol=np.trace(np.asarray(end_gfunc[1]))

print("KRCCSD energy (per unit cell) =", mycc.e_tot)


#
# Running HF and CCSD for single k-point
#
kpts = cell.make_kpts(nmp)
kpts -= kpts[0]
kmf = scf.KRHF(cell, kpts, exxdiv=None)
kmf.conv_tol_grad = 10e-10
kmf.conv_tol_grad = 10e-10
kmf.kpts = kpts
kmf.diis = None
kmf.verbose = 5
kmf.output = '~/work/gfunc_kpts/indextest/molectest/log.txt'
ehf = kmf.kernel()
mycc = cc.KRCCSD(kmf)
mycc.conv_tol_normt = 10e-10
mycc.conv_tol = 10e-10
mycc.ip_partition = None
mycc.ea_partition = None
mycc.kernel()
print 'nocc ',mycc.nocc
print 'nmo ', mycc.nmo
p = [0,1,2,3]
q = [0,1,2,3]
#p = [4,5,6,7]
#q = [4,5,6,7]
#omegas = np.arange(-10.99479191, -11.04387487,-0.002454148)
#omegas = [-10.99479191, -11.04387487]
omegas = [-10.99479191]
gfunccc = kpts_gf.OneParticleGF(mycc)
end_gfunc = gfunccc.kernel(kpts,p,q,omegas)
print 'kpts',kpts
print 'p',p
print 'q',q
print 'dim gfunc', np.asarray(end_gfunc).shape
print np.asarray(end_gfunc)
print 'ip trace at kpt 1', np.trace(np.asarray(end_gfunc[0][0]))
print 'ip trace at kpt 2', np.trace(np.asarray(end_gfunc[0][1]))
print 'sum ip', np.trace(np.asarray(end_gfunc[0][0]))+np.trace(np.asarray(end_gfunc[0][1]))
print 'ipmol ',ipmol
print 'ea trace at kpt 1', np.trace(np.asarray(end_gfunc[1][0]))
print 'ea trace at kpt 2', np.trace(np.asarray(end_gfunc[1][1]))
print 'sum ea', np.trace(np.asarray(end_gfunc[1][0]))+np.trace(np.asarray(end_gfunc[1][1]))
print 'eamol ',eamol
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
