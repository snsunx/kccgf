#!/usr/bin/env python

'''
CCSD with k-point sampling
'''

import numpy as np
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto, scf, cc
from pyscf import cc as mol_cc
import kpts_gf
import gf

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
cell.verbose = 5
cell.precision = 1e-12
cell.build()

cell.rcut *= 2.0
cell.build()

omegas = [-10.99479191]

#
#Run old code
#
supcell = super_cell(cell, nmp)
mf = scf.RHF(supcell, exxdiv=None)
mf.diis = None
mf.conv_tol_grad = 1e-8
mf.conv_tol = 1e-8
ehf = mf.kernel()
mf.analyze()
mycc = cc.RCCSD(mf)
mycc.ip_partition = None
mycc.ea_partition = None
mycc.conv_tol_normt = 1e-10
mycc.conv_tol = 1e-10
mycc.kernel()
p = range(mycc.nocc)
mos = mycc.nmo
q = range(mos-mycc.nocc)
gfunccc = gf.OneParticleGF(mycc)
gf_ip = gfunccc.kernel(p,q,omegas)
print("KRCCSD energy (per unit cell) =", mycc.e_tot)

print "gf"
print gf_ip

bal = 0.0
for iocc in range(8):
    bal += gf_ip[iocc, iocc, 0]
print "trace gf ", bal

#Run new

kpts = cell.make_kpts(nmp)
kpts -= kpts[0]
#kpts = cell.make_kpts([1,1,1])
kmf = scf.KRHF(cell, kpts, exxdiv=None)
kmf.kpts = kpts
kmf.diis = None
kmf.conv_tol_grad = 1e-8
kmf.conv_tol = 1e-8
ehf = kmf.kernel()
kmf.analyze()
mycc = cc.KRCCSD(kmf)
mycc.ip_partition = None
mycc.ea_partition = None
mycc.conv_tol_normt = 1e-10
mycc.conv_tol = 1e-10
mycc.kernel()
p = range(mycc.nocc)
mos = mycc.nmo
q = range(mos-mycc.nocc)
gfunccc = kpts_gf.OneParticleGF(mycc)
kpts_gf_ip = gfunccc.kernel(kpts,p,q,omegas)

print "kpts gf"
print kpts_gf_ip
#print "gf"
#print gf_ip

val = 0.0
for k in range(2):
    for iocc in range(4):
        val += kpts_gf_ip[k, iocc, iocc, 0]
print "trace kpts gf ", val
#print 'molectrace     (-0.679067423896-0.000595898244016j)'
print 'molectrace     ', bal
#print 'molectrace     (-0.700699968762-0.000614590301974j)'
