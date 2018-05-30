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
mycc.frozen = [0,1,3,4,5,6,9,10,11,12,14,15]
mycc.ip_partition = None
mycc.ea_partition = None
mycc.conv_tol_normt = 1e-10
mycc.conv_tol = 1e-10
mycc.kernel()
#p=[8,9,10,11,12,13,14,15]
#q=[8,9,10,11,12,13,14,15]
p=[2,3]
q=[2,3]
gfunccc = gf.OneParticleGF(mycc)
gf_ea = gfunccc.kernel(p,q,omegas)
print("KRCCSD energy (per unit cell) =", mycc.e_tot)

print "gf"
print gf_ea

bal = 0.0
#for iocc in range(8):
#    bal += gf_ea[iocc, iocc, 0]
sc_nocc = 2
if gf_ea.ndim==3:
    for iocc in range(sc_nocc):
        bal += gf_ea[iocc,iocc,0]
else:
    bal += gf_ea[0]

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
mycc.frozen = [[0,1,2,5,6,7],[0,2,3,4,5,7]]
mycc.ip_partition = None
mycc.ea_partition = None
mycc.conv_tol_normt = 1e-10
mycc.conv_tol = 1e-10
mycc.kernel()
#p=[4,5,6,7]
#q=[4,5,6,7]
p=[1]
q=[1]
#p = range(mycc.nocc)
#mos = mycc.nmo
#q = range(mos-mycc.nocc)
gfunccc = kpts_gf.OneParticleGF(mycc)
kpts_gf_ea = gfunccc.kernel(kpts,p,q,omegas)

print "kpts gf"
print kpts_gf_ea
#print "gf"
#print gf_ip

#val = 0.0
#for k in range(2):
#    for iocc in range(4):
#        val += kpts_gf_ea[k, iocc, iocc, 0]
nkpts=len(kpts)
val = 0.0
nocc_per_kpt = [2,2]
if kpts_gf_ea.ndim==4:
    for k in range(nkpts):
        for iocc in range(nocc_per_kpt[k]):
            val+=kpts_gf_ea[k,iocc,iocc,0]
else:
    for k in range(nkpts):
        val+=kpts_gf_ea[k,0]

print "trace kpts gf ", val
print "molec ip ",      bal
#print 'molec ip       (-0.164216181839+0.000134847437435j)'
#with 2 bottom einsums
#print 'trace gf       (-0.650489787201+0.000529207363046j)'
#all einsums
#print 'molectrace     (-0.679067423896-0.000595898244016j)'
#print 'molectrace     (-0.700699968762-0.000614590301974j)'
#print 'molectrace     (-0.629996672535+0.000512440754188j)'
#two middle between einsums, no einsums
#print 'molectrace     (-0.650489987856+0.000529207681403j)'
