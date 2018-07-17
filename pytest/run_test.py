'''
CCSD with k-point sampling
'''

import unittest
import numpy as np
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto, scf, cc
from pyscf import cc as mol_cc
import kpts_gf
import gf
import kpts_gf_eascript

class Test(unittest.TestCase):
    def test_ip(self):
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

        kpts = cell.make_kpts(nmp)
        kpts -= kpts[0]
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
        #p=[4,5,6,7]
        #q=[4,5,6,7]
        p=[0,1,2,3]
        q=[0,1,2,3]
        gfunccc = kpts_gf.OneParticleGF(mycc)
        kpts_gf_ea = gfunccc.kernel(kpts,p,q,omegas)

        val = 0.0
        for k in range(2):
            for iocc in range(4):
                val += kpts_gf_ea[k, iocc, iocc, 0]

        print "trace kpts gf ", val
        print 'molectrace     (-0.679067423896-0.000595898244016j)'

        assert (val.real)-(-0.679067423986) < 0.0000001
        assert (val.imag)-(-0.000595898244016) < 0.0000001

    def test_ea(self):
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

        kpts = cell.make_kpts(nmp)
        kpts -= kpts[0]
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
        p=[4,5,6,7]
        q=[4,5,6,7]
        gfunccc = kpts_gf_eascript.OneParticleGF(mycc)
        kpts_gf_ea = gfunccc.kernel(kpts,p,q,omegas)

        val = 0.0
        for k in range(2):
            for iocc in range(4):
                val += kpts_gf_ea[k, iocc, iocc, 0]

        print "trace kpts gf ", val
        print 'molectrace     (-0.679067423896-0.000595898244016j)'

        assert (val.real)-(-0.629996672535) < 0.0000001
        assert (val.imag)-(0.000512440754188) < 0.0000001



#print "molec ip      ",bal
#print 'molec ip       (-0.6108319803645543+0.0004967588252764139j)'
#ea everything except second part of first einsum block
#print  'molec ip       (-0.161690898186+0.000132762266567j)'

#ea both bottom einsums
#print 'trace gf        (-0.164215950151+0.000134847089655j)'
#ea bottom first einsum only
#print 'molec ip       (-0.164215926408+0.000134847052123j)'
#print 'molec ip       (-0.164216181839+0.000134847437435j)'
#with 2 botstdom einsums
#print 'trace afgf       (-0.650489787201+0.000529207363046j)'
#all einsumsdsf
#print 'molectrace     (-0.679067423896-0.000595898244016j)'
#print 'molectrace     (-0.700699968762-0.000614590301974j)'
#print 'molectrace     (-0.629996672535+0.000512440754188j)'
#two middle between einsums, no einsums
#print 'molectrace     (-0.650489987856+0.000529207681403j)'
