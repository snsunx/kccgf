#!/usr/bin/env python

import sys
from numpy import linalg as LA
import numpy as np
import scipy.sparse.linalg as spla
from math import *

from pyscf import gto
from pyscf import scf
from pyscf import cc

import antisymeri
from ccsd import ccsd
import eom_driver
#from eom_utils import *
#import eom_ee
#import eom_ea
#import eom_ip
#from eom_ip import *

import matplotlib.pyplot as plt

class DIIS:
# J. Mol. Struct. 114, 31-34
# PCCP, 4, 11
# GEDIIS, JCTC, 2, 835
# C2DIIS, IJQC, 45, 31
# SCF-EDIIS, JCP 116, 8255
# DIIS try to minimize the change of the input vectors. It rotates the vectors
# to minimize the error in the least square sense.
    '''
diis.space is the maximum of the allowed space
diis.min_space is the minimal number of vectors to store before damping'''
    def __init__(self):
        self._vec_stack = []
        self._error_vec_stack = []
        self.conv_tol = 1e-6
        self.space = 20
        self.min_space = 1

    def push_vec(self, x):
        self._vec_stack.append(x.copy())
        if self._vec_stack.__len__() > self.space:
            self._vec_stack.pop(0)

    def push_error_vec(self, x):
        self._error_vec_stack.append(x.copy())
        if self._error_vec_stack.__len__() > self.space:
            self._error_vec_stack.pop(0)

    def get_err_vec(self, idx):
        return self._error_vec_stack[idx+1]

    def get_vec(self, idx):
        return self._vec_stack[idx+1]

    def get_num_diis_vec(self):
        return self._vec_stack.__len__() - 1

    def update(self, x, err):
        '''use DIIS method to solve Eq.  operator(x) = x.'''
        self.push_vec(x)
        self.push_error_vec(err)

        nd = self.get_num_diis_vec()
        if nd <= self.min_space:
            return x

        H = np.ones((nd+1,nd+1), x.dtype)
        H[0,0] = 0
        G = np.zeros(nd+1, x.dtype)
        G[0] = 1
        for i in range(nd):
            dti = self.get_err_vec(i)
            for j in range(i+1):
                dtj = self.get_err_vec(j)
                H[i+1,j+1] = np.dot(np.array(dti).ravel(), \
                                       np.array(dtj).ravel())
                H[j+1,i+1] = H[i+1,j+1].conj()
        try:
            c = np.linalg.solve(H, G)
        except np.linalg.linalg.LinAlgError:
            log.warn(self, 'singularity in diis')
            #c = pyscf.lib.solve_lineq_by_SVD(H, G)
            ## damp diagonal elements to avoid singularity
            #for i in range(H.shape[0]):
            #    H[i,i] = H[i,i] + 1e-9
            #c = numpy.linalg.solve(H, G)
            for i in range(1,nd):
                H[i,i] = H[i,i] + 1e-10
            c = np.linalg.solve(H, G)
            #c = np.linalg.solve(H[:nd,:nd], G[:nd])

        x = np.zeros_like(x)
        #print c[1:]
        for i, ci in enumerate(c[1:]):
            x += self.get_vec(i) * ci
        return x













def gf_solver_ip( cc ):

    eom_ccsd_ip_greenfunction( cc )
    #eom_ccsd_ea_greenfunction(args)

    """
    eom_ccsd_ip(t1,t2,F,W,nroots)

    """

def eom_ccsd_ip_greenfunction(cc):
    nocc, nvir = cc.t1.shape
    print "NOCC : ", nocc
    print "NVIR : ", nvir

    print  " :: Making A diagonal :: "

    # Creating diagonal preconditioner
    Adiag = np.zeros((nocc+nocc*(nocc-1)*nvir/2),dtype=complex)
    index = 0
    for i in xrange(nocc):
        Adiag[index] = -1.0 * cc.f_oo[i,i]
        index += 1
    for i in xrange(nocc):
        for j in xrange(i):
            for a in xrange(nvir):
                Adiag[index] = cc.f_vv[a,a] - cc.f_oo[i,i] - cc.f_oo[j,j]
                index +=1
    print Adiag
    print  "  Finished! =^.^="

    ones = np.ones( Adiag.shape[ 0 ] )
    # Creating a better diagonal preconditioner
    vec = cc.ip_eom_ccsd_diag( ones, 0 )
    print vec
    Adiag = vec

    theta = 0.05
    omega_init = 0.0
    domega = 0.000734986475817
    nomega = 1

    CORE = 0
    HOMO = nocc - 1
    LUMO = nocc

    plist = []
    plist.append( CORE )

    # Loop over orbital list
    for p in plist:
        omega = omega_init
        b = ip_bpvec(cc,p)
        print "BVEC"
        print_vec( b )
        x0 = np.zeros( b.shape[ 0 ], dtype = complex )
        # Create initial guess from diagonal
        for i in xrange(len(x0)):
            fac = (-theta / ( (omega-Adiag[ i ])**2 + theta**2 ))
            x0[i] += fac*1j
            x0[i] += fac*(Adiag[ i ]-omega)/theta
            x0[i] *= b[i]
        #x0 = get_good_guess()
        data = []
        omega_grid = []
        # Loop over omega values
        for i in xrange(nomega):
            omega = omega_init + domega * i
            eq  = ip_eqvec( cc, p )
            print "EQ :"
            print_vec( eq )
            print " Solving IP at omega(%3d) = %14.10f" % (i,omega)
            sp  = greenfunction_solver_ip1(cc,x0,b,Adiag,omega,theta)
            x0 = sp

            # Get imaginary part of the green's function (the spectral function)
            for q in plist:
                eq  = ip_eqvec( cc, q )
                print "EQ :"
                print_vec( eq )
                imgf = -1.*np.dot(sp,eq).imag
                print " :: GF value = ", imgf
                data.append(imgf)
                omega_grid.append(omega)

        print "DATA"
        print data
        plt.plot(omega_grid,data,marker='.',ls='')
        #plt.show()

def ip_bpvec(cc,p):
    nocc,nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=complex)
    if p >= nocc:
        vector1 = cc.t1[:,p-nocc]
        vector2 = 1.0 * cc.t2[:,:,:,p-nocc]
    else:
        vector1[ p ] += 1.0
    return cc.amplitudes_to_vector_ip(vector1,vector2)


def print_vec( invec ):
    for i in xrange( len( invec ) ):
      if abs( invec[ i ] ) > 1e-10 :
          print i, invec[ i ]


def ip_eqvec(cc,q):
    nocc,nvir = cc.t1.shape
    lamb1 = cc.L1
    lamb2 = cc.L2
    nocc,nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=complex)
    if q < nocc:
        vector1[q] -= 1.0
        vector1 += - np.einsum('ia,a->i',lamb1, cc.t1[q,:])
        vector1 += 0.25 * np.einsum( 'ilcd,lcd->i', lamb2, cc.t2[q,:,:,:] )
        vector1 += 0.25 * np.einsum( 'ildc,ldc->i', lamb2, cc.t2[q,:,:,:] )
    else:
        vector1 += lamb1[:,q-nocc]
        vector2 += 0.5 * ( lamb2[:,:,q-nocc,:] - lamb2[:,:,:,q-nocc] )

    return cc.amplitudes_to_vector_ip(vector1,vector2)

def greenfunction_solver_ip1(cc,x0,b,Adiag,inomega,intheta):
    #solve equation (w-A+1j*theta)x = b
    args = 0
    size = b.shape[0]
    res = np.zeros( size, dtype = complex )
    x = np.zeros( size, dtype = complex )
    errtol = 1e-7
    maxcyc = 100

    omega = inomega
    theta = intheta
    x_arr = np.zeros( (maxcyc, size), dtype = complex )
    qj    = np.zeros( size, dtype = complex )

    pk    = np.zeros( (maxcyc, size), dtype = complex )
    qk    = np.zeros( (maxcyc, size), dtype = complex )

    x = np.array( x0 )
    r = b - cc.ip_eom_ccsd_matvec(x,args) - omega*x + (theta*1j)*x

    iteration = 0

    # Solves the linear equation (w-A+1j*theta)x = b for x
    for i in xrange(maxcyc):

        pk[ iteration ] = r
        qk[ iteration ] = cc.ip_eom_ccsd_matvec(r,args) + omega*r - (theta*1j)*r

        for j in xrange( 0, iteration ):
            betakj = np.vdot( qk[j], qk[iteration] )
            qk[iteration] -= betakj * qk[j]
            pk[iteration] -= betakj * pk[j]

        norm = LA.norm( qk[iteration] )
        qk[iteration] /= norm

        pk[iteration] /= norm
        alphak = np.dot( r, qk[iteration] )
        alphak = alphak.conj()
        x = x + alphak * pk[iteration]
        r = r - alphak * qk[iteration]
        norm = LA.norm( r )
        print " :: arnoldi res norm = %24.16f" % norm
        #print x

        error = 10.
        iteration += 1

        #if error < errtol:
        if norm < errtol:
            break
        if error > 50:
            sys.exit( "EXIT : IP-GF error > 50" )
        if i == maxcyc - 1:
            sys.exit( "EXIT : IP-GF maxcyc reached!" )
    print " :: CONVERGED IN %3d CYCLES!" % iteration
    #print "::CONVERGED X::"
    #for i in xrange(len( x )):
    #    print x[ i ]
    #print "::END OF CONVERGED X::"
    #print b
    c = np.zeros( size, dtype = complex )
    c = omega*x + cc.ip_eom_ccsd_matvec(x,args) - (theta*1j)*x
    gf_error = np.amax(abs(c-b))
    if( gf_error > 10*errtol ):
        sys.exit( "ERROR: IP-GF converged... but not to a correct solution." )
    return x


def greenfunction_solver_ip(cc,x0,b,Adiag,inomega,intheta):
    #solve equation (w-A+1j*theta)x = b
    size = b.shape[0]
    res = np.zeros( size, dtype = complex )
    x = np.zeros( size, dtype = complex )
    #xlist = np.zeros( 100, size, dtype = complex )
    errtol = 1e-7
    maxcyc = 5

    omega = inomega
    theta = intheta
    x_arr = np.zeros( (maxcyc, size), dtype = complex )
    qj    = np.zeros( size, dtype = complex )
    H_mat = np.zeros( (maxcyc+1, maxcyc), dtype = complex )

    norm = LA.norm(b)

    #args = 0
    #diis_solver = DIIS()
    #x = np.array( x0 )
    #norm = LA.norm( x )
    #x = x/norm
    #x_arr[ 0 ] = x
    #H_mat[ 0, 0 ] = norm



    #res = b - cc.ip_eom_ccsd_matvec(x,args) - omega*x + (theta*1j)*x
    iteration = 0

    for i in xrange(maxcyc):

        iteration += 1



        xold = x.copy()
        #for j in xrange(size):
        #    x[j] = res[j]/Adiag[j] + xold[ j ]

        r = cc.ip_eom_ccsd_matvec(x_arr[iteration-1],args)
        # Orthogonalizing against all the previous iterations
        for j in xrange( 0, iteration ):
            qj = x_arr[ j ]
            qjdotx = np.vdot( qj, r )
            H_mat[ j, iteration] = qjdotx
            x = x - H_mat[ j, iteration] * x_arr[ j ]
        norm = LA.norm( x )
        H_mat[iteration, iteration - 1] = norm
        print " :: arnoldi new norm = %24.16f" % norm
        x = x/norm
        x_arr[ iteration ] = x

        #res = b - cc.ip_eom_ccsd_matvec(x,args) - omega*x + (theta*1j)*x
        #error= np.amax( abs( res ) )
        #print " :: IP-GF ERROR (iter=%3d)= %14.10f" % (i,error)
        #print "x iter( %3d )" % i
        #print_vec( x )
        #x = np.conjugate(x)

        error = 10.

        #if error < errtol:
        if norm < errtol:
            break
        if error > 50:
            sys.exit( "EXIT : IP-GF error > 50" )
        if i == maxcyc - 1:
            sys.exit( "EXIT : IP-GF maxcyc reached!" )
    print " :: CONVERGED IN %3d CYCLES!" % iteration
    Hsub = H_mat[ 0:iteration, 0:iteration ]
    Qmat = x_arr[ 0:iteration ]
    print "INVERSE"
    print LA.inv( Hsub )
    print "H"
    print Hsub
    print "QMAT"
    print Qmat
    new = Qmat.conj().T * LA.inv( Hsub ) * Qmat * b
    print new.shape
    #print "::CONVERGED X::"
    #for i in xrange(len( x )):
    #    print x[ i ]
    #print "::END OF CONVERGED X::"
    #print b
    c = np.zeros( size, dtype = complex )
    c = omega*x + cc.ip_eom_ccsd_matvec(x,args) - (theta*1j)*x
    gf_error = np.amax(abs(c-b))
    if( gf_error > 10*errtol ):
        sys.exit( "ERROR: IP-GF converged... but not to a correct solution." )
    return x









def ea_bra(p,args):
    t1,t2,F,W = args
    nocc,nvir = t1.shape
    lamb1 = np.zeros((nocc,nvir))
    lamb2 = np.zeros((nocc,nocc,nvir,nvir))
    vector1 = np.zeros((nvir),dtype=complex)
    vector2 = np.zeros((nocc,nvir,nvir),dtype=complex)
    if p>= nocc:
        vector1[p-nocc] += 1
        vector1 += np.einsum('ia,i->a',lamb1,t1[:,p-nocc])

        vector1 +=  np.einsum('ijab,ijb->b',lamb2,t2[:,:,p-nocc,:])\
                -   np.einsum('ijba,ijb->b',lamb2,t2[:,:,p-nocc,:])\
                -   np.einsum('jiab,ijb->b',lamb2,t2[:,:,p-nocc,:])\
                +   np.einsum('jiba,ijb->b',lamb2,t2[:,:,p-nocc,:])\
                +   np.einsum('ijab,ija->b',lamb2,t2[:,:,:,p-nocc])\
                -   np.einsum('ijba,ija->b',lamb2,t2[:,:,:,p-nocc])\
                -   np.einsum('jiab,ija->b',lamb2,t2[:,:,:,p-nocc])\
                +   np.einsum('jiba,ija->b',lamb2,t2[:,:,:,p-nocc])\

        vector2[:,:,p-nocc]  += lamb1
        vector2 += -np.einsum('ijab,i->jab',lamb2,t1[:,p-nocc])
        vector2 += +np.einsum('jiab,i->jab',lamb2,t1[:,p-nocc])

    else:
        vector1 += -lamb1[p,:]
        vector2 += -lamb2[:,p,:,:] - lamb2[p,:,:,:].tranpose(0,2,1)

    return amplitudes_to_vector_ea(vector1,vector2,nocc,nvir)






if __name__ == '__main__':
    main()
