import sys
from numpy import linalg as LA
import numpy as np
import scipy.sparse.linalg as spla
from math import *

from pyscf import gto
from pyscf import scf
from pyscf import cc

import gminres
import arnoldi_solver
import antisymeri
from ccsd import ccsd
#import eom_driver

def print2Vec(title,vec):
    print "*******************************"
    print title
    print "*******************************"
    shape0 = vec.shape[0]
    shape1 = vec.shape[1]
    for i in xrange(shape0):
        for j in xrange(shape1):
            dVal = vec[i,j]
            if abs(dVal) > 1e-13:
                print "%3d %3d  %20.16f" %(i,j,dVal)

def print4Vec(title,vec):
    print "*******************************"
    print title
    print "*******************************"
    shape0 = vec.shape[0]//2
    shape1 = vec.shape[1]//2
    shape2 = vec.shape[2]//2
    shape3 = vec.shape[3]//2
    print " :: AABB spin combination ::"
    for i in range(shape0):
        spin_i = 1 #i%2
        for j in range(shape1):
            spin_j = 0 #j%2
            for a in range(shape2):
                spin_a = 1 #1a%2
                for b in range(shape3):
                    spin_b = 0 #b%2
                    si = 2*i + spin_i
                    sj = 2*j + spin_j
                    sa = 2*a + spin_a
                    sb = 2*b + spin_b
                    dval = vec[si,sj,sa,sb]
                    if abs(dval) > 1e-13:
                        print "%3d %3d %3d %3d %20.16f" %(si,sj,sa,sb,vec[si,sj,sa,sb])
    print ""
    print " :: AAAA spin combination ::"
    for i in range(shape0):
        spin_i = 0 #i%2
        for j in range(shape1):
            spin_j = 0 #j%2
            for a in range(shape2):
                spin_a = 0 #1a%2
                for b in range(shape3):
                    spin_b = 0 #b%2
                    si = 2*i + spin_i
                    sj = 2*j + spin_j
                    sa = 2*a + spin_a
                    sb = 2*b + spin_b
                    dval = vec[si,sj,sa,sb]
                    if abs(dval) > 1e-13:
                        print "%3d %3d %3d %3d %20.16f" %(si,sj,sa,sb,vec[si,sj,sa,sb])
    print "*******************************"

def ea_b_vec(cc,p):
    nocc,nvir = cc.t1.shape
    vector1 = np.zeros((nvir),dtype=complex)
    vector2 = np.zeros((nocc,nvir,nvir),dtype=complex)
    if p >= nocc:
        vector1[ p-nocc ] += 1.0
    else:
        vector1 = cc.t1[p,:]
        vector2 = 1.0 * cc.t2[p,:,:,:]
    return cc.amplitudes_to_vector_ea(vector1,vector2)

def ea_e_vec(cc,q):
    nocc,nvir = cc.t1.shape
    lamb1 = cc.L1
    lamb2 = cc.L2
    nocc,nvir = cc.t1.shape
    vector1 = np.zeros((nvir),dtype=complex)
    vector2 = np.zeros((nocc,nvir,nvir),dtype=complex)
    if q >= nocc:
        vector1[q-nocc] += 1.0
        vector1 += np.einsum('ia,i->a',lamb1, cc.t1[:,q-nocc])
        vector1 += (1./4.) * np.einsum( 'ilcd,ild->c', lamb2, cc.t2[:,:,q-nocc,:] )
        vector1 -= (1./4.) * np.einsum( 'ildc,ild->c', lamb2, cc.t2[:,:,q-nocc,:] )
#
#
#  Notice the change in signs between this and the green's function
#  equations in the Nooijen paper.  The sign comes from working with
#  s(i,j,a) versus s(j,i,a)
#
#
        vector2[:,q-nocc,:] += 1.0*lamb1
        vector2[:,:,q-nocc] -= 1.0*lamb1
        vector2 += (1./2.) * np.einsum( 'i,ijcb->jcb', cc.t1[:,q-nocc], lamb2 )
        vector2 -= (1./2.) * np.einsum( 'i,jicb->jcb', cc.t1[:,q-nocc], lamb2 )
    else:
        vector1 += lamb1[q,:]
        vector2 += - 0.5 * ( lamb2[:,q,:,:] - lamb2[q,:,:,:] )

    return cc.amplitudes_to_vector_ea(vector1,vector2)

def eom_ccsd_ip_greenfunction(cc):
    eta = 0.1
    omega = [0.0];
    state = 6

    nocc,nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=complex)

#
#
#  Making initial guess
#
#
    #vector1 += np.ones(nocc)
    #vector1 += 1j*np.ones(nocc)
    #vector2 += np.ones((nocc,nocc,nvir))
    #vector2 += 1j*np.ones((nocc,nocc,nvir))
    v0 = cc.amplitudes_to_vector_ip(vector1,vector2)
    v0 = v0.reshape(v0.shape[0],1)

#
#
#  Making pre conditioner
#
#
    vector1 -= vector1
    vector2 -= vector2
    vector1 += np.ones(nocc)
    vector2 += np.ones((nocc,nocc,nvir))
    P  = cc.amplitudes_to_vector_ip(vector1,vector2)
    P  = P.reshape( P.shape[0], 1 )
    #print "precon : ", P

#
#
#  Making 'b' vector
#
#
    b  = ip_b_vec(cc,state)
    b  = b.reshape( b.shape[0], 1 )
    print "bvector..."
    for i in xrange(b.shape[0]):
        dval = b[i]
        if abs(dval) > 1e-15:
            print "b : ", i, dval

#
#
#  Making 'e' vector
#
#
    e  = ip_e_vec(cc,state)
    e  = e.reshape( e.shape[0], 1 )
    print "evector..."
    for i in xrange(e.shape[0]):
        dval = e[i]
        if abs(dval) > 1e-15:
            print "e : ", i, dval


    gg = arnoldi_solver.arnoldi(cc,v0,P)
    solution = gg.getSolution()
    #print "solution = "
    #print solution
    print "green's function = "
    print np.vdot(e,solution)

def eom_ccsd_ea_greenfunction(cc):
    eta = 0.04777412092810763
    omega = [0.0];
    state = 0

    nocc,nvir = cc.t1.shape
    vector1 = np.zeros((nvir),dtype=complex)
    vector2 = np.zeros((nocc,nvir,nvir),dtype=complex)

#
#
#  Making initial guess
#
#
    vector1 += np.ones(nvir)
    #vector1 += 1j*np.ones(nvir)
    vector2 += np.ones((nocc,nvir,nvir))
    #vector2 += 1j*np.ones((nocc,nvir,nvir))
    v0 = cc.amplitudes_to_vector_ea(vector1,vector2)
    v0 = v0.reshape(v0.shape[0],1)

#
#
#  Making pre conditioner
#
#
    vector1 -= vector1
    vector2 -= vector2
    vector1 += np.ones(nvir)
    vector2 += np.ones((nocc,nvir,nvir))
    P  = cc.amplitudes_to_vector_ea(vector1,vector2)
    P  = P.reshape( P.shape[0], 1 )
    #print "precon : ", P

#
#
#  Making 'b' vector
#
#
    b  = ea_b_vec(cc,state)
    b  = b.reshape( b.shape[0], 1 )
    #for i in xrange(b.shape[0]):
    #    dval = b[i]
    #    if abs(dval) > 1e-15:
    #        print "b : ", i, dval

#
#
#  Making 'e' vector
#
#
    e  = ea_e_vec(cc,state)
    e  = e.reshape( e.shape[0], 1 )
    #print "evector..."
    #for i in xrange(e.shape[0]):
    #    dval = e[i]
    #    if abs(dval) > 1e-15:
    #        print "e : ", i, dval
    #for i in xrange(e.shape[0]):
    #    print e[i]


    gg = gminres.gMinRes(cc.ea_eom_ccsd_matvec,b,v0,P)
    #gg = arnoldi_solver.arnoldi(cc,v0,P)
    solution = gg.getSolution()
    print "green's function = "
    print np.vdot(e,solution)


def gf_solver_ea( cc ):
    eom_ccsd_ea_greenfunction( cc )

def gf_solver_ip( cc ):
    eom_ccsd_ip_greenfunction( cc )

def greens_b_vector_ip_rhf(cc,p):
    nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=complex)
    if p < nocc:
        vector1[ p ] += 1.0
    else:
        vector1 = cc.t1[:,p-nocc]
        vector2 = cc.t2[:,:,:,p-nocc]
    return cc.amplitudes_to_vector_ip(vector1,vector2)

def greens_e_vector_ip_rhf(cc,p):
    nocc, nvir = cc.t1.shape
    nocc,nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=complex)
    if p < nocc:
        vector1[ p ] = -1.0
        vector1 += np.einsum('ia,a->i', cc.l1, cc.t1[p,:])
        vector1 += 2*np.einsum('ilcd,lcd->i', cc.l2, cc.t2[p,:,:,:])
        vector1 -=   np.einsum('ilcd,ldc->i', cc.l2, cc.t2[p,:,:,:])

        vector2[p,:,:] += -2.*cc.l1
        vector2[:,p,:] += cc.l1
        vector2 += 2*np.einsum('c,ijcb->ijb', cc.t1[p,:], cc.l2)
        vector2 -=   np.einsum('c,jicb->ijb', cc.t1[p,:], cc.l2)
    else:
        vector1 += -lamb1[:,p-nocc]
        vector2 += -2*lamb2[:,:,p-nocc,:] + lamb2[:,:,:,p-nocc]

def greens_func_multiply(ham,vector,imag_part,real_part,args=None):
    return ham(vector) + (1j*imag_part + real_part)*vector


def ip_b_vec(cc,p):
    nocc,nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=complex)
    if p < nocc:
        vector1[ p ] += 1.0
    else:
        vector1 = cc.t1[:,p-nocc]
        vector2 = 1.0 * cc.t2[:,:,:,p-nocc]
    return cc.amplitudes_to_vector_ip(vector1,vector2)

def ip_e_vec(cc,p):
    nocc,nvir = cc.t1.shape
    lamb1 = cc.L1
    lamb2 = cc.L2
    nocc,nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nocc,nocc,nvir),dtype=complex)
    print "LAM1"
    print lamb1
    #print "LAM2"
    #print lamb2
    print4Vec("LAMBDA-2",lamb2)

    if p < nocc:
        vector1[p] -= 1.0
        vector1 += np.einsum('ia,a->i',lamb1, cc.t1[p,:])
        vector1 += (1./4.) * np.einsum( 'ilcd,lcd->i', lamb2, cc.t2[p,:,:,:] )
        vector1 -= (1./4.) * np.einsum( 'ilcd,ldc->i', lamb2, cc.t2[p,:,:,:] )
#
#
#  Notice the change in signs between this and the green's function
#  equations in the Nooijen paper.  The sign comes from working with
#  s(i,j,a) versus s(j,i,a)
#
#
        vector2[p,:,:] += 1.0*lamb1
        vector2[:,p,:] -= 1.0*lamb1
        vector2 -= (1./2.) * np.einsum( 'c,ijcb->ijb', cc.t1[p,:], lamb2 )
        vector2 += (1./2.) * np.einsum( 'c,jicb->ijb', cc.t1[p,:], lamb2 )
    else:
        vector1 += lamb1[:,p-nocc]
        vector2 += - 0.5 * ( lamb2[:,:,p-nocc,:] - lamb2[:,:,:,p-nocc] )
    return cc.amplitudes_to_vector_ip(vector1,vector2)

#def eom_ccsd_ip_greenfunction(cc):
#    eta = 0.1
#    omega = [0.0];
#    state = 0
#
#    nocc,nvir = cc.t1.shape
#    vector1 = np.zeros((nocc),dtype=complex)
#    vector2 = np.zeros((nocc,nocc,nvir),dtype=complex)
#    if p < nocc:
#        vector1[ p ] -= 1.0
#        vector1 += np.einsum('ic,c->i',cc.lamb1,cc.t1[p,:])
#        vector1 += 0.25 * np.einsum('ilcd,lcd->i', lamb2, cc.t2[p,:,:,:] )
#        vector1 -= 0.25 * np.einsum('ilcd,ldc->i', lamb2, cc.t2[p,:,:,:] )
#        #vector2
#    else:
#        vector1 += cc.lamb1[:,p-nocc]
#        vector2 += 0.5 * (cc.t2[:,:,p-nocc,:] - cc.t2[:,:,:,p-nocc])
#    return cc.amplitudes_to_vector_ip(vector1,vector2)
