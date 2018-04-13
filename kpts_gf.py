import collections
import numpy as np
#import gminres
import scipy.sparse.linalg as spla

###################
# EA Greens       #
###################

def greens_b_vector_ea_rhf(cc, p, kp=None):
    nkpts, nocc, nvir = cc.t1.shape
    ds_type = cc.t1.dtype
    vector1 = np.zeros((nvir),dtype=ds_type)
    vector2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir),dtype=ds_type)
    if p < nocc:
        # Changed both to minus
        vector1 += -cc.t1[kp,p,:]
        for ki in range(nkpts):
            for kj in range(nkpts):
                vector2[ki,kj] += -cc.t2[ki,kj,kp,p,:,:,:]
    else:
        vector1[ p-nocc ] = 1.0
    return cc.amplitudes_to_vector_ea(vector1,vector2)

def greens_e_vector_ea_rhf(cc, p, kp=None):
    nkpts, nocc, nvir = cc.t1.shape
    ds_type = cc.t1.dtype
    vector1 = np.zeros((nvir),dtype=ds_type)
    vector2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir),dtype=ds_type)

    if hasattr(cc, 'l1') and cc.l1 is not None:
        l1 = cc.l1
        l2 = cc.l2
    else:
        l1 = cc.t1
        l2 = cc.t2

    if p < nocc:
        # Changed both to plus
        vector1 += l1[kp,p,:]
        for ki in range(nkpts):
            for kj in range(nkpts):
                vector2[ki, kj] += 2*l2[ki,kj,kp,p,:,:,:] - \
                                     l2[kj,ki,kp,:,p,:,:]
        pass

    else:
        vector1[ p-nocc ] = -1.0
        vector1 += np.einsum('ia,i->a', l1[kp], cc.t1[kp,:,p-nocc])
        for kl in range(nkpts):
            for kc in range(nkpts):
                vector1 += 2 * np.einsum('klca,klc->a', l2[kp,kl,kc], \
                           cc.t2[kp,kl,kc,:,:,:,p-nocc])
                vector1 -= np.einsum('klca,lkc->a', l2[kp,kl,kc], \
                           cc.t2[kp,kl,kc,:,:,:,p-nocc])

        for kj in range(nkpts):
            vector2[kp,kj,:,p-nocc,:] += -2.*l1[kj]

        for ki in range(nkpts):
            vector2[ki,kp,:,:,p-nocc] += l1[ki]
            
            for kj in range(nkpts):
                vector2[ki,kj] += 2*np.einsum('k,jkba->jab', \
                                  cc.t1[kp,:,p-nocc], l2[ki,kj,kp,:,:,:,:])
                vector2[ki,kj] -= np.einsum('k,jkab->jab', \
                                  cc.t1[kp,:,p-nocc], l2[ki,kj,kp,:,:,:,:])

    return cc.amplitudes_to_vector_ea(vector1,vector2)

###################
# IP Greens       #
###################

def greens_b_vector_ip_rhf(cc,p,kp=None):
    nkpts, nocc, nvir = cc.t1.shape

    #Changed dimensions to account for kpts. 3 kpt indices? 
    # 1 nvir index only?
    # b(ki, kk, i, k)     ki == kk == kp
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir),dtype=complex)

    #Added kp index for p<nocc. In else added for loop summing new v1 kp
    #and over ki, kj, kp for v2
    if p < nocc:
        vector1[p] = 1.0
    else:
        vector1 += cc.t1[kp,:,p-nocc]
        for ki in range(nkpts):
            for kj in range(nkpts):
                vector2[ki,kj] += cc.t2[ki,kj,kp,:,:,:,p-nocc]
    return cc.amplitudes_to_vector_ip(vector1,vector2)

def greens_e_vector_ip_rhf(cc,p,kp=None):
    nkpts, nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir),dtype=complex)

    if hasattr(cc, 'l1') and cc.l1 is not None:
        l1 = cc.l1
        l2 = cc.l2
    else:
        l1 = cc.t1
        l2 = cc.t2

    if p < nocc:
        vector1[p] = -1.0
        vector1 += np.einsum('ia,a->i', l1[kp], cc.t1[kp,p,:])
        for kl in range(nkpts):
            for kc in range(nkpts):
                vector1 += 2 * np.einsum('ilcd,lcd->i', \
                       l2[kp,kl,kc], cc.t2[kp,kl,kc,p,:,:,:])
                vector1 -= np.einsum('ilcd,ldc->i',   \
                       l2[kp,kl,kc], cc.t2[kp,kl,kc,p,:,:,:])
      
        for kj in range(nkpts):
            vector2[kp,kj,p,:,:] += -2*l1[kj]

        for ki in range(nkpts):
            # kj == kk == kp, ki == kb
            vector2[ki,kp,:,p,:] +=  l1[ki]

            for kj in range(nkpts):
                # kc == kk == kp
                vector2[ki,kj] += 2*np.einsum('c,ijcb->ijb', \
                       cc.t1[kp,p,:], l2[ki,kj,kp,:,:,:,:])
                vector2[ki,kj] -= np.einsum('c,jicb->ijb', \
                       cc.t1[kp,p,:], l2[ki,kj,kp,:,:,:,:])
    else:
        vector1 += -l1[kp,:,p-nocc]
        for ki in range(nkpts):
            for kj in range(nkpts):
                vector2[ki, kj] += -2*l2[ki,kj,kp,:,:,p-nocc,:] + \
                                   l2[ki,kj,kp,:,:,:,p-nocc]

    return cc.amplitudes_to_vector_ip(vector1,vector2)

def greens_func_multiply(ham,vector,linear_part,args=None):
    return np.array(ham(vector) + (linear_part)*vector)

def initial_ip_guess(cc):
    nkpts, nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nocc),dtype=complex)
    vector2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir),dtype=complex)
    return cc.amplitudes_to_vector_ip(vector1,vector2)

def initial_ea_guess(cc):
    nkpts, nocc, nvir = cc.t1.shape
    vector1 = np.zeros((nvir),dtype=complex)
    vector2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir),dtype=complex)
    return cc.amplitudes_to_vector_ea(vector1,vector2)


class OneParticleGF(object):
    def __init__(self, cc, eta=0.01):
        self.cc = cc
        self.eta = eta

    def solve_ip(self, kptlist, ps, qs, omegas):
        if not isinstance(ps, collections.Iterable): ps = [ps]
        if not isinstance(qs, collections.Iterable): qs = [qs]
        cc = self.cc
        print("solving ip portion")
        Sw = initial_ip_guess(cc)
        e_vector = list() 
        for kp, ikpt in enumerate(kptlist):
            for q in qs:
                e_vector.append(greens_e_vector_ip_rhf(cc,q,kp))
        gfvals = np.zeros((len(kptlist), len(ps),len(qs),len(omegas)),dtype=complex)
        for kp, ikpt in enumerate(kptlist):
            for ip, p in enumerate(ps):
                b_vector = greens_b_vector_ip_rhf(cc,p,kp)
                cc.kshift = kp
                diag = cc.ipccsd_diag()
                for iw, omega in enumerate(omegas):
                    invprecond_multiply = lambda x: x/(omega + diag + 1j*self.eta)
                    def matr_multiply(vector,args=None):
                        return greens_func_multiply(cc.ipccsd_matvec, vector, omega + 1j*self.eta)
                    size = len(b_vector)
                    Ax = spla.LinearOperator((size,size), matr_multiply)
                    mx = spla.LinearOperator((size,size), invprecond_multiply)
                    Sw, info = spla.gmres(Ax, b_vector, x0=Sw, tol=1e-10, M=mx)
                    for iq,q in enumerate(qs):
                        gfvals[kp,ip,iq,iw]  = -np.dot(e_vector[iq],Sw)
        if len(ps) == 1 and len(qs) == 1:
            print 'ip 1',gfvals[:,0,0,:]
            print 't1 ',cc.t1[0,0,0]
            print 't2 ',cc.t2[0,0,0,0,0,0,0]
            return gfvals[:,0,0,:]
        else:
            print 'ip 2',gfvals
            return gfvals

    def solve_ea(self, kptlist, ps, qs, omegas):
        if not isinstance(ps, collections.Iterable): ps = [ps]
        if not isinstance(qs, collections.Iterable): qs = [qs]
        cc = self.cc
        print("solving ea portion")
        Sw = initial_ea_guess(cc)
        e_vector = list()
        for kp, ikpt in enumerate(kptlist): 
            for p in ps:
                e_vector.append(greens_e_vector_ea_rhf(cc,p,kp))
        gfvals = np.zeros((len(kptlist),len(ps),len(qs),len(omegas)),dtype=complex)
        for kp, ikpt in enumerate(kptlist):
            for iq, q in enumerate(qs):
                b_vector = greens_b_vector_ea_rhf(cc,q,kp)
                cc.kshift = kp
                diag = cc.eaccsd_diag()
                for iw, omega in enumerate(omegas):
                    invprecond_multiply = lambda x: x/(-omega + diag + 1j*self.eta)
                    def matr_multiply(vector,args=None):
                        return greens_func_multiply(cc.eaccsd_matvec, vector, -omega + 1j*self.eta)
                    size = len(b_vector)
                    Ax = spla.LinearOperator((size,size), matr_multiply)
                    mx = spla.LinearOperator((size,size), invprecond_multiply)
                    Sw, info = spla.gmres(Ax, b_vector, x0=Sw, M=mx)
                    for ip,p in enumerate(ps):
                        gfvals[kp,ip,iq,iw] = np.dot(e_vector[ip],Sw)
        if len(ps) == 1 and len(qs) == 1:
            print 'ea 1',gfvals[:,0,0,:]
            return gfvals[:,0,0,:]
        else:
            print 'ea 2',gfvals
            return gfvals

    def kernel(self, k, p, q, omegas):
        return self.solve_ip(k, p, q, omegas), self.solve_ea(k, p, q, omegas)

