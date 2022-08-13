import numpy as np
import math
import os
import sympy
import matplotlib.pyplot as plt
import multiprocessing as mp
import scipy
import torch
import time
from math import factorial
from itertools import combinations
from thewalrus import tor
from tqdm import tqdm
from numba import jit

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

@jit(nopython=True)
def BinToDecimal(B):
    L = len(B)
    sum = 0
    for i in range(L):
        sum =sum + (2**(L-1-i))*B[i]
    return sum

@jit(nopython=True)
def BinHexOct(n,x,L):
    res = np.zeros((L,),dtype=np.int32)
    i=0
    temp = n
    while True:
        res[L-1-i] = temp % x
        temp = temp//x
        i = i + 1
        if temp==0:
            break
    return res

@jit(nopython=True)
def _click_events(Z):
    b = np.zeros((2**Z,Z),dtype = np.int32)
    a = np.arange(0,2**Z,1)
    for i in range(0,len(a)):
        b[i:] = BinHexOct(a[i],2,Z)
    return b

@jit(nopython=True)
def _Prob_ABtoC_2(Probs_A,Probs_B):
    Probs_C = np.zeros(256,dtype=np.float64)
    for i in range(256):
        for j in range(256):
            Probs_C[i|j] += Probs_A[i]*Probs_B[j]
    return Probs_C

@jit(nopython=True)
def _Prob_ABtoC(events,Probs_A,Probs_B):
    Probs_C = np.zeros(len(Probs_A),dtype=np.float64)
    N = len(events[0])
    for k in range(0,len(events)):
        events_C = events[k]
        index = np.argwhere(events_C).astype(np.int32)[:,0]
        tol = np.sum(events_C)
        A_list = np.zeros((3 ** tol, tol),dtype=np.int32)
        B_list = np.zeros((3 ** tol, tol),dtype=np.int32)
        events_A_list = np.zeros((3 ** tol, N),dtype=np.int32)
        events_B_list = np.zeros((3 ** tol, N),dtype=np.int32)
        index_A = np.zeros((3 ** tol,),dtype=np.int32)
        index_B = np.zeros((3 ** tol,),dtype=np.int32)
        for i in range(0,3 ** tol):
            index_list = BinHexOct(i,3,tol)
            for j in range(0,tol):
                if index_list[j]==0:
                    A_list[i,j] = 1
                    B_list[i,j] = 0
                if index_list[j]==1:
                    A_list[i,j] = 0
                    B_list[i,j] = 1
                if index_list[j]==2:
                    A_list[i,j] = 1
                    B_list[i,j] = 1
            events_A_list[i][index] = A_list[i]
            events_B_list[i][index] = B_list[i]
            index_A[i] = BinToDecimal(events_A_list[i])
            index_B[i] = BinToDecimal(events_B_list[i])
        Probs_C[k] = np.sum(np.multiply(Probs_A[index_A], Probs_B[index_B]))
    return Probs_C

def _A_s_tor(A,S):
    N = len(S)
    index_0 = np.argwhere(S)[0:]
    index = np.vstack((index_0,index_0+N)).reshape(2*len(index_0))
    A_S = A[index,:][:,index]
    return A_S

class Torontonian:

    def A_s_tor(self,A,S):
        N = len(S)
        index_0 = np.argwhere(S)[0:]
        index = np.array([index_0,index_0+N]).reshape(-1)
        A_S = A[index,:][:,index]
        return A_S
    #def A_s_tor(self,A,S):
    #    res = _A_s_tor(A,S)
    #    return res

    def Prob(self,sample):
        sigma_inv_s = self.A_s_tor(self.sigma_inv,sample)
        O_s = np.eye(len(sigma_inv_s)) - sigma_inv_s
        Prob = tor(O_s)/self.sqrt_det_sigma
        return Prob.real

    def Probs(self,samples):
        return np.array([self.Prob(i) for i in samples])

    def ProbsAll(self):
        N = np.array([len(self.sigma)/2]).astype(np.int32)
        Eve = self.click_events(N[0])
        return self.Probs(Eve),Eve

    def sigma_set(self,sigma):
        self.sigma = sigma
        self.sigma_inv = np.linalg.pinv(self.sigma, rcond = 1e-15)
        self.sqrt_det_sigma = np.sqrt(np.linalg.det(self.sigma))
        return self

    def sigma_SMSS(self,rall,T):
        N = len(rall)
        S = np.zeros((2 * N, 2 * N))
        S[0: N, 0: N] = np.diag(np.cosh(rall))
        S[0: N, N: 2 * N] = np.diag(np.sinh(rall))
        S[N: 2 * N,0: N] = np.diag(np.sinh(rall))
        S[N: 2 * N,N: 2 * N] = np.diag(np.cosh(rall))

        sigma_vac = np.eye(2 * N) / 2
        sigma_in = np.dot(S, np.dot(sigma_vac, S.T.conj()))
    
        matrix_1 = np.zeros((2 * N, 2 * N), dtype=complex)
        matrix_1[0: N, 0: N] = T
        matrix_1[N: 2 * N, N: 2 * N] = T.conj()
        matrix_2 = matrix_1.T.conj()

        self.sigma = np.eye(2 * N) - 0.5 * np.dot(matrix_1, matrix_2) + np.dot(matrix_1, np.dot(sigma_in, matrix_2))
        self.sigma_inv = np.linalg.pinv(self.sigma, rcond = 1e-15)
        self.sqrt_det_sigma = np.sqrt(np.linalg.det(self.sigma))
        return self

    def sigma_Thermal(self,n_mean,N,T):
        sigma_in = np.eye(2*N) * n_mean+ 1/2 * np.eye(2*N)

        matrix_1 = np.zeros((2 * N, 2 * N), dtype=complex)
        matrix_1[0: N, 0: N] = T
        matrix_1[N: 2 * N, N: 2 * N] = T.conj()
        matrix_2 = matrix_1.T.conj()

        self.sigma = np.eye(2 * N) - 0.5 * np.dot(matrix_1, matrix_2) + np.dot(matrix_1, np.dot(sigma_in, matrix_2))
        #print(type(self.sigma))
        self.sigma_inv = np.linalg.pinv(self.sigma, rcond = 1e-15)
        self.sqrt_det_sigma = np.sqrt(np.linalg.det(self.sigma))

        return self

    def sigma_Squashed(self,n_mean,T):
        sigma_in = np.zeros((2 * N, 2 * N))
        sigma_in[0:N,0:N] = np.eye(2 * N) * n_mean
        sigma_in[N : 2 * N,0:N] = np.eye(2 * N) * n_mean
        sigma_in[0:N,0:N] = np.eye(2 * N) * n_mean
        sigma_in[0:N,0:N] = np.eye(2 * N) * n_mean
        return -1

    def Prob_Coherent(self,Distance,sample):
        A = np.exp( - np.sum(pow(abs(Distance), 2)))
        B = np.prod(np.array([factorial(i) for i in sample]))
        C = np.prod(np.array([abs(Distance[i]) ** (2 * sample[i]) for i in range(0,len(sample))]))
        return (A / B) * C

    def traceto(self):
        return -1
    
    @classmethod
    def Prob_ABtoC(self,events,Probs_A,Probs_B):
        #return _Prob_ABtoC_2(Probs_A,Probs_B)
        return _Prob_ABtoC(events,Probs_A,Probs_B)

    @classmethod
    def click_events(self,Z):
        return _click_events(Z)
    
def data():
    dirc = os.path.abspath('.')
    profix = '0812-F5'

    names = np.load(dirc+'\\names{}.npy'.format(profix))
    results = np.load(dirc+'\\results{}.npy'.format(profix))
    binNum = 8
    #print(results.shape)

    #拆分
    part_num = 0
    split_index = np.arange(0, len(results), 1, dtype = np.int32) + part_num
    results = np.array(results[split_index][:])
    Pr = np.zeros((len(results), binNum+1))
    
    for i in range(len(results[0])):
        phoNum = bin(i).count('1')  # 光子数
        Pr[:, phoNum] += results[:, i]

    # 概率归一化
    for i in range(len(results)):
        Pr[i] /= np.sum(Pr[i])

    events = np.zeros((256,8),dtype=np.int32)
    for i in range(0,256):
        E = bin(i)[2:]
        events[i][8-len(E):] = np.array([int(j) for j in E])

    @jit(nopython=True)
    def Pr_Pi_0_generate(L,results,list):
        Pr_Pi_0 = list
        for i in range(L):
            for j in range(len(events)):
                for k in range(len(events[j])):
                    if events[j,k] == 0:
                        Pr_Pi_0[i,k] = Pr_Pi_0[i,k] + results[i,j]
        return Pr_Pi_0
    @jit(nopython=True)
    def Pr_Pi_1_generate(L,results,list):
        Pr_Pi_1 = list
        for i in range(L):
            for j in range(len(events)):
                for k in range(len(events[j])):
                    if events[j,k] == 1:
                        Pr_Pi_1[i,k] = Pr_Pi_1[i,k] + results[i,j]
        return Pr_Pi_1
    Pr_Pi_0 = Pr_Pi_0_generate(len(Pr),results,np.zeros((len(Pr),8)))
    Pr_Pi_1 = Pr_Pi_1_generate(len(Pr),results,np.zeros((len(Pr),8)))
    for i in range(len(results)):#归一化
        Pr_Pi_0[i] = Pr_Pi_0[i]/np.sum(results[i])
        Pr_Pi_1[i] = Pr_Pi_1[i]/np.sum(results[i])

    index = np.array(Pr[:,0]).argsort()[::-1]
    Pr = Pr[index]
    Pr_Pi_0 = Pr_Pi_0[index]
    Pr_Pi_1 = Pr_Pi_1[index]

    #取后子集
    sub = 5
    K = 0
    N = len(Pr)
    Pr = Pr[N-sub-K:N-K]
    #for i in range(len(Pr[0])):
    #    print(Pr[0][i],', ',end = '')
    return Pr,Pr_Pi_0

def getNM(g2,Np):
    n = sympy.symbols('n')
    m = (Np - n)
    A = (2+n)*(2+m)*(2*n*n+2*m*m+2*m*n+3*m*m*n+3*n*n*m+m*m*n*n)
    B = (1+n)*(1+m)*(2*n+2*m+m*n)*(2*n+2*m+m*n)
    C = ((A)/(B)-g2)
    if C.evalf(subs={n:0},n=5)<=0:
        print('C(0)=',C.evalf(subs={n:0},n=5),'|','getNM() false')
        return -1, -1
    res = list(sympy.solveset(C,n,sympy.Interval(0,Np)))
    m = np.float64(res[0])
    n = np.float64(res[1])
    return m,n

def get_eta(n_mean,m_mean,p):
    eta = sympy.symbols('eta')
    fun_get_eta = p * (1+ n_mean * eta ) * (1+ m_mean * eta ) - 1
    res = list(sympy.solveset(fun_get_eta, eta, sympy.Interval(0,1)))
    return np.float64(res[0])

def get_eta2(n_mean,m_mean,l_mean,p):
    eta = sympy.symbols('eta')
    fun_get_eta = p * (1+ n_mean * eta ) * (1+ m_mean * eta ) * (1+ l_mean * eta ) - 1
    res = list(sympy.solveset(fun_get_eta, eta, sympy.Interval(0,1)))
    return np.float64(res[0])

def get_g2(n,m):
    A = (2+n)*(2+m)*(2*n*n+2*m*m+2*m*n+3*m*m*n+3*n*n*m+m*m*n*n)
    B = (1+n)*(1+m)*(2*n+2*m+m*n)*(2*n+2*m+m*n)
    return A/B

def get_taueta(n,m,Pr):

    eta = sympy.symbols('eta')
    tau = []
    for i in range(0,len(Pr)):
        tau.append(sympy.symbols('tau_'+str(i)))
    fun = []
    for i in range(0,len(Pr)):
        fun_i = Pr[i,0] * (1 + m * tau[i] * eta) * math.exp(n * tau[i] * eta) - 1
        fun.append(fun_i)
    para = tau
    para.append(eta)

    return sympy.solve(fun, para)

@jit(nopython=True)
def get_TVD(Pr,Pr_throry):
    sum = 0
    for i in range(len(Pr)):
        sum = sum + abs(Pr[i]-Pr_throry[i])/Pr_throry[i]
    return sum

@jit(nopython=True)
def get_tau_i(a,b,c,p):
    #(1+ax)(1+bx)(1+cx)-1/p
    A = a * b * c
    B = (a*b+b*c+c*a)
    C = a + b + c
    D = 1-1/p

    e = 1e-6

    t_0 = 0.125
    t_1 = t_0 - ((A * t_0 ** 3 + B * t_0 ** 2 + C * t_0 + D) / (3 * A * t_0 ** 2 + 2 * B * t_0 + C))
    while abs(t_1-t_0)>e:
        t_0 = t_1
        t_1 = t_0 - ((A * t_0 ** 3 + B * t_0 ** 2 + C * t_0 + D) / (3 * A * t_0 ** 2 + 2 * B * t_0 + C))
    return t_1

def get_tau(a,b,c,Pr_Pi_0):
    tau = np.zeros(8,dtype=np.float64)
    for i in range(8):
        tau[i] = get_tau_i(a,b,c,Pr_Pi_0[i])
    return tau

def main(k):
    start=time.time()
    k_1 = 1
    k_2 = k[0]
    k_3 = 0

    N = 8
    T = np.zeros((N,N))
    #T[:,0] = np.sqrt(np.array([0.13372587,0.12251623,0.12846983,0.1198264,0.12912523,0.11829967,0.1307555,0.12195197]))

    #Pr= data_2()
    Pr, Pr_Pi_0=data()

    Pr_throry = np.zeros((Pr.shape[0], Pr.shape[1]))
    Probs_C = np.zeros((Pr.shape[0], 2 ** N))
    Probs_C2 = np.zeros((Pr.shape[0], 2 ** N))
    P_click = 12/250
    eta = np.zeros((5,1))
    n_mean = 20#任意值
    n_mean_A = k_1/(k_1+k_2+k_3) * n_mean
    n_mean_B = k_2/(k_1+k_2+k_3) * n_mean
    n_mean_B2 = k_3/(k_1+k_2+k_3) * n_mean

    for i in range(0,len(Pr)):

        eta_i = get_eta2(n_mean_A,n_mean_B,n_mean_B2,Pr[i,0])
        #print(eta_i * n_mean_A,eta_i * n_mean_B, eta_i * n_mean_B2,Pr_Pi_0[i])
        #print(Pr[i,0],Pr_Pi_0[i])
        tau = get_tau(eta_i*n_mean_A, eta_i*n_mean_B, eta_i*n_mean_B2,Pr_Pi_0[i])
        tau = tau/np.sum(tau)

        T[:,0] = np.sqrt(tau.reshape(-1))

        A = Torontonian().sigma_Thermal(n_mean_A, N, T * np.sqrt(eta_i))
        B = Torontonian().sigma_Thermal(n_mean_B, N, T * np.sqrt(eta_i))

        A_ProbsAll,Eve = A.ProbsAll()
        B_ProbsAll,Eve = B.ProbsAll()

        Probs_C[i] = Torontonian.Prob_ABtoC(Eve, A_ProbsAll, B_ProbsAll)

        print('n_mean = ',eta_i*n_mean,end=' ')
    print('\n')
    for j in range(0,2 ** N):
        ProbNum = len(np.argwhere(Eve[j]))
        Pr_throry[:, ProbNum] += Probs_C[:, j]
    np.save('Pr_throry.npy',Pr_throry)

    Pr = Pr.reshape(-1)
    Pr_throry = Pr_throry.reshape(-1)
    TVD = get_TVD(Pr,Pr_throry)
    print('k=',k,'TVD = ',TVD,sep='\t',end='|\t')
    end=time.time()
    print(end-start)
    return TVD                                     #TVD

def test():
    Pr, Pr_Pi_0=data()
    Pr_throry = np.load('Pr_throry.npy')
    delta_Pr = np.zeros((Pr.shape[0],Pr.shape[1]))
    for i in range(len(Pr)):
        for j in range(len(Pr_throry[0])):
            print('%.11f' % Pr_throry[i,j],'%.11f' % Pr[i,j],'%.5f' % ((Pr_throry[i,j]-Pr[i,j])/Pr_throry[i,j]),sep = '\t')
            delta_Pr[i,j] = (Pr_throry[i,j]-Pr[i,j])/Pr_throry[i,j]
        print('-----------------------------------------------')
    X = np.array([0,1,2,3,4,5,6,7,8])
    plt.title('1')
    for i in range(len(Pr)):
        name = str(i)
        plt.plot(X,delta_Pr[i],label=str(i))

    plt.legend(loc = 'lower left')
    plt.xlabel('0~8')
    plt.ylabel('$\Delta$')
    plt.show()


if __name__=='__main__':

    x_0 = np.array([0])
    res_min = scipy.optimize.fmin(main,x_0,xtol = 0.0001,ftol = 1e-5)
    x_1 = res_min
    for i in res_min:
        print(i)
    main(x_1)
    test()
