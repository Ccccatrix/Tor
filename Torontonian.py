import numpy as np
import math
import os
import sympy
import matplotlib.pyplot as plt
import scipy
import time
from math import factorial
from thewalrus import tor
from numba import jit

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
profix = '0815-G11'

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

@jit(nopython=True)
def _MD_Threshold(events,Probs):
    D = events.shape[0]
    M = events.shape[1]
    C_1 = np.zeros((M,2),dtype=np.float64)
    C_2 = np.zeros((M,M,2,2),dtype=np.float64)
    for m in range(0,M):
        for d in range(0,D):
            j = events[d,m]
            C_1[m,j] += Probs[d]
            for n in range(0,M):
                if n!=m:
                    k = events[d,n]
                    C_2[m,n,j,k] += Probs[d]
    return  C_1, C_2

class Torontonian:

    def A_s_tor(self,A,S):
        N = len(S)
        index_0 = np.argwhere(S)[0:]
        index = np.array([index_0,index_0+N]).reshape(-1)
        A_S = A[index,:][:,index]
        return A_S

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

    results = np.load('results{}.npy'.format(profix))
    binNum = 8
    print(results.shape)

    #拆分
    part_num = 0
    split_index = np.arange(0, len(results), 1, dtype = np.int32) + part_num
    results = np.array(results[split_index][:])
    Pr = np.zeros((len(results), binNum+1))
    Probs = np.zeros((len(results),256),dtype=np.float64)

    for i in range(len(results)):
        Probs[i] = results[i]/np.sum(results[i])

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

    index = np.array(Pr[:,0]).argsort()[::-1]#平均光子数从小到大排列
    Pr = Pr[index]
    Pr_Pi_0 = Pr_Pi_0[index]
    Pr_Pi_1 = Pr_Pi_1[index]
    Probs = Probs[index]

    #取子集
    N = len(Pr)
    sub = 5
    K = 0
    if sub > N:
        sub = N

    index_sub = np.arange(N-sub-K,N-K,1,dtype=np.int8)
    Pr = Pr[index_sub]
    Pr_Pi_0 = Pr_Pi_0[index_sub]
    Pr_Pi_1 = Pr_Pi_1[index_sub]
    Probs = Probs[index_sub]

    np.save('Pr.npy',Pr)
    np.save('Pr_Pi_0.npy',Pr_Pi_0)
    np.save('Probs.npy',Probs)
   
    return Pr, Pr_Pi_0, Probs

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
    A = m_mean * n_mean
    B = m_mean + n_mean
    C = 1 - 1 / p
    if A ==0:
        eta = - C / B
    else:
        eta = (- B + math.sqrt(B ** 2 - 4 * A *C)) / (2 * A)
    return eta

def get_eta_3(n_mean,m_mean,l_mean,p):
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
        sum = sum + abs(Pr[i]-Pr_throry[i])#/Pr_throry[i]
    return sum

@jit(nopython=True)
def get_tau_i_3(a,b,c,p):
    #(1+ax)(1+bx)(1+cx)-1/p
    A = a * b * c
    B = a * b + b * c + c * a
    C = a + b + c
    D = 1 - 1 / p

    e = 1e-6

    t_0 = 0.125
    t_1 = t_0 - ((A * t_0 ** 3 + B * t_0 ** 2 + C * t_0 + D) / (3 * A * t_0 ** 2 + 2 * B * t_0 + C))
    while abs(t_1-t_0)>e:
        t_0 = t_1
        t_1 = t_0 - ((A * t_0 ** 3 + B * t_0 ** 2 + C * t_0 + D) / (3 * A * t_0 ** 2 + 2 * B * t_0 + C))
    return t_1

def get_tau_3(a,b,c,Pr_Pi_0):
    tau = np.zeros(8,dtype=np.float64)
    for i in range(8):
        tau[i] = get_tau_i_3(a,b,c,Pr_Pi_0[i])
    return tau

def get_tau_i(a,b,p):
    A = a * b
    B = a + b
    C = 1 - 1 / p
    if A == 0:
        if B!=0:
            tau_i = - C / B
        if B==0:
            tau_i = 1/8
    else:
        tau_i = (- B + math.sqrt(B ** 2 - 4 * A *C)) / (2 * A)
    return tau_i

def get_tau(a,b,Pr_Pi_0):
    N = len(Pr_Pi_0)
    tau = np.zeros(N,dtype=np.float64)
    for i in range(N):
        tau[i] = get_tau_i(a,b,Pr_Pi_0[i])
    return tau

def main(k):

    start=time.time()
    k_1 = 1
    k_2 = k[0]

    N = 8
    T = np.zeros((N,N))

    Pr = np.load('Pr.npy')
    Pr_Pi_0 = np.load('Pr_Pi_0.npy')

    Pr_throry = np.zeros((len(Pr), Pr.shape[1]))
    Probs_C = np.zeros((len(Pr), 2 ** N))

    n_mean_all = np.zeros(len(Pr),dtype=np.float64)
    tau_all = np.zeros((len(Pr), N),dtype=np.float64)

    C_1_Theory_all = np.zeros((len(Pr),N,2),dtype=np.float64)
    C_2_Theory_all = np.zeros((len(Pr),N,N,2,2),dtype=np.float64)

    n_mean = 10#任意值
    n_mean_A = k_1 / (k_1 + k_2) * n_mean
    n_mean_B = k_2 / (k_1 + k_2) * n_mean

    for i in range(0,len(Pr)):

        eta_i = get_eta(n_mean_A,n_mean_B,Pr[i,0])
        tau = get_tau(eta_i*n_mean_A, eta_i*n_mean_B, Pr_Pi_0[i]).reshape(-1)
        tau = tau/np.sum(tau)
        tau_all[i] = tau
        T[:,0] = np.sqrt(tau)

        A = Torontonian().sigma_Thermal(n_mean_A, N, T * np.sqrt(eta_i))
        B = Torontonian().sigma_Thermal(n_mean_B, N, T * np.sqrt(eta_i))

        A_ProbsAll,Eve = A.ProbsAll()
        B_ProbsAll,Eve = B.ProbsAll()

        Probs_C[i] = Torontonian.Prob_ABtoC(Eve, A_ProbsAll, B_ProbsAll)
        C_1_Theory_all[i], C_2_Theory_all[i] = _MD_Threshold(Eve,Probs_C[i])

        n_mean_all[i] = eta_i*n_mean
        print('n_mean = ',eta_i*n_mean,end=' ')

    np.save('n_mean_all.npy',n_mean_all)
    np.save('tau_all.npy',tau_all)
    np.save('C_1_Theory_all.npy',C_1_Theory_all)
    np.save('C_2_Theory_all.npy',C_2_Theory_all)
    np.save('Probs_Theory_all',Probs_C)

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
    print('run time =',end - start,'s',sep=' ')

    return TVD                                     #TVD

def TVD_plot():

    Pr = np.load('Pr.npy')
    Pr_throry = np.load('Pr_throry.npy')
    n_mean_all = np.load('n_mean_all.npy')
    tau_all = np.load('tau_all.npy')
    k = np.load('k.npy')

    delta_Pr = np.zeros((Pr.shape[0],Pr.shape[1]))
    print('-----------------------------------------------')
    for i in range(Pr.shape[0]):
        print('splitting ratio =',tau_all[i])
        print('n_mean =',n_mean_all[i])
        print('----------------------')
        for j in range(Pr.shape[1]):
            print('%.11f' % Pr_throry[i,j],'%.11f' % Pr[i,j],'%.5f' % ((Pr_throry[i,j]-Pr[i,j])/Pr_throry[i,j]), sep = '\t')
            delta_Pr[i,j] = (Pr_throry[i,j]-Pr[i,j])/Pr_throry[i,j]
        print('-----------------------------------------------')
    X = np.array([0,1,2,3,4,5,6,7,8])
    plt.title(profix+'\n purity '+'%.6f'%(1/(1+k))+'(1: %.6f)'%k)
    for i in range(len(Pr)):
        plt.plot(X,delta_Pr[i],label='$\\bar{n}$=%.4f'%n_mean_all[i])

    plt.legend(loc = 'upper left')
    plt.xlabel('$\pi_0$~$\pi_8$')
    plt.ylabel('$\Delta$')
    plt.show()

def data_check():

    Pr = np.load('Pr.npy')
    Pr_Pi_0 = np.load('Pr_Pi_0.npy')
    print('-----------------------------')
    print('Pr')
    print('-----------------------------')
    print(Pr)
    print('-----------------------------')
    print('Pr_Pi_0')
    print('-----------------------------')
    print(Pr_Pi_0)
    print('-----------------------------')
    return 0

def MD_check():
    n_mean_all = np.load('n_mean_all.npy')
    C_1_Theory = np.load('C_1_Theory_all.npy')
    C_2_Theory = np.load('C_2_Theory_all.npy')

    Eve = _click_events(8)
    Probs = np.load('Probs.npy')
    for i in range(len(Probs)):
        C_1_i ,C_2_i = _MD_Threshold(Eve,Probs[i])
        print('n_mean =',n_mean_all[i],'一阶边缘分布TVD =',np.sum(abs(C_1_i-C_1_Theory[i])/2),'二阶边缘分布TVD =', np.sum(abs(C_2_i-C_2_Theory[i])/2))

    return 0

if __name__=='__main__':
    data()#数据预处理
    print('data check')
    data_check()
    print('Enter 0: Continue...','|','else: End...')
    num = input()
    if num == '0':
        x_0 = np.array([0])
        res_min = scipy.optimize.fmin(main,x_0,xtol = 0.00001,ftol = 1e-6)
        x_1 = res_min
        np.save('k.npy',x_1)
        for i in res_min:
            print(i)
        main(x_1)
        MD_check()
        TVD_plot()
        
