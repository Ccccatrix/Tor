import numpy as np
import os
import sympy
from thewalrus import tor
from tqdm import tqdm
from numba import jit

class Torontonian:
    
    def A_s_tor(self,A,S):
        N = len(S)
        index_0 = np.argwhere(S)
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
        N = int(len(self.sigma)/2)
        Eve = self.click_events(N)
        return self.Probs(Eve),Eve

    def sigma_other(self,sigma):
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

    def sigma_Coherent(self,alpha,T):
        return -1

    def traceto(self):
        return -1

    @classmethod
    def click_events(self,Z):
        b = np.zeros((2**Z,Z)).astype(int)
        a = np.arange(0,2**Z,1)
        for i in range(0,len(a)):
            x = np.array([int(i) for i in bin(a[i])[2:]])
            b[i,Z-len(x):Z] = x
        return b

    @classmethod
    def BinHexOct(self,n,x):
        #任意进制转换
        #n为待转换的十进制数，x为进制，取值为2-16
        a=[0,1,2,3,4,5,6,7,8,9,'A','b','C','D','E','F']
        b=[]
        while True:
            s=n//x  # 商
            y=n%x  # 余数
            b=b+[y]
            if s==0:
                break
            n=s
        b.reverse()
        res = np.array([a[i] for i in b]).reshape(-1)
        return res

    @classmethod
    def Prob_ABtoC(self,events,Probs_A,Probs_B):
        Probs_C = np.zeros((len(Probs_A),1)).reshape(-1)
        N = len(events[0])
        for k in tqdm(range(0,len(events))):
            events_C = events[k]
            index = np.argwhere(events_C).reshape(-1)
            tol = np.sum(events_C)
            A_list = np.zeros((3 ** tol , tol),dtype=int)
            B_list = np.zeros((3 ** tol , tol),dtype=int)
            events_A_list = np.zeros((3 ** tol , N),dtype=int)
            events_B_list = np.zeros((3 ** tol , N),dtype=int)
            index_A = np.zeros((3 ** tol , 1),dtype=int).reshape(-1)
            index_B = np.zeros((3 ** tol , 1),dtype=int).reshape(-1)
            for i in range(0,3 ** tol):
                change = self.BinHexOct(i,3)
                index_list = np.zeros((tol,1)).reshape(-1)
                index_list[tol-len(change):tol] = change
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
                index_A[i] = int(''.join(str(x) for x in events_A_list[i]), 2)
                index_B[i] = int(''.join(str(x) for x in events_B_list[i]), 2)
            Probs_C[k] = np.sum(np.multiply(Probs_A[index_A], Probs_B[index_B]))
        return Probs_C
    
    @classmethod
    def getNM(self,g2,Np):
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
