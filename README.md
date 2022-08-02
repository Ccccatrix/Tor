# Gaussian boson sampling

## 1 Introduction
### Gaussian State
 对于一个 N 模式的高斯态，可用 2N×2N 的协方差矩阵来描述
 
 $$d_{i} = \braket{a_{i}}$$
 
 $$n_{ij} = \braket{a_{i}^{\dagger}a_{j}} - d_{i}^* d_{j}$$
 
 $$m_{ij} = \braket{a_{i}a_{j}} - d_{i}d_{j}$$
 
 经过干涉仪矩阵T (可以是非酉矩阵) 的变换后
 
  $$d^{\rm out} = Td^{\rm in}$$
  
  $$n^{\rm out} = T^* n^{\rm in}T^{T}$$
  
  $$m^{\rm out} = Tm^{\rm in}T^{T}$$
  
 对应的协方差矩阵为
 
 $$
  \sigma = 
  \begin{pmatrix}
   n & m\\
   m^* & n*
  \end{pmatrix}+\frac{1}{2}\mathbb{1}
 $$
 
 对于光子数可分辨探测器 ${[1]}$ (eg: PNR)
 
 $$
  1
 $$
 
 对于阈值探测器 ${[2]}$ (eg: SNSPD)
 
 $$
  1
 $$
 
 常见的几种高斯态: 
 
 $$n_{\rm squeezed} = \sinh^2{r},~m_{\rm squeezed}=\cosh^2{r},~d_{\rm squeezed}=0$$
 
 $$n_{\rm thermal} = \bar{n},~m_{\rm thermal}=0,~d_{\rm thermal}=0$$
 
 $$n_{\rm squashed} = \bar{n},~m_{\rm squashed}=\bar{n},~d_{\rm squashed}=0$$
 
 $$n_{\rm coherent} = 0,~m_{\rm coherent}=0,~d_{\rm coherent}=\beta$$
 
 
 
---
## Reference
[1] Madsen, Lars S., et al. "Quantum computational advantage with a programmable photonic processor." Nature 606.7912 (2022): 75-81.

[2] Villalonga, Benjamin, et al. "Efficient approximation of experimental Gaussian boson sampling." arXiv preprint arXiv:2109.11525 (2021).

[3] Kruse, Regina, et al. "Detailed study of Gaussian boson sampling." Physical Review A 100.3 (2019): 032326.

---
## 未完待续...
