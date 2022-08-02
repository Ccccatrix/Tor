# Tor

## Introduction
### Gaussian State
 对于一个n模式的高斯态，可用2n×2n的协方差矩阵来描述
 $$d_{i} = \braket{a_{i}}$$
 $$n_{ij} = \braket{a_{i}^{\dagger}a_{j}} - d_{i}^* d_{j}$$
 $$m_{ij} = \braket{a_{i}a_{j}} - d_{i}d_{j}$$
 经过干涉仪矩阵T(可以是非酉矩阵)的变换后
  $$d^{out} = Td^{in}$$
  $$n^{out} = T^* n^{in}T^{T}$$
  $$m^{out} = Tm^{in}T^{T}$$
 对应的协方差矩阵为
 $$
 \begin{matrix} n & m\\
 n & m\\ \end{matrix}$$
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
