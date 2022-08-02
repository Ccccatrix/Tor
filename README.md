# Gaussian boson sampling
---
## 1 Introduction
### Gaussian State
 对于一个 $\ell$ 模式的高斯态，可用 $2\ell × 2\ell$ 的协方差矩阵来描述
 
 $$d_{i} = \braket{a_{i}}$$
 
 $$n_{ij} = \braket{a_{i}^{\dagger}a_{j}} - d_{i}^* d_{j}$$
 
 $$m_{ij} = \braket{a_{i}a_{j}} - d_{i}d_{j}$$
  
 对应的协方差矩阵 $\sigma$ 为
 
 $$
  __\sigma__ = 
  \begin{pmatrix}
   __n & m\\
   m^* & n*__
  \end{pmatrix}+\frac{1}{2}\mathbb{__I___{\mathrm{2}\ell}}
 $$
 
 经过干涉仪矩阵T (可以是非酉矩阵) 的变换后
 
 $$d^{\rm out} = Td^{\rm in}$$
  
 $$n^{\rm out} = T^* n^{\rm in}T^{T}$$
  
 $$m^{\rm out} = Tm^{\rm in}T^{T}$$
 
 $$
 __\sigma__ = \mathbb{__I___{\mathrm{2}\ell}}
 -\frac{1}{2}
 \begin{pmatrix}
 T & 0 \\
 0 & T^*
 \end{pmatrix}
 \begin{pmatrix}
 T^{\dagger} & 0 \\
 0 & T^T
 \end{pmatrix}
 +
 \begin{pmatrix}
 T & 0 \\
 0 & T^*
 \end{pmatrix}
 \sigma_{in}
 \begin{pmatrix}
 T^{\dagger} & 0 \\
 0 & T^T
 \end{pmatrix}
 $$
 
 常见的几种高斯态: 
 
 $$n_{\rm squeezed} = \sinh^2{r},~m_{\rm squeezed}=\cosh^2{r},~d_{\rm squeezed}=0$$
 
 $$n_{\rm thermal} = \bar{n},~m_{\rm thermal}=0,~d_{\rm thermal}=0$$
 
 $$n_{\rm squashed} = \bar{n},~m_{\rm squashed}=\bar{n},~d_{\rm squashed}=0$$
 
 $$n_{\rm coherent} = 0,~m_{\rm coherent}=0,~d_{\rm coherent}=\beta$$
 
 对于**光子数可分辨探测器** $^{[1]}$ (eg: PNR)
 
  $$
   __A__ = __X___{2\ell}(\mathbb{__I___{\mathrm{2}\ell}} - __\sigma__^{-1})
  $$
 
 $$
  __X___{2\ell} = 
  \begin{pmatrix}
  0 & \mathbb{__I___{\ell}} \\
  \mathbb{__I___{\ell}} & 0 \\
  \end{pmatrix}
 $$
 
 $$
 {\rm Prob}({__S__})=\frac{1}{\sqrt{{\rm det}(__\sigma__)}} \frac{{\rm Haf}(__{A_{S}}__)}{\prod\nolimits_{i=1}^{\ell}S_i!}
 $$
 
 对于**阈值探测器** $^{[2]}$ (eg: SNSPD)
 
 $$
  __{O_S}__=\mathbb{__I___{\mathrm{2}\ell}} - __{(\sigma^{-1})_{S}}__
 $$
 
 $$
  {\rm Prob}({__S__})=\frac{{\rm Tor}( __{O_S}__ )}{\sqrt{{\rm det}__{\sigma}__}}
 $$
 
 选取 $\sigma$ 的 $[m_0,m_1,...,m_k,m_0+\ell,m_1+\ell...,m_k+\ell]$ 行和列交叉的元素作为子矩阵, 可得到子模式 $[m_0,m_1,...,m_k]$ 的 $\sigma$
 
---
## Reference
[1] Madsen, Lars S., et al. "Quantum computational advantage with a programmable photonic processor." Nature 606.7912 (2022): 75-81.

[2] Villalonga, Benjamin, et al. "Efficient approximation of experimental Gaussian boson sampling." arXiv preprint arXiv:2109.11525 (2021).

[3] Kruse, Regina, et al. "Detailed study of Gaussian boson sampling." Physical Review A 100.3 (2019): 032326.

---
## 未完待续...
