import matplotlib.pyplot as plt
import numpy as np
import math




def expanded_div(a,b): 
 """Computes the piecewise divison of list like variables a, b with the convension
    0/0 = 0
Parameters
----------
a : list or 1D ndarray (n_samples)

b : list or 1D ndarray (n_samples)

Returns
-------
a/b or 0 if a = b = 0
"""
  s = np.zeros(len(a))
  for i in range(len(a)):
    if a[i] == b[i] == 0:
      s[i] = 0
    else: 
      s[i] = a[i] / b[i]
  return s
    


def m_minus(X, x, h): 
   """Computes the left Kernel K_-(x-X/h) given a bandwidth h. See [1] 
Parameters
----------
a : list or 1D ndarray (n_samples)

b : list or 1D ndarray (n_samples)

Returns
-------
1 or 0

References
----------
.. [1] M.Fitzke, Schätzung von Sprungstellen einer Regressionsfunktion durch Kernschätzung,
  Masterarbeit
"""
  if 0 > (x - X)/h >= -1: 
    return 1
  else:
    return 0

    
def m_plus(X, x, h): 
     """Computes the left Kernel K_+(x-X/h) given a bandwidth h. See [1] 
Parameters
----------
a : list or 1D ndarray (n_samples)

b : list or 1D ndarray (n_samples)

Returns
-------
1 or 0

References
----------
.. [1] M.Fitzke, Schätzung von Sprungstellen einer Regressionsfunktion durch Kernschätzung,
  Masterarbeit
"""
  if 0 < (x - X)/h <= 1: 
    return 1
  else:
    return 0 


def jp_est(X,y):
       """Computes jump point estimation with the bandwidth heuristic h= 1/sqrt{n}
----------
X : list or 1D ndarray (n_samples)

y : list or 1D ndarray (n_samples)

Returns
-------
\hat{\Delta}_n, \hat{z}_n

References
----------
.. [1] M.Fitzke, Schätzung von Sprungstellen einer Regressionsfunktion durch Kernschätzung,
  Masterarbeit
"""
  h = 1/np.sqrt(np.sqrt(X.size))
  x = np.linspace(min(X)+h, max(X)-h, 10000)
  
  K_minus = pairwise_kernels(X,x,m_minus, h)
  K_plus = pairwise_kernels(X,x,m_plus, h)
 
  y_minus = expanded_div((K_minus*y[:,None]).sum(axis=0),K_minus.sum(axis=0))
  y_plus = expanded_div((K_plus*y[:,None]).sum(axis=0),K_plus.sum(axis=0))
  j = abs(y_plus - y_minus)
  return max(j), x[np.argmax(j)]
  
def jp_est2(X,y):
         """Computes jump point estimation with data dependent bandwidth h_n(x).
----------
X : list or 1D ndarray (n_samples)

y : list or 1D ndarray (n_samples)

Returns
-------
\hat{\Delta}_n, \hat{z}_n

References
----------
.. [1] M.Fitzke, Schätzung von Sprungstellen einer Regressionsfunktion durch Kernschätzung,
  Masterarbeit
"""
  h = 1/np.sqrt(np.sqrt(X.size))
  x = np.linspace(min(X)+h, max(X)-h, 10000)
  l = np.zeros(len(x))
  for i in range(len(x)):
    l[i] = bandwidth(x[i],X,0)
  
  x = x[l != np.inf]
  l = l[l != np.inf]
  
  K_minus = pairwise_kernels(X,x,m_minus, l)
  K_plus = pairwise_kernels(X,x,m_plus, l)
 
  y_minus = expanded_div((K_minus*y[:,None]).sum(axis=0),K_minus.sum(axis=0))
  y_plus = expanded_div((K_plus*y[:,None]).sum(axis=0),K_plus.sum(axis=0))
  j = abs(y_plus - y_minus)
  return max(j), x[np.argmax(j)]
  



def pairwise_kernels(X, Y, metric, h):
         """Computes the kernels between arrays X,Y given a metric and a bandwidth h
----------
X : list or 1D ndarray (n_samples)

X : list or 1D ndarray (n_samples)

metric: callable function m(x,y)



Returns
-------
nxn array K with K[i][j] = m(X[i], Y[j]) 

References
----------
.. [1] M.Fitzke, Schätzung von Sprungstellen einer Regressionsfunktion durch Kernschätzung,
  Masterarbeit
"""
  if callable(metric):
    isArray = isinstance(h, np.ndarray)
# Check matrices first (this is usually done by the metric).
#  X, Y = check_pairwise_arrays(X, Y)
    n_x, n_y = X.shape[0], Y.shape[0]
    # Calculate kernel for each element in X and Y.
    K = np.zeros((n_x, n_y), dtype='float')
    for i in range(n_x):
      start = 0
      if X is Y:
        start = i
      for j in range(start, n_y):
        if isArray: 
          K[i][j] = metric(X[i], Y[j], h[j])
        else:  
          K[i][j] = metric(X[i], Y[j], h)
        if X is Y:
          K[j][i] = K[i][j]
    return K
  else:
    raise AttributeError("Unknown metric %s" % metric)
    
    
    
def bandwidth(x, X,min_h):
"""the data dependent bandwidth h_n(x). See [1]
----------
x: point to adress h_n(x) to. 

X : list or 1D ndarray (n_samples)

min_h: minimal bandwidth h_n (See [1])


Returns
-------
h_n(x) or \infty

References
----------
.. [1] M.Fitzke, Schätzung von Sprungstellen einer Regressionsfunktion durch Kernschätzung,
  Masterarbeit
"""
  n_log = np.ceil(X.size/math.log(X.size))
  
  
  X_r = np.sort(X[X>x])
  X_l = np.sort(X[X<x])[::-1]
  
  if X_r.size < n_log or X_l.size < n_log: 
    return np.inf 
  
  else:
    h_r = X_r[n_log - 1] - x
    h_l = x- X_l[n_log - 1]
  
    h= max(h_r, h_l)
    if h >= min_h:
      return h 
    else:
      return np.inf
        