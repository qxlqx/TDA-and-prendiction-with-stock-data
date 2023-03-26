## PCA application in time series data

To analysis data features in different dimensions, PCA can analyse principal data distributions by eliminating disturbance. It also can be used to verify data features and performance, which can simplify the tedious work and be useful for analysis.

With eigenvalues approaching to infinitesimal, the time series data is similar to linear distribution. That means variance calculated from data is tiny, obvious disturbance in data set can be excluded. While an calculable eigenvalue represent the main values in need for analysis model construction.

And I try to use kPCA in original data, making dimension reduction with nonlinear function. Calculate symmetric kernel matrix and use its eigenvalues and eigenvectors to extract principal components. Using computers for calculation make less hassle, but the process in kPCA is suitable for referance.\\

```python
import numpy as np
import matplotlib.pyplot as plt

dml = np.loadtxt(open("stockdata_per.csv"),delimiter=",").astype(np.int64)
dmla = 176
p1 = 100 # enlargement parameter
sli = []
for i in range(len(dml)):
    sli.append([dml[i, 0], dml[i, 3], dml[i, 4]])

slik = list(map(lambda x: p1*np.log(x), list(map(lambda y: y, sli))))
K0 = np.asarray(np.matrix(slik))
K = np.matrix(slik)
KL = []
KLp = []
for i in range(np.shape(K)[1]):
    for j in range(np.shape(K)[0]):
        KLp.append(K[j,i])
    KL.append(KLp)
    KLp = []
K = KL*K

d = np.shape(K)[1]
L = np.ones((d,d))*(1/d)
KLc = K - L*K - K*L + L*K*L
print(np.linalg.det(KLc))
u,v = np.linalg.eig(KLc)
print(u)

from sklearn.decomposition import PCA
pca = PCA(n_components=d)
pca.fit(K0)
print(pca.components_)
print(pca.explained_variance_)
v0 = pca.mean_
for v,l in zip(pca.components_,pca.explained_variance_):
    v1 = v*3*np.sqrt(l)
axs = plt.figure().add_subplot(projection='3d')
t = np.linspace(0,1)
vp = []
print(v0)
for i in range(len(t)):
    vp.append(v0+t[i]*v1)
axs.plot(vp[:][0],vp[:][1],vp[:][2])
# axs.scatter(K0[:,0],K0[:,1],K0[:,2])
plt.show()
```

```python
-9.231899050620735e-05
[ 2.27438148e+02  4.08600393e+01 -9.93410146e-09]
```

<img src="C:%5CUsers%5C%E8%8D%89%E8%8A%A5%5CDesktop%5Cpca.png" style="zoom:80%;" />