## Data Correlation In Different Time Intervals

Adopt WILL5000IND data from 2020 to 2023 in FRED with topological model. Set time intervals as partitions, the spatial features of related data implied correlation with complexes construction.

```python
import numpy as np
import pandas as pd
from dateutil.parser import parse
import matplotlib.pyplot as plt
```

```python
lstw = pd.read_csv('WILL5000INDlot.csv')


# Process missing value
pr = lstw[lstw['WILL5000IND']!='.']
pr.to_csv('pr.csv',index = True,header = True,encoding = 'utf-8')

prt = pd.read_csv('pr.csv')
prt.drop('Unnamed: 0',axis=1,inplace=True)


# Construct time series
sld = []
slw = []
prld = pd.Series(prt['DATE'],index=range(len(prt)))
for i in range(len(prld)):
    prld[i] = parse(prld[i])
    sld.append(prld[i])
prtw = pd.Series(prt['WILL5000IND'],index=range(len(prt)))
for i in range(len(prtw)):
    slw.append(prtw[i])

lstw = pd.Series(slw,index=sld)
```

```python
# Generate time series image
lsmean = lstw.rolling(10,min_periods=1).mean()
lsewm = lstw.ewm(span=10).mean()
lsmean.plot(style='k--',label='mean')
lsewm.plot(style='k-',label='ewm')
plt.show()
```

```python
sld20 = []
sld21 = []
sld22 = []
print(lstw['2020'].index)
slt = []
for k in range(len(lstw['2020'])):
    slt.append(lstw['2020'].index[k])
print(len(slt))
for ic in range(len(lstw['2020'])):
    sld20.append(lstw['2020'][ic])
for ik in range(len(lstw['2021'])):
    sld21.append(lstw['2021'][ik])
for ie in range(len(lstw['2022'])):
    sld22.append(lstw['2022'][ie])

slda = []
numd = min(len(lstw['2020']),len(lstw['2021']),len(lstw['2022']))
numd = numd-210
# Make 3-dimentional data
for i in range(numd):
    slda.append([sld20[i],sld21[i],sld22[i]])

def fipt(sli):
    axs = plt.figure().add_subplot(projection='3d')

    global slistor, slindor, slirdor, rstd, rndd, rrdd
    ran = len(sli) - 1
    print(ran)
    slior = []
    axi = 0
    k = 20

    u = np.linspace(0, 2 * np.pi, 200)
    v = np.linspace(0, np.pi, 200)
    x = k * np.outer(np.cos(u), np.sin(v))
    y = k * np.outer(np.sin(u), np.sin(v))
    z = k * np.outer(np.ones(np.size(u)), np.cos(v))


    nsk = []
    nsl = []
    dek = []
    dekc = []
    k = 0

    while len(sli) > 1:
        t = sli[0]
        i = 0
        while i < len(sli):
            s = np.array(t) - np.array(sli[i])
            ns = np.sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2)
            if ns > 0:
                nsk.append(i)
                nsl.append(ns)
            i += 1
        rk = nsk[np.argmin(nsl)]
        xs = [t[0], sli[rk][0]]
        ys = [t[1], sli[rk][1]]
        zs = [t[2], sli[rk][2]]
        axs.plot(xs, ys, zs, 'o-')

        dek.append(sli[0][:3])
        dekc.append(dek)
        if len(dek) == 2:
            xsd = [sli[rk][0], dek[0][0]]
            ysd = [sli[rk][1], dek[0][1]]
            zsd = [sli[rk][2], dek[0][2]]
            axs.plot(xsd, ysd, zsd, 'o-')

            dek = []

        del sli[0]
        nsk = []
        nsl = []
        slmp = []
        sr = sli[rk - 1]
        sli[rk - 1] = sli[0]
        sli[0] = sr
        k += 1
    xsf = [dekc[-1][0][0], sli[0][0]]
    ysf = [dekc[-1][0][1], sli[0][1]]
    zsf = [dekc[-1][0][2], sli[0][2]]
    axs.plot(xsf, ysf, zsf, 'o-')
    axs.set_xlabel('x')
    axs.set_ylabel('y')
    axs.set_zlabel('z')
    plt.show()

fipt(slda)
```

