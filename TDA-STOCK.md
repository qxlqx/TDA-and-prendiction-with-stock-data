# Topological data analysis in stock data



In Topological data analysis, it's vital to build topological data model for integrated data in multi-dimension. Make the values which has been forecast according to the data distribution as a centre of sphere, and adjust radius value, by plotting it in a 3DAxes, forecast new data distribution in reasonable probability, and analyse data trend in spatial-temporal duality.

Import packages for model construction.

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
```



This code apply the model to typical stocks data analysis and trend analysis according to the stock K-line. It uses sectional stock data including opening price, closing price and preclosing price from 2021 to 2022, in which data in Sept have been chosen as reference for forecast. To make visualization distinctly, the code uses ten stocks as a part to plot topological figure, increasing the number in every part properly is triable. And change radius in six levels from adjusted radius.

From the trend implied from the whole stocks, data set follow direct ratio on the whole, and the data of chosen part is in downtrend. Just consider the new data have high probability to approach the proportional curve. By setting suitable magnification parameters to define six levels, get different values in levels, and assign weight to these radius values. If the range approaches the last known stock, the data volatility is slight, the stock is steady. In the range of highest level or even over the range, data volatility is significant, may influence stock value to a great extent. In the medium range, the probability is higher than the former, and the distribution is fit for known data distribution.

In every two-dimensional plane, to follow direct ratio more possibly and ensure the reference data deviation acceptable, choose the minimum value in three planes as basic unchanged value, And estimate the forecast data distribution according to the adjusted radius.

```python
dml = np.loadtxt(open("stockdata_per.csv"),delimiter=",")
slist = []
axi = 0
# Set original radius
k = 20
# Sept. data from line 174 to 194
dmla = 176

# Normalize data
def normalization(x):
    g = np.sum(x)/len(x)
    xc = np.array(x) - g
    xt = xc.T
    a = np.sum(xc*xt)/len(x)
    x = (x - g)/a
    return x

slivn0 = normalization(slivn0)
slivn1 = normalization(slivn1)
slivn2 = normalization(slivn2)
for i in range(len(sli)):
    sli[i][0] = slivn0[i]
    sli[i][1] = slivn1[i]
    sli[i][2] = slivn2[i]

# Forecast model
axs = plt.figure().add_subplot(projection='3d')
for i in range(10):
    sli.append([dml[dmla + i, 1],dml[dmla + i, 2],dml[dmla + i, 4], i])
    slior.append([dml[dmla + i, 1], dml[dmla + i, 2], dml[dmla + i, 4]])
    slivn0.append(dml[dmla + i, 1])
    slivn1.append(dml[dmla + i, 2])
    slivn2.append(dml[dmla + i, 4])

u = np.linspace(0,2*np.pi,200)
v = np.linspace(0,np.pi,200)
x = k*np.outer(np.cos(u),np.sin(v))
y = k*np.outer(np.sin(u),np.sin(v))
z = k*np.outer(np.ones(np.size(u)),np.cos(v))

# Adjust and assign weight to radius value
rlist = []
rstd = abs((sli[9][0] - sli[9][1])*.7 + k *.3)
rndd = abs((sli[9][1] - sli[9][2])*.7 + k *.3)
rrdd = abs((sli[9][0] - sli[9][2])*.7 + k *.3)
for r in [rstd,rndd,rrdd]:
    r = (r*.1+r*1.5*.25+r*(1.5**2)*.23+r*(1.5**3)*.22+r*(1.5**4)*.1+r*(1.5**5)*.1)/2
    rlist.append(r)
space_radius = np.sqrt(rlist[0]**2+rlist[1]**2+rlist[2]**2)

# Forecast in every 2D plane
slist_min = min(sli[9][0],sli[9][1],sli[9][2])
while axi < 6:
    if sli[9][0] == slist_min:
        axs.plot_surface(space_radius*x/k + sli[9][0], space_radius*y/k + sli[9][1] - rstd, space_radius*z/k + sli[9][2] - rrdd, cmap=cm.afmhot, alpha=.2)
        slist_x = sli[9][0]
        slist_y = sli[9][1] - rstd
        slist_z = sli[9][2] - rrdd
    if sli[9][1] == slist_min:
        axs.plot_surface(space_radius*x/k + sli[9][0] - rstd, space_radius*y/k + sli[9][1], space_radius*z/k + sli[9][2] - rndd, cmap=cm.afmhot, alpha=.2)
        slist_x = sli[9][0] - rstd
        slist_y = sli[9][1]
        slist_z = sli[9][2] - rndd
    if sli[9][2] == slist_min:
        axs.plot_surface(space_radius*x/k + sli[9][0] - rrdd, space_radius*y/k + sli[9][1] - rndd, space_radius*z/k + sli[9][2], cmap=cm.afmhot, alpha=.2)
        slist_x = sli[9][0] - rrdd
        slist_y = sli[9][1] - rndd
        slist_z = sli[9][2]
    space_radius *= 1.5
    axi += 1
axs.plot_surface(x + sli[9][0], y + sli[9][1], z + sli[9][2], cmap=cm.GnBu, alpha=.3)
```



Then try to build the topological data model and combine it with the forecast model. This code tries to build quadrangle components to reflects quadrilateral correlation data, referring to the thought of KNN, of course you can try triangular shape to build model. Each quadrangle implies the correlation data in space. And there are reflected characteristics of data change between different quadrangles.

```python
# Topological model
distance_k = []
distance = []
shape_k = []
shape_kall = []
tg = []
tgn = []
dotlen = []
deki = []

slrk = []
slmp = []
slmpl = []
cs_k = 60

slist0 = sli[0]
while len(sli) > 1:
    t = sli[0]
    i = 0
    while i < len(sli):
        s = np.array(t) - np.array(slist[i])
        ns = np.sqrt(s[0]**2+s[1]**2+s[2]**2)
        if ns > 0:
            distance_k.append(i)
            distance.append(ns)
        i += 1
    # Get the near one
    rk = distance_k[np.argmin(distance)]
    
    # Slope and edge length calculation
    tgpl0 = max(abs(sli[rk][1]-t[1]), abs(sli[rk][0]-t[0]))
    tg0_a = np.sign((sli[rk][1]-t[1])*(sli[rk][0]-t[0]))*min(abs((sli[rk][1]-t[1])/tgpl0),abs((sli[rk][0]-		t[0])/tgpl0))
    tg0_b = (sli[rk][2] - t[2])/np.sqrt((sli[rk][0]-t[0])**2 + (sli[rk][1]-t[1])**2)
    tg.append([tg0_a,tg0_b])
    tgd0 = np.sign(t[2] - sli[rk][2])
    tgn0_a = abs(tg0_a) * tgd0
    tgn0_b = abs(tg0_b) * tgd0
    if t[3] < sli[rk][3]:
        tgn0_a = -tgn0_a
        tgn0_b = -tgn0_b
    tgn.append([tgn0_a,tgn0_b])
    dotlen.append(np.min(nsl))
    
    xs = [t[0],sli[rk][0]]
    ys = [t[1],sli[rk][1]]
    zs = [t[2],sli[rk][2]]
    axs.plot(xs,ys,zs,'o-')

    # Build component diagram
    shape_k.append(sli[0])
    deki.append(sli[0][-1])
    if len(shape_k) == 2:
        shape_kall.append(shape_k)
        
        # Data extraction for contraction
        for i in range(3):
        	mp1 = .5*sli[rk][i] + .25*shape_k[0][i] + .25*shape_k[1][i]
            mp2 = .5*shape_k[0][i] + .25*sli[rk][i] + .25*shape_k[1][i]
            mp3 = .5*shape_k[1][i] + .25*shape_k[0][i] + .25*sli[rk][i]
            mp = np.mean([mp1,mp2,mp3])
            slmp.append(mp)
        sp = np.array(slmp) - np.array(sli[rk][:3])
        nsp = np.sqrt(sp[0]**2+sp[1]**2+sp[2]**2)
        if nsp <= 60:
            slrk.append(sli[rk][3])
            slmpl.append(slmp)
        
        tgpl1 = max(abs(shape_k[0][1]-sli[rk][1]), abs(shape_k[0][0]-sli[rk][0]))
        tg1_a = np.sign((shape_k[0][1]-sli[rk][1]) * (shape_k[0][0]-sli[rk][0])) * min(abs((shape_k[0][1]-sli[rk][1])/tgpl1),abs((shape_k[0][0]-sli[rk][0])/tgpl1))
        tg1_b = (shape_k[0][2] - sli[rk][2]) / np.sqrt((shape_k[0][0]-sli[rk][0])**2 + (shape_k[0][1]-sli[rk][1])**2)
        tg.append([tg1_a, tg1_b])
        tgd1 = np.sign(shape_k[0][2] - sli[rk][2])
        tgn1_a = abs(tg1_a) * tgd1
        tgn1_b = abs(tg1_b) * tgd1
        if dek[0][3] < sli[rk][3]:
            tgn1_a = -tgn1_a
            tgn1_b = -tgn1_b
        tgn.append([tgn1_a, tgn1_b])
        dotlen1 = np.linalg.norm(np.array(sli[rk])-np.array(shape_k[0]))
        dotlen.append(dotlen1)
        
        xsd = [sli[rk][0],shape_k[0][0]]
        ysd = [sli[rk][1],shape_k[0][1]]
        zsd = [sli[rk][2],shape_k[0][2]]
        axs.plot(xsd,ysd,zsd,'o-')
        shape_k = []
    
    # Refresh data
    del sli[0]
    distance_k = []
    distance = []
    slmp = []
    sr = slist[rk-1]
    slist[rk-1] = slist[0]
    slist[0] = sr
    
    # Tail connection
    if n % 2 != 0:
    tgpl2 = max(abs(sli[0][1]-deke[-1][0][1]), abs(sli[0][0]-deke[-1][0][0]))
    tg2_a = np.sign((sli[0][1]-deke[-1][0][1]) * (sli[0][0]-deke[-1][0][0])) * min(abs((sli[0][1]-deke[-1]		[0][1]) / tgpl2),abs((sli[0][0]-deke[-1][0][0]) / tgpl2))
    tg2_b = (sli[0][2] - deke[-1][0][2])/np.sqrt((sli[0][0]-deke[-1][0][0])**2 + (sli[0][1]-deke[-1][0]			[1])**2)
    tg.append([tg2_a, tg2_b])
    tgd2 = np.sign(deke[-1][0][2] - sli[0][2])
    tgn2_a = abs(tg2_a) * tgd2
    tgn2_b = abs(tg2_b) * tgd2
    if deke[-1][0][3] < sli[0][3]:
        tgn2_a = -tgn2_a
        tgn2_b = -tgn2_b
    tgn.append([tgn2_a, tgn2_b])
    dotlen2 = np.linalg.norm(np.array(deke[-1][0])-np.array(sli[0]))
    dotlen.append(dotlen2)

    xsf = [deke[-1][0][0],sli[0][0]]
    ysf = [deke[-1][0][1],sli[0][1]]
    zsf = [deke[-1][0][2],sli[0][2]]
else:
    xa = deke[-1][0]
    xb = deke[-1][1]
    dxa = np.array(sli[0][:3]) - np.array(xa[:3])
    dxaf = np.linalg.norm(dxa)
    dxb = np.array(sli[0][:3]) - np.array(xb[:3])
    dxbf = np.linalg.norm(dxb)
    A = np.mat([np.array(deke[-1][0][:3]) - np.array(deke[-1][1][:3]), np.array(deke[-2][0][:3]) - np.array(deke[-1][1][:3])])
    p0 = np.mat(np.array(sli[0][:3]) - np.array(deke[-1][1][:3]))
    B = np.bmat('A; p0')
    if matrix_rank(B) == 1:
        tgpl3 = max(abs(sli[0][1] - xa[1]), abs(sli[0][0] - xa[0]))
        tg3_a = np.sign((sli[0][1] - xa[1]) * (sli[0][0] - xa[0])) * min(abs((sli[0][1] - xa[1]) / tgpl3), abs((sli[0][0] - xa[0]) / tgpl3))
        tg3_b = (sli[0][2]-xa[2]) / np.sqrt((sli[0][0]-xa[0]) ** 2 + (sli[0][1] - xa[1]) ** 2)
        tg.append([tg3_a, tg3_b])
        tgd3 = np.sign(deke[-1][0][2] - sli[0][2])
        tgn3_a = abs(tg3_a) * tgd3
        tgn3_b = abs(tg3_b) * tgd3
        if deke[-1][0][3] < sli[0][3]:
            tgn3_a = -tgn3_a
            tgn3_b = -tgn3_b
        tgn.append([tgn3_a, tgn3_b])
        dotlen.append(dxaf)

        xsf = [xa[0], sli[0][0]]
        ysf = [xa[1], sli[0][1]]
        zsf = [xa[2], sli[0][2]]
    else:
        if dxaf <= dxbf:
            tgpl3 = max(abs(sli[0][1] - xa[1]), abs(sli[0][0] - xa[0]))
            tg3_a = np.sign((sli[0][1] - xa[1]) * (sli[0][0] - xa[0])) * min(
                abs((sli[0][1] - xa[1]) / tgpl3),
                abs((sli[0][0] - xa[0]) / tgpl3))
            tg3_b = (sli[0][2] - xa[2]) / np.sqrt(
                (sli[0][0] - xa[0]) ** 2 + (sli[0][1] - xa[1]) ** 2)
            tg.append([tg3_a, tg3_b])
            tgd3 = np.sign(deke[-1][0][2] - sli[0][2])
            tgn3_a = abs(tg3_a) * tgd3
            tgn3_b = abs(tg3_b) * tgd3
            if deke[-1][0][3] < sli[0][3]:
                tgn3_a = -tgn3_a
                tgn3_b = -tgn3_b
            tgn.append([tgn3_a, tgn3_b])
            dotlen.append(dxaf)
            xsf = [xa[0], sli[0][0]]
            ysf = [xa[1], sli[0][1]]
            zsf = [xa[2], sli[0][2]]
        else:
            tgpl3 = max(abs(sli[0][1] - xb[1]), abs(sli[0][0] - xb[0]))
            tg3_a = np.sign((sli[0][1] - xb[1]) * (sli[0][0] - xb[0])) * min(
                abs((sli[0][1] - xb[1]) / tgpl3),
                abs((sli[0][0] - xb[0]) / tgpl3))
            tg3_b = (sli[0][2] - xb[2]) / np.sqrt(
                (sli[0][0] - xb[0]) ** 2 + (sli[0][1] - xb[1]) ** 2)
            tg.append([tg3_a, tg3_b])
            tgd3 = np.sign(deke[-1][0][2] - sli[0][2])
            tgn3_a = abs(tg3_a) * tgd3
            tgn3_b = abs(tg3_b) * tgd3
            if deke[-1][0][3] < sli[0][3]:
                tgn3_a = -tgn3_a
                tgn3_b = -tgn3_b
            tgn.append([tgn3_a, tgn3_b])
            dotlen.append(dxbf)
			xsf = [shape_kall[-1][1][0],slist[0][0]]
			ysf = [shape_kall[-1][1][1],slist[0][1]]
			zsf = [shape_kall[-1][1][2],slist[0][2]]

axs.plot(xsf,ysf,zsf,'o-')
axs.set_xlabel('x')
axs.set_ylabel('y')
axs.set_zlabel('z')
plt.show()
```



<img src="C:%5CUsers%5C%E8%8D%89%E8%8A%A5%5CDesktop%5Cfigure%5Ctdafigure%5CFigure_176_tms.png" alt="Figure_176_tms" style="zoom:72%;" />

<img src="C:%5CUsers%5C%E8%8D%89%E8%8A%A5%5CDesktop%5Cfigure%5Ctdafigure%5Cfigure_tdav.png" alt="figure_tdav" style="zoom:72%;" />

<img src="C:%5CUsers%5C%E8%8D%89%E8%8A%A5%5CDesktop%5Cfigure%5Ctdafigure%5CFigure_tmf.png" alt="Figure_tmf" style="zoom:72%;" />





Here make a probability and accuracy rate analysis. based on the forecast values and range, the code calculates the agree of aggregation and error. With applying model to several ten-stock parts, the average values have been worked out. The agree of aggregation is 87.99% and the error is 0.75%. The accuracy is acceptable.

```python
# Agree of aggregation and error

dmlf = dmla + 10
poft = np.array([dml[dmlf,1],dml[dmlf,2],dml[dmlf,4]])
npolen = np.linalg.norm(poft - np.array(ndot))
npoflen = np.linalg.norm(poft - np.array(slif))
npoclen = np.linalg.norm(np.array(ndot) - np.array(slif))
ae = npoclen/npolen
st = 0
pdot = [dml[dmlf,1],dml[dmlf,2],dml[dmlf,4]]
rf = r*rk**5
npoflenil = []
while np.linalg.norm(np.array(pdot) - np.array(slif)) <= rf:
	st += 1
    npofleni = np.linalg.norm(np.array(pdot) - np.array(slif))
    npoflenil.append(npofleni)
   pdot = [dml[dmlf+st,1],dml[dmlf+st,2],dml[dmlf+st,4]]
range_possibility = (st+1)/6
aggregation_agree = 1 - max(npoflenil)/rf
error = abs(npolen - npoflen)/npolen
```



Compared with the real values of data which have been obtained and forecast, almost are distribute in the forecast range. By analysing their distribution feature, the different level range they're in implies the stocks volatility. For example, from Sept 16th to 22nd in 2022, the six data respectively in 3rd, 5th, 4th, 5th, 6th, 6th level, which implies the data declining trend, but have slight volatility. from Sept 19th to 23rd, the six data all in 2nd level, which implies the steady trend in this stock term. Compared with linear prediction in two-dimensional plane, topological prediction can predict data volatility more correctly based on spatial-temporal duality. With distinct data fluctuation, several points may deviate the prediction smooth curve or repetitive translation curve obviously, and decline the accuracy rate.



The topological model is corresponding to persistence which extracted with GUDHI.

```python
import gudhi

# Persistence barcode
rips_complex = gudhi.RipsComplex(points=distance_matrix, max_edge_length=250) #max_edge_length = filtration
simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
diag = simplex_tree.persistence(min_persistence=0.4)
gudhi.plot_persistence_barcode(diag)


# Persistence diagram
diag = simplex_tree.persistence()
gudhi.plot_persistence_diagram(diag)
plt.show()


# Persistence density
import gudhi.point_cloud.knn

gudhi.plot_persistence_density(diag) #dg
plt.show()
```

<img src="C:%5CUsers%5C%E8%8D%89%E8%8A%A5%5CDesktop%5Cfigure%5Cperfigure%5Cpersistence_barcode.png" alt="persistence_barcode" style="zoom:50%;" />

<img src="C:%5CUsers%5C%E8%8D%89%E8%8A%A5%5CDesktop%5Cfigure%5Cperfigure%5Cpersistence_diagram.png" alt="persistence_diagram" style="zoom:50%;" />

<img src="C:%5CUsers%5C%E8%8D%89%E8%8A%A5%5CDesktop%5Cfigure%5Cperfigure%5Cpersistence_density.png" alt="persistence_density" style="zoom:50%;" />

<img src="C:%5CUsers%5C%E8%8D%89%E8%8A%A5%5CDesktop%5Cfigure%5Cperfigure%5Cpersistence_mesh_ribbon_v2.png" alt="persistence_mesh_ribbon_v2" style="zoom:50%;" />
