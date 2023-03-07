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

# Forecast model
axs = plt.figure().add_subplot(projection='3d')
for i in range(10):
    slist.append([dml[dmla+i,0],dml[dmla+i,3],dml[dmla+i,4]])

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
    if slist[9][0] == slist_min:
        axs.plot_surface(space_radius*x/k + slist[9][0], space_radius*y/k + slist[9][1] - rstd, space_radius*z/k + slist[9][2] - rrdd, cmap=cm.afmhot, alpha=.2)
        slist_x = slist[9][0]
        slist_y = slist[9][1] - rstd
        slist_z = slist[9][2] - rrdd
    if slist[9][1] == slist_min:
        axs.plot_surface(space_radius*x/k + slist[9][0] - rstd, space_radius*y/k + slist[9][1], space_radius*z/k + slist[9][2] - rndd, cmap=cm.afmhot, alpha=.2)
        slist_x = slist[9][0] - rstd
        slist_y = slist[9][1]
        slist_z = slist[9][2] - rndd
    if slist[9][2] == slist_min:
        axs.plot_surface(space_radius*x/k + slist[9][0] - rrdd, space_radius*y/k + slist[9][1] - rndd, space_radius*z/k + slist[9][2], cmap=cm.afmhot, alpha=.2)
        slist_x = slist[9][0] - rrdd
        slist_y = slist[9][1] - rndd
        slist_z = slist[9][2]
    space_radius *= 1.5
    axi += 1
axs.plot_surface(x + slist[9][0], y + slist[9][1], z + slist[9][2], cmap=cm.GnBu, alpha=.3)
```



Then try to build the topological data model and combine it with the forecast model. This code tries to build quadrangle components to reflects quadrilateral correlation data, referring to the thought of KNN, of course you can try triangular shape to build model. Each quadrangle implies the correlation data in space. And there are reflected characteristics of data change between different quadrangles.

```python
# Topological model
distance_k = []
distance = []
shape_k = []
shape_kall = []
slist0 = slist[0]
while len(slist) > 1:
    t = slist[0]
    i = 0
    while i < len(slist):
        s = np.array(t) - np.array(slist[i])
        ns = np.sqrt(s[0]**2+s[1]**2+s[2]**2)
        if ns > 0:
            distance_k.append(i)
            distance.append(ns)
        i += 1
    # Get the near one
    rk = distance_k[np.argmin(distance)]
    xs = [t[0],slist[rk][0]]
    ys = [t[1],slist[rk][1]]
    zs = [t[2],slist[rk][2]]
    axs.plot(xs,ys,zs,'o-')

    # Build component diagram
    shape_k.append(sli[0])
    shape_kall.append(shape_k)
    if len(shape_k) == 3:
        xsd = [slist[rk][0],shape_k[0][0]]
        ysd = [slist[rk][1],shape_k[0][1]]
        zsd = [slist[rk][2],shape_k[0][2]]
        axs.plot(xsd,ysd,zsd,'o-')
        shape_k = []
    
    # Refresh data
    del slist[0]
    distance_k = []
    distance = []
    sr = slist[rk-1]
    slist[rk-1] = slist[0]
    slist[0] = sr
xsf = [shape_kall[-1][0][0],slist[0][0]]
ysf = [shape_kall[-1][0][1],slist[0][1]]
zsf = [shape_kall[-1][0][2],slist[0][2]]
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
slist_f = []
slist_f.append([slist_x,slist_y,slist_z])
dmlf = dmla + 10
point_f = np.array([dml[dmlf,0],dml[dmlf,3],dml[dmlf,4]])
npo = point_f - np.array(slist[0])
npof = point_f - np.array(slist_f[0])
length_npo = np.sqrt(npo[0]**2+npo[1]**2+npo[2]**2)
length_npof = np.sqrt(npof[0]**2+npof[1]**2+npof[2]**2)
aggregation_agree = abs(length_npof - space_radius - 5)/length_npo
print(aggregation_agree)
error = abs(length_npof - space_radius)*3/sum(point_f)
print(error)
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
