# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:42:30 2017

@author: Dell
"""
import numpy as np
import pylab as pl
from scipy import interpolate
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi+np.pi/4, 10)
y = np.sin(x)

x_new = np.linspace(0, 2*np.pi+np.pi/4, 100)
f_linear = interpolate.interp1d(x, y)
tck = interpolate.splrep(x, y)#使用splrep()计算欲插值曲线的样条系数（对于N-维空间使用splprep）
y_bspline = interpolate.splev(x_new, tck)   #计算样条插值结果。

plt.xlabel(u'安培/A')
plt.ylabel(u'伏特/V')

plt.plot(x, y, "o",  label=u"原始数据")
plt.plot(x_new, f_linear(x_new), label=u"线性插值")
plt.plot(x_new, y_bspline, label=u"B-spline插值")

pl.legend()
pl.show()