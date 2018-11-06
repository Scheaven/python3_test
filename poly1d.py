# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
#多项式拟合(从给定的x,y中解析出最接近数据的方程式)
#要拟合的x,y数据
x = np.arange(1, 17, 1)
y = np.array([4.00, 6.40, 8.00, 8.80, 9.22, 9.50, 9.70, 9.86, 10.00, 10.20, 10.32, 10.42, 10.50, 10.55, 10.58, 10.60])
z1 = np.polyfit(x, y, 4)#3为多项式最高次幂，结果为多项式的各个系数
#最高次幂3，得到4个系数,从高次到低次排列
#最高次幂取几要视情况而定
p1 = np.poly1d(z1)#将系数代入方程，得到函式p1
print(z1)#多项式系数
print(p1)#多项式方程
print(p1(18))#调用，输入x值，得到y
x1=np.linspace(x.min(),x.max(),100)#x给定数据太少，方程曲线不光滑，多取x值得到光滑曲线
pp1=p1(x1)#x1代入多项式，得到pp1,代入matplotlib中画多项式曲线
plt.rcParams['font.sans-serif']=['SimHei']#显示中文
plt.scatter(x,y,color='g',label='散点图')#x，y散点图
plt.plot(x,y,color='r',label='连线图')#x,y线形图
plt.plot(x1,pp1,color='b',label='拟合图')#100个x及对应y值绘制的曲线
#可应用于各个行业的数值预估
plt.legend(loc='best')
plt.savefig('polyfit.png',dpi=400,bbox_inches='tight')
