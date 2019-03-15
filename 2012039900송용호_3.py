#과제 3-3

import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pylab as plt
import matplotlib as mpl
import seaborn as sns
from seaborn import distplot
import random


N1 = random.randint(1,51)
N2 = random.randint(5,46)

X1 = sp.stats.norm().rvs(N1)
X2 = sp.stats.norm().rvs(N2)

sns.distplot(X1)
sns.distplot(X2)


y = sp.stats.ks_2samp(X1,X2)

if (y.pvalue>=0.05):
    type = "같은"
else:
    type = "다른"

print("유의확률이",int(y.pvalue*100),"% 이므로 유의수준이 5%라면",type,"분포라고 할 수 있다.")
plt.show()