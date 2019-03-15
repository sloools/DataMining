#과제 3-2

import numpy as np
import scipy as sp
from scipy import stats

N = 100 #데이터 개수
theta_0 = 0.5 # 실제 모수=0.5
np.random.seed(0)
x = sp.stats.bernoulli(theta_0).rvs(N)
n = np.count_nonzero(x)
y = sp.stats.binom_test(n,N)
print("P-value = ",y)

if (y>=0.05):
    print("유의확률은 ",int(y*100),"% 로, 귀무가설을 기각할 수 없다.")
else:
    print("유의확률은 ",int(y*100),"% 로 귀무가설을 기각할 수 있다.")
