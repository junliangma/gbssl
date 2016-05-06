import numpy as np
from scipy import sparse
from gbssl import LGC,HMN,PARW,MAD

G = sparse.lil_matrix((5,5))
G[0,1]=1
G[1,0]=1
G[1,2]=1
G[2,1]=1
G[2,3]=1
G[3,2]=1
G[2,0]=1
G[0,2]=1
G[3,4]=1
G[4,3]=1
G.tocsr()

lgc = LGC(graph=G,alpha=0.50)
hmn = HMN(graph=G)
parw = PARW(graph=G,lamb=10)
mad = MAD(graph=G)

x = np.array([1,2,3])
y = np.array([0,0,1])

lgc.fit(x,y)
hmn.fit(x,y)
parw.fit(x,y)
mad.fit(x,y)

print lgc.predict_proba(np.arange(5))
print hmn.predict_proba(np.arange(5))
print parw.predict_proba(np.arange(5))
print mad.predict_proba(np.arange(5))
