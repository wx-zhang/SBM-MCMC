from graph_tool.all import *
from sklearn.metrics.cluster import normalized_mutual_info_score
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

b1n = 19
b2n = 75149

g1 = collection.data['football'] # n nodes, O(n) edges n~10e4
cl1 = global_clustering(g1)
print (cl1)
b1t = joblib.load('truthfb.pkl')
def pat1(g1,b1n,b1t):
    state1 = minimize_blockmodel_dl(g1, B_min = b1n, B_max = b1n, deg_corr = False)
    b1 = state1.get_blocks()
    s1 = normalized_mutual_info_score(list(b1), b1t)
    #print (s1)
    #state1.draw(pos=g1.vp.pos, output="fb1.png")
    return s1



d = [42,80,82,36,63,58,59,97]
g11 = Graph(g1)
g11.remove_vertex(d)
b11t = b1t.copy()
for i in d:
    b11t[i] = -1
while -1 in b11t:
    b11t.remove(-1)
b11n = 11


sss11 = []
sss1 = []
for i in range(100):
    sss11.append(pat1(g11,b11n,b11t))
    sss1.append(pat1(g1,b1n,b1t))

plt.figure()
plt.title('NMI scores for model data')
#plt.xlim(0, 1)
plt.hist(sss11)

plt.savefig('b11.png')
g1 = collection.data['football']
plt.figure()
plt.title('NMI scores for the real world data')
#plt.xlim(0, 1)
plt.hist(sss1)

plt.savefig('b1.png')