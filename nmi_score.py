from graph_tool.all import *
from sklearn.metrics.cluster import normalized_mutual_info_score
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np




b_noise = 19
g_noise = collection.data['football'] # n nodes, O(n) edges n~10e4
cl_noise = global_clustering(g_noise)
bt_noise = joblib.load('truthfb.pkl')
state_noise = minimize_blockmodel_dl(g_noise, B_min = b_noise, B_max = b_noise, deg_corr = False)
bp_noise = state_noise.get_blocks()
nmi_noise = normalized_mutual_info_score(list(bp_noise), bt_noise)
print (f'NMI score of full data is {nmi_noise}')




d = [42,80,82,36,63,58,59,97]
g_clear = Graph(g_noise)
g_clear.remove_vertex(d)
bt_clear = b1t.copy()
for i in d:
    bt_clear[i] = -1
while -1 in bt_clear:
    b11t.remove(-1)
b_clear = 11
state_clear = minimize_blockmodel_dl(g_clear, B_min = b_clear, B_max = b_clear, deg_corr = False)
bp_clear = state_clear.get_blocks()
nmi_clear = normalized_mutual_info_score(list(bp_clear), bt_clear)
print (f'NMI score of full data is {nmi_clear}')

'''
sss11 = []
sss1 = []
for i in range(100):
    sss11.append(pat1(g11,b11n,b11t))
    sss1.append(pat1(g1,b1n,b1t))

plt.figure()
plt.title('NMI scores for model data')
#plt.xlim(0, 1)
plt.hist(sss11

plt.savefig('b11.png')
g1 = collection.data['football']
plt.figure()
plt.title('NMI scores for the noise data')
plt.hist(sss1)

plt.savefig('b1.png')
'''
