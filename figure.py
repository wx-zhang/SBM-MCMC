from graph_tool.all import *
import joblib

g1 = collection.data['football']
b1t = joblib.load('truthfb.pkl')
b1n  = 19
b11n = 11
n1 = 115
n2 = 107
d = [42,80,82,36,63,58,59,97]
g11 = Graph(g1)
g11.remove_vertex(d)
b11t = b1t.copy()
for i in d:
    b11t[i] = -1
while -1 in b11t:
    b11t.remove(-1)

state1 = minimize_blockmodel_dl(g1, B_min=b1n, B_max=b1n, deg_corr=False)
state1t = state1
for i in range(n1):
    state1t.move_vertex(i,b1t[i])
    g1.vp.value[i] = b1t[i]

state1 = minimize_blockmodel_dl(g1, B_min=b1n, B_max=b1n, deg_corr=False)

state1t.draw(pos=g1.vp.pos, vertex_text = g1.vp.value,output = '1t.png')
state1.draw(pos=g1.vp.pos, output = '1.png')

state11 = minimize_blockmodel_dl(g11, B_min=b1n, B_max=b1n, deg_corr=False)
state11t = state11
for i in range(n2):
    state11t.move_vertex(i,b11t[i])
    g11.vp.value[i] = b11t[i]

state11 = minimize_blockmodel_dl(g11, B_min=b11n, B_max=b11n, deg_corr=False)

state11t.draw(pos=g11.vp.pos, vertex_text = g11.vp.value, output = '11t.png')
state11.draw(pos=g11.vp.pos, output = '11.png')