import random


def kager(adjlist):
    i = len(adjlist) + 1
    while len(adjlist) > 2:
        u, v = random.sample(list(adjlist.keys()), k=2)
        u_edges = adjlist[u]
        v_edges = adjlist[v]
        
        w = i
        w_edges = []
        adjlist[w] = w_edges
        
        for out_v in u_edges:
            while u in adjlist[out_v]:
                adjlist[out_v].remove(u)
            adjlist[out_v].append(w)
            w_edges.append(out_v)
        for out_v in v_edges:
            while v in adjlist[out_v]:
                adjlist[out_v].remove(v)
            adjlist[out_v].append(w)
            w_edges.append(out_v)
        while w in w_edges:
            w_edges.remove(w)
        
        del adjlist[u]
        del adjlist[v]

        i += 1
    
    v1, v2 = adjlist.keys()
    assert len(adjlist[v1]) == len(adjlist[v2]), "degree of v1 v2 not equal"
    return len(adjlist[v1])

lines = []
with open('kargergraph.txt', 'r') as f:
    lines = [line.split() for line in f.readlines()]
adjlist = { l[0]:l[1:] for l in lines }

import copy
mincut = 200 ** 2
for k in range(5000):
    # print(k)
    # lines = []
    # with open('kargergraph.txt', 'r') as f:
    #     lines = [line.split() for line in f.readlines()]
    # adjlist = { l[0]:l[1:] for l in lines }
    adjlist_copy = copy.deepcopy(adjlist)
    cut = kager(adjlist_copy)
    if mincut > cut:
        mincut = cut

print(mincut)
