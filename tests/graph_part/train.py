import numpy as np, random

def random_graph(n_nodes, n_edges, intergroup_link_ratio):
    edges = []
    while len(edges) < n_edges:
        n1 = random.randint(0, n_nodes-1)
        n2 = random.randint(0, n_nodes-1)
        if n1%2 != n2%2:
            if intergroup_link_ratio < random.random():
                continue
            edges.append((n1, n2, -1.0))
        else:
            edges.append((n1, n2, 1.0))
    return edges
    
def generate_minibatch(mb_size, n_nodes, graph):
    eye = np.eye(n_nodes)
    mb = np.empty((mb_size, n_nodes*2+1))
    for _ in range(mb_size):
        edge = random.choice(graph)
        row = np.empty(())
    
    