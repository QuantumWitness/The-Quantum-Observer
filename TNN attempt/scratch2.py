# %%
# testing out some bipartite graph stuff

import networkx as nx
from networkx.algorithms import bipartite

def create_bipartite_from_2d(size):
    # Create a 2D grid graph
    G = nx.grid_2d_graph(*size)
    G = nx.convert_node_labels_to_integers(G)

    # Assign each node to one of two sets based on its position.
    for node in G:
        x, y = divmod(node,size[0])   # Convert single integer back into (i,j) coordinates.
        G.nodes[node]["bipartite"] = (x + y) % 2

    return G

# Test function with a small example.
B = create_bipartite_from_2d((5 ,5))

# Check if B is bipartite. This should print True.
print(bipartite.is_bipartite(B))

# Draw using NetworkX built-in drawing functions.
color_map = ["blue" if B.nodes[node]['bipartite'] == 0 else "red" for node in B]
nx.draw(B, node_color=color_map, with_labels=False)

# %%
G=nx.grid_2d_graph(5,5)
nx.draw(G, with_labels=False)
# %%
from network import Network
import numpy as np
from tqdm.notebook import tqdm
array_size = 10
temperature = 100
connectivity = 'bipartite'
TNN = Network((array_size,array_size), connectivity=connectivity, num_states=2, temperature=temperature)
G = TNN.get_graph()
G.edges(1)
len(G.edges(0))
# %%
import networkx as nx
G = nx.grid_2d_graph(10,10, periodic=True, create_using=nx.MultiGraph)
G = nx.convert_node_labels_to_integers(G)
nnn = set()

for node in G:
    for nbr in G[node]:
        for nbr_nbr in G[nbr]:
            for nbr_nbr_nbr in G[nbr_nbr]:
                if nbr_nbr_nbr not in G[node]:
                    nnn.add((node,nbr_nbr_nbr,0))
    
G.add_edges_from(nnn)
# %%
print(len(G.edges(3)))   
# %%
G.add_edges_from(nnn)
# %%
