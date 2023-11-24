# Project: TNN
# %%
from network import Network
import numpy as np
from tqdm.notebook import tqdm
array_size = 40
temperature = (10,10) # (node, edge)
anneal = True
connectivity = 'bipartite'
bias_node_list = [(800,1,1000)] # (node, bias, weight)
TNN = Network((array_size,array_size), connectivity=connectivity, num_states=2, temperature=temperature)
TNN.set_bias_nodes(bias_node_list)

max_time = 1000
scale_factor = 1
array_series = np.empty((array_size, array_size, max_time//scale_factor),dtype=int)
array_series_flip = np.empty((array_size, array_size, max_time//scale_factor),dtype=int)
state_mask = np.ones((array_size,array_size))
for i in range(array_size):
    for j in range(array_size):
        if (i+j)%2 == 1:
            state_mask[i,j] = -1

for t in tqdm(range(max_time)):
    TNN.update_network_state()
    if t % scale_factor == 0:
        curr_state = TNN.get_state()
        if TNN._connectivity == 'bipartite':
            array_series[:,:,t//scale_factor] = curr_state
            array_series_flip[:,:,t//scale_factor] = state_mask*curr_state
        else:
            array_series[:,:,t//scale_factor] = curr_state
            array_series_flip[:,:,t//scale_factor] = state_mask*curr_state

        TNN.update_energy_series()
    if anneal:
        new_node_temperature = np.max([temperature[0]*(1- 1*t/max_time),1])
        new_edge_temperature = np.max([temperature[1]*(1- 1*t/max_time),1])
        TNN.set_node_temperature(new_node_temperature)
        TNN.set_edge_temperature(new_edge_temperature)
        
# %%  
from plotting import create_gif

node_temp = TNN.get_node_temperature()
edge_temp = TNN.get_edge_temperature()
gif_name = f'random_{connectivity}_grid_{array_size}x{array_size}_node_temp_{node_temp}_edge_temp_{edge_temp}_time_{max_time}.gif'
flip_gif_name = f'random_{connectivity}_grid_{array_size}x{array_size}_node_temp_{node_temp}_edge_temp_{edge_temp}_time_{max_time}_flip.gif'
create_gif(array_series, gif_name)
create_gif(array_series_flip, flip_gif_name)

# %%
from IPython.display import Image
with open(gif_name,'rb') as file:
    display(Image(file.read()))


with open(flip_gif_name,'rb') as file:
    display(Image(file.read()))


# %%
import matplotlib.pyplot as plt

time_axis = np.linspace(0,max_time-1,max_time//scale_factor)
energy_series = TNN.get_energy_series()
min_energy = np.min(energy_series)
plt.figure()
plt.plot(time_axis,energy_series, label=f'Min Energy={min_energy:.2f}', linewidth=3)
plt.xlabel('Time', fontsize=18)
plt.ylabel('Energy', fontsize=18)
plt.title(f'TNN with {connectivity} connectivity', fontsize=18)
plt.legend(fontsize=18)
plt.tight_layout()
plt.show()


# %%
from moviepy.editor import VideoFileClip

# Load your GIF file
gif_filename = 'random_NN_grid_100x100_node_temp_10_edge_temp_100_time_1000.gif'
video_clip = VideoFileClip(gif_filename)

# Save as an MPEG (The format will be inferred from the filename extension)
mpeg_filename = 'output.mpeg'
video_clip.write_videofile(mpeg_filename)
# %%
#Plot convergence

import matplotlib.pyplot as plt
energy_series = TNN.get_energy_series()
max_abs = np.max(np.abs(energy_series))
energy_diff = np.abs(np.diff(energy_series))
rel_energy_diff = energy_diff/max_abs

plt.figure()
plt.plot(energy_diff)
plt.show()

plt.figure()
plt.semilogy(rel_energy_diff)
plt.show()

# %%
# Let's figure out which pixels remain unchanged
from plotting import unchanged_pixels, plot_unchanged_pixels

unchanged = unchanged_pixels(array_series)

plot_unchanged_pixels(unchanged)


# %%

# I want to do some code profiling.

import cProfile
import pstats
profiler = cProfile.Profile()
profiler.enable()
TNN.update_network_state()

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats()
# %%
# Seems like __update_edge_weights is slow.

def edge_weights():
    for _ in range(10000):
        TNN._Network__update_edge_weights(10)

import cProfile
import pstats
profiler = cProfile.Profile()
profiler.enable()
edge_weights()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats()
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
B = create_bipartite_from_2d((3 ,3))

# Check if B is bipartite. This should print True.
print(bipartite.is_bipartite(B))

# Draw using NetworkX built-in drawing functions.
color_map = ["blue" if B.nodes[node]['bipartite'] == 0 else "red" for node in B]
nx.draw(B, node_color=color_map)

# %%

# Convert graph to sparce matrix

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from network import Network
array_size = 10
connectivity = 'NN'
temperature = (100,100)

G = nx.grid_2d_graph(array_size,array_size,periodic=True)
G = nx.convert_node_labels_to_integers(G)
A = nx.to_scipy_sparse_array(G)
print(A.todense())

sparse = A.tocsr()

fig = plt.figure()
plt.spy(sparse, markersize=2)
plt.show()

TNN = Network((array_size,array_size), connectivity=connectivity,
              num_states=2, temperature=temperature)
G = TNN.get_graph()
A = nx.to_scipy_sparse_array(G)
sparse = A.tocsr()
fig = plt.figure()
plt.spy(sparse, markersize=2)
plt.xlim(12,18)
plt.ylim(12,18)
plt.show()

# %%
