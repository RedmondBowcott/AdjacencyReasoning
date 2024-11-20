import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import Counter
from itertools import combinations

with open('out.pkl', 'rb') as f:
    losses, zs, all_visited, first_visit, nodes, edges = pickle.load(f)

plt.figure(figsize=(10,6))
unique_visited, counts = np.unique(all_visited, return_counts=True)
plt.bar(unique_visited, counts, label='Visited states', color='orange')
plt.xlabel('State')
plt.ylabel('Visit count')
plt.title('Visited States Distribution')
plt.savefig('visited_states_plot.png') 
plt.close()

n = 5
state_counts = Counter(all_visited)
top_n_states = state_counts.most_common(n)

print(f"Top {n} Most Visited States:")
for state, count in top_n_states:
    print(f"State: {bin(state)[2:].zfill(edges)}, Visited: {count} times")