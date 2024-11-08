import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

with open('out.pkl', 'rb') as f:
    losses, zs, all_visited, first_visit, l1log = pickle.load(f)

plt.figure(figsize=(10, 6))
plt.plot(losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss during training')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png') 
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(zs, label='Z', color='red')
plt.xlabel('Iteration')
plt.ylabel('Z')
plt.title('Z Value during training')
plt.legend()
plt.grid(True)
plt.savefig('z_plot.png')
plt.close()

l1_iterations, l1_values = zip(*l1log)
plt.figure(figsize=(10, 6))
plt.plot(l1_iterations, l1_values, label='L1 Distance')
plt.xlabel('Iterations')
plt.ylabel('L1 Distance')
plt.title('L1 Distance between empirical and true distribution')
plt.legend()
plt.grid(True)
plt.savefig('l1_distance_plot.png')
plt.close()

plt.figure(figsize=(10, 6))
unique_visited, counts = np.unique(all_visited, return_counts=True)
plt.bar(unique_visited, counts, label='Visited states', color='orange')
plt.xlabel('State')
plt.ylabel('Visit count')
plt.title('Visited States Distribution')
plt.legend()
plt.grid(True)
plt.savefig('visited_states_plot.png') 
plt.close()

first_visit_times = np.where(first_visit >= 0, first_visit, np.nan)
plt.figure(figsize=(10, 6))
plt.scatter(range(len(first_visit_times)), first_visit_times, label='First Visit Time', color='green', s=10)
plt.xlabel('State')
plt.ylabel('First Visit Iteration')
plt.title('First Visit Time for Each State')
plt.legend()
plt.grid(True)
plt.savefig('first_visit_plot.png') 
plt.close()

state_counts = Counter(all_visited)
top_10_states = state_counts.most_common(10)

print("Top 10 Most Visited States:")
for state, count in top_10_states:
    print(f"State: {state}, Visited: {count} times")
