import torch as T
import numpy as np
import tqdm
import pickle
from collections import deque
from itertools import combinations

device = T.device('cpu')

nodes = 5
edges = int(nodes*(nodes-1)/2)
var = 1

n_hid = 256
n_layers = 2

bs = 16
uniform_pb = False

def make_mlp(l, act=T.nn.LeakyReLU(), tail=[]):
    return T.nn.Sequential(*(sum(
        [[T.nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))

def make_adj_list(x):
    links = list(combinations(range(1, nodes + 1), 2))
    adjacency_list = {i: [] for i in range(1, nodes + 1)}

    for i, link in enumerate(x):
        if link:  
            a, b = links[i]
            adjacency_list[a].append(b)
            adjacency_list[b].append(a)

    return adjacency_list

def bfs_from_node(x, a):
    queue = deque([a])
    d = {a: 0} 

    while queue:
        current_node = queue.popleft()

        for b in x.get(current_node, []):
            if b not in d:
                d[b] = d[current_node] + 1
                queue.append(b)

    max_node = max(x.keys())
    return [d.get(node, -1) for node in range(1, max_node + 1)]

def bfs_total(x):
    shortest = []

    for n in x:
        n_distances = bfs_from_node(x, n)

        for m in x:
            if m > n:
                shortest.append(n_distances[list(x.keys()).index(m)])

    return shortest

def shortest_paths(z):    
    return([bfs_total(make_adj_list(z))])

def log_reward(x):
    true_paths = T.tensor([1,2,3,4,1,2,3,1,2,1])
    shortest = shortest_paths(x)
    paths = T.tensor([0 if tp == 0 else s for tp, s in zip(true_paths, shortest)])
    return -1/(2*var)*(((paths-true_paths + 2*nodes*(x==-1)) ** 2 ).sum())

Z = T.zeros((1,)).to(device)
model = make_mlp([edges] + [n_hid] * n_layers + [2*edges+1]).to(device)
opt = T.optim.Adam([ {'params':model.parameters(), 'lr':0.001}, {'params':[Z], 'lr':0.1} ])
Z.requires_grad_()

losses = []
zs = []
all_visited = []
first_visit = -1 * np.ones((2**edges))
l1log = []

for it in tqdm.trange(2000):
    opt.zero_grad()
    
    z = T.zeros((bs,edges), dtype=T.long).to(device)
    done = T.full((bs,), False, dtype=T.bool).to(device)
        
    action = None
    
    ll_diff = T.zeros((bs,)).to(device)
    ll_diff += Z

    while T.any(~done):
        
        pred = model(z[~done].float())
        
        edge_mask = T.cat([ (z[~done]==1).float(), T.zeros(((~done).sum(),1), device=device) ], 1)
        logits = (pred[...,:edges+1] - 1000000000*edge_mask).log_softmax(1)

        init_edge_mask = (z[~done]== 0).float()
        back_logits = ( (0 if uniform_pb else 1)*pred[...,edges+1:2*edges+1] - 1000000000*init_edge_mask).log_softmax(1)
        
        if action is not None: 
            ll_diff[~done] -= back_logits.gather(1, action[action!=edges].unsqueeze(1)).squeeze(1)
            
        exp_weight= 0.
        temp = 1
        sample_ins_probs = (1-exp_weight)*(logits/temp).softmax(1) + exp_weight*(1-edge_mask) / (1-edge_mask+0.0000001).sum(1).unsqueeze(1)
        
        action = sample_ins_probs.multinomial(1)
        ll_diff[~done] += logits.gather(1, action).squeeze(1)

        terminate = (action==edges).squeeze(1)
        for x in z[~done][terminate]: 
            state = (x.cpu()*(2**T.arange(edges))).sum().item()
            if first_visit[state]<0: first_visit[state] = it
            all_visited.append(state)
        
        done[~done] |= terminate

        with T.no_grad():
            z[~done] = z[~done].scatter_add(1, action[~terminate], T.ones(action[~terminate].shape, dtype=T.long, device=device))

    lr = T.tensor([log_reward(z[i].flip(0)) for  i in range (z.size(0))])
    ll_diff -= lr

    loss = (ll_diff**2).sum()/(bs)
        
    loss.backward()

    opt.step()

    losses.append(loss.item())
  
    zs.append(Z.item())

    if it%100==0: 
        print('loss =', np.array(losses[-100:]).mean(), 'Z =', Z.item())
        emp_dist = np.bincount(all_visited[-50000:], minlength=2**edges).astype(float)
        emp_dist /= emp_dist.sum()

pickle.dump([losses,zs,all_visited,first_visit, nodes, edges], open(f'out.pkl','wb'))