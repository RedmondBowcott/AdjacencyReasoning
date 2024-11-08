import torch as T
import numpy as np
import tqdm
import pickle
from collections import deque, defaultdict
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
#Creates an adjacency list from the connections tensor
    links = list(combinations(range(1, nodes + 1), 2))
    adjacency_list = defaultdict(list)

    for i, link in enumerate(x):
        if link:
            a,b = links[i]
            adjacency_list[a].append(b)
            adjacency_list[b].append(a)

    return dict(adjacency_list)

def bfs_from_node(x,a):
    #x our adajcency list, a our starting node, d the distances
    queue = deque([a])
    d = {a: 0}  

    while queue:
        current_node = queue.popleft()

        for b in x.get(current_node, []):
            if b not in d:  # If not visited yet
                d[b] = d[current_node] + 1
                queue.append(b)

    for node in x:
        if node not in d:
            d[node] = -1

    return list[d.values()]

def bfs_total(x):
    shortest = []

    for n in x:
        n_distances = bfs_from_node(x, n)

        for m in x:
            if m > n:
                shortest.append(n_distances[m])

    return shortest

def shortest_paths(z):    
    return([bfs_total(make_adj_list(x)) for x in z])

def log_reward(x):
    true_paths = T.tensor([1,2,3,4,1,2,3,1,2,1])
    shortest = shortest_paths(x)
    paths = T.where(true_paths==0, 0, shortest)
    return -1/(2*var)*(((paths-true_paths + 2*nodes*(x==-1)) ** 2 ).sum())

all_graphs = T.tensor([ [int(digit) for digit in bin(x)[2:].zfill(edges)] for x in range(2**edges)])
print(all_graphs)
print(all_graphs[0])
print(log_reward(all_graphs[0]))
truelr = T.tensor([log_reward(g) for g in all_graphs])
print(truelr[124], truelr[125],truelr[127],truelr[551], truelr[511], truelr[1022],truelr[1023])
print(truelr.argmax().item())
print('total reward', truelr.logsumexp(0))
true_dist = truelr.softmax(0).cpu().numpy()


Z = T.zeros((1,)).to(device)
model = make_mlp([edges] + [n_hid] * n_layers + [2*edges+1]).to(device)
opt = T.optim.Adam([ {'params':model.parameters(), 'lr':0.001}, {'params':[Z], 'lr':0.1} ])
Z.requires_grad_()

losses = []
zs = []
all_visited = []
first_visit = -1 * np.ones_like(true_dist)
l1log = []

for it in tqdm.trange(10000):
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

    lr = truelr[(z * (2 ** T.arange(edges))).sum(dim=1)]
    ll_diff -= lr

    loss = (ll_diff**2).sum()/(bs)
        
    loss.backward()

    opt.step()

    losses.append(loss.item())
  
    zs.append(Z.item())

    if it%100==0: 
        print('loss =', np.array(losses[-100:]).mean(), 'Z =', Z.item())
        emp_dist = np.bincount(all_visited[-2000:], minlength=len(true_dist)).astype(float)
        emp_dist /= emp_dist.sum()
        l1 = np.abs(true_dist-emp_dist).mean()
        print('L1 =', l1)
        l1log.append((len(all_visited), l1))

pickle.dump([losses,zs,all_visited,first_visit,l1log], open(f'out.pkl','wb'))
