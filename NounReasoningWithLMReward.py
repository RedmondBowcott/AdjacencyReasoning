import torch as T
import numpy as np
import tqdm
import pickle
from LMReward import LMReward

device = T.device('cpu')
lmreward = LMReward()

nodes = 5
edges = int(nodes*(nodes-1)/2)
pairs = [(j, k) for j in range(0, nodes - 1) for k in range(j + 1, nodes)]
nouns = ['lying', 'cheating', 'helping', 'supporting', 'deceiving']

n_hid = 256
n_layers = 2

bs = 16
uniform_pb = False

def make_mlp(l, act=T.nn.LeakyReLU(), tail=[]):
    return T.nn.Sequential(*(sum(
        [[T.nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
         for n, (i, o) in enumerate(zip(l, l[1:]))], []) + tail))

def connected_nodes(z):
#Given adjacency list, returns list of tuples containing connected paths and distance
    return [(x, y) for i, (x, y) in enumerate(pairs) if z[i-1] == 1]       

def graph_str(z):
#Given adjacency list, returns string describing this list
#If shortest is true, then gives string stating these shortest paths
    str = ""
    z = connected_nodes(z)
    for (a, b) in z:
        str += f"{nouns[a]} and {nouns[b]} are connected. "
    return str

def str_convert(z):
    z = graph_str(z)
    return (z)

Z = T.zeros((1,)).to(device)
model = make_mlp([edges] + [n_hid] * n_layers + [2*edges+1]).to(device)
opt = T.optim.Adam([ {'params':model.parameters(), 'lr':0.001}, {'params':[Z], 'lr':0.1} ])
Z.requires_grad_()

losses = []
zs = []
all_visited = []
first_visit = -1 * np.ones((2**edges))
l1log = []

for it in tqdm.trange(1000):
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

    strs = [str_convert(row.flip(0)) for row in z]
    lr = lmreward.str_likelihood(strs)
    ll_diff -= lr

    loss = (ll_diff**2).sum()/(bs)
        
    loss.backward()

    opt.step()

    losses.append(loss.item())
  
    zs.append(Z.item())

    if it%100==0: 
        print('loss =', np.array(losses[-50000:]).mean(), 'Z =', Z.item())
        emp_dist = np.bincount(all_visited[-20000:], minlength=2**edges).astype(float)
        emp_dist /= emp_dist.sum()

pickle.dump([losses,zs,all_visited,first_visit, nodes, edges], open(f'out.pkl','wb'))