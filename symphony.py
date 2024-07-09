import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import time
import random
import torch.nn.functional as F
import torch.jit as jit


#==============================================================================================
#==============================================================================================
#=========================================SYMPHONY=============================================
#==============================================================================================
#==============================================================================================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# random seeds
#r1, r2, r3 = random.randint(0,2**32-1), random.randint(0,2**32-1), random.randint(0,2**32-1)
r1, r2, r3 = 590334781, 4271685945, 3353565093
print(r1, ", ", r2, ", ", r3)
torch.manual_seed(r1)
np.random.seed(r2)
random.seed(r3)

class LogFile(object):
    def __init__(self, log_name):
        self.log_name= log_name
    def write(self, text):
        with open(self.log_name, 'a+') as file:
            file.write(text)

log_name = "history_" + str(r1) + "_" + str(r2) + "_" + str(r3) + ".log"
log_file = LogFile(log_name)

#Rectified Huber Symmetric Error Loss Function via JIT Module
# nn.Module -> JIT C++ graph
class ReHSE(jit.ScriptModule):
    def __init__(self):
        super(ReHSE, self).__init__()

    @jit.script_method
    def forward(self, y1, y2):
        ae = torch.abs(y1-y2)
        ae = ae*torch.tanh(ae)
        return ae.mean()

#Rectified Huber Asymmetric Error Loss Function via JIT Module
# nn.Module -> JIT C++ graph
class ReHAE(jit.ScriptModule):
    def __init__(self):
        super(ReHAE, self).__init__()

    @jit.script_method
    def forward(self, y1, y2):
        e = (y1-y2)
        e = torch.abs(e)*torch.tanh(e)
        return e.mean()



#Inplace Dropout function created with the help of ChatGPT
# It is not recommended to use JIT compilation decorator with random generator as Symphony updates seeds each time
# We did exception only for this module as it is used inside neural networks.
# nn.Module -> JIT C++ graph
class InplaceDropout(jit.ScriptModule):
    def __init__(self, p=0.5):
        super(InplaceDropout, self).__init__()
        self.p = p

    @jit.script_method
    def forward(self, x):
        mask = (torch.rand_like(x) > self.p).float()
        return  mask * x + (1.0-mask) * x.detach()

#ReSine Activation Function
# nn.Module -> JIT C++ graph
class ReSine(jit.ScriptModule):
    def __init__(self, hidden_dim=256):
        super(ReSine, self).__init__()
        self.s = nn.Parameter(data=0.5*torch.rand(hidden_dim)+0.5, requires_grad=False)

    @jit.script_method
    def forward(self, x):
        scale = self.s
        x = scale*torch.sin(x/scale)
        return F.prelu(x, 0.1*scale)

#Shared Feed Forward Module
# nn.Module -> JIT C++ graph
class FeedForward(jit.ScriptModule):
    def __init__(self, f_in, f_out=1):
        super(FeedForward, self).__init__()

        self.ffw = nn.Sequential(
            nn.Linear(f_in, 384),
            nn.LayerNorm(384),
            nn.Linear(384, 256),
            ReSine(256),
            nn.Linear(256, 128),
            nn.Linear(128, f_out),
        )

    @jit.script_method
    def forward(self, x):
        return self.ffw(x)


#We recommend not to JIT compile Actor
# underlying logic for different shape input is unclear
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1.0, prob=0.25):
        super(Actor, self).__init__()

        hidden_dim = 384
        
        self.inA = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            InplaceDropout(prob)
        )
        self.inB = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            InplaceDropout(prob)
        )
        self.inC = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            InplaceDropout(prob)
        )
        
        
        self.ffw = nn.Sequential(
            FeedForward(3*hidden_dim, action_dim),
            nn.Tanh()
        )

        self.max_action = torch.mean(max_action).item()
        self.scale = 0.1*self.max_action
        self.lim = 2.5*self.scale
    

    def forward(self, state):
        x = torch.cat([self.inA(state), self.inB(state), self.inC(state)], dim=-1)
        return self.max_action*self.ffw(x)
    
    # Do not use any decorators with random generators (Symphony updates seed each time)
    def soft(self, state):
        x = self.forward(state)
        x += self.scale*torch.randn_like(x).clamp(-self.lim, self.lim)
        return x.clamp(-self.max_action, self.max_action)

 

# nn.Module -> JIT C++ graph
class Critic(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, prob=0.75):
        super(Critic, self).__init__()


        qA = nn.Sequential(
            FeedForward(state_dim+action_dim, 128),
            InplaceDropout(prob)
        )
        qB = nn.Sequential(
            FeedForward(state_dim+action_dim, 128),
            InplaceDropout(prob)
        )
        qC = nn.Sequential(
            FeedForward(state_dim+action_dim, 128),
            InplaceDropout(prob)
        )

        self.nets = nn.ModuleList([qA, qB, qC])

    @jit.script_method
    def forward(self, state, action):
        x = torch.cat([state, action], -1)
        return [net(x) for net in self.nets]

    @jit.script_method
    def min(self, state, action):
        xs = self.forward(state, action)
        xs = torch.cat([torch.mean(x, dim=-1, keepdim=True) for x in xs], dim=-1)
        return torch.min(xs, dim=-1, keepdim=True).values



# Define the algorithm
class Symphony(object):
    def __init__(self, state_dim, action_dim, device, max_action=1.0, tau=0.0057, prob_a=0.25, prob_c = 0.75, capacity=200000, batch_lim = 384, fade_factor=10.0):

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, device, capacity, batch_lim, fade_factor)

        self.actor = Actor(state_dim, action_dim, max_action=max_action, prob=prob_a).to(device)

        self.critic = Critic(state_dim, action_dim, prob=prob_c).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.rehse = ReHSE()
        self.rehae = ReHAE()


        self.max_action = max_action
        self.tau = tau
        self.tau_ = 1.0 - tau
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_next_old_policy = torch.zeros(1).to(device)
        
        


    def select_action(self, state, mean=False):
        state = torch.FloatTensor(state).reshape(-1,self.state_dim).to(self.device)
        with torch.no_grad(): action = self.actor(state) if mean else self.actor.soft(state)
        return action.cpu().data.numpy().flatten()


    def train(self, tr_per_step=5):
        for _ in range(tr_per_step): self.update()
        #time.sleep(0.05)


    def update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample()

        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau_*target_param.data + self.tau*param)
        
        #Actor Update
        next_action = self.actor.soft(next_state)
        q_next_target = self.critic_target.min(next_state, next_action)
        actor_loss = -self.rehae(q_next_target, self.q_next_old_policy)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        #Critic Update
        q = reward + (1-done) * 0.99 * q_next_target.detach()
        qs = self.critic(state, action)
        critic_loss = self.rehse(q, qs[0]) + self.rehse(q, qs[1]) + self.rehse(q, qs[2])

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        with torch.no_grad(): self.q_next_old_policy = q_next_target.detach().mean()




class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device, capacity, batch_lim, fade_factor=7.0):
        self.capacity, self.length, self.device = capacity, 0, device
        self.batch_size = min(max(128, self.length//100), batch_lim) #in order for sample to describe population
        self.random = np.random.default_rng()
        self.indices, self.indexes, self.probs = [], np.array([]), np.array([])
        self.fade_factor = fade_factor
        self.probs_ready = False
        self.batch_lim = batch_lim

        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32).to(device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32).to(device)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.float32).to(device)


    def add(self, state, action, reward, next_state, done):
        if self.length<self.capacity:
            self.length += 1
            self.indices.append(self.length-1)
            self.indexes = np.array(self.indices)
            self.batch_size = min(max(128, self.length//100), self.batch_lim)

        idx = self.length-1

        self.states[idx,:] = torch.FloatTensor(state).to(self.device)
        self.actions[idx,:] = torch.FloatTensor(action).to(self.device)
        self.rewards[idx,:] = torch.FloatTensor([reward]).to(self.device)
        self.next_states[idx,:] = torch.FloatTensor(next_state).to(self.device)
        self.dones[idx,:] = torch.FloatTensor([done]).to(self.device)


        if self.length==self.capacity:
            self.states = torch.roll(self.states, shifts=-1, dims=0)
            self.actions = torch.roll(self.actions, shifts=-1, dims=0)
            self.rewards = torch.roll(self.rewards, shifts=-1, dims=0)
            self.next_states = torch.roll(self.next_states, shifts=-1, dims=0)
            self.dones = torch.roll(self.dones, shifts=-1, dims=0)


    def generate_probs(self):
        if self.probs_ready: return self.probs
        def fade(norm_index): return np.tanh(self.fade_factor*norm_index**2.3) # linear / -> non-linear _/‾
        weights = 1e-7*fade(self.indexes/self.length)# weights are based solely on the history, highly squashed
        self.probs = weights/np.sum(weights)
        if self.length>=self.capacity: self.probs_ready = True
        return self.probs
    
    # Do not use any decorators with random generators (Symphony updates seed each time)
    def sample(self):
        indices = self.random.choice(self.indexes, p=self.generate_probs(), size=self.batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )


    def __len__(self):
        return self.length
