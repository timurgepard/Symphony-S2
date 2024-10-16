import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import math
import random
import torch.nn.functional as F
import torch.jit as jit
import os, re


#==============================================================================================
#==============================================================================================
#=========================================LOGGING=============================================
#==============================================================================================
#==============================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# to continue writing to the same history file and derive its name. This function created with the help of ChatGPT
def extract_r1_r2_r3():
    pattern = r'history_(\d+)_(\d+)_(\d+)\.log'

    # Iterate through the files in the given directory
    for filename in os.listdir():
        # Match the filename with the pattern
        match = re.match(pattern, filename)
        if match:
            # Extract the numbers r1, r2, and r3 from the filename
            return map(int, match.groups())
    return None


#write or append to the history log file
class LogFile(object):
    def __init__(self, log_name_main, log_name_opt):
        self.log_name_main = log_name_main
        self.log_name_opt = log_name_opt
    def write(self, text):
        with open(self.log_name_main, 'a+') as file:
            file.write(text)
    def write_opt(self, text):
        with open(self.log_name_opt, 'a+') as file:
            file.write(text)

numbers = extract_r1_r2_r3()
if numbers != None:
    # derive random numbers from history file
    r1, r2, r3 = numbers
else:
    # generate new random seeds
    r1, r2, r3 = random.randint(0,10), random.randint(0,10), random.randint(0,10)

torch.manual_seed(r1)
np.random.seed(r2)
random.seed(r3)

print(r1, ", ", r2, ", ", r3)

log_name_main = "history_" + str(r1) + "_" + str(r2) + "_" + str(r3) + ".log"
log_name_opt = "episodes_" + str(r1) + "_" + str(r2) + "_" + str(r3) + ".log"
log_file = LogFile(log_name_main, log_name_opt)




#==============================================================================================
#==============================================================================================
#=========================================SYMPHONY=============================================
#==============================================================================================
#==============================================================================================


#Rectified Huber Symmetric Error Loss Function via JIT Module
# nn.Module -> JIT C++ graph
class ReHSE(jit.ScriptModule):
    def __init__(self):
        super(ReHSE, self).__init__()

    @jit.script_method
    def forward(self, y1, y2, k:float):
        ae = torch.abs(y1-y2) + 1e-6
        ae = ae**k*torch.tanh(k*ae/2)
        return ae.mean()


#Rectified Huber Asymmetric Error Loss Function via JIT Module
# nn.Module -> JIT C++ graph
class ReHAE(jit.ScriptModule):
    def __init__(self):
        super(ReHAE, self).__init__()

    @jit.script_method
    def forward(self, y1, y2, k:float):
        e = (y1-y2) + 1e-6
        e = torch.abs(e)**k*torch.tanh(k*e/2)
        return e.mean()







#Silent Dropout function created with the help of ChatGPT
# nn.Module -> JIT C++ graph
class SilentDropout(jit.ScriptModule):
    def __init__(self, p=0.5):
        super(SilentDropout, self).__init__()
        self.p = p

    # It is not recommended to use JIT compilation decorator with online random generator as Symphony updates seeds each time
    # We did exception only for this module as it is used inside neural networks.
    @jit.script_method
    def forward(self, x):
        mask = (torch.rand_like(x) > self.p).float()
        return  mask * x + (1.0-mask) * x.detach()




#Linear followed by Silent Dropout
# nn.Module -> JIT C++ graph
class LinearSDropout(jit.ScriptModule):
    def __init__(self, f_in, f_out, p=0.5):
        super(LinearSDropout, self).__init__()
        self.ffw = nn.Linear(f_in, f_out)
        self.p = p

    @jit.script_method
    def forward(self, x):
        x = self.ffw(x)
        mask = (torch.rand_like(x) > self.p).float()
        return  mask * x + (1.0-mask) * x.detach()



#ReSine Activation Function
# nn.Module -> JIT C++ graph
class ReSine(jit.ScriptModule):
    def __init__(self, hidden_dim=256):
        super(ReSine, self).__init__()


    @jit.script_method
    def forward(self, x):
        x = 0.75*torch.sin(x/0.75)
        return F.leaky_relu(x, 0.15)




#Shared Feed Forward Module
# nn.Module -> JIT C++ graph
class FeedForward(jit.ScriptModule):
    def __init__(self, f_in, f_out):
        super(FeedForward, self).__init__()

        self.ffw = nn.Sequential(
            nn.Linear(f_in, 320),
            nn.LayerNorm(320),
            nn.Linear(320, 256),
            ReSine(256),
            LinearSDropout(256, 192, 0.5),
            LinearSDropout(192, f_out, 0.5)
        )

    @jit.script_method
    def forward(self, x):
        return self.ffw(x)



# nn.Module -> JIT C++ graph
class Actor(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super(Actor, self).__init__()

        
        self.inA = nn.Linear(state_dim, 320)
        self.inB = nn.Linear(state_dim, 256)
        self.inC = nn.Linear(state_dim, 192)

        
        self.ffw = nn.Sequential(
            FeedForward(768+state_dim, action_dim),
            nn.Tanh()
        )

        self.max_action = torch.mean(max_action).item()
        self.scale = 0.2*self.max_action
        self.lim = 3.0*self.scale

    
    @jit.script_method
    def forward(self, state):
        x = torch.cat([self.inA(state), self.inB(state), self.inC(state), state], dim=-1)
        return self.max_action*self.ffw(x)
    
    # Do not use any decorators with online random generators (Symphony updates seed each time)
    def soft(self, state):
        x = self.forward(state)
        x += (self.scale*torch.randn_like(x)).clamp(-self.lim, self.lim)
        return x.clamp(-self.max_action, self.max_action)




# nn.Module -> JIT C++ graph
class Critic(jit.ScriptModule):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()


        qA = FeedForward(state_dim+action_dim, 128)
        qB = FeedForward(state_dim+action_dim, 128)
        qC = FeedForward(state_dim+action_dim, 128)


        self.nets = nn.ModuleList([qA, qB, qC])


    @jit.script_method
    def forward(self, state, action):
        x = torch.cat([state, action], -1)
        return [net(x) for net in self.nets]
    
    # take means of 3 distributions and concatenate them
    @jit.script_method
    def cmin(self, state, action, k:float):
        xs = self.forward(state, action)
        xs = torch.cat([torch.mean(x, dim=-1, keepdim=True) for x in xs], dim=-1)
        xs = torch.sort(xs, dim=-1).values
        return (0.97*xs[:,0]+0.029*xs[:,1]+0.001*xs[:,2]).unsqueeze(1)


# Define the algorithm
class Symphony(object):
    def __init__(self, state_dim, action_dim, device, max_action=1.0, tau=0.005, capacity=300000, batch_lim = 768, fade_factor=7.0):

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, device, capacity, batch_lim, fade_factor)

        self.actor = Actor(state_dim, action_dim, max_action=max_action,).to(device)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = optim.RMSprop(self.critic.parameters(), lr=3.33e-4)
        self.actor_optimizer = optim.RMSprop(self.actor.parameters(), lr=2.33e-4)


        self.rehse = ReHSE()
        self.rehae = ReHAE()


        self.max_action = max_action
        self.tau = tau
        self.tau_ = 1.0 - tau
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.q_next_old_policy = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.weights =  torch.FloatTensor([math.exp(-0.9), math.exp(-0.8), math.exp(-0.7), math.exp(-0.6), math.exp(-0.5), math.exp(-0.4), math.exp(-0.3), math.exp(-0.2), math.exp(-0.1), math.exp(0)])
        self.weights = self.weights/self.weights.sum()
        #self.scaler = torch.amp.GradScaler('cuda')

        


    def select_action(self, state, mean=False):
        state = torch.FloatTensor(state).reshape(-1,self.state_dim).to(self.device)
        with torch.no_grad(): action = self.actor(state) if mean else self.actor.soft(state)
        return action.cpu().data.numpy().flatten()



    def train(self, tr_per_step=5):
        #---------------------decreases dependence on random seed: ---------------
        r1, r2, r3 = random.randint(0,2**32-1), random.randint(0,2**32-1), random.randint(0,2**32-1)
        torch.manual_seed(r1)
        np.random.seed(r2)
        random.seed(r3)

        for _ in range(tr_per_step): self.update()


        

    def q_next_prev(self, q_next_target):
        with torch.no_grad():
            # cut list of the last 5 elements [Qn-3, Qn-2, Qn-1]
            self.q_next_old_policy = self.q_next_old_policy[-10:]
            # multiply last 5 elements with exp weights and sum, creating exponential weighted average
            out = (torch.FloatTensor(self.q_next_old_policy)*self.weights).sum() # [0.06 Qn-5 + 0.1 Qn-4 + 0.16 Qn-3 + 0.21 Qn-2 + 0.43 Qn-1]
            # append new q next target value to the list
            self.q_next_old_policy.append(q_next_target.mean().detach())
            # return exp weighted average
            return out

    def update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample()
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        k = self.replay_buffer.ratio


        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau_*target_param.data + self.tau*param)

        with torch.amp.autocast('cuda', dtype=torch.float32):
            next_action = self.actor.soft(next_state)
            q_next_target = self.critic_target.cmin(next_state, next_action, k)
            actor_loss = -self.rehae(q_next_target,  self.q_next_prev(q_next_target), k)

            q = 0.01 * reward + (1-done) * 0.99 * q_next_target.detach()
            qs = self.critic(state, action)
            critic_loss = self.rehse(q, qs[0], k) + self.rehse(q, qs[1], k) + self.rehse(q, qs[2], k)

        
        actor_loss.backward()
        self.actor_optimizer.step()

        critic_loss.backward()
        self.critic_optimizer.step()
        """
        #Actor Update
        self.scaler.scale(actor_loss).backward()
        self.scaler.step(self.actor_optimizer)

        #Critic Update
        self.scaler.scale(critic_loss).backward()
        self.scaler.step(self.critic_optimizer)

        self.scaler.update()
        """
        
        
        






class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device, capacity, batch_lim, fade_factor=7.0):

        self.capacity, self.length, self.device = capacity, 0, device
        self.batch_size = min(max(64, self.length//300), batch_lim) #in order for sample to describe population
        self.random = np.random.default_rng()
        self.indices, self.indexes, self.probs = [], np.array([]), np.array([])
        self.fade_factor = fade_factor
        self.batch_lim = batch_lim
        self.ratio, self.cnt = 0.0, 0

        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.float32, device=device)


    #Normalized index conversion into fading probabilities
    def fade(self, norm_index):
        weights = np.tanh(self.fade_factor*norm_index**2.7) # linear / -> non-linear _/‾
        return weights/np.sum(weights) #probabilities



    def add(self, state, action, reward, next_state, done):
        idx = self.length-1
        self.cnt += 1
        if self.length<self.capacity:
            self.length += 1
            self.indices.append(self.length-1)
            self.indexes = np.array(self.indices)
            self.probs = self.fade(self.indexes/self.length) if self.length>1 else np.array([0.0])
            self.batch_size = min(max(64, self.length//300), self.batch_lim)
            
        if self.cnt<2.0*self.capacity: self.ratio = self.cnt/self.capacity
            

        
        self.states[idx,:] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[idx,:] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[idx,:] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.next_states[idx,:] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[idx,:] = torch.tensor([done], dtype=torch.float32, device=self.device)


        if self.length==self.capacity:
            self.states = torch.roll(self.states, shifts=-1, dims=0)
            self.actions = torch.roll(self.actions, shifts=-1, dims=0)
            self.rewards = torch.roll(self.rewards, shifts=-1, dims=0)
            self.next_states = torch.roll(self.next_states, shifts=-1, dims=0)
            self.dones = torch.roll(self.dones, shifts=-1, dims=0)



   
    # Do not use any decorators with random generators (Symphony updates seed each time)
    def sample(self):
        indices = self.random.choice(self.indexes, p=self.probs, size=self.batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )


    def __len__(self):
        return self.length
