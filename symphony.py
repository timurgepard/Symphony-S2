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
    def clean(self):
        with open(self.log_name_main, 'w') as file:
            file.write("")
        with open(self.log_name_opt, 'w') as file:
            file.write("")


numbers = extract_r1_r2_r3()
if numbers != None:
    # derive random numbers from history file
    r1, r2, r3 = numbers
else:
    # generate new random seeds
    r1, r2, r3 = random.randint(0,5), random.randint(0,5), random.randint(0,5)

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


class RMSprop(optim.Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.99):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid learning rate: {alpha}")
        
        defaults = dict(lr=lr, alpha=alpha)
        super().__init__(params, defaults)

        self.lr = lr
        self.alpha = alpha
        self.alpha_ = 1-alpha
    
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            
            for p in group['params']:
                if p.grad is None:
                    continue

                
                state = self.state[p]
                if len(state) == 0: state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                v = state['v']

                grad = p.grad

                # Update denominator
                v.mul_(self.alpha).addcmul_(grad, grad, value=(1-self.alpha))

                p.addcdiv_(grad, v.sqrt() + 1e-8, value=-self.lr)



#Rectified Huber Symmetric Error Loss Function via JIT Module
# nn.Module -> JIT C++ graph
class ReHSE(jit.ScriptModule):
    def __init__(self):
        super(ReHSE, self).__init__()

    @jit.script_method
    def forward(self, e):
        return (e * torch.tanh(e/2)).mean()


#Rectified Huber Asymmetric Error Loss Function via JIT Module
# nn.Module -> JIT C++ graph
class ReHAE(jit.ScriptModule):
    def __init__(self):
        super(ReHAE, self).__init__()

    @jit.script_method
    def forward(self, e):
        return (torch.abs(e) * torch.tanh(e/2)).mean()
    


#ReSine Activation Function
# nn.Module -> JIT C++ graph
class ReSine(jit.ScriptModule):
    def __init__(self, hidden_dim=256):
        super(ReSine, self).__init__()
        self.s = nn.Parameter(data=2.0*torch.rand(hidden_dim)-1.0, requires_grad=True)

    @jit.script_method
    def forward(self, x):
        s = torch.sigmoid(self.s)
        x = s*torch.sin(x/s)
        return x/(1+torch.exp(-1.5*x/s))



#Linear Layer followed by Silent Dropout
# nn.Module -> JIT C++ graph
class LinearSDropout(jit.ScriptModule):
    def __init__(self, f_in, f_out=False):
        super(LinearSDropout, self).__init__()
        self.ffw = nn.Linear(f_in, f_out)

    @jit.script_method
    def forward(self, x):
        x = self.ffw(x)
        #Silent Dropout function created with the help of ChatGPT
        # It is not recommended to use JIT compilation decorator with online random generator as Symphony updates seeds each time
        # We did exception only for this module as it is used inside neural networks.
        mask = (torch.rand_like(x) > 0.5).float()
        return  mask * x + (1.0-mask) * x.detach()


#Shared Feed Forward Module
# nn.Module -> JIT C++ graph
class FeedForward(jit.ScriptModule):
    def __init__(self, f_in, f_out):
        super(FeedForward, self).__init__()

        self.ffw = nn.Sequential(
            nn.Linear(f_in, 384),
            nn.LayerNorm(384),
            nn.Linear(384, 256),
            ReSine(256),
            LinearSDropout(256, 256),
            LinearSDropout(256, f_out)
        )

    @jit.script_method
    def forward(self, x):
        return self.ffw(x)

   

# nn.Module -> JIT C++ graph
class ActorCritic(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super().__init__()

        self.action_dim = action_dim

        self.a = FeedForward(state_dim, action_dim)
        self.a_max = nn.Parameter(data= max_action, requires_grad=False)
        self.noise_std = nn.Parameter(data= math.sqrt(0.05) * max_action, requires_grad=False)
        #self.eps = nn.Parameter(data= torch.zeros(1), requires_grad=False)
        

        self.qA = FeedForward(state_dim+action_dim, 128)
        self.qB = FeedForward(state_dim+action_dim, 128)
        self.qC = FeedForward(state_dim+action_dim, 128)

        self.qnets = nn.ModuleList([self.qA, self.qB, self.qC])


    #========= Actor Forward Pass =========
    @jit.script_method
    def actor(self, state):
        a = self.a(state)
        a_ = torch.tanh(a/2)
        a_out =  a_ + self.noise_std * torch.randn_like(a_).clamp(-math.e, math.e)
        #a_out =  torch.tanh(self.eps)**2 * a_ + self.noise_std * torch.randn_like(a_).clamp(-math.e, math.e)
        return self.a_max*torch.tanh(a_out/self.a_max),  (a*a_)/(2*math.e)


    #========= Critic Forward Pass =========
    # take 3 distributions and concatenate them
    @jit.script_method
    def critic(self, state, action):
        #self.eps += 0.0002/4
        x = torch.cat([state, action], -1)
        return torch.cat([qnet(x) for qnet in self.qnets], dim=-1)



    # take average in between min and mean
    @jit.script_method
    def critic_soft(self, state, action):
        q = self.critic(state, action)#.mean(dim=-1, keepdim=True)
        q = 0.618 * q.min(dim=-1, keepdim=True)[0] + 0.382 * q.mean(dim=-1, keepdim=True)
        return q, q.detach()




# Define the algorithm
class Symphony(object):
    def __init__(self, state_dim, action_dim, device, max_action=1.0):

        self.tau = 0.005

        self.tau_ = 1.0 - self.tau
        self.learning_rate = 3e-4
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device
        
        self.utd = 1
        self.k = 1/self.utd

        self.q_next_ema = 0.0


        self.replay_buffer = ReplayBuffer(state_dim, action_dim, device)

        self.nets = ActorCritic(state_dim, action_dim, max_action=max_action).to(device)
        self.nets_target = ActorCritic(state_dim, action_dim, max_action=max_action).to(device)
        self.nets_target.load_state_dict(self.nets.state_dict())


        self.nets_optimizer = RMSprop(self.nets.parameters(), lr=self.learning_rate)


        self.rehse = ReHSE()
        self.rehae = ReHAE()


    
    def select_action(self, state, explore=False):
        if explore: return self.max_action.numpy()*np.random.uniform(-0.5, 0.75, size=self.action_dim)
        state = torch.FloatTensor(state).reshape(-1,self.state_dim).to(self.device)
        with torch.no_grad(): action = self.nets.actor(state)[0]
        return action.cpu().data.numpy().flatten()

    """
    def select_action(self, state, mean=False, explore=False):
        if explore: return self.max_action*torch.FloatTensor(self.action_dim).uniform_(-0.5, 0.75).to(device)
        with torch.no_grad(): action = self.nets.actor(state)[0]
        return action
    """


    def train(self):
        # decreases dependence on random seeds:
        r1, r2, r3 = random.randint(0,2**32-1), random.randint(0,2**32-1), random.randint(0,2**32-1)
        torch.manual_seed(r1)
        np.random.seed(r2)
        random.seed(r3)

        for _ in range(self.utd): self.update()
        





    def update(self):


        state, action, reward, next_state, not_done_gamma = self.replay_buffer.sample()
        self.nets_optimizer.zero_grad(set_to_none=True)


        with torch.no_grad():
            for target_param, param in zip(self.nets_target.qnets.parameters(), self.nets.qnets.parameters()):
                target_param.data.copy_(self.tau_*target_param.data + self.tau*param.data)
    


        next_action, next_s2 = self.nets.actor(next_state)
        q_next_target, q_next_target_value = self.nets_target.critic_soft(next_state, next_action)
        q_target =  self.replay_buffer.r_scale * reward  + not_done_gamma * q_next_target_value
        q_pred = self.nets.critic(state, action)

        q_next_ema = 0.618 * self.q_next_ema + 0.382 * q_next_target_value
        nets_loss = -self.rehae(q_next_target - q_next_ema) + self.rehse(q_pred-q_target) + self.tau * next_s2.mean()

        (self.k * nets_loss).backward()
        self.nets_optimizer.step()
        self.q_next_ema =  q_next_ema.detach().mean()









class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device):

        self.capacity, self.length, self.device = 1024000, 0, device

        self.random = np.random.default_rng()

        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=device)
        self.not_dones_gamma = torch.zeros((self.capacity, 1), dtype=torch.float32, device=device)



    def add(self, state, action, reward, next_state, done):

        if self.length<self.capacity:
            self.length += 1
            self.batch_size = 64 + self.length//3200
            self.indexes = np.arange(0, self.length, 1)
            self.probs = self.fade(self.indexes/self.length)

        idx = self.length-1
        
        self.states[idx,:] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[idx,:] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[idx,:] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.next_states[idx,:] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.not_dones_gamma[idx,:] = 0.99 * (1.0 - torch.tensor([done], dtype=torch.float32, device=self.device))


        if self.length>=self.capacity:
            self.states = torch.roll(self.states, shifts=-1, dims=0)
            self.actions = torch.roll(self.actions, shifts=-1, dims=0)
            self.rewards = torch.roll(self.rewards, shifts=-1, dims=0)
            self.next_states = torch.roll(self.next_states, shifts=-1, dims=0)
            self.not_dones_gamma = torch.roll(self.not_dones_gamma, shifts=-1, dims=0)



   
    # Do not use any decorators with random generators (Symphony updates seed each time)
    def sample(self):
        indices = self.random.choice(self.indexes, p=self.probs, size=self.batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.not_dones_gamma[indices]
        )


    def __len__(self):
        return self.length
    

    #==============================================================
    #==============================================================
    #===========================HELPERS============================
    #==============================================================
    #==============================================================


    #Normalized index conversion into fading probabilities
    def fade(self, norm_index):
        weights = np.tanh((math.e*norm_index)**math.e) # linear - => non-linear ~
        return weights/np.sum(weights) #probabilities

    def norm_Q(self):
        Q = torch.sum(torch.abs(self.rewards[:self.length]))/self.length
        return Q.item()

    def fill(self, capacity):

        self.r_scale = 0.01/self.norm_Q()


        print("copying replay data, current length", self.length)

        def repeat(tensor, times):
            temp = tensor[:self.length]
            return temp.repeat(times, 1)

        times = capacity//self.length
        self.temp_length = times*self.length
        


        self.states[:self.temp_length] = repeat(self.states, times)
        self.actions[:self.temp_length] = repeat(self.actions, times)
        self.rewards[:self.temp_length] = repeat(self.rewards, times)
        self.next_states[:self.temp_length] = repeat(self.next_states, times)
        self.not_dones_gamma[:self.temp_length] = repeat(self.not_dones_gamma, times)
        
        self.left = capacity - self.temp_length

        if self.left>0:

            self.states[-self.left:] = self.states[:self.left]
            self.actions[-self.left:] = self.actions[:self.left]
            self.rewards[-self.left:] = self.rewards[:self.left]
            self.next_states[-self.left:] = self.next_states[:self.left]
            self.not_dones_gamma[-self.left:] = self.not_dones_gamma[:self.left]


        self.length = self.temp_length + self.left
        
        self.indexes = np.arange(0, self.length, 1)
        self.probs = self.fade(self.indexes/self.length)

        print("new replay buffer length: ", self.length)
