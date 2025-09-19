import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
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


class Adam(optim.Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.5, 0.995)):
        defaults = dict(lr=lr, betas=betas)
        super().__init__(params, defaults)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.beta1_, self.beta2_ = 1-self.beta1, 1-self.beta2
        self.eps = 1e-8  # You can make this configurable if needed


    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)


                m = state['m']
                v = state['v']

                # Update biased first moment estimate
                m.mul_(self.beta1).add_(grad, alpha=self.beta1_)
                # Update biased second raw moment estimate
                v.mul_(self.beta2).addcmul_(grad, grad, value=self.beta2_)

                # Update parameters
                p.addcdiv_(m, v.sqrt() + self.eps, value=-self.lr)
                



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
        self.s = nn.Parameter(data=torch.sigmoid(2.0*torch.rand(hidden_dim)-1.0), requires_grad=False)

    @jit.script_method
    def forward(self, x):
        x = self.s*torch.sin(x/self.s)
        return x/(1+torch.exp(-1.5*x/self.s))

# Swaddling function
# nn.Module -> JIT C++ graph
class Omega(jit.ScriptModule):
    def __init__(self):
        super(Omega, self).__init__()
        self.min = 1e-3
        self.max = 1.0-self.min

    @jit.script_method
    def forward(self, x):
        x = (0.5 - x.clamp(self.min, self.max)**0.25)**2
        return 0.5*torch.log((0.25+x)/(0.25-x)).mean()
        


class FeedForward(jit.ScriptModule):
    def __init__(self, f_in, f_out):
        super(FeedForward, self).__init__()

        self.ffw = nn.Sequential(
            nn.Linear(f_in, 512),
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            ReSine(512),
            nn.Linear(512, f_out)
        )


    @jit.script_method
    def forward(self, x):
        return self.ffw(x)






# nn.Module -> JIT C++ graph
class ActorCritic(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super().__init__()


        self.action_dim = action_dim
        q_nodes = 128

        self.a = FeedForward(state_dim, 2*action_dim)
        self.a_max = nn.Parameter(data= torch.ones(action_dim), requires_grad=False)
        self.noise_std = math.sqrt(0.05)
        
        self.qA = FeedForward(state_dim+action_dim, q_nodes)
        self.qB = FeedForward(state_dim+action_dim, q_nodes)
        self.qnets = nn.ModuleList([self.qA, self.qB])

        q_dist = len(self.qnets)*q_nodes
        indexes = torch.arange(0, q_dist, 1)/q_dist
        weights = torch.tanh((math.pi*(1-indexes))**math.e)
        self.probs = nn.Parameter(data= weights/torch.sum(weights), requires_grad=False)

    #========= Actor Forward Pass =========
    
    @jit.script_method
    def actor(self, state, noise:bool=True):
        AS = torch.tanh(self.a(state)/2).reshape(-1, 2, self.action_dim)
        A, S =   AS[:, 0], (AS[:, 1]+1)/2
        A = self.a_max * torch.tanh(S * A +  noise * self.noise_std * torch.randn_like(A).clamp(-math.pi, math.pi)) 
        return A, S




    #========= Critic Forward Pass =========
    # take 3 distributions and concatenate them
    @jit.script_method
    def critic(self, state, action):
        x = torch.cat([state, action], -1)
        return torch.cat([qnet(x) for qnet in self.qnets], dim=-1)



    @jit.script_method
    def critic_soft(self, state, action):
        q = self.probs * self.critic(state, action).sort(dim=-1)[0]
        q =  q.sum(dim=-1, keepdim=True)
        return q, q.detach()



class Nets(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, max_action, device):
        super(Nets, self).__init__()
        self.online = ActorCritic(state_dim, action_dim, max_action=max_action).to(device)
        self.target = ActorCritic(state_dim, action_dim, max_action=max_action).to(device)
        self.target.load_state_dict(self.online.state_dict())

        self.rehse = ReHSE()
        self.rehae = ReHAE()
        self.omega = Omega()
        self.tau = 0.005
        self.tau_ = 1.0 - self.tau
        self.alpha = 0.5
        self.alpha_= 1.0 - self.alpha
        self.q_next_ema = torch.zeros(1, device=device)
        self.eta = torch.zeros(1, device=device)
        self.beta = 0.005

    @torch.no_grad()
    def tau_update(self):
        for target_param, param in zip(self.target.qnets.parameters(), self.online.qnets.parameters()):
            target_param.data.copy_(self.tau_*target_param.data + self.tau*param.data) 

    @jit.script_method
    def forward(self, state, action, reward, next_state, not_done_gamma):

        next_action, next_scale = self.online.actor(next_state)
        q_next_target, q_next_target_value = self.target.critic_soft(next_state, next_action)
        q_target =  self.eta * reward  + not_done_gamma * q_next_target_value
        q_pred = self.online.critic(state, action)

        q_next_ema = self.alpha * self.q_next_ema + self.alpha_ * q_next_target_value
        nets_loss = self.rehae(q_next_ema - q_next_target) + self.rehse(q_pred-q_target) + self.beta * self.omega(next_scale)
        self.q_next_ema = q_next_ema.mean()

        return nets_loss, next_scale.detach()



# Define the algorithm
class Symphony(object):
    def __init__(self, state_dim, action_dim, device, max_action=1.0, learning_rate=3e-4):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, device)
        self.nets = Nets(state_dim, action_dim, max_action, device)
        self.nets_optimizer = Adam(self.nets.online.parameters(), lr=learning_rate)



    def select_action(self, state, noise=True):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(-1,self.state_dim)
        with torch.no_grad(): action, scale = self.nets.online.actor(state, noise)
        return action.cpu().data.numpy().flatten(), scale.detach()

    """
    def select_action(self, state, noise=1.0, explore=False):
        with torch.no_grad(): action = self.nets.online.actor(state, noise)[0]
        return action
    """

    def norm_fill(self):
        if self.nets.eta==0: self.nets.eta = self.replay_buffer.norm_R(); self.replay_buffer.fill(100)


    def train(self):
        # decreases dependence on random seeds:
        r1, r2, r3 = random.randint(0,2**32-1), random.randint(0,2**32-1), random.randint(0,2**32-1)
        torch.manual_seed(r1)
        np.random.seed(r2)
        random.seed(r3)

        state, action, reward, next_state, not_done_gamma = self.replay_buffer.sample()
        self.nets_optimizer.zero_grad(set_to_none=True)
        self.nets.tau_update()

        nets_loss, scale = self.nets(state, action, reward, next_state, not_done_gamma)

        nets_loss.backward()
        self.nets_optimizer.step()

        return scale







class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device):

        self.capacity, self.length, self.idx, self.device = 1024000, 0, 0, device
        #self.batch_size = np.minimum(np.maximum(np.arange(1, self.capacity+1)//4000, 128), 512)
        #print("batch_size: ", self.batch_size)


        self.random = np.random.default_rng()
        self.indices = []
        self.indexes = np.array(self.indices)


        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=device)
        self.not_dones_gamma = torch.zeros((self.capacity, 1), dtype=torch.float32, device=device)




    def add(self, state, action, reward, next_state, done):


        if self.length<self.capacity:
            self.length += 1
            self.indices.append(self.length-1)
            self.indexes = np.array(self.indices)
            self.probs = self.fade(self.indexes/self.length)
        
        self.idx = self.length-1
        
        self.states[self.idx,:] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.idx,:] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.idx,:] = torch.tensor([reward], dtype=torch.float32, device=self.device)
        self.next_states[self.idx,:] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.not_dones_gamma[self.idx,:] = 0.99 * (1.0 - torch.tensor([done], dtype=torch.float32, device=self.device))


        if self.length>=self.capacity:
            self.states = torch.roll(self.states, shifts=-1, dims=0)
            self.actions = torch.roll(self.actions, shifts=-1, dims=0)
            self.rewards = torch.roll(self.rewards, shifts=-1, dims=0)
            self.next_states = torch.roll(self.next_states, shifts=-1, dims=0)
            self.not_dones_gamma = torch.roll(self.not_dones_gamma, shifts=-1, dims=0)



    # Do not use any decorators with random generators (Symphony updates seed each time)
    def sample(self):
        indices = self.random.choice(self.indexes, p=self.probs, size=256)

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
        weights = np.tanh((math.pi*norm_index)**math.e)
        return weights/np.sum(weights) #probabilities

    def norm_R(self):
        return 0.01/torch.mean(torch.abs(self.rewards[:self.length]))


    def fill(self, times):


        print("copying replay data, current length", self.length)

        def repeat(tensor, times):
            temp = tensor[:self.length]
            return temp.repeat(times, 1)

        self.temp_length = times*self.length
        
        self.states[:self.temp_length] = repeat(self.states, times)
        self.actions[:self.temp_length] = repeat(self.actions, times)
        self.rewards[:self.temp_length] = repeat(self.rewards, times)
        self.next_states[:self.temp_length] = repeat(self.next_states, times)
        self.not_dones_gamma[:self.temp_length] = repeat(self.not_dones_gamma, times)

        self.length = self.temp_length
        
        self.indices = np.arange(0, self.length, 1).tolist()
        self.indexes = np.array(self.indices)
        self.probs = self.fade(self.indexes/self.length)

        print("new replay buffer length: ", self.length)
