import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.jit as jit
import random



#==============================================================================================
#==============================================================================================
#=========================================SYMPHONY=============================================
#==============================================================================================
#==============================================================================================


class Adam(optim.Optimizer):
    def __init__(self, params, lr=3e-4, weight_decay=0.01, betas=((math.sqrt(5)-1)/2, 0.995)):
        defaults = dict(lr=lr, betas=betas)
        super().__init__(params, defaults)
        self.wd = weight_decay
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
                p.add_(m/(v.sqrt() + self.eps) + self.wd*p, alpha=-self.lr)
                





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
        k = 1/math.sqrt(hidden_dim)
        self.s = nn.Parameter(data=2.0*k*torch.rand(hidden_dim)-k, requires_grad=True)
 
    @jit.script_method
    def forward(self, x):
        s = torch.sigmoid(self.s)
        x = s*torch.sin(x/s)
        return x/(1+torch.exp(-1.5*x/s))

#GradientDropout
# nn.Module -> JIT C++ graph
class GradientDropout(jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.std = 1/math.e

    @jit.script_method
    def forward(self, x):
        p = torch.sigmoid(self.std * torch.randn_like(x).clamp(-math.e, math.e))
        mask = (torch.rand_like(x) > p).float()
        return mask * x + (1.0 - mask) * x.detach()



class Swaddling(jit.ScriptModule):
    def __init__(self):
        super(Swaddling, self).__init__()

    @jit.script_method
    def Omega(self, x):
        return torch.log((1+x)/(1-x))

    @jit.script_method
    def omega(self, x):
        return x*torch.log(x)


    @jit.script_method
    def forward(self, x, k):
        return (self.Omega(x**(1/k.detach())) + k * self.omega(x) + self.Omega(k**2)).mean()



class FeedForward(jit.ScriptModule):
    def __init__(self, f_in, h_dim, f_out):
        super(FeedForward, self).__init__()


        self.ffw = nn.Sequential(
            nn.Linear(f_in, h_dim),
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, h_dim),
            ReSine(h_dim),
            nn.Linear(h_dim, f_out),
            GradientDropout()
        )



    @jit.script_method
    def forward(self, x):
        return self.ffw(x)






# nn.Module -> JIT C++ graph
class ActorCritic(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, h_dim, max_action=1.0):
        super().__init__()


        self.action_dim = action_dim
        q_nodes = h_dim//4

        self.a = FeedForward(state_dim, h_dim, 3*action_dim)
        self.a_max = nn.Parameter(data= max_action, requires_grad=False)
        self.std = 1/math.e

        self.qA = FeedForward(state_dim+action_dim, h_dim, q_nodes)
        self.qB = FeedForward(state_dim+action_dim, h_dim, q_nodes)
        self.qC = FeedForward(state_dim+action_dim, h_dim, q_nodes)
        self.qnets = nn.ModuleList([self.qA, self.qB, self.qC])


        self.q_dist = q_nodes*len(self.qnets)
        indexes = torch.arange(0, self.q_dist, 1)/self.q_dist
        weights = torch.tanh((math.pi*(1-indexes))**math.pi) - 0.02*torch.exp(-(indexes/0.02)**2)
        self.probs = nn.Parameter(data= weights/torch.sum(weights), requires_grad=False)

        self.e = 1e-3
        self.e_ = 1-self.e

    #========= Actor Forward Pass =========
    
    @jit.script_method
    def actor(self, state, action:bool = True, noise:bool=True):
        ASB = torch.tanh(self.a(state)/2).reshape(-1, 3, self.action_dim)
        A, S, B =   ASB [:, 0], ASB[:, 1].abs(), ASB[:, 2].abs()
        N = self.std * torch.randn_like(A).clamp(-math.e, math.e)
        return self.a_max * torch.tanh(float(action) * S * A + float(noise) * N), S.clamp(self.e, self.e_), B.clamp(self.e, self.e_)

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
    def __init__(self, state_dim, action_dim, h_dim, max_action, device):
        super(Nets, self).__init__()

        self.online = ActorCritic(state_dim, action_dim, h_dim, max_action=max_action).to(device)
        self.target = ActorCritic(state_dim, action_dim, h_dim, max_action=max_action).to(device)
        self.target.load_state_dict(self.online.state_dict())

        self.rehse = ReHSE()
        self.rehae = ReHAE()
        self.sw = Swaddling()
        self.tau = 0.005
        self.tau_ = 1.0 - self.tau
        self.alpha = (math.sqrt(5)-1)/2
        self.alpha_= 1.0 - self.alpha
        self.q_next_ema = torch.zeros(1, device=device)


    @torch.no_grad()
    def tau_update(self):
        for target_param, param in zip(self.target.qnets.parameters(), self.online.qnets.parameters()):
            target_param.data.copy_(self.tau_*target_param.data + self.tau*param.data) 


    @jit.script_method
    def forward(self, state, action, reward, next_state, not_done_gamma):

        next_action, next_scale, next_beta = self.online.actor(next_state)
        q_next_target, q_next_target_value = self.target.critic_soft(next_state, next_action)
        q_target = reward + not_done_gamma * q_next_target_value
        q_pred = self.online.critic(state, action)

        q_next_ema = self.alpha * self.q_next_ema + self.alpha_ * q_next_target_value
        nets_loss = -self.rehae((q_next_target - q_next_ema)/q_next_ema.abs()) + self.rehse(q_pred-q_target) + self.sw(next_scale, next_beta) 
        self.q_next_ema = q_next_ema.mean()

        return nets_loss, next_scale.detach()



# Define the algorithm
class Symphony(object):
    def __init__(self, capacity, state_dim, action_dim, h_dim, device, max_action, learning_rate=3e-4):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device


        self.replay_buffer = ReplayBuffer(capacity, state_dim, action_dim, device)
        self.nets = Nets(state_dim, action_dim, h_dim, max_action, device)
        self.nets_optimizer = Adam(self.nets.online.parameters(), lr=learning_rate)
        self.batch_size = self.nets.online.q_dist


    def select_action(self, state, action = True, noise=True):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(-1,self.state_dim)
        with torch.no_grad(): action = self.nets.online.actor(state, action, noise)[0]
        return action.cpu().data.numpy().flatten()

    """
    def select_action(self, state,  action = True, noise=True):
        with torch.no_grad(): return self.nets.online.actor(state, action, noise)[0]
    """



    def train(self):

        torch.manual_seed(random.randint(0,2**32-1))

        state, action, reward, next_state, not_done_gamma = self.replay_buffer.sample(self.batch_size)
        self.nets_optimizer.zero_grad(set_to_none=True)
        
        nets_loss, scale = self.nets(state, action, reward, next_state, not_done_gamma)

        nets_loss.backward()
        self.nets_optimizer.step()
        self.nets.tau_update()

        return scale






class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim, device):

        self.capacity, self.length, self.device = capacity, 0, device

        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32, device=device)
        self.not_dones_gamma = torch.zeros((self.capacity, 1), dtype=torch.float32, device=device)

        self.norm = 1.0


    def add(self, state, action, reward, next_state, done):

        if self.length<self.capacity: self.length += 1

        idx = self.length-1

        self.states[idx,:] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[idx,:] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[idx,:] = torch.tensor([reward/self.norm], dtype=torch.float32, device=self.device)
        self.next_states[idx,:] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.not_dones_gamma[idx,:] = torch.tensor([0.99 * (1.0 - float(done))], dtype=torch.float32, device=self.device)

        if self.length>=self.capacity:
            shift = 2 if self.not_dones_gamma[0,:].item() == 0.0 else 1
            self.states = torch.roll(self.states, shifts=-shift, dims=0)
            self.actions = torch.roll(self.actions, shifts=-shift, dims=0)
            self.rewards = torch.roll(self.rewards, shifts=-shift, dims=0)
            self.next_states = torch.roll(self.next_states, shifts=-shift, dims=0)
            self.not_dones_gamma = torch.roll(self.not_dones_gamma, shifts=-shift, dims=0)



    def sample(self, batch_size):

        indices = torch.multinomial(self.probs, num_samples=batch_size, replacement=True)

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

    def norm_fill(self, times:int):


        print("copying replay data, current length", self.length)

        self.states = self.states[:self.length].repeat(times, 1)
        self.actions = self.actions[:self.length].repeat(times, 1)
        self.rewards = self.rewards[:self.length].repeat(times, 1)
        self.next_states = self.next_states[:self.length].repeat(times, 1)
        self.not_dones_gamma = self.not_dones_gamma[:self.length].repeat(times, 1)

        self.norm = torch.mean(torch.abs(self.rewards)).item()

        self.rewards /= self.norm

        self.length = times*self.length

        indexes = torch.arange(0, self.length, 1)/self.length
        weights = torch.tanh((math.pi*indexes)**math.pi) - 0.02*torch.exp(-((indexes-1)/0.02)**2)
        self.probs =  weights/torch.sum(weights)

        print("new replay buffer length: ", self.length)
