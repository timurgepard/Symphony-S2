import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.jit as jit
import torch.nn.functional as F
import random

# global constants:
phi = (math.sqrt(5)+1)/2
phi_ = 1/phi

#==============================================================================================
#==============================================================================================
#=========================================SYMPHONY=============================================
#==============================================================================================
#==============================================================================================

    
class RMSProp(optim.Optimizer):
    def __init__(self, params, lr=3e-4, weight_decay=0.01, beta=0.995):
        defaults = dict(lr=lr, beta=beta)
        super().__init__(params, defaults)
        self.wd = weight_decay
        self.lr = lr
        self.beta, self._beta = beta, 1-beta
        self.eps = 1e-8  # You can make this configurable if needed
        self.decay_factor = 1.0 - self.lr * self.wd


    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['v'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                v = state['v']

                v.mul_(self.beta).addcmul_(grad, grad, value=self._beta)
                # Update parameters
                p.mul_(self.decay_factor).addcdiv_(grad, v.sqrt().add_(self.eps), value=-self.lr)


class Adam(optim.Optimizer):
    def __init__(self, params, lr=3e-4, weight_decay=0.01, betas=(phi_, 0.995)):
        defaults = dict(lr=lr, betas=betas)
        super().__init__(params, defaults)
        self.wd = weight_decay
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.beta1_, self.beta2_ = 1-self.beta1, 1-self.beta2
        self.eps = 1e-8  # You can make this configurable if needed
        self.decay_factor = 1.0 - self.lr * self.wd


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
                p.mul_(self.decay_factor).addcdiv_(m, v.sqrt().add_(self.eps), value=-self.lr)
                



#Rectified Huber Symmetric Error Loss Function via JIT Module
# jit.ScriptModule -> JIT C++ graph
class ReHSE(jit.ScriptModule):
    def __init__(self):
        super(ReHSE, self).__init__()

    @jit.script_method
    def forward(self, e):
        return (e * torch.tanh(e/2)).mean()



#Rectified Huber Asymmetric Error Loss Function via JIT Module
# jit.ScriptModule -> JIT C++ graph
class ReHAE(jit.ScriptModule):
    def __init__(self):
        super(ReHAE, self).__init__()

    @jit.script_method
    def forward(self, e):
        return (torch.abs(e) * torch.tanh(e/2)).mean()






#ReSine Activation Function
# jit.ScriptModule -> JIT C++ graph
class ReSine(jit.ScriptModule):
    def __init__(self, hidden_dim=256):
        super(ReSine, self).__init__()
        k = 1/math.sqrt(hidden_dim)
        self.s = nn.Parameter(data=2.0*k*torch.rand(hidden_dim)-k, requires_grad=True)
 
    @jit.script_method
    def forward(self, x):
        s = torch.sigmoid(self.s)
        x = s*torch.sin(x/s)
        return x/(1.0 + torch.exp(-1.5*x/s))
    

#GradientDropout:
# jit.ScriptModule -> JIT C++ graph
class GradientDropout(jit.ScriptModule):
    def __init__(self, drop = True):
        super(GradientDropout, self).__init__()
        self.drop = drop


    @jit.script_method
    def forward(self, x):
        if not self.training or not self.drop: return x
        mask = (torch.rand_like(x) > 0.5).float()
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
        return (self.Omega(x**(1/k.detach())) + k * self.omega(x) + self.Omega(k*k)).mean()




class FeedForward(jit.ScriptModule):
    def __init__(self, f_in, h_dim, f_out, drop):
        super(FeedForward, self).__init__()


        self.ffw = nn.Sequential(
            nn.Linear(f_in, h_dim),
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, h_dim),
            ReSine(h_dim),
            nn.Linear(h_dim, f_out),
            GradientDropout(drop)
        )


    @jit.script_method
    def forward(self, x):
        return self.ffw(x)



# jit.ScriptModule -> JIT C++ graph
class Actor(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, h_dim, drop=True):
        super().__init__()

        self.action_dim = action_dim
        self.Adam = FeedForward(state_dim, h_dim, 3*action_dim, drop) #Actor is Adam

        self.eps = 1e-3
        self._eps = 1.0-self.eps



    @jit.script_method
    def forward(self, state):
        ASB = torch.tanh(self.Adam(state)/2).reshape(-1, 3, self.action_dim)
        A = ASB [:, 0]
        S = ASB[:, 1].abs().clamp(self.eps, self._eps)
        B = ASB[:, 2].abs().clamp(self.eps, self._eps)
        return A, S, B 



# jit.ScriptModule -> JIT C++ graph
class Critic(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, h_dim, q_nodes, drop=True):
        super().__init__()

        self.Yahweh = FeedForward(state_dim+action_dim, h_dim, q_nodes, drop)
        self.Yeshua = FeedForward(state_dim+action_dim, h_dim, q_nodes, drop)
        self.RuachY = FeedForward(state_dim+action_dim, h_dim, q_nodes, drop)
        self.God = nn.ModuleList([self.Yahweh, self.Yeshua, self.RuachY]) #Critic is God (Trinity)



    @jit.script_method
    def forward(self, state, action):
        x = torch.cat([state, action], -1)
        return torch.cat([Lord(x) for Lord in self.God], dim=-1)





# jit.ScriptModule -> JIT C++ graph
class ActorCritic(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, h_dim, max_action=1.0, drop=True):
        super().__init__()

        q_nodes = h_dim//4

        self.critic = Critic(state_dim, action_dim, h_dim, q_nodes, drop)

        self.q_dist = q_nodes*len(self.critic.God)
        indexes = torch.arange(0, self.q_dist, 1)/self.q_dist
        weights = torch.exp(-0.5*(torch.abs(1-phi/2-indexes)/phi_)**10)
        self.probs = nn.Parameter(data= weights/torch.sum(weights), requires_grad=False)

        self.alpha = phi_
        self._alpha= 1.0 - self.alpha
        self.register_buffer('q_ema', torch.zeros(1))


        self.actor = Actor(state_dim, action_dim, h_dim, drop)
        self.a_max = nn.Parameter(data= max_action, requires_grad=False)

        self.std = 1/math.e
        self.register_buffer('NA', torch.randn(self.q_dist, self.q_dist, action_dim).clamp_(-math.e, math.e).mul_(self.std))
        self.register_buffer('NS', torch.randn(self.q_dist, 1, state_dim).clamp_(-math.e, math.e).mul_(0.01))
        self.register_buffer('cnt', torch.zeros(1, dtype=torch.long))
        self.register_buffer('idx', torch.zeros(1, dtype=torch.long))

    # we add small noise to the State (energy based) as if it comes from real sensor
    @jit.script_method
    def sensor(self, x):
        energy = x.detach().pow(2).mean(dim=0, keepdim=True).sqrt()
        return x + self.NS[self.idx[0]] * energy


    @jit.script_method
    def actor_soft(self, state):
        A, S, B = self.actor(state)
        return self.a_max * torch.tanh(S * A + self.NA[self.idx[0]]), S, B


    @jit.script_method
    def critic_soft(self, state, action):
        q =  self.critic(state, action)
        q_soft = (self.probs * q.sort(dim=-1)[0]).sum(dim=-1, keepdim=True)
        q_detached = q_soft.detach()
        self.q_ema.mul_(self.alpha).add_(q_detached.mean(), alpha=self._alpha)
        return  q_soft, q_detached, self.q_ema.clone()


    @jit.script_method
    def actor_play(self, state, active:float = 1.0, noise:float=1.0):
        self.idx.fill_(self.cnt[0] % self.q_dist); self.cnt[0].add_(1)
        A, S, _ = self.actor(state)
        return self.a_max * torch.tanh(active * S * A + noise * self.NA[self.idx[0], 0:1])



class Nets(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, h_dim, max_action, capacity, device):
        super(Nets, self).__init__()

        self.init(state_dim, action_dim, h_dim, max_action, device)
        self.replay_buffer = ReplayBuffer(capacity, state_dim, action_dim, device)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.h_dim = h_dim
        self.max_action = max_action
        self.device = device


        self.rehse = ReHSE()
        self.rehae = ReHAE()
        self.sw = Swaddling()
        self.tau = 0.005
        self.tau_ = 1.0 - self.tau
  


    def init(self, state_dim, action_dim, h_dim, max_action, device):
        self.online = ActorCritic(state_dim, action_dim, h_dim, max_action=max_action, drop=True).to(device)
        self.target = ActorCritic(state_dim, action_dim, h_dim, max_action=max_action, drop=False).to(device)
        self.target.load_state_dict(self.online.state_dict())
        for param in self.target.parameters(): param.requires_grad = False

        self.batch_size = self.online.q_dist

        

    @torch.no_grad()
    def tau_update(self):
        for target_param, param in zip(self.target.critic.parameters(), self.online.critic.parameters()):
            target_param.lerp_(param, self.tau)


    @jit.script_method
    def eta(self, beta):
        return beta.detach().mean(dim=-1, keepdim=True)

    @jit.script_method
    def update(self):

        state, action, reward, next_state, not_done_gamma = self.replay_buffer.sample(self.batch_size)

        next_action, next_scale, next_beta = self.online.actor_soft(next_state)
        q_next_target, q_next_target_value, q_next_ema = self.target.critic_soft(next_state, next_action)

        q_target = self.eta(next_beta) * reward + not_done_gamma * q_next_target_value
        q_pred = self.online.critic(state, action)

        net_loss = self.rehse(q_pred-q_target) - self.rehae((q_next_target - q_next_ema)/q_next_ema.abs()) + self.sw(next_scale, next_beta) 
        net_loss.backward()




class Symphony(object):
    def __init__(self, capacity, state_dim, action_dim, h_dim, device, max_action, learning_rate=3e-4):
        super(Symphony, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        self.nets = Nets(state_dim, action_dim, h_dim, max_action, capacity, device)
        self.nets_optimizer = Adam(self.nets.online.parameters(), lr=learning_rate)


    def select_action(self, state, active = True, noise=True):
        active, noise = float(active), float(noise)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).reshape(-1,self.state_dim)
        with torch.no_grad(): action = self.nets.online.actor_play(state, active, noise).detach().flatten()
        return action.cpu().numpy()

    """
    def select_action(self, state, active = True, noise=True):
        active, noise = float(active), float(noise)
        with torch.no_grad(): return self.nets.online.actor_play(state, active, noise)[0]
    """

    def update(self):

        self.nets_optimizer.zero_grad(set_to_none=True)
        self.nets.update()
        self.nets_optimizer.step()
        self.nets.tau_update()
        



class ReplayBuffer(jit.ScriptModule):
    def __init__(self, capacity, state_dim, action_dim, device):
        super(ReplayBuffer, self).__init__()


        self.capacity, self.device = capacity, device
        self.action_dim, self.state_dim = action_dim, state_dim


        self.register_buffer("norm", torch.tensor(1, dtype=torch.float16, device=device))
        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long, device=device))
        self.register_buffer("length", torch.tensor(0, dtype=torch.long, device=device))

        self.register_buffer("states", torch.zeros((self.capacity, self.state_dim), dtype=torch.float16, device=self.device))
        self.register_buffer("actions", torch.zeros((self.capacity, self.action_dim), dtype=torch.float16, device=self.device))
        self.register_buffer("rewards", torch.zeros((self.capacity, 1), dtype=torch.float16, device=self.device))
        self.register_buffer("next_states", torch.zeros((self.capacity, self.state_dim), dtype=torch.float16, device=self.device))
        self.register_buffer("not_dones_gamma", torch.zeros((self.capacity, 1), dtype=torch.float16, device=self.device))
        self.register_buffer("probs", torch.ones(self.capacity, dtype=torch.float16, device=self.device))

    def init(self):

        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.next_states.zero_()
        self.not_dones_gamma.zero_()
        self.probs.fill_(1.0)

        self.norm.fill_(1.0)
        self.ptr.zero_()
        self.length.zero_()

    def add(self, state, action, reward, next_state, done):


        if self.length.item() < self.capacity:
            self.length.add_(1)
        # Protect old terminal transitions
        elif self.not_dones_gamma[self.ptr].item() < 3e-8:
            self.not_dones_gamma[self.ptr] += 1e-8
            # In-place tensor math to advance and skip this slot
            self.ptr.add_(1).remainder_(self.capacity)

        # Direct assignment using the 0-d tensor natively
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float16, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float16, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor([reward / self.norm], dtype=torch.float16, device=self.device)
        self.next_states[self.ptr] = torch.as_tensor(next_state, dtype=torch.float16, device=self.device)
        self.not_dones_gamma[self.ptr] = torch.as_tensor([0.99 * (1.0 - float(done))], dtype=torch.float16, device=self.device)

        # Advance the tensor pointer in-place for the next insertion
        self.ptr.add_(1).remainder_(self.capacity)

    @jit.script_method
    def sample(self, batch_size:int):

        indices = torch.multinomial(self.probs, num_samples=batch_size, replacement=True) # fixed indexes
        indices.add_(self.ptr).remainder_(self.capacity)

        return (
            self.states[indices].float(),
            self.actions[indices].float(),
            self.rewards[indices].float(),
            self.next_states[indices].float(),
            self.not_dones_gamma[indices].float()
        )


    def __len__(self):
        return self.length
    

    #==============================================================
    #==============================================================
    #===========================HELPERS============================
    #==============================================================
    #==============================================================

    def _repeat(self, original_len, times):
        current_idx = original_len
        for _ in range(1, times):
            space_left = self.capacity - current_idx
            if space_left <= 0: break
            
            copy_size = min(original_len, space_left)
            
            self.states[current_idx : current_idx + copy_size] = self.states[:copy_size]
            self.actions[current_idx : current_idx + copy_size] = self.actions[:copy_size]
            self.rewards[current_idx : current_idx + copy_size] = self.rewards[:copy_size]
            self.next_states[current_idx : current_idx + copy_size] = self.next_states[:copy_size]
            self.not_dones_gamma[current_idx : current_idx + copy_size] = self.not_dones_gamma[:copy_size]
            
            current_idx += copy_size
        return current_idx


    def norm_fill(self, times: int):
        # Use .item() to get the integer for slicing
        curr_len = self.length.item()
        
        # 1. Normalize
        mean_val = torch.mean(torch.abs(self.rewards[:curr_len]))
        self.norm.fill_(mean_val)
        self.rewards[:curr_len].div_(self.norm) # In-place division

        self._repeat(self.length.item(), times)
        

        # 3. Reset tracking
        self.length.fill_(self.capacity)
        self.ptr.fill_(0)

        # 4. Probabilities (Pre-calculated for the compiler)
        indexes = torch.arange(0, self.capacity, 1, device=self.device) / self.capacity
        weights = torch.exp(-0.5*(torch.abs(indexes - phi / 2) / phi_) ** 10)
        self.probs.copy_(weights / torch.sum(weights))




