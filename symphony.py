import torch
import torch.nn as nn
import math
import torch.jit as jit
from runtime import Adam, ReHSE, ReHAE, FourierSeries, FeedForward, FeedForwardTrinity, Swaddling

# global constants:
phi = (math.sqrt(5)+1)/2 #1.618...
phi_ = 1/phi #0.618...

#==============================================================================================
#==============================================================================================
#=========================================SYMPHONY=============================================
#==============================================================================================
#==============================================================================================

    


class FeatureExtractor(jit.ScriptModule):
    def __init__(self, state_dim, h_dim, z_dim, action_dim, drop):
        super(FeatureExtractor, self).__init__()



        self.ffw = FourierSeries(state_dim, h_dim, z_dim)
        self.norm1 = nn.RMSNorm(state_dim + z_dim)

        self.rew = FeedForward(state_dim + z_dim + action_dim, h_dim, z_dim, 1, drop)
        self.norm2 = nn.RMSNorm(state_dim + action_dim + 2*z_dim)


    @jit.script_method
    def z(self, s):
        return self.norm1(torch.cat([s, self.ffw(s)], dim=-1))


    @jit.script_method
    def za(self, s, a):
        za = torch.cat([a, self.z(s)], dim=-1)
        with torch.no_grad(): fr = self.rew.latent(za)
        return self.norm2(torch.cat([za, fr], dim=-1))



    @jit.script_method
    def transition(self, s, a):
        return self.rew(torch.cat([a, self.z(s)], dim=-1))





# jit.ScriptModule -> JIT C++ graph
class Actor(jit.ScriptModule):
    def __init__(self, state_dim, h_dim, z_dim, action_dim, drop=True):
        super().__init__()

        self.action_dim = action_dim
        self.Adam = FeedForward(state_dim, h_dim, z_dim, 3*action_dim, drop)


    @jit.script_method
    def forward(self, state):
        A, S, B = torch.tanh(self.Adam(state)/2).chunk(3, dim=-1)
        return A, S.abs(), B.abs()



class Critic(jit.ScriptModule):
    def __init__(self, state_action_dim: int, h_dim: int, z_dim: int, q_dist: int, drop: bool = True):
        super().__init__()
        q_nodes = q_dist // 3
        self.God = FeedForwardTrinity(state_action_dim, h_dim, z_dim, q_nodes, drop)

    @jit.script_method
    def forward(self, state_action):
        x3d = state_action.unsqueeze(0).expand(3, -1, -1)
        # (3, batch, q_nodes) -> (batch, 3, q_nodes) -> (batch, q_dist)
        return self.God(x3d).transpose(0, 1).reshape(state_action.size(0), -1)

# jit.ScriptModule -> JIT C++ graph
class ActorCritic(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, h_dim, alpha, q_dist, max_action, drop=True):
        super().__init__()

        fn = h_dim//12
        z_dim = q_dist


        self.fe = FeatureExtractor(state_dim, h_dim, fn, action_dim, drop=False)

        self.actor = Actor(state_dim + fn, h_dim, z_dim, action_dim, drop)
        self.register_buffer('a_max', torch.as_tensor(max_action, dtype=torch.float32))


        self.std = 1/math.e
        self.register_buffer('N', torch.empty((q_dist, action_dim)))



        self.critic = Critic(state_dim + action_dim + 2*fn, h_dim, z_dim, q_dist, drop)
        
        indexes = torch.arange(0, q_dist, 1)/q_dist
        weights = torch.exp(-(torch.abs(1-phi/2-indexes)/phi_)**(2*math.e))
        self.probs = nn.Parameter(data= weights/torch.sum(weights), requires_grad=False)

        self.alpha = alpha
        self._alpha = 1.0 - alpha
        self.register_buffer('q_ema', torch.zeros(1))


    @jit.script_method
    def actor_soft(self, state):
        A, S, B = self.actor(self.fe.z(state))
        return self.a_max * torch.tanh(S * A + self.N), S, B


    @jit.script_method
    def critic_soft(self, state, action):
        q =  self.critic(self.fe.za(state, action))
        q_soft = (self.probs * q.sort(dim=-1)[0]).sum(dim=-1, keepdim=True)
        q_soft_detached = q_soft.detach()
        self.q_ema.mul_(self.alpha).add_(q_soft_detached.mean(), alpha=self._alpha)
        return  q_soft, q_soft_detached, self.q_ema.clone()

    # helper functions ===========================================================

    @jit.script_method
    def actor_play(self, state, active:float = 1.0, test:float=0.0):
        A, S, B = self.actor(self.fe.z(state))
        self.N.normal_(0.0, 1.0).clamp_(-math.e, math.e).mul_(self.std)
        return self.a_max * torch.tanh(active * S * A + (1.0-test) * self.N[0:1])


    @jit.script_method
    def critic_direct(self, state, action):
        q_pred = self.critic(self.fe.za(state, action))
        r_pred = self.fe.transition(state, action)
        return q_pred, r_pred
        


    @jit.script_method
    def critic_info(self, state, action):
        q =  self.critic(self.fe.za(state, action))
        q_std = q.std(dim=-1, keepdim=True)/q.detach().pow(2).mean(dim=-1, keepdim=True).sqrt()
        return self.q_ema.clone(), q_std
    
    # ============================================================


class Nets(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, h_dim, alpha, tau, q_dist, batch_size, max_action, capacity, learning_rate, device):
        super(Nets, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device

        self.rehse = ReHSE()
        self.rehae = ReHAE()
        self.sw = Swaddling()
        self.tau = tau


        self.init(state_dim, action_dim, h_dim, alpha, q_dist, max_action, device)
        self.replay_buffer = ReplayBuffer(capacity, state_dim, action_dim, batch_size, device)
        self.optimizer = Adam(self.online.parameters(), lr=learning_rate, betas=(alpha, 1-tau))



    def init(self, state_dim, action_dim, h_dim, alpha, q_dist, max_action, device):

        self.online = ActorCritic(state_dim, action_dim, h_dim, alpha, q_dist, max_action, drop=True).to(device)
        self.target = ActorCritic(state_dim, action_dim, h_dim, alpha, q_dist, max_action, drop=False).to(device)
        self.target.load_state_dict(self.online.state_dict())
        for param in self.target.parameters(): param.requires_grad = False
                    
                

    @torch.no_grad()
    def tau_update(self):


        for target_param, param in zip(self.target.critic.parameters(), self.online.critic.parameters()):
            target_param.lerp_(param, self.tau)

        for target_param, param in zip(self.target.fe.parameters(), self.online.fe.parameters()):
            target_param.lerp_(param, self.tau)


    @jit.script_method
    def update(self):

        state, action, reward, next_state, not_done_gamma = self.replay_buffer.sample()

        next_action, next_scale, next_beta = self.online.actor_soft(next_state)
        q_next_target, q_next_target_value, q_next_ema = self.target.critic_soft(next_state, next_action)


        sw_and_beta_loss, sw_value = self.sw(next_scale, next_beta)
        q_target = reward + not_done_gamma * (q_next_target_value - sw_value)
        q_pred, r_pred = self.online.critic_direct(state, action)


        net_loss = self.rehse(r_pred - reward) + self.rehse(q_pred-q_target) - self.rehae((q_next_target - q_next_ema)/q_next_ema.abs()) + sw_and_beta_loss
        net_loss.backward()



    @jit.script_method
    def info(self):
        next_state = self.replay_buffer.sample()[3]
        with torch.no_grad(): next_action, next_scale, next_beta = self.online.actor_soft(next_state)
        with torch.no_grad(): q_ema, q_std = self.target.critic_info(next_state, next_action)
        return next_action.detach().mean(), next_scale.detach().mean(), next_beta.detach().mean(), q_ema.mean(), q_std.mean()




class Symphony(object):
    def __init__(self, capacity, state_dim, action_dim, h_dim, alpha, tau, q_dist, batch_size, max_action, state_high, state_low, learning_rate, device):
        super(Symphony, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        self.nets = Nets(state_dim, action_dim, h_dim, alpha, tau, q_dist, batch_size, max_action, capacity, learning_rate, device)

    
    def select_action(self, state, active = True, test=False):
        active, test = float(active), float(test)
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(-1,self.state_dim)
        with torch.no_grad(): action = self.nets.online.actor_play(state, active, test).detach().flatten()
        return action.cpu().numpy()


    """
    def select_action(self, state, active = True, noise=True):
        active, test = float(active), float(noise)
        with torch.no_grad(): return self.nets.online.actor_play(state, active, noise)[0]
    """



    def train(self):
        self.update()


    def update(self):
        self.nets.optimizer.zero_grad(set_to_none=True)
        self.nets.update()
        self.nets.optimizer.step()
        self.nets.tau_update()
        


    def info(self):
        action, scale, beta, q_ema, q_std = self.nets.info()
        return action.item(), scale.item(), beta.item(), q_ema.item(), q_std.item()


class ReplayBuffer(jit.ScriptModule):
    # TorchScript attribute annotations
    capacity: int
    batch_size: int
    action_dim: int
    state_dim: int

    def __init__(self, capacity: int, state_dim: int, action_dim: int, batch_size: int, device: torch.device):
        super(ReplayBuffer, self).__init__()

        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.action_dim = action_dim
        self.state_dim = state_dim



        self.register_buffer("ptr", torch.tensor(0, dtype=torch.long, device="cpu"))
        self.register_buffer("length", torch.tensor(0, dtype=torch.long, device="cpu"))
        # CPU terminal lifecycle tracker (1 MB RAM per 1M transitions):
        # 0 = normal/expired transition (can overwrite)
        # 1 = fresh terminal transition (protected for 1 cycle)
        self.terminal_lifecycle = torch.zeros(self.capacity, dtype=torch.uint8, device="cpu")

        # Normalization and Replay Buffers on GPU
        self.register_buffer("norm", torch.tensor(1.0, dtype=torch.float16, device=self.device))
        self.register_buffer("states", torch.zeros((self.capacity, self.state_dim), dtype=torch.float16, device=self.device))
        self.register_buffer("actions", torch.zeros((self.capacity, self.action_dim), dtype=torch.float16, device=self.device))
        self.register_buffer("rewards", torch.zeros((self.capacity, 1), dtype=torch.float16, device=self.device))
        self.register_buffer("next_states", torch.zeros((self.capacity, self.state_dim), dtype=torch.float16, device=self.device))
        self.register_buffer("not_dones_gamma", torch.zeros((self.capacity, 1), dtype=torch.float16, device=self.device))
        self.register_buffer("cdf", torch.zeros(self.capacity, dtype=torch.float32, device=self.device))

        indexes = torch.arange(0, self.capacity, 1, device=self.device, dtype=torch.float32) / self.capacity
        weights = torch.exp(-((torch.abs(indexes - phi / 2) / phi_) ** (2 * math.e)))

        # Precompute cumulative distribution function once:
        cdf = torch.cumsum(weights, dim=0)
        cdf = cdf / cdf[-1]
        cdf[-1] = 1.0  # Guard against float rounding to prevent out-of-bounds in searchsorted
        self.cdf.copy_(cdf)

    def init(self):
        self.states.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.next_states.zero_()
        self.not_dones_gamma.zero_()
        self.norm.fill_(1.0)

        # Reset CPU state
        self.ptr.zero_()
        self.length.zero_()
        self.terminal_lifecycle.zero_()

    def add(self, state, action, reward, next_state, done):
        if self.length.item() < self.capacity:
            self.length.add_(1)
        elif self.terminal_lifecycle[self.ptr] == 1:
            # Grant 1 extra cycle: demote to 0 so it gets overwritten NEXT loop, then advance
            self.terminal_lifecycle[self.ptr] = 0
            self.ptr.add_(1).remainder_(self.capacity)

        # Mark lifecycle state in host RAM
        self.terminal_lifecycle[self.ptr] = 1 if done else 0

        # Asynchronous device writes (no GPU-to-CPU sync stalls)
        self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float16, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float16, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor([reward], dtype=torch.float16, device=self.device) / self.norm
        self.next_states[self.ptr] = torch.as_tensor(next_state, dtype=torch.float16, device=self.device)
        self.not_dones_gamma[self.ptr] = torch.as_tensor([0.99 * (1.0 - float(done))], dtype=torch.float16, device=self.device)

        self.ptr.add_(1).remainder_(self.capacity)

    @jit.script_method
    def sample(self):
        #Parallel binary search on GPU - takes single-digit microseconds
        indices = torch.searchsorted(self.cdf, torch.rand(self.batch_size, device=self.cdf.device), right=False)
        # Circular offset
        indices.add_(self.ptr).remainder_(self.capacity)

        return (
            self.states[indices].float(),
            self.actions[indices].float(),
            self.rewards[indices].float(),
            self.next_states[indices].float(),
            self.not_dones_gamma[indices].float()
        )

    def __len__(self):
        return self.length.item()

    # ========================== Helper functions =======================================


    def _repeat(self, original_len: int, times: int):
        current_idx = original_len
        for _ in range(1, times):
            space_left = self.capacity - current_idx
            if space_left <= 0:
                break
            copy_size = min(original_len, space_left)

            self.states[current_idx : current_idx + copy_size].copy_(self.states[:copy_size])
            self.actions[current_idx : current_idx + copy_size].copy_(self.actions[:copy_size])
            self.rewards[current_idx : current_idx + copy_size].copy_(self.rewards[:copy_size])
            self.next_states[current_idx : current_idx + copy_size].copy_(self.next_states[:copy_size])
            self.not_dones_gamma[current_idx : current_idx + copy_size].copy_(self.not_dones_gamma[:copy_size])

            # Mirror repeats to CPU lifecycle buffer
            self.terminal_lifecycle[current_idx : current_idx + copy_size].copy_(self.terminal_lifecycle[:copy_size])

            current_idx += copy_size
        return current_idx

    def norm_fill(self, times: int):
        self._repeat(self.length.item(), times)

        mean_val = torch.mean(torch.abs(self.rewards))
        self.norm.fill_(mean_val)
        self.rewards.div_(self.norm)

        self.length.fill_(self.capacity)
        self.ptr.fill_(0)


