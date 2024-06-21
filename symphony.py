import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import math
import random
import torch.nn.functional as F



#==============================================================================================
#==============================================================================================
#=========================================SYMPHONY=============================================
#==============================================================================================
#==============================================================================================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# random seeds
r1, r2, r3 = random.randint(0,2**32-1), random.randint(0,2**32-1), random.randint(0,2**32-1)
#r1, r2, r3 = 3412506066, 3270540065, 862847336
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


#Rectified Huber Symmetric Error Loss Function
def ReHSE(error):
    ae = torch.abs(error).mean()
    return ae*torch.tanh(ae)

#Rectified Huber Asymmetric Error Loss Function
def ReHAE(error):
    e = error.mean()
    return torch.abs(e)*torch.tanh(e)

#This function created with the help of ChatGPT
class InplaceDropout(nn.Module):
    def __init__(self, p=0.5):
        super(InplaceDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) > self.p).float()
            return  mask * x + (1-mask) * x.detach()
        else:
            return x * (1 - self.p)

class ReSine(nn.Module):
    def __init__(self, hidden_dim=256):
        super(ReSine, self).__init__()
        self.scale = nn.Parameter(data=torch.randn(hidden_dim))
    def forward(self, x):
        scale = torch.sigmoid(1e-4*self.scale)
        x = scale*torch.sin(x/scale)
        return F.prelu(x, 0.1*scale)

  



class FeedForward(nn.Module):
    def __init__(self, f_in, hidden_dim=256, f_out=1):
        super(FeedForward, self).__init__()

        self.ffw = nn.Sequential(
            nn.Linear(f_in, 384),
            nn.LayerNorm(384),
            nn.Linear(384, 256),
            ReSine(256),
            nn.Linear(256, 128),
            nn.Linear(128, f_out),
        )

    def forward(self, x):
        return self.ffw(x)

# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dim=256, max_action=1.0):
        super(Actor, self).__init__()
        self.device = device
        
        self.inA = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            InplaceDropout(0.05)
        )
        self.inB = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            InplaceDropout(0.05)
        )
        self.inC = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            InplaceDropout(0.05)
        )
        

        self.ffw = nn.Sequential(
            FeedForward(3*hidden_dim, hidden_dim, action_dim),
            nn.Tanh()
        )

        self.max_action = torch.mean(max_action).item()
        self.scale = 0.05*self.max_action
        self.lim = 2.5*self.scale
    
    
    def forward(self, state):
        x = torch.cat([self.inA(state), self.inB(state), self.inC(state)], dim=-1)
        return self.max_action*self.ffw(x)
    
    def soft(self, state):
        x = self(state)
        x += self.scale*torch.randn_like(x).clamp(-self.lim, self.lim)
        return x.clamp(-self.max_action, self.max_action)
    



        
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()



        qA = nn.Sequential(
            FeedForward(state_dim+action_dim, hidden_dim, 128),
            InplaceDropout(0.5)
        )
        qB = nn.Sequential(
            FeedForward(state_dim+action_dim, hidden_dim, 128),
            InplaceDropout(0.5)
        )
        qC = nn.Sequential(
            FeedForward(state_dim+action_dim, hidden_dim, 128),
            InplaceDropout(0.5)
        )

        self.nets = nn.ModuleList([qA, qB, qC])


       
    def forward(self, state, action, united=False):
        x = torch.cat([state, action], -1)
        xs = [net(x) for net in self.nets]
        if not united: return xs
        xs = torch.cat([torch.mean(x, dim=-1, keepdim=True) for x in xs], dim=-1)
        return torch.min(xs, dim=-1, keepdim=True).values



# Define the actor-critic agent
class Symphony(object):
    def __init__(self, state_dim, action_dim, hidden_dim, device, max_action=1.0, tau=0.005, capacity=192000, fade_factor=10.0):

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, device, capacity, fade_factor)


        self.actor = Actor(state_dim, action_dim, device, hidden_dim, max_action=max_action).to(device)

        self.critic = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=5e-4)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=5e-4)


        self.max_action = max_action
        self.tau = tau
        self.tau_ = 1.0 - tau
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim



    def select_action(self, states):
        states = np.array(states)
        state = torch.FloatTensor(states).reshape(-1,self.state_dim).to(self.device)
        with torch.no_grad(): action = self.actor(state)
        return action.cpu().data.numpy().flatten()


    def train(self, tr_per_step=10):
        for i in range(tr_per_step):
            state, action, reward, next_state, done = self.replay_buffer.sample()
            self.update(state, action, reward, next_state, done)

            


    def update(self, state, action, reward, next_state, done):

        next_action = self.actor(next_state)

        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau_*target_param.data + self.tau*param)
            
            q_next_target = self.critic_target(next_state, next_action, united=True)
            q_value = reward +  (1-done) * 0.99 * q_next_target

       
        qs = self.critic(state, action, united=False)
        critic_loss = ReHSE(q_value - qs[0]) + ReHSE(q_value - qs[1]) + ReHSE(q_value - qs[2])

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        q_next_target_new = self.critic_target(next_state, next_action, united=True)
        actor_loss = -ReHAE(2.0*(q_next_target_new - q_next_target))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        

            

            

        
            



class ReplayBuffer:
    def __init__(self, state_dim, action_dim, device, capacity, fade_factor=7.0):
        self.capacity, self.length, self.device = capacity, 0, device
        self.batch_size = min(max(128, self.length//250), 256) #in order for sample to describe population
        self.indices = []

        self.states = torch.zeros((self.capacity, state_dim), dtype=torch.float32).to(device)
        self.actions = torch.zeros((self.capacity, action_dim), dtype=torch.float32).to(device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32).to(device)
        self.next_states = torch.zeros((self.capacity, state_dim), dtype=torch.float32).to(device)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.float32).to(device)


    def add(self, state, action, reward, next_state, done):
        if self.length<self.capacity:
            self.length += 1
            self.indices.append(self.length-1)


        idx = self.length-1
        

        self.states[idx,:] = torch.FloatTensor(state).to(self.device)
        self.actions[idx,:] = torch.FloatTensor(action).to(self.device)
        self.rewards[idx,:] = torch.FloatTensor([reward]).to(self.device)
        self.next_states[idx,:] = torch.FloatTensor(next_state).to(self.device)
        self.dones[idx,:] = torch.FloatTensor([done]).to(self.device)

        self.batch_size = min(max(64, self.length//150), 256)


        if self.length==self.capacity:
            self.states = torch.roll(self.states, shifts=-1, dims=0)
            self.actions = torch.roll(self.actions, shifts=-1, dims=0)
            self.rewards = torch.roll(self.rewards, shifts=-1, dims=0)
            self.next_states = torch.roll(self.next_states, shifts=-1, dims=0)
            self.dones = torch.roll(self.dones, shifts=-1, dims=0)

        



    def sample(self):
        indices = random.choices(self.indices, k=self.batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )




    def __len__(self):
        return self.length
