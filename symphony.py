
# nn.Module -> JIT C++ graph
class ActorCritic(jit.ScriptModule):
    def __init__(self, state_dim, action_dim, max_action=1.0):
        super().__init__()



        self.a = FeedForward(state_dim, 2*action_dim)
        self.a_max = nn.Parameter(data= max_action, requires_grad=False)
        self.x_max = nn.Parameter(data= max_action, requires_grad=True)


        self.qA = FeedForward(state_dim+action_dim, 128)
        self.qB = FeedForward(state_dim+action_dim, 128)
        self.qC = FeedForward(state_dim+action_dim, 128)

        self.qnets = nn.ModuleList([self.qA, self.qB, self.qC])

        self.max_action = max_action
        self.action_dim = action_dim




    #========= Actor Forward Pass =========

    @jit.script_method
    def actor(self, state):
        x = self.a(state).clamp(-3.0, 3.0).reshape(-1,2,self.action_dim)
        self.x_max = self.a_max*torch.sigmoid(2*x[:,0]/self.a_max)
        return self.x_max*torch.tanh(x[:,1]/self.x_max)



    #========= Critic Forward Pass =========
    # take 3 distributions and concatenate them
    @jit.script_method
    def critic(self, state, action):
        x = torch.cat([state, action], -1)
        return torch.cat([qnet(x) for qnet in self.qnets], dim=-1)


    # take average in between min and mean
    @jit.script_method
    def critic_soft(self, state, action):
        s2 = (0.5 * torch.log(1/self.x_max - 1) + 1e-6)**2
        x = self.critic(state, action)
        x = 0.5 * (x.min(dim=-1, keepdim=True)[0] + x.mean(dim=-1, keepdim=True)) * (1 - 0.01 * s2)
        return x, x.detach()
