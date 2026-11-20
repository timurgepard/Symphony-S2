
import torch
import torch.nn as nn
import torch.jit as jit
import torch.optim as optim
import math

# ========================== ENSEMBLED VERSIONS TO RUN 3 Critics Simultaneusly ------------------------------------

#ReSine Activation Function
# jit.ScriptModule -> JIT C++ graph
class EnReSine(jit.ScriptModule):
    def __init__(self, hidden_dim: int =256, n: int =3):
        super(EnReSine, self).__init__()

        k = 1.0 / math.sqrt(hidden_dim)
        # Parameter shape: (N, 1, 2 * hidden_dim) for automatic broadcasting over batch
        initial_weight = 2.0 * k * torch.rand(n, 1, 2 * hidden_dim) - k
        self.sb_ = nn.Parameter(data=initial_weight, requires_grad=True)

    @jit.script_method
    def forward(self, x: torch.Tensor):
        # x is expected to have shape (N, batch_size, hidden_dim)
        s, b_ = torch.sigmoid(self.sb_).chunk(2, dim=-1)
        x = s * torch.sin(x / s)
        return x * torch.sigmoid(x / (b_ * s))



class EnLinear(jit.ScriptModule):
    def __init__(self, in_features: int, out_features: int, n: int = 3):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n, 1, out_features))

        bound = 1.0 / math.sqrt(in_features)
        for i in range(n):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            nn.init.uniform_(self.bias[i], -bound, bound)

    @jit.script_method
    def forward(self, x):
        # x: (n, batch, in_features)
        return torch.baddbmm(self.bias, x, self.weight)


class EnLayerNorm(jit.ScriptModule):
    def __init__(
        self,
        normalized_shape: int,
        n: int = 3,
        eps: float = 1e-5,
    ):
        super(EnLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.n = n
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(n, 1, normalized_shape))
        self.bias = nn.Parameter(torch.zeros(n, 1, normalized_shape))

    @jit.script_method
    def forward(self, x: torch.Tensor):

        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias



class HolyTrinity(jit.ScriptModule):
    def __init__(self, f_in: int, h_dim: int, f_out: int, drop: bool):
        super().__init__()
        
        self.ffw = nn.Sequential(
            EnLinear(f_in, h_dim),
            EnLayerNorm(h_dim),
            EnLinear(h_dim, h_dim),
            EnReSine(h_dim),
            EnLinear(h_dim, h_dim),
            GradientDropout(drop),
            EnLinear(h_dim, f_out)
        )

    @jit.script_method
    def forward(self, x):
        return self.ffw(x)


# ===========================================================================================

class Adam(optim.Optimizer):
    def __init__(self, params, lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps))
        self.lr, self.eps = lr, eps
        self.b1, self.b2 = betas
        self.b1_, self.b2_ = 1.0 - self.b1, 1.0 - self.b2
        self.decay = 1.0 - lr * weight_decay
        self.init = False

    def _setup(self):
        self.p = [p for g in self.param_groups for p in g['params'] if p.requires_grad]
        self.m = [torch.zeros_like(p) for p in self.p]
        self.v = [torch.zeros_like(p) for p in self.p]
        # Scalar per tensor matching your exact shape logic
        self.e = [torch.tensor(1.0 - self.eps, device=p.device, dtype=p.dtype) for p in self.p]
        self.init = True

    @torch.no_grad()
    def step(self):
        if not self.init: 
            self._setup()

        # Filter out parameters whose grad is None
        p_active, grads, m_active, v_active, e_active  = [], [], [], [], []


        for p, m, v, e in zip(self.p, self.m, self.v, self.e):
            if p.grad is None: continue
            p_active.append(p)
            grads.append(p.grad)
            m_active.append(m)
            v_active.append(v)
            e_active.append(e)

        # If all gradients are None, exit early
        if not p_active: return
            

        # Vectorized operations only over active parameters
        torch._foreach_mul_(m_active, self.b1)
        torch._foreach_add_(m_active, grads, alpha=self.b1_)
        torch._foreach_mul_(v_active, self.b2)
        torch._foreach_addcmul_(v_active, grads, grads, value=self.b2_)
        torch._foreach_mul_(e_active, self.b2)
        torch._foreach_add_(e_active, self.eps * self.b2_)

        denom = torch._foreach_add(torch._foreach_sqrt(v_active), e_active)
        torch._foreach_mul_(p_active, self.decay)
        torch._foreach_addcdiv_(p_active, m_active, denom, value=-self.lr)


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
        self.sb_ = nn.Parameter(data=2.0*k*torch.rand(2*hidden_dim)-k, requires_grad=True)


    @jit.script_method
    def forward(self, x):
        s, b_ = torch.sigmoid(self.sb_).chunk(2, dim=-1)
        x = s*torch.sin(x/s)
        return x * torch.sigmoid(x/(b_*s))




#GradientDropout:
# jit.ScriptModule -> JIT C++ graph
class GradientDropout(jit.ScriptModule):
    def __init__(self, drop = True):
        super(GradientDropout, self).__init__()
        self.drop = drop


    @jit.script_method
    def forward(self, x):
        if not self.training or not self.drop: return x
        p = torch.sigmoid(torch.randn_like(x))
        mask = (torch.rand_like(x) > p).float()
        return mask * x + (1.0 - mask) * x.detach()



class Swaddling(jit.ScriptModule):
    def __init__(self):
        super(Swaddling, self).__init__()

        self.eps = 1e-8
        self._eps = 1.0-self.eps


    @jit.script_method
    def Omega(self, x):
        return torch.log((1+x)/(1-x))

    @jit.script_method
    def omega(self, x):
        return x*torch.log(x)


    @jit.script_method
    def forward(self, x, k):
        x, k = x.clamp(self.eps, self._eps), k.clamp(self.eps, self._eps)
        sw = (self.Omega(x**(1/k.detach())) + k * self.omega(x)).mean(dim=-1, keepdim=True)
        k2 = self.Omega(k*k).mean(dim=-1, keepdim=True)
        return sw.mean() + k2.mean(), sw.detach()




class FourierSeries(jit.ScriptModule):
    def __init__(self, f_in, h_dim, f_out):
        super(FourierSeries, self).__init__()

        self.ffw = nn.Sequential(
            nn.Linear(f_in, h_dim),
            ReSine(h_dim),
            nn.Linear(h_dim, f_out)
        )

    @jit.script_method
    def forward(self, x):
        return self.ffw(x)




class FeedForward(jit.ScriptModule):
    def __init__(self, f_in, h_dim, f_out, drop):
        super().__init__()


        self.ffw = nn.Sequential(
            nn.Linear(f_in, h_dim),
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, h_dim),
            ReSine(h_dim),
            nn.Linear(h_dim, h_dim),
            GradientDropout(drop),
            nn.Linear(h_dim, f_out)
        )


    @jit.script_method
    def forward(self, x):
        return self.ffw(x)

