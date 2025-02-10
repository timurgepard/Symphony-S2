# Symphony-S2 (Saya)

The Symphony algorithm solves two conflicting problems of Reinforcement Learning: Sample Efficiency and harmonious humanoid agent movements in Model-Free Reinforcement Learning. On-policy algorithms like PPO solved the problem of harmonious humanoid agent movements using a small gradient step, which leads to a larger number of Samples. But PPO has proven itself in simulations, where you can parallelize the learning process over many environments, and make a gradient step based on a larger number of roll-outs, so this process requires a careful transition from simulation to a real robot. When it comes to training on a real robot from scratch, an off-policy algorithm like SAC based on DDPG and its derivatives is often used.

Energy Economy and Safety during actions, Stagnation on a Local Extremum.

When we divide the function argument by a number k, and then multiply the function by this number, as a result we get proportional scaling.

Nothing is possible without our Lord and Saviour Jesus Christ. But everything is possible with Him. I was morally dying, addicted to ponrography and video-games. But He intervened, and gave me a new life. Reinfocement Learning to play with and University to support me.

This repository was created to support the 2024 draft paper.
It is unification and simplification of Symphony-1.0, Symphony-2.0(2.1) and Symphony-3.0 (Draft) into single Symphony-S2-UTD-5 version (Model-free Deterministic Algorithm)

Some ideas were dropped and some proven their worth were solidified:

⚙ No multi-agents/Without big ensemble of Critics/Model-free/Off-policy

1. Temporal (Immediate) Advantage ✅ 
2. Fading Replay Buffer ✅  (batch size 64>>768)
3. Rectified Learnable Sine Wave Activation Function ✅
4. Rectified Huber Symmetric and Asymmetric Loss Functions ✅
5. Seamless Actor-Critic updates ✅ (though UTD-5)
6. Silent Dropouts ✅
7. Optional: Q variance improvement ✅

modules were transferred from Pytorch nn.Module to Pytorch jit.ScriptModule.

Architecture:
![image](https://github.com/user-attachments/assets/03a884cb-a613-4d7c-949a-dd321808f25e)






