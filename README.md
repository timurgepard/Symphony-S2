# Symphony-S2 (Saya)

The Symphony algorithm solves two conflicting problems of Reinforcement Learning: Sample Efficiency and harmonious humanoid agent movements in Model-Free Reinforcement Learning. 

It is unification and simplification of Symphony-1.0, Symphony-2.0(2.1) and Symphony-3.0 (Draft) into single Symphony-S2-UTD-5 version (Model-free Deterministic Algorithm)

Some ideas were dropped and some proven their worth were solidified:

⚙ No multi-agents/Without big ensemble of Critics/Model-free/Off-policy

1. Temporal (Immediate) Advantage ✅ 
2. Fading Replay Buffer ✅  (batch size 64>>768)
3. Rectified Learnable Sine Wave Activation Function ✅
4. Rectified Huber Symmetric and Asymmetric Loss Functions ✅
5. Seamless Actor-Critic updates ✅ (though UTD-5)
6. Silent Dropouts ✅

PS: No matter how far you’ve wandered, Jesus is still waiting with open arms. It’s never too late to come home

Architecture:
![image](https://github.com/user-attachments/assets/03a884cb-a613-4d7c-949a-dd321808f25e)






