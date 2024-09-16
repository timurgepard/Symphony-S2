# Symphony-S

Nothing is possible without our Lord and Saviour Jesus Christ. But everything is possible with Him. I was morally dying, addicted to ponrography and video-games. But He intervened, and gave me a new life. Reinfocement Learning to play with and University to support me.

This repository was created to support the 2024 paper.
It is unification and simplification of Symphony-1.0, Symphony-2.0(2.1) and Symphony-3.0 (Draft) into single Symphony-Classic-UTD-5 version (Model-free Deterministic Algorithm)

Some ideas were dropped and some proven their worth were solidified:

⚙ No multi-agents/Without big ensemble of Critics/Model-free/Off-policy

1. Temporal (Immediate) Advantage ✅ (though UTD-5, batch size 128>>768)
2. Fading Replay Buffer ✅
3. Rectified Learnable Sine Wave Activation Function ✅
4. Rectified Huber Symmetric and Asymmetric Loss Functions ✅
5. Seamless Actor-Critic updates ✅
6. Inplace Dropouts ✅

1. <del>"movement is life" concept</del> ❌
2. <del>reduced objective to learn Bellman's sum of dumped reward's variance</del> ❌
3. <del>improve reward variance through immediate Advantage</del> ❌

Some modules were transferred from Pytorch nn.Module to Pytorch jit.ScriptModule.

Architecture:
![image](https://github.com/timurgepard/Symphony-Classic/assets/13238473/459a9e9b-250f-467c-ad04-4d7e76d0f8c7)


