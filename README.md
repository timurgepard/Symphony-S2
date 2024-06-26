# Symphony-Classic

This repository was created to support the 2024 paper.
It is unification and simplification of Symphony-1.0, Symphony-2.0(2.1) and draft of Symphony-3.0.

I believe that industrial applications need not only a fast growth but an assymptotically stability in the learning process.
Some ideas was dropped and some proven their worth was solidified:

1. Temporal Advantage
2. Fading Replay Buffuer
3. Rectified Learnable Sine Wave Activation Function
4. Rectified Huber Symmetric and Asymmetric Loss Functions
5. Inplace Dropouts

Some modules were transmitted from Pytorch nn.Module to Pytorch jit.ScriptModule.

Architecture:
![image](https://github.com/timurgepard/Symphony-Classic/assets/13238473/459a9e9b-250f-467c-ad04-4d7e76d0f8c7)
