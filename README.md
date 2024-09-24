# <div align="center">Robo-MUTUAL:<br>Robotic Multimodal Task Specification via Unimodal Learning</div>

<div align="center">
    <img src="assets/web/intro.png" width="850">
    <p>
        The official implementation of "Robo-MUTUAL: Robotic Multimodal Task Specification via Unimodal Learning"
    </p>

[Paper](https://arxiv.org/) | [Project page](https://zh1hao.wang/Robo_MUTUAL/)

</div>

## <div align="center">Introduction</div>

We propose Robo-MUTUAL (Robotic Multimodal Task specifications via Unimodal Learning). This new framework enhances the Cross-modality Alignment capability of existing multimodal encoders by consuming a broader spectrum of robot-relevant data. Specifically, we retrain DecisionNCE, a state-of-the-art robotic multimodal encoder on an all-encompassing dataset, which not only consists of large-scale robot datasets including Open-X and DROID, but also incorporates a large human-activity dataset EPICK-KITCHEN. Combined, these datasets form the most comprehensive collection to date for robotic multimodal encoder pretraining. Building on the pretrained encoders, we explore two training-free methods to bridge the modality gap within the representation space, where we further introduce an effective cosine-similarity noise to facilitate efficient data augmentation in representation space to enable generalization to new task prompts. 

<div align='center'>
    <img src="assets/web/overview.png">
</div>

Tested across over 130 tasks and 4000 evaluations on both simulated LIBERO environments and real robot platforms, extensive experiments showcase a promising avenue towards enabling robots to understand multimodal instructions via unimodal training.

## <div align="center">Quick Start</div>
Code will come soon


## <div align="center">Citation</div>
Coming soon