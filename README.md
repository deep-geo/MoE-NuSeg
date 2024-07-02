# MoE-NuSeg (Mixture of Experts Nuclei Seg)

## Introduction

 Accurate nuclei segmentation is crucial for
extracting quantitative information from histology images
to support disease diagnosis and treatment decisions.
However, precise segmentation remains challenging due to
the presence of clustered nuclei, varied morphologies, and the need to capture global spatial correlations. While state-of-the-art Transformer-based models have made progress
by employing specialized tri-decoder architectures that
separately segment nuclei, edges, and clustered edges,
their complexity and long inference times hinder integration into clinical workflows. To address these  challenges, we proposed MoE-NuSeg


## Our Contributions

1. **MoE-NuSeg**:we introduce
MoE-NuSeg, a novel domain-knowledge-driven MoE nuclei
segmentation network based on the Swin Transformer.

2. **attention-based gating
network**:  we propose an innovative attention-based gating
network in MoE that dynamically modulates the contributions
of the three specialized experts based on input data, enhancing
overall segmentation performance and robustness
3. **Novel Edge Prompt**: we
design a novel two-stage training scheme to optimize the
collaboration between the experts and the gating network.
In the initial stage, the three specialized experts are trained
independently to excel in their respective tasks, consolidating
the roles previously spread across multiple decoders. In the
subsequent stage, these experts are trained in conjunction
with the gating network, fostering enhanced collaboration and
co-evolution among the components. This two-stage training
strategy ensures that the experts can leverage their specialized
knowledge while adaptively collaborating based on the input
data, ultimately leading to superior segmentation performance.


## Results

Extensive experiments validate our model's effectiveness on test images. Our approach is designed for easy integration into existing clinical workflows.

## Architecture

![image](https://github.com/deep-geo/MoE-NuSeg/assets/112611011/ab194456-bacd-4941-aa5b-9a0dd3281568)




## Citation

If you find this work useful for your research, please consider citing:

