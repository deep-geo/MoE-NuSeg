# MoE-NuSeg (Mixture of Experts Nuclei Segementation)

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

Extensive experiments validate our model's effectiveness on test images. From left to right, the layout displays the following sequence:

	1.	Raw Input Image
	2.	Ground Truth (GT) Mask
	3.	Predicted Mask
	4.	Ground Truth Edge Mask
	5.	Predicted Edge Mask
	6.	Ground Truth Cluster Edge Mask
	7.	Predicted Cluster Edge Mask

<img width="927" alt="image" src="https://github.com/user-attachments/assets/a4480fff-8093-4ae2-a6f4-96baab946d2d">

This layout provides a side-by-side comparison of the input image, ground truth, and predictions, allowing for a clear assessment of the model’s segmentation accuracy.


## Architecture

![image](https://github.com/deep-geo/MoE-NuSeg/assets/112611011/ab194456-bacd-4941-aa5b-9a0dd3281568)




## How to Run the Code

MoE-NuSeg training is organized into two stages. Each stage is run using a shell script that calls the respective Python script for that stage. Follow the steps below to train and generate the final model weights.

### Stage 1: Initial Training


1.	Run Stage 1 by executing the following command:
    
	sh main.sh

•	This will call train_step1.py to initiate the first stage of training.
•	The script generates a weight file upon completion, which will be used in the second stage of training.

2.	Update the Weight File Path:
•	Once Stage 1 completes, locate the generated weight file.
•	Open train_step2.py and update the path to the generated weight file from Stage 1 so it can be loaded in Stage 2.

### Stage 2: Fine-Tuning

1.	Run Stage 2 by executing the following command:

	sh main_p2.sh

•	This command calls train_step2.py, which continues training using the weight file from Stage 1.

By following these steps, you’ll complete both stages of training, generating the final MoE-NuSeg model weights for evaluation and deployment.


## Comparison of Validation Loss Curves for Stage 1 and Stage 2 Training Across Six Loss Components

<img width="965" alt="image" src="https://github.com/user-attachments/assets/0ded5f3b-3d63-4e6f-8289-fae5f90f782c">

The plot shows the validation loss curves for each of the six components (Cross-Entropy and Dice Loss for each of the three experts) over the course of training in both Stage 1 and Stage 2, providing a detailed view of performance trends and convergence at each stage. Training incorporates early stopping, halting after 50 epochs without improvement in nuclei segmentation accuracy, indicating that the lowest loss is reached before the full 50 epochs.




## Citation

If you find this work useful for your research, please consider citing:

Wu, X., Shen, Y., Zhao, Q., Kang, Y., & Zhang, W. (2025). MoE-NuSeg: Enhancing nuclei segmentation in histology images with a two-stage Mixture of Experts network. Alexandria Engineering Journal, 110, 557-566. 

You can access the full paper through ScienceDirect at the following link: MoE-NuSeg on ScienceDirect:
https://www.sciencedirect.com/science/article/pii/S1110016824011669

