# KEPP
[CVPR 2024] KEPP: Why Not Use Your Textbook? Knowledge-Enhanced Procedure Planning of Instructional Videos

This repository gives the official implementation of KEPP:Why Not Use Your Textbook? Knowledge-Enhanced Procedure Planning of Instructional Videos (CVPR 2024)

In our project, we explore the capability of an agent to construct a logical sequence of action steps, thereby assembling a strategic procedural plan. This plan is crucial for navigating from an initial visual observation to a target visual outcome, as depicted in real-life instructional videos. Existing works have attained partial success by extensively leveraging various sources of information available in the datasets, such as heavy intermediate visual observations, procedural names, or natural language step-by-step instructions, for features or supervision signals. However, the task remains formidable due to the implicit causal constraints in the sequencing of steps and the variability inherent in multiple feasible plans. To tackle these intricacies that previous efforts have overlooked, we propose to enhance the agent's capabilities by infusing it with procedural knowledge. This knowledge, sourced from training procedure plans and structured as a directed weighted graph, equips the agent to better navigate the complexities of step sequencing and its potential variations. We coin our approach KEPP, a novel Knowledge-Enhanced Procedure Planning system, which harnesses a probabilistic procedural knowledge graph extracted from training data, effectively acting as a comprehensive textbook for the training domain. Experimental evaluations across three widely-used datasets under settings of varying complexity reveal that KEPP attains superior, state-of-the-art results while requiring only minimal supervision. Code and trained models will be made publicly available.

## Setup
In a conda env with cuda available, run:
```shell
pip install -r requirements.txt
```
## Data Preparation
### CrossTask
1. Download datasets&features
```shell
cd {root}/dataset/crosstask
bash download.sh
```
2. move your datasplit files and action one-hot coding file to `{root}/dataset/crosstask/crosstask_release/`
```shell
mv *.json crosstask_release
mv actions_one_hot.npy crosstask_release
```
### COIN
1. Download datasets&features
```shell
cd {root}/dataset/coin
bash download.sh
```
### NIV
1. Download datasets&features
```shell
cd {root}/dataset/NIV
bash download.sh
```
## Train Step model
## Generate paths from procedure knowlege graph
## Train plan model
## Inference