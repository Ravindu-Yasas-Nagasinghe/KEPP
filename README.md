# KEPP: Why Not Use Your Textbook? Knowledge-Enhanced Procedure Planning of Instructional Videos-CVPR 2024

<!-- [![paper](https://img.shields.io/badge/arXiv-Paper-42FF33)](https://arxiv.org/abs/2306.08271) 
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://bimsarapathiraja.github.io/mccl-project-page/)  -->

This repository gives the official implementation of KEPP:Why Not Use Your Textbook? Knowledge-Enhanced Procedure Planning of Instructional Videos (CVPR 2024)

In our project, we explore the capability of an agent to construct a logical sequence of action steps, thereby assembling a strategic procedural plan. This plan is crucial for navigating from an initial visual observation to a target visual outcome, as depicted in real-life instructional videos. Existing works have attained partial success by extensively leveraging various sources of information available in the datasets, such as heavy intermediate visual observations, procedural names, or natural language step-by-step instructions, for features or supervision signals. However, the task remains formidable due to the implicit causal constraints in the sequencing of steps and the variability inherent in multiple feasible plans. To tackle these intricacies that previous efforts have overlooked, we propose to enhance the agent's capabilities by infusing it with procedural knowledge. This knowledge, sourced from training procedure plans and structured as a directed weighted graph, equips the agent to better navigate the complexities of step sequencing and its potential variations. We coin our approach KEPP, a novel Knowledge-Enhanced Procedure Planning system, which harnesses a probabilistic procedural knowledge graph extracted from training data, effectively acting as a comprehensive textbook for the training domain. Experimental evaluations across three widely-used datasets under settings of varying complexity reveal that KEPP attains superior, state-of-the-art results while requiring only minimal supervision. The main architexture of our model is as follows.

<!-- ## Citation
If you find our work useful. Please consider giving a star :star: and a citation.
```bibtex
@InProceedings{Pathiraja_2023_CVPR,
        author    = {Pathiraja, Bimsara and Gunawardhana, Malitha and Khan, Muhammad Haris},
        title     = {Multiclass Confidence and Localization Calibration for Object Detection},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2023},
        pages     = {19734-19743}
``` -->

![kepp (2)_page-0001](https://github.com/Ravindu-Yasas-Nagasinghe/KEPP/assets/56619402/ef7a12f5-bf7d-461d-a03b-4630ccd23751)

### Contents  
1) [Setup](#Setup) 
2) [Data Preparation](#Data-Preparation)
3) [Train Step model](#Train-Step-model)
4) [Generate paths from procedure knowlege graph](#Generate-paths-from-procedure-knowlege-graph)
5) [Inference](#Inference)
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
1. First generate the training and testing dataset json files. You can modify the dataset, train steps, horizon(prediction length), json files savepath etc. in `args.py`. Set the `--json_path_train`, and `--json_path_val` in `args.py` as the dataset json file paths.
```shell
cd {root}/step
python loading_data.py 
```
Dimensions for different datasets are listed below:

| Dataset	| observation_dim |	action_dim	| class_dim |
|----| ----| ----| ----| 
| CrossTask	| 1536(how) 9600(base) | 105	| 18 | 
| COIN 	| 1536	| 778	| 180 | 
| NIV	| 1536	| 48	| 5 | 

2. Train the step model
```shell
python main_distributed.py --multiprocessing-distributed --num_thread_reader=8 --cudnn_benchmark=1 --pin_memory --checkpoint_dir=whl --resume --batch_size=256 --batch_size_val=256 --evaluate
```
The trained models will be saved in {root}/step/save_max.

3. Generate first and last action predictions for train and test dataset.
* Modify the checkpoint path(L329) as the evaluated model(in save_max) in inference.py.
* Modify the `--json_path_val` , `--steps_path` ,  and `--step_model_output` arguments in `args.py` to generate step predicted dataset json file paths for train and test datasets seperately. Run following command for train and test datasets seperately by modifying as afore mentioned.

```shell
python inference.py --multiprocessing-distributed --num_thread_reader=8 --cudnn_benchmark=1 --pin_memory --checkpoint_dir=whl --resume --batch_size=256 --batch_size_val=256 --evaluate > output.txt
```
## Generate paths from procedure knowlege graph
1. Train the graph for the relavent dataset (Not compulsory)
```shell
cd {root}/PKG
python graph_creation.py
Select mode "train_out_n" 
```
Trained graphs for CrossTask, COIN, NIV datasets are available on `cd {root}/PKG/graphs`. Change (L13) `graph_save_path` of `graph_creation.py` to load procedure knowledge graphs trained on different datasets.

2. Obtain PKG conditions for train and test datasets.
* Modify line 540 of `graph_creation.py` as the output of step model (`--step_model_output`).
* Modify line 568 of `graph_creation.py` to set the output path for the generated procedure knowlwdge graph conditioned train and test dataset json files. 
* run the following for both train and test dataset files generated from the step model by modifying `graph_creation.py` file as afore mentioned.
```shell
python graph_creation.py
Select mode "validate"
```
## Train plan model
1. Modify the `json_path_train` and `json_path_val` arguments of `args.py` in plan model as the outputs generated from procedure knowlwdge graph for train and test data respectively.

Modify the parameter `--num_seq_PKG` in `args.py` to match the generated amount of PKG conditions. (Modify `--num_seq_LLM` to the same number as well if LLM conditions are not used seperately.)
```shell
cd {root}/plan
python main_distributed.py --multiprocessing-distributed --num_thread_reader=8 --cudnn_benchmark=1 --pin_memory --checkpoint_dir=whl --resume --batch_size=256 --batch_size_val=256 --evaluate
```
## Inference

For Metrics
​Modify the max checkpoint path(L339) as the evaluated model in inference.py and run:
```shell
python inference.py --multiprocessing-distributed --num_thread_reader=8 --cudnn_benchmark=1 --pin_memory --checkpoint_dir=whl --resume --batch_size=256 --batch_size_val=256 --evaluate > output.txt
```
Results of given checkpoints:

| dataset | SR | mAcc |	MIoU |
| ---- | -- | -- | -- |
| Crosstask_T=4 |	21.02	| 56.08	| 64.15 |
| COIN_T=4 | 15.63 | 39.53 |	53.27 |
| NIV_T=4 |	22.71 |	41.59 |	91.49 |

Here we present the qualitative examples of our proposed method. Intermediate steps are padded in the step model because it only predicts the start and end actions.

<p align="center">
  <img src="https://github.com/Ravindu-Yasas-Nagasinghe/KEPP/assets/56619402/686ee6ab-3256-43bd-a9e7-d6ef1bec8044" width="400" />
  <img src="https://github.com/Ravindu-Yasas-Nagasinghe/KEPP/assets/56619402/28492179-d19d-4589-90c7-d1b773629750" width="390" /> 
</p>


Checkpoint links will be uploaded soon


### Citation
```shell
@misc{nagasinghe2024use,
      title={Why Not Use Your Textbook? Knowledge-Enhanced Procedure Planning of Instructional Videos}, 
      author={Kumaranage Ravindu Yasas Nagasinghe and Honglu Zhou and Malitha Gunawardhana and Martin Renqiang Min and Daniel Harari and Muhammad Haris Khan},
      year={2024},
      eprint={2403.02782},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
### Contact
In case of any query, create issue or contact ravindunagasinghe1998@gmail.com

### Acknowledgement
This codebase is built on <a href="https://github.com/MCG-NJU/PDPP?tab=readme-ov-file">PDPP</a>
