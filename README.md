# What Makes Graph Neural Networks Miscalibrated?

Source code of out NeurIPS 2022 paper "What Makes Graph Neural Networks Miscalibrated?" [Paper]

### Factors that influence GNN calibration
1. General under-confident tendency
2. Diversity of nodewise predictive distributions
3. Distance to training nodes
4. Relative confidence level
5. Neighborhood similarity

### Graph Attention Temperature Scaling (GATS)
![alt text](https://github.com/hans66hsu/GATS/blob/main/figure/GATS_new.png?raw=true)
*Illustration of GATS for a graph with four nodes*

## Requirements

* python >= 3.6
* matplotlib >= 3.2.2
* numpy >= 1.19.5
* pathlib2 2.3.5
* scipy 1.5.1
* seaborn 0.11.1
* sklearn 0.0
* torch 1.7.0+cu101
* torch-geometric 2.0.1

Install the dependencies from requirements file. PyTorch and PyTorch-Geometric are installed with Cuda 10.1.

```
pip install -r requirements.txt
```

## Fast Usage

The implementation consists of two stages. We first train GNNs using the training script `src/train.py` and then calibrate the model using post-hoc calibration methods with the script `src/calibration.py`. We provided the following bash files to reporduce our results in the paper.

### Train

Run `./reproduce_train.sh` to first train GCN and GAT.

### Calibration

Run `./reproduce_cal.sh` to reproduce the whole table in the main paper.

Run `./reproduce_cal_suppl.sh` to reproduce the results of additional baselines in the supplementary material.

Note that the numeric results may be slightly different due to the non-deterministic Ops on GPU.

## Detailed Usage - Example with GCN trained on Cora

We can first train GCN by running the following command

```
PYTHONPATH=. python src/train.py --dataset Cora --model GCN --wdecay 5e-4
```

### Calibration with GATS



### Calibration with other Baselines

We implemented the muliple basline methods and compare them with GATS:

| Baseline Methods  |`--calibration` | Hyperparameters|
| ------------- | ------------- | ------------- |
| [Temperature Scaling](https://arxiv.org/pdf/1706.04599.pdf) | `TS`  | None |
| [Vector Scaling](https://arxiv.org/pdf/1706.04599.pdf)  | `VS`  | None |
| [Ensemble Temperature Scaling](http://proceedings.mlr.press/v119/zhang20k/zhang20k.pdf)  | `ETS`  | None |
| [CaGCN](https://arxiv.org/pdf/2109.14285.pdf) |`CaGCN`| `--cal_wdecay`, `--cal_dropout_rate` |
| Multi-class isotonic regression |`IRM`| None |
| Calibration using spline |`Spline`| None |
| Dirichlet calibration |`Dirichlet`| `--cal_wdecay` |
| Order invariant calibration |`OrderInvariant`| `--cal_wdecay` |

Note that one can simpliy run with the argument `--config` to use the tuned hyperparameters stored in `/config`.

### Argument details

Both scripts `src/train.py` and `src/calibration.py` share the same arguments.

```
train.py and calibration.py share the same arguments

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Random Seed
  --dataset {Cora,Citeseer,Pubmed,Computers,Photo,CS,Physics,CoraFull}
  --split_type SPLIT_TYPE
                        k-fold and test split
  --model {GCN,GAT}
  --verbose             Show training and validation loss
  --wdecay WDECAY       Weight decay for training phase
  --dropout_rate DROPOUT_RATE
                        Dropout rate. 1.0 denotes drop all the weights to zero
  --calibration CALIBRATION
                        Post-hoc calibrators
  --cal_wdecay CAL_WDECAY
                        Weight decay for calibration phase
  --cal_dropout_rate CAL_DROPOUT_RATE
                        Dropout rate for calibrators
  --folds FOLDS         K folds cross-validation for calibration
  --ece-bins ECE_BINS   number of bins for ece
  --ece-scheme {equal_width,uniform_mass}
                        binning scheme for ece
  --ece-norm ECE_NORM   norm for ece
  --save_prediction
  --config

optional GATS arguments:
  --heads HEADS         Number of heads for GATS. Hyperparameter set:
                        {1,2,4,8,16}
  --bias BIAS           Bias initialization for GATS

```

## Citation

Please consider citing our work if you find our work useful for your research:

```
@InProceedings{gnn_miscal_2020,
title={What Makes Graph Neural Networks Miscalibrated?},
author={Hans Hao-Hsun Hsu, Yuesong Shen, Christian Tomani, Daniel Cremers},
booktitle = {NeurIPS},
year = {2022}
}
```
