# What Makes Graph Neural Networks Miscalibrated?

Source code of our NeurIPS 2022 paper "What Makes Graph Neural Networks Miscalibrated?" [[Paper](http://arxiv.org/abs/2210.06391)]

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

Run `./reproduce_train.sh` to first train GCN and GAT. The trained models will be saved in the `/model` directory.

### Calibration

Run `./reproduce_cal.sh` to reproduce the whole table in the main paper.

Run `./reproduce_cal_suppl.sh` to reproduce the results of additional baselines in the supplementary material.

Note that the numeric results may be slightly different due to the non-deterministic Ops on GPU.

## Detailed Usage - Example with GCN trained on Cora

We can first train GCN by running the following command:

```
PYTHONPATH=. python src/train.py --dataset Cora --model GCN --wdecay 5e-4
```

### Calibration with GATS

The train/val/test splits are saved in `/data/split`. In the calibration stage, GATS is trained on the validation set and validated on the training set for early stopping. For details of the experimental setup, please refer to Appendix A in our paper.

To calibrate the trained GCN with GATS run:

```
PYTHONPATH=. python src/calibration.py --dataset Cora --model GCN --wdecay 5e-4 --calibration GATS --config
```

or

```
PYTHONPATH=. python src/calibration.py --dataset Cora --model GCN --wdecay 5e-4 --calibration GATS --cal_wdecay 0.005 --heads 8 --bias 1
```

The `--config` argument will load the hyperparameters (`--cal_wdecay`, `--heads`, `--bias`) from the `.yaml` files stored in `/config`.

The GATS layer can be found in `/src/calibrator/attention_ts.py`. 

GATS assigns nodes with different scaling factor depending on the **distance to training nodes**. We computed this information offline and stored them in `/data/dist_to_train`. If you have a different splitting from ours, you can either pass `dist_to_train=None` to the GATS layer to generate the information online or run the following comand to generate it offline:

```
PYTHONPATH=. python -c 'from src.data.data_utils import *; generate_node_to_nearest_training(name="Cora", split_type="5_3f_85", bfs_depth=2)'
```

### Calibration with other Baselines

We implemented multiple baseline methods and compare them with GATS. The implemenation can be found in `/src/calibrator/calibrator.py`. To run the following baseline methods, simpliy set the argument `--calibration` to the following values:

| Baseline Methods  |`--calibration` | Hyperparameters|
| ------------- | ------------- | ------------- |
| [Temperature Scaling](https://arxiv.org/pdf/1706.04599.pdf) | `TS`  | None |
| [Vector Scaling](https://arxiv.org/pdf/1706.04599.pdf)  | `VS`  | None |
| [Ensemble Temperature Scaling](http://proceedings.mlr.press/v119/zhang20k/zhang20k.pdf)  | `ETS`  | None |
| [CaGCN](https://arxiv.org/pdf/2109.14285.pdf) |`CaGCN`| `--cal_wdecay`, `--cal_dropout_rate` |
| [Multi-class isotonic regression](http://proceedings.mlr.press/v119/zhang20k/zhang20k.pdf) |`IRM`| None |
| [Calibration using spline](https://arxiv.org/pdf/2006.12800.pdf) |`Spline`| None |
| [Dirichlet calibration](https://arxiv.org/pdf/1910.12656.pdf) |`Dirichlet`| `--cal_wdecay` |
| [Order invariant calibration](https://arxiv.org/pdf/2003.06820.pdf) |`OrderInvariant`| `--cal_wdecay` |

Similarly, one can run with the argument `--config` to use the tuned hyperparameters stored in `/config`.

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
@InProceedings{hsu2022what,
title={What Makes Graph Neural Networks Miscalibrated?},
author={Hans Hao-Hsun Hsu and Yuesong Shen and Christian Tomani and Daniel Cremers},
booktitle = {NeurIPS},
year = {2022}
}
```
