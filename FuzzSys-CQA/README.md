# Tutorial

This repository contains the implementation for our under-review paper, **An Efficient Fuzzy System for Complex Query Answering on Knowledge Graphs**.

As our approach is a novel reasoning method for complex query answering that can adapt to existing link prediction-based methods, we replace the fuzzy set operations with our Fuzzy System (FS) in the reasoning workflow of CQD and QTO, denoted as **CQD+FS** and **QTO+FS**.

Our code is based on the implementation of [CQD](https://github.com/pminervini/KGReasoning/) and [QTO](https://github.com/bys0318/QTO) available.



## 0. Environment

```bash
$ conda create -n fuzzsys python=3.9
$ conda activate fuzzsys
$ pip install -r requirements.txt
```



## 1. Download the datasets

The KG data (FB15k, FB15k-237, NELL995) mentioned in the BetaE paper and the Query2box paper can be downloaded [here](http://snap.stanford.edu/betae/KG_data.zip).



## 2. Obtain the pre-trained backbone models

### 2.1. Pre-trained CQD model

As mentioned above, the proposed FS is built on a pre-trained backbone model, i.e., CQD.

To download and decompress the pre-trained CQD models, execute the following commands:

```bash
$ mkdir models/
$ for i in "fb15k" "fb15k-237" "nell"; do for j in "betae" "q2b"; do wget -c http://data.neuralnoise.com/kgreasoning-cqd/$i-$j.tar.gz; done; done
$ for i in *.tar.gz; do tar xvfz $i; done
```

In case you need to re-train the CQD model from scratch, use the following command lines:

```bash
$ PYTHONPATH=. python3 main.py --do_train --do_valid --do_test --data_path data/FB15k-237-betae -n 1 -b 5000 -d 1000 -lr 0.1 --warm_up_steps 100000000 --max_steps 100000 --cpu_num 0 --geo cqd --valid_steps 500 --tasks 1p --print_on_screen --test_batch_size 1000 --optimizer adagrad --reg_weight 0.1 --log_steps 500 --cuda --use-qa-iterator
$ PYTHONPATH=. python3 main.py --do_train --do_valid --do_test --data_path data/FB15k-betae -n 1 -b 5000 -d 1000 -lr 0.1 --warm_up_steps 100000000 --max_steps 100000 --cpu_num 0 --geo cqd --valid_steps 500 --tasks 1p --print_on_screen --test_batch_size 1000 --optimizer adagrad --reg_weight 0.01 --log_steps 500 --cuda --use-qa-iterator
$ PYTHONPATH=. python3 main.py --do_train --do_valid --do_test --data_path data/NELL-betae -n 1 -b 5000 -d 1000 -lr 0.1 --warm_up_steps 100000000 --max_steps 100000 --cpu_num 0 --geo cqd --valid_steps 500 --tasks 1p --print_on_screen --test_batch_size 1000 --optimizer adagrad --reg_weight 0.1 --log_steps 500 --cuda --use-qa-iterator
```



### 2.2. Pre-trained QTO model

To train the QTO model, run the following commands under the `QTO/` folder.

```bash
$ cd kbc/src
$ bash ../scripts/preprocess.sh
$ bash ../scripts/train_[dataset].sh
$ cd ../..
$ bash scripts/build_neural_adj.sh
```



## 3. Hyperparameter estimation of FS

**1)** Estimate the hyperparameters of the membership functions in our FS by running the following command:

```bash
$ bash scripts/valid_memb_func.sh [Backbone] [Dataset] [Cuda ID] [Membship Function Type]
# Example: bash scripts/valid_memb_func.sh QTO NELL 0 bspline
# Backbone = QTO/CQD
# Dataset = NELL/FB15k/FB15k-237
# Membship Function Type = bspline/symbolic
```

The hyperparameter setting of the B-spline membership functions is saved in `params/[Dataset]_[Backbone]_params.yml` (e.g., `params/NELL_QTO_params.yml`), while that of the symbolic membership functions is saved in `params/[Dataset]_[Backbone]_symbolic_params.yml` (e.g., `params/NELL_QTO_symbolic_params.yml`).



**2)** Perform grid search on the validation set to determine the optimal value of hyperparameters of fuzzy rules:

```bash
$ bash scripts/grid_valid.sh [Backbone] [Dataset] [Cuda ID] [Membship Function Type] > log/grid_valid_[Backbone]_[Dataset]_[Membship Function Type].log
# Example: bash scripts/grid_valid.sh QTO NELL 0 bspline
```

The hyperparameter setting of the B-spline membership functions is saved in `params/[Dataset]_[Backbone]_rule_params.yml` (e.g., `params/NELL_QTO_rule_params.yml`), while that of the symbolic membership functions is saved in `params/[Dataset]_[Backbone]_rule_symbolic_params.yml` (e.g., `params/NELL_QTO_rule_symbolic_params.yml`).



## 3. Answer the complex queries

### 3.1. Evaluate QTO+FS

You can evaluate the performance of QTO+FS by running the following commands:

```bash
# Using B-spline membership functions
$ bash scripts/eval_fs_fb15k-QTO.sh > log/fb15k-qto.log
$ bash scripts/eval_fs_fb15k-237-QTO.sh > log/fb15k-237-qto.log
$ bash scripts/eval_fs_nell-QTO.sh > log/nell-qto.log

# Using symbolic membership functions
$ bash scripts/eval_fs_fb15k-QTO_symbolic.sh > log/fb15k-qto-symbolic.log
$ bash scripts/eval_fs_fb15k-237-QTO_symbolic.sh > log/fb15k-237-qto-symbolic.log
$ bash scripts/eval_fs_nell-QTO_symbolic.sh > log/nell-qto-symbolic.log
```



### 3.2. Evaluate CQD+FS

You can evaluate the performance of CQD+FS by running the following commands:

```bash
# Using B-spline membership functions
$ bash scripts/eval_fs_fb15k.sh > log/fb15k-cqd.log
$ bash scripts/eval_fs_fb15k-237.sh > log/fb15k-237-cqd.log
$ bash scripts/eval_fs_nell.sh > log/nell-cqd.log

# Using symbolic membership functions
$ bash scripts/eval_fs_fb15k_symbolic.sh > log/fb15k-cqd-symbolic.log
$ bash scripts/eval_fs_fb15k-237_symbolic.sh > log/fb15k-237-cqd-symbolic.log
$ bash scripts/eval_fs_nell_symbolic.sh > log/nell-cqd-symbolic.log
```



