## A Unified Transferable Model in PyTorch

Pytorch implementation of a unified model based transformer to explore transferability across databases and across tasks in ML-enhanced DBMS.

### Requirment

PyTorch 1.8.1

Python 3.8.5

### Environment configuration

If your cuda version is **10.2**, you can run the following code to set up a conda environment:

```
conda env create -f join-opt-env.yml
source activate join-opt-env
```

### Running experiments

There are two basic tasks, the first is multi-task learning on query optimization to explore transferability across tasks, and the second is meta learning on join order selection to explore transferability across databases. 

#### Multi-task learning on query optimization

Run `python run_multi_task.py --help` to see list of tunable knobs.

Setting `--train_data_path` and `--test_data_path` is neccessary, It represents the path of training set and test set respectively.

`--mode` knob has two options. `--mode="multi-task"` to run our Transformer model, and `--cost`,`--card` and `--join` to run corresponding task.`--mode="tree-lstm"` to run Tree-LSTM model.

```python
# Train and test three tasks at the same time
python run_multi_task.py --mode="mulit-task" --card=1 --cost==1 --join=1 --train_data_path="/train_data" --test_data_path="/test_data"

# Train and test tree-lstm model
python run_multi_task.py --mode="tree-lstm" --train_data_path="/train_data" --test_data_path="/test_data"
```

#### Meta learning on join order selection

Run `python run_meta_learning.py --help` to see list of tunable knobs.

Setting `--train_db` , `--test_db` and `--db_path` is neccessary, It represents id of database for training, id of database for testing and the directory of databases.

```python
# Train on db[0,1,2,3], test on db[4]
python run_meta_learning.py --train_db 0 1 2 3 --test_db 4 --db_path="/data/meta_learner"
```

### Reference

If you find this repository useful in your work, please cite [our paper](https://arxiv.org/pdf/2105.02418.pdf).

```
@article{wu2021unified,
  title={A Unified Transferable Model for ML-Enhanced DBMS},
  author={Wu, Ziniu and Yang, Peilun and Yu, Pei and Zhu, Rong and Han, Yuxing and Li, Yaliang and Lian, Defu and Zeng, Kai and Zhou, Jingren},
  journal={arXiv preprint arXiv:2105.02418},
  year={2021}
}
```

