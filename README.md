# HIP network

## Environment
    python 3.6.9
    torch==1.7.0+cu101
    torch==1.7.0+cu101
    torch-scatter==2.0.5
    torchvision==0.8.1+cu101
    dgl-cu100==0.5.2

## Dataset
There are five datasets (from previous RE-NET and CyGNet): ICEWS18, ICEWS14, GDELT, WIKI, and YAGO. 
These datasets are for the *extrapolation* problem. 
- Times of test set should be larger than times of train and valid sets. 
- Times of valid set also should be larger than times of train set.

Each data folder has 'stat.txt', 'train.txt', 'valid.txt', 'test.txt'
- 'stat.txt': First value is the number of entities, and second value is the number of relations.
- 'train.txt', 'valid.txt', 'test.txt': First column is subject entities, second column is relations, and third column is object entities. The fourth column is time. The fifth column is ignored in our experiment.

## Run the main experiment
Train the model and test.
python train.py -d ICEWS18 --gpu 2 --dropout 0.5 --n-hidden 200 --lr 1e-3 --max-epochs 100 --batch-size 1024 --valid-every 10 --test-every 2
For other datasets, the only thing need to do is replacing the ICEWS18 with other names (YAGO, WIKI, ICEWS14, GDELT).

## Other parameters
You can find more details at train.py.

## Train time
- All models achieve their best results within 100 epochs.
- Since we cached the information for each time step, an epoch takes one to ten minutes on NVIDIA TITAN RTX Graphics Processing Units.

## solid seed
In train.py:
- seed = 999
- np.random.seed(seed)
- torch.manual_seed(seed)
- torch.cuda.manual_seed_all(seed)

## reference
```bib
@inproceedings{DBLP:conf/ijcai/HeZL0ZZ21,  
  author    = {Yongquan He and  
               Peng Zhang and  
               Luchen Liu and  
               Qi Liang and  
               Wenyuan Zhang and  
               Chuang Zhang},  
  title     = {HIP Network: Historical Information Passing Network for Extrapolation  
               Reasoning on Temporal Knowledge Graph},  
  booktitle = {Proceedings of the Thirtieth International Joint Conference on Artificial
               Intelligence, IJCAI 2021, Virtual Event / Montreal, Canada, 19-27
               August 2021},  
  pages     = {1915--1921},  
  year      = {2021},  
  url       = {https://doi.org/10.24963/ijcai.2021/264}  
}
```

