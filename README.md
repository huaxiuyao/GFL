# GFL (Graph Few-shot Learning)

## About
Source code of the paper [Graph Few-shot Learning via Knowledge Transfer
](https://arxiv.org/abs/1910.03053). This code is built based upon the pytorch implementation of few-shot learning [few-shot](https://github.com/oscarknagg/few-shot).

## Data Format:
1. Meta-training Graphs: 
- put all graphs in ./data/graph/, each graph is named as graph_#id.txt. Each line represents one link and the format is 'node_id_1 node_id_2'
- put the correponding label in ./data/graph/, each graph is named as graph_#id_label.txt. The format is 'node_id label'
- put all features in ./data/feature.txt. The format is 'node_id, feature_1, ..., feature_n'

2. Meta-testing Graphs: 
- put all graphs in ./data/graph/, each graph is named as test_graph_#id.txt. Each line represents one link and the format is 'node_id_1 node_id_2'
- put the correponding label in ./data/graph/, each graph is named as test_graph_#id_label.txt. The format is 'node_id label'
- put all features in feature.txt, the format is 'node_id, feature_1, ..., feature_n'

## How to use
- **meta-training**: python main.py --datapath=./data/xxx/ --graphpath=./data/xxx/graph/ --in_f_d=xxx --nclasses=xxx --meta_lr=0.01 --update_batch_size=50 --logdir=../logs --hidden=32 --proto=graph --train=1 --inner_train_steps=5 --module_type=sigmoid --structure_dim=32 --hop_concat_type=attention --metatrain_iterations=xxx
- **meta-testing**: python main.py --datapath=./data/xxx/ --graphpath=./data/xxx/graph/ --in_f_d=xxx --nclasses=xxx --meta_lr=0.01 --update_batch_size=50 --logdir=../logs --hidden=32 --proto=graph --train=0 --inner_train_steps=5 --module_type=sigmoid --structure_dim=32 --hop_concat_type=attention --metatrain_iterations=xxx --test_load_epoch=xxx


If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{yao2020graph,
  title={Graph Few-shot Learning via Knowledge Transfer},
  author={Yao, Huaxiu and Zhang, Chuxu and Wei, Ying and Jiang, Meng and Wang, Suhang and Huang, Junzhou and Chawla, Nitesh V and Li, Zhenhui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2020} 
}
```