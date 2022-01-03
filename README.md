# Official Pytorch Implementation of "AdaGCN: Adaboosting Graph Convolutional Networks into Deep Models" (ICLR 2021)

Please refer to [openreview](https://openreview.net/forum?id=QkRbdiiEjM) (ICLR 2021) to look into the details of our paper.

![Alt text](https://github.com/datake/AdaGCN/raw/main/AdaGCN.png)


## Enviromment

```
python3.6  
cuda11.0  
torch1.7.1
```

## Run the code （Datasets: citeseer, cora_ml, pubmed and ms_academic）



#### Baseline: GCN

```
python main.py --trainsize 20 --dataset citeseer --niter 5 --nseed 20 --model GCN --dropout 0.5 --reg 5e-4
```

#### Baseline: APPNP or PPNP

```
python main.py --trainsize 20 --dataset citeseer --niter 5 --nseed 20 --model APPNP --dropout 0.5 --early 1 --patience 300 --max 500 --reg 5e-3
```

#### AdaGCN on Four datasets:

```
python main.py --trainsize 20 --dataset citeseer --niter 5 --nseed 20 --model AdaGCN --layers 15 --hid_AdaGCN 5000 --dropout 0.0 --weight_decay 1e-3 --early 1 --patience 300 --max 500 --reg 5e-3   
python main.py --trainsize 20 --dataset cora_ml --niter 5 --nseed 20 --model AdaGCN --layers 12 --hid_AdaGCN 5000 --dropout 0.0 --weight_decay 1e-4 --early 1 --patience 300 --max 500 --reg 5e-3   
python main.py --trainsize 20 --dataset pubmed --niter 5 --nseed 20 --model AdaGCN --layers 20 --hid_AdaGCN 5000 --dropout 0.2 --weight_decay 1e-4 --early 1 --patience 300 --max 500 --reg 5e-3   
python main.py --trainsize 20 --dataset ms_academic --niter 5 --nseed 20 --model AdaGCN --layers 5 --hid_AdaGCN 3000 --dropout 0.2 --weight_decay 1e-4 --early 1 --patience 300 --max 500 --reg 5e-3
```

**Results:** 

| Dataset  | Average Accuracy | Std |
| ------------- | ------------- | ------------- |
| Citeseer  | 76.68  | 0.20  |
| Cora-ML  | 85.97  | 0.20  |
| PubMed  | 79.95  | 0.21  |
| MS Academic  | 93.17  | 0.07  |

## Acknowledgement

Our code is directly adapted from PPNP paper **Predict then Propagate: Graph Neural Networks meet Personalized PageRank** (ICLR 2019) github: https://github.com/klicperajo/ppnp.

## Contact

Please refer to ajksunke@pku.edu.cn in case you have any questions. 

## Cite
Please cite our paper if you use the model or this code in your own work:
```
@inproceedings{sun2020adagcn,
  title={AdaGCN: Adaboosting Graph Convolutional Networks into Deep Models},
  author={Sun, Ke and Zhu, Zhanxing and Lin, Zhouchen},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```
