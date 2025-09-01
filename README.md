# Proxy-Validated Importance-Aware Federated Sample Selection with Meta Learning 

# Abstract

Federated data selection strategically chooses a group of high-quality samples to train a global model, and it is promising to optimize the convergence and resource overhead of federated learning (FL). However, existing studies either fail to account for the dynamic importance of training samples or rely on external unbiased validation datasets. These shortcomings can compromise FL model performance, potentially complicating their application in real-world scenarios. In this paper, we propose a novel proxy-validated importance-aware federated sample selection framework, termed FedSelect. It employs a novel meta learning approach with a proxy validation dataset to select the most positively important clients and their most important samples, thereby accelerating the training process and optimizing FL model performance. To eliminate the dependency on external unbiased data, we present a momentum-based meta-margin function to discover influential samples as the proxy validation dataset, providing an adaptive reward for sample selection. Furthermore, we also develop an online meta model update strategy to guarantee the efficiency of FedSelect. Comprehensive experiments on four benchmark datasets demonstrate that FedSelect is superior in both effectiveness and efficiency, while maintaining strong scalability across diverse scenarios.


# Run

```python
nohup bash run.sh &
```




# Citation
If you find our paper or this code useful for your research, please consider citing the corresponding paper:

```
@inproceedings{zhang2025proxy,
  title={Proxy-Validated Importance-Aware Federated Sample Selection with Meta Learning},
  author={Zhang, Yan and Miao, Xiaoye and Li, Bin and Wu, Yangyang and Shang, Yongheng},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={3855--3866},
  year={2025}
}
```
