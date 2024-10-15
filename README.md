# FedCSS
This is the code repository of paper **FedCSS: Joint Client-and-Sample Selection for Hard Sample-Aware Noise-Robust Federated Learning**.

## Operation
For example, when FedCSS with a selection ratio of 0.4 is used for training on MNIST dataset with 10 clients, 5 of which are corrupted with a corruption ratio of 0.6,

```
python main.py --dataset_name=mnist \
	--select_client \
	--client_num 10 \
	--select_ratio 0.4 \
	--corrupt_num=5 \
 	--train_type=meta \
	--test_name=test \
	--corruption_prob=0.6 \
	--epochs=100 \
	--batch_size=100 \
	--lr=1e-2 \
	--momentum=0.5
```



## References

If you find our library helpful in your research, please consider citing it:

```
@article{li2023fedcss,
  title={FedCSS: Joint Client-and-Sample Selection for Hard Sample-Aware Noise-Robust Federated Learning},
  author={Li, Anran and Cao, Yue and Guo, Jiabao and Peng, Hongyi and Guo, Qing and Yu, Han},
  journal={Proceedings of the ACM on Management of Data},
  volume={1},
  number={3},
  pages={1--24},
  year={2023},
  publisher={ACM New York, NY, USA}
}
```

