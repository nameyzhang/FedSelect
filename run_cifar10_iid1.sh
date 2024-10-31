# nohup bash run_cifar10_iid1.sh &

algorithm=(FedSelect)

model=resnet32
dataset_name=CIFAR10

client_num=10
select_client=1
select_ratio=0.4

select_samples=1
select_samples_ratio=0.6

corruption_type=uniform
corrupt_num=10
corruption_prob=0.4

epochs=130
local_epochs=1
batch_size=100
lr=1e-2
momentum=0.5

seed=(1 2 3 4 5)



for (( j = 0; j < ${#algorithm[@]}; j++ )); do
    for (( i = 0; i<${#seed[@]}; i++ ))     # 遍历不同的seed
        do
            python -u main.py --algorithm "${algorithm[j]}"\
            --model $model \
            --dataset_name $dataset_name \
            --client_num $client_num \
            --select_client $select_client\
            --select_ratio $select_ratio \
            --select_samples $select_samples \
            --select_samples_ratio $select_samples_ratio \
            --corruption_type $corruption_type \
            --corrupt_num $corrupt_num  \
            --corruption_prob $corruption_prob \
            --epochs $epochs \
            --local_epochs $local_epochs \
            --batch_size $batch_size \
            --lr $lr \
            --momentum $momentum\
            --seed "${seed[i]}"
    done
done

