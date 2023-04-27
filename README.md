

# Compilation
```
$ git clone https://github.com/cnclabs/PMiPR
$ mkdir result_embedding
$ cd PMiPR/code
$ make
```

# Task
Given a user-item interaction input:
```txt
userA itemA 1
userA itemC 1
userB itemA 1
userB itemB 1
userC itemA 1
```
The model learns the representations of each vertex:
```
userA 0.0815412 0.0205459 0.288714 0.296497 0.394043
itemA -0.207083 -0.258583 0.233185 0.0959801 0.258183
itemC 0.0185886 0.138003 0.213609 0.276383 0.45732
userB -0.0137994 -0.227462 0.103224 -0.456051 0.389858
itemB -0.317921 -0.163652 0.103891 -0.449869 0.318225
userC -0.156576 -0.3505 0.213454 0.10476 0.259673
```
Note that we can input more than one txt file (more than one behavior)

# Usage
```
./code/air+ -train_ui ./data/Beibei/processed/purc.txt.air,./data/Beibei/processed/cart.txt.air,./data/Beibei/processed/pv.txt.air,./data/Beibei/processed/all-v.txt.air \
-save ./result_embedding/Beibei-embeddings -update_times 100 -dimension 128 -worker 50 -l2_reg 0.001 -init_alpha 0.025
```
Note that in above example, we input four txt file, so the model will output the following four embedding txt files respectively.
``` ./result_embedding/Beibei-embeddings.0, ./result_embedding/Beibei-embeddings.1, ./result_embedding/Beibei-embeddings.2, ./result_embedding/Beibei-embeddings.3 
```


# Run and Eval example
```
sh run.sh
sh eval.sh
```