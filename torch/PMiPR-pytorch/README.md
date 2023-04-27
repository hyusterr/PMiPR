# PMiPR Pytorch version
Pytorch implementation for PMiPR model

## Environment Requirement
The code has been tested running under Python 3.7.11 The required packages are as follows:
* torch==1.10.0
* torchaudio==0.10.0
* torchvision==0.11.1
* numpy==1.20.3
* scikit-learn==1.0.2
* scipy==1.7.3
Or you can visit ```requirements.txt``` for more details, note that this requirement contains many unnecessary packages (for example, transfomers, tokenizers, jupyterlab, you can just install the suggested packages listed as above first)

## Description
* The code writing style is refer to GHCF: https://github.com/chenchongthu/GHCF

Download and unzip the data from https://drive.google.com/file/d/1hTqG7rkjl0S77OU0jF9DMZ93A_YdRqpX/view?usp=share_link

* Since the GHCF is implemented by tensorflow, so it is difficult to modify (especially evaluation part)

* PMiPR.py: PMiPR model
    * Default: Use all behavior, but only evaluate on purchase behavior
    * For other behavior (for example, if you want to evaluate on cart behavior)
        * Step1: go ```load_data.py```, change the following code:
            * line20: ```train_file = path + '/cart.txt'```
            * line21: ```test_file = path + '/cart_test.txt'```
            * line22: ```cart_file = path + '/train.txt'```
            * line23: ```pv_file = path + '/pv.txt'```
        * Step2: go ```PMiPR.py```, change the following code:
            * line148: ```R_purc_u = [ 3*us + 2   for us in users_to_test] # +0 : purchase, +1: pv, +2: cart```
            * line149: ```R_purc_i = [ 3*us + 2   for us in range(data_generator.n_items)] # +0 : purchase, +1: pv, +2: cart```

* Disadvantage: 
    * Only support 3 behaviors
    * Does not include 'global' behavior
    * Need to change some code if you want to see the result on othe behavior

## Run and Eval example
```
python PMiPR.py
```
