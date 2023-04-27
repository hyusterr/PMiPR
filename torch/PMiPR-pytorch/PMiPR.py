import torch
import torch.nn as nn
import torch.optim as optim

import os
import time
import argparse
import numpy as np
import multiprocessing
import pickle

from batch_test import *

class PMiPR(nn.Module):
    def __init__(self, user_num, item_num, rel_u_num, rel_i_num, factor_num):
        super(PMiPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """		
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)
        self.embed_rel_u = nn.Embedding(rel_u_num, factor_num)
        self.embed_rel_i = nn.Embedding(rel_i_num, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        nn.init.normal_(self.embed_rel_u.weight, std=0.01)
        nn.init.normal_(self.embed_rel_i.weight, std=0.01)

    def forward(self, user, item, user_pos, item_pos, user_neg, item_neg, rel_u, pos_rel_u, neg_rel_u, rel_i, pos_rel_i, neg_rel_i):
        
        users = self.embed_user(user)
        items = self.embed_item(item)

        user_pos = self.embed_user(user_pos)
        item_pos = self.embed_item(item_pos)

        user_neg = self.embed_user(user_neg)
        item_neg = self.embed_item(item_neg)

        rel_u = self.embed_rel_u(rel_u)
        pos_rel_u = self.embed_rel_u(pos_rel_u)
        neg_rel_u = self.embed_rel_u(neg_rel_u)

        rel_i = self.embed_rel_i(rel_i)
        pos_rel_i = self.embed_rel_i(pos_rel_i)
        neg_rel_i = self.embed_rel_i(neg_rel_i)        


        u_i_ = users + items + rel_u + rel_i # g
        u_i_pos = user_pos + item_pos + pos_rel_u + pos_rel_i # g+
        u_i_neg = user_neg + item_neg + neg_rel_u + neg_rel_i # g-

        #######
        prediction_i = torch.multiply(u_i_, u_i_pos).sum(dim=-1) # g g+
        prediction_j = torch.multiply(u_i_, u_i_neg).sum(dim=-1) # g g-
        
        loss = torch.mean(nn.functional.softplus(prediction_j - prediction_i)) # equal to -log softmax
        reg_loss = (1/2)*(users.norm(2).pow(2) + items.norm(2).pow(2) + 
                            user_pos.norm(2).pow(2) + item_pos.norm(2).pow(2) +
                            user_neg.norm(2).pow(2) + item_neg.norm(2).pow(2) + 
                            rel_u.norm(2).pow(2) + pos_rel_u.norm(2).pow(2) + neg_rel_u.norm(2).pow(2) + 
                            rel_i.norm(2).pow(2) + pos_rel_i.norm(2).pow(2) + neg_rel_i.norm(2).pow(2)) / float(len(user))
        
        return loss, reg_loss

########  main  ########

user_num = data_generator.n_users
item_num = data_generator.n_items
train_items = data_generator.train_items
test_set =  data_generator.test_set
users_to_test = list(data_generator.test_set.keys())

items_to_test = list(data_generator.test_set.values())
items_to_test = [i[0] for i in items_to_test]

batch_size = data_generator.batch_size

count = 0
verbose=1
weight_decay = 1e-4
rel_num = 3

path = 'output_log.txt'

model = PMiPR(user_num, item_num, rel_num*user_num , rel_num*item_num, 64)
model.cuda()       

optimizer = optim.Adam(model.parameters(), lr=0.001)

Best_rec = 0

for epoch in range(1, 1001 +1):
    model.train() 
    start_time = time()

    n_batch = data_generator.n_train // batch_size + 1
    Loss, Reg_loss = 0., 0.
    for idx in range(n_batch):  # like train loader
        users, items, pos_users, pos_items, neg_users, neg_items, nor_u_r, pos_u_r, neg_u_r, nor_i_r, pos_i_r, neg_i_r  = data_generator.sample_negrel_ui()

        users = torch.tensor(users).cuda()
        items = torch.tensor(items).cuda()
        pos_users = torch.tensor(pos_users).cuda()
        pos_items = torch.tensor(pos_items).cuda()
        neg_users = torch.tensor(neg_users).cuda()
        neg_items = torch.tensor(neg_items).cuda()
        nor_u_r = torch.tensor(nor_u_r).cuda()
        pos_u_r = torch.tensor(pos_u_r).cuda()
        neg_u_r = torch.tensor(neg_u_r).cuda()

        nor_i_r = torch.tensor(nor_i_r).cuda()
        pos_i_r = torch.tensor(pos_i_r).cuda()
        neg_i_r = torch.tensor(neg_i_r).cuda()
        
        model.zero_grad()
        #optimizer.zero_grad()
    
      
        loss, reg_loss = model(users, items, pos_users, pos_items, neg_users, neg_items, nor_u_r, pos_u_r, neg_u_r, nor_i_r, pos_i_r, neg_i_r)
        reg_loss = reg_loss* weight_decay
        loss_ = loss + reg_loss
           
        loss_.backward()
        optimizer.step()
        count += 1
        
        Loss += loss_ / n_batch
        Reg_loss += reg_loss / n_batch
        
    elapsed_time = time() - start_time

    if verbose > 0 :
        perf_str = 'Epoch %d [%.1fs]: train==[%.5f + %.5f]' % (
            epoch, elapsed_time , Loss, Reg_loss)


    if (epoch % 10) == 0:
        # this will slow the evaluation since it load the parameters and infer 
        U = model.embed_user.weight.cpu().detach().numpy()
        I = model.embed_item.weight.cpu().detach().numpy()
        RU = model.embed_rel_u.weight.cpu().detach().numpy()
        RI = model.embed_rel_i.weight.cpu().detach().numpy()

        R_purc_u = [ 3*us + 0   for us in users_to_test] # +0 : purchase, +1: pv, +2: cart
        R_purc_i = [ 3*us + 0   for us in range(len(I))] # +0 : purchase, +1: pv, +2: cart

        ret = test_rel_multi_ui(  users_to_test, U, I, RU, RI, R_purc_u, R_purc_i)
        #print('recall: ',ret['recall'])
        #print('ndcg: ',ret['ndcg'])

        perf_str = 'Epoch %d [%.1fs]: train==[%.5f + %.5f]   [%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f]  [%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f]' % (
            epoch, elapsed_time , Loss, Reg_loss, 
            ret['recall'][0], ret['recall'][1], ret['recall'][2], ret['recall'][3], ret['recall'][4], ret['recall'][5], ret['recall'][6],
            ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][2], ret['ndcg'][3], ret['ndcg'][4], ret['ndcg'][5], ret['ndcg'][6] )

        if ret['recall'][2] > Best_rec:
            Best_rec = ret['recall'][2]
            Best_perf = perf_str

            # print('Save the best result...')

            # with open('/tmp2/weile/AIR_python/AIR_pytorch/result_emb/U_rel_multi_ui', 'wb') as fp:
            #     pickle.dump(U, fp)
            # with open('/tmp2/weile/AIR_python/AIR_pytorch/result_emb/I_rel_multi_ui', 'wb') as fp:
            #     pickle.dump(I, fp)
            # with open('/tmp2/weile/AIR_python/AIR_pytorch/result_emb/RU_rel_multi_ui', 'wb') as fp:
            #     pickle.dump(RU, fp)
            # with open('/tmp2/weile/AIR_python/AIR_pytorch/result_emb/RI_rel_multi_ui', 'wb') as fp:
            #     pickle.dump(RI, fp)

        else:
            count += 1
            if count == 50 :
                print('BEST: ', Best_perf)
                break

    print(perf_str)

    with open(path, 'a+') as f:
        f.write(f'{perf_str}\n')

