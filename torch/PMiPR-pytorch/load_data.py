'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import pickle

class Data(object):
    def __init__(self, path, batch_size, neg_num):
        self.path = path
        self.batch_size = batch_size
        self.neg_num = neg_num

        # default: purchase test
        train_file = path + '/train.txt'  # train.txt is puchase behavior
        test_file = path + '/test.txt'   # test.txt is puchase test
        cart_file = path + '/cart.txt'
        pv_file = path + '/pv.txt'
        
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0 # purchase
        self.n_train_pv, self.n_train_cart = 0, 0 

        self.exist_users = [] # purchase
        self.exist_users_pv = []
        self.exist_users_cart = []
        
        self.train_items, self.train_items_pv, self.train_items_cart = {}, {}, {}
        self.train_users, self.train_users_pv, self.train_users_cart = {}, {}, {}
        self.test_set = {}

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.train_items[uid] = items
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)
                    for iid in items:
                        if iid not in self.train_users:
                            self.train_users[iid] = []
                        self.train_users[iid] += [uid]
                    
        with open(cart_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.train_items_cart[uid] = items
                    self.exist_users_cart.append(uid)
                    #self.n_items = max(self.n_items, max(items))
                    #self.n_users = max(self.n_users, uid)
                    self.n_train_cart += len(items)
                    for iid in items:
                        if iid not in self.train_users_cart:
                            self.train_users_cart[iid] = []
                        self.train_users_cart[iid] += [uid]

        with open(pv_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.train_items_pv[uid] = items
                    self.exist_users_pv.append(uid)
                    #self.n_items = max(self.n_items, max(items))
                    #self.n_users = max(self.n_users, uid)
                    self.n_train_pv += len(items)
                    for iid in items:
                        if iid not in self.train_users_pv:
                            self.train_users_pv[iid] = []
                        self.train_users_pv[iid] += [uid]

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    try:
                        items = [int(i) for i in l[1:]]
                    except Exception:
                        continue
                    uid = int(l[0])
                    self.test_set[uid] = items
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1
        self.print_statistics()
        
        # self.train_items : dictionary:  {u1 : [i1, i2, i5],  u2: [i10, i3, i4]}
        # self.train_users : dictionary:  {i1 : [u1, u7, u9],  i2: [u1, u3, u11]}
        self.exist_users_all = [self.exist_users, self.exist_users_pv, self.exist_users_cart] # purchase, pv, cart
        self.train_items_all = [self.train_items, self.train_items_pv, self.train_items_cart] # purchase, pv, cart
        self.train_users_all = [self.train_users, self.train_users_pv, self.train_users_cart] # purchase, pv, cart
        self.exist_items_all = [list(i.keys()) for i in self.train_users_all] 

        self.item_list = list(set(range(self.n_items)))
        self.total_interaction = self.n_train+self.n_train_cart+self.n_train_pv
        self.prob = [self.n_train/(self.total_interaction), self.n_train_pv/self.total_interaction, 1-self.n_train/(self.total_interaction)-self.n_train_pv/self.total_interaction] # purchase, pv, cart
        print(self.prob)

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_purchase_interactions=%d' % (self.n_train ))
        print('n_cart_interactions=%d' % (self.n_train_cart ))
        print('n_pv_interactions=%d' % (self.n_train_pv ))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))


########################

    def sample_pos_u(self,i, r, num): # get user from i under relation r
        pos_users = self.train_users_all[r][i]
        n_pos_users = len(pos_users)
        pos_batch = []
        while True:
            if len(pos_batch) == num: break
            pos_id = np.random.randint(low=0, high=n_pos_users, size=1)[0]
            pos_u_id = pos_users[pos_id]
            if pos_u_id not in pos_batch:
                pos_batch.append(pos_u_id)
        return pos_batch

    def sample_pos_i(self,i, r, num): # get item from u under relation r
        pos_users = self.train_items_all[r][i]
        n_pos_users = len(pos_users)
        pos_batch = []
        while True:
            if len(pos_batch) == num: break
            pos_id = np.random.randint(low=0, high=n_pos_users, size=1)[0]
            pos_u_id = pos_users[pos_id]
            if pos_u_id not in pos_batch:
                pos_batch.append(pos_u_id)
        return pos_batch


    def sample_negrel_ui(self):
        relation_batch = [np.random.choice(np.arange(0, 3), p=[self.prob[0], self.prob[1], self.prob[2]])  for _ in range(self.batch_size)]
        #relation_batch = [rd.choice([0,1,2]) for _ in range(self.batch_size)] # r, r+
        users = [rd.choice(self.exist_users_all[i]) for i in relation_batch] # u
        items = [self.sample_pos_i(users[u], relation_batch[u], 1)[0] for u in range(self.batch_size)]
        u_rel_idx = [ 3*users[i] + relation_batch[i] for i in range(self.batch_size) ]
        i_rel_idx = [ 3*items[i] + relation_batch[i] for i in range(self.batch_size) ]

        neg_relation_batch = [np.random.choice(np.arange(0, 3), p=[self.prob[0], self.prob[1], self.prob[2]])  for _ in range(self.batch_size* self.neg_num)]
        neg_items = [rd.choice(self.exist_items_all[i]) for i in neg_relation_batch]
        neg_users = [self.sample_pos_u(neg_items[i], neg_relation_batch[i], 1)[0] for i in range(self.batch_size * self.neg_num) ]
        neg_u_rel_idx =  [ 3*neg_users[i] + neg_relation_batch[i] for i in range(self.batch_size * self.neg_num) ]
        neg_i_rel_idx =  [ 3*neg_items[i] + neg_relation_batch[i] for i in range(self.batch_size * self.neg_num) ]

        pos_users, pos_items = [], []
        pos_u_rel_idx, pos_i_rel_idx = [], []
        multi_users, multi_items = [], []
        multi_u_rel_idx, multi_i_rel_idx = [], []

        for u in range(len(users)): 
            for n in range(self.neg_num):
                pos_items += self.sample_pos_i(users[u], relation_batch[u], 1) # i+
                pos_users += self.sample_pos_u(pos_items[-1], relation_batch[u], 1) # u+
                pos_u_rel_idx += [ 3*pos_users[-1] + relation_batch[u] ] 
                pos_i_rel_idx += [ 3*pos_items[-1] + relation_batch[u] ] 

            multi_users += [users[u]]*self.neg_num
            multi_items += [items[u]]*self.neg_num
            multi_u_rel_idx += [u_rel_idx[u]]*self.neg_num
            multi_i_rel_idx += [i_rel_idx[u]]*self.neg_num

        return multi_users, multi_items, pos_users, pos_items, neg_users, neg_items, multi_u_rel_idx, pos_u_rel_idx, neg_u_rel_idx, multi_i_rel_idx, pos_i_rel_idx, neg_i_rel_idx
