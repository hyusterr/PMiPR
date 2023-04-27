import multiprocessing
import heapq
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse


def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(precision_at_k(r, K))
        recall.append(recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(ndcg_at_k(r, K))
        hit_ratio.append(hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}

def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = train_items_r1[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

   # if args.test_flag == 'part':
    r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    #else:
   #     r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


def test( users_to_test, A):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        item_batch = range(ITEM_NUM)
        
        # (u+r) dot i = sum (u+r)xi  : np.matmul((A[0][user_batch] + A[2][rel_batch]), np.transpose(A[1]))
        # (uxr) dot i :  np.matmul(np.multiply(A[0][user_batch] , A[2][rel_batch]), np.transpose(A[1]))
        rate_batch = np.matmul(A[0][user_batch] , np.transpose(A[1]))

        user_batch_rating_uid = zip(rate_batch, user_batch)

        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            #result['hit_ratio'] += re['hit_ratio']/n_test_users
            #result['auc'] += re['auc']/n_test_users
    assert count == n_test_users
    pool.close()
    return result


def read_emb(path):
    max_u, max_i, dim = 0, 0, 0
    E_u, E_i = {}, {}
    with open(path, 'r') as f:
        for line in f:
            _id, emb = line.rstrip('\n').split('\t')
            if 'u' in _id:
                uid = int(_id[1:])
                emb = np.array(emb.split(' '))
                E_u[uid] = emb
                dim = len(emb)
                if uid > max_u:
                    max_u = uid
            else:
                iid =  int(_id[1:]) #int(_id)
                emb = np.array(emb.split(' '))
                E_i[iid] = emb
                dim = len(emb)
                if iid > max_i:
                    max_i = iid
        U = np.zeros((max_u+1, dim), dtype=float)
        I = np.zeros((max_i+1, dim), dtype=float)
        for u in E_u:
            U[u] = E_u[u]
        for i in E_i:
            I[i] = E_i[i]
    return U, I

def test_rel( users_to_test, U, I):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE
    i_batch_size = BATCH_SIZE

    test_users = users_to_test
    
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        item_batch = range(ITEM_NUM)  

        rate_batch = np.matmul(U[user_batch] , np.transpose(I[item_batch]))

        user_batch_rating_uid = zip(rate_batch, user_batch)

        batch_result = pool.map(test_one_user, user_batch_rating_uid)

        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            #result['hit_ratio'] += re['hit_ratio']/n_test_users
            #result['auc'] += re['auc']/n_test_users
    assert count == n_test_users
    pool.close()
    return result

############################################
parser=argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--dataset', type=str, help='Beibei, Taobao, Rees46, Ecommerce')
parser.add_argument('--r1_train', type=str, help='')
parser.add_argument('--r1_test', type=str, help='')
parser.add_argument('--r2_train', type=str, help='')
parser.add_argument('--r3_train', type=str, help='')
parser.add_argument('--r1_emb', type=str, help='')
parser.add_argument('--all_emb', type=str, help='')
args=parser.parse_args()


cores = multiprocessing.cpu_count() // 2
dataset = args.dataset

batch_size = 1024 
path = f'./data/{dataset}/original'
r1_train_file = path + f'/{args.r1_train}'
r1_test_file = path + f'/{args.r1_test}'
r2_train_file = path + f'/{args.r2_train}'
r3_train_file = path + f'/{args.r3_train}'


n_users, n_items = 0, 0
n_train_r1, n_test_r1 = 0, 0 # purchase
n_train_r2, n_train_r3 = 0, 0 
neg_pools = {}

exist_users_r1 = [] # purchase
exist_users_r2 = [] # pv
exist_users_r3 = []

train_items_r1, train_items_r2, train_items_r3 = {}, {}, {}
train_users_r1, train_users_r2, train_users_r3 = {}, {}, {}
test_set = {}

with open(r1_train_file) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n').split(' ')
            items = [int(i) for i in l[1:]]
            uid = int(l[0])
        
            train_items_r1[uid] = items
            
            exist_users_r1.append(uid)
            n_items = max(n_items, max(items))
            n_users = max(n_users, uid)
            n_train_r1 += len(items)
            for iid in items:
                if iid not in train_users_r1:
                    train_users_r1[iid] = []
                train_users_r1[iid] += [uid]  
                
with open(r3_train_file) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n').split(' ')
            items = [int(i) for i in l[1:]]
            uid = int(l[0])
            train_items_r3[uid] = items
            exist_users_r3.append(uid)
            n_items = max(n_items, max(items))
            n_users = max(n_users, uid)
            n_train_r3 += len(items)
            for iid in items:
                if iid not in train_users_r3:
                    train_users_r3[iid] = []
                train_users_r3[iid] += [uid]

with open(r2_train_file) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n').split(' ')
            items = [int(i) for i in l[1:]]
            uid = int(l[0])
            train_items_r2[uid] = items
            exist_users_r2.append(uid)
            n_items = max(n_items, max(items))
            n_users = max(n_users, uid)
            n_train_r2 += len(items)
            for iid in items:
                if iid not in train_users_r2:
                    train_users_r2[iid] = []
                train_users_r2[iid] += [uid]                
                
with open(r1_test_file) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n').split(' ')
            try:
                items = [int(i) for i in l[1:]]
            except Exception:
                continue
            uid = int(l[0])
            
            test_set[uid] = items
            n_users = max(n_users, uid)
            n_items = max(n_items, max(items))
            n_test_r1 += len(items)
            
n_items += 1
n_users += 1

exist_users_all = [exist_users_r1, exist_users_r2, exist_users_r3] # purchase, pv, cart
train_items_all = [train_items_r1, train_items_r2, train_items_r3] # purchase, pv, cart
train_users_all = [train_users_r1, train_users_r2, train_users_r3] # purchase, pv, cart

user_relation_prob = {}
for u in exist_users_r1:
    s = []
    for i in range(len(train_items_all)):
        if u in train_items_all[i]:
            s.append(len(train_items_all[i][u]))
        else:
            s.append(0)
    user_relation_prob[u] = [j/sum(s) for j in s]

item_list = list(set(range(n_items)))

Ks = [10,50,100]
USR_NUM, ITEM_NUM = n_users, n_items
N_TRAIN, N_TEST = n_train_r1, n_test_r1
BATCH_SIZE=264
users_to_test = list(test_set.keys())

 
u_tr, i_tr = read_emb(f'{args.r1_emb}')
u_all, i_all =  read_emb(f'{args.all_emb}')
u_tr, i_tr = np.concatenate((u_tr, u_all), axis=1), np.concatenate((i_tr, i_all), axis=1)
ret = test_rel(  users_to_test , u_tr  ,i_tr )
print(f'Dataset: {dataset}')
print('Recall@10,50,100 : ', ret['recall'])
print('NDCG@10,50,100 : ', ret['ndcg'])


