import metrics as metrics
from load_data import *
import multiprocessing
import heapq

cores = multiprocessing.cpu_count() // 2


Ks = [1,5,10,20,50,80,100]
# Data(path='/tmp2/weile/GHCF/Data/' + 'Beibei/', batch_size=256)
data_generator = Data(path='Data/' + 'Beibei/', batch_size=1024, neg_num=5)
USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test

BATCH_SIZE = 1024




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
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_items[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

   # if args.test_flag == 'part':
    r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    #else:
   #     r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)


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



def test_rel_global( users_to_test, U, I, R):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE

    test_users = users_to_test
    
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        
        item_batch = range(ITEM_NUM)
        
        # u+r+i
        rate_batch = np.matmul(U[user_batch] + R[start: end] , np.transpose(I[item_batch]))

        # u*r + i
        #rate_batch = np.matmul(U[user_batch] * R[start: end] , np.transpose(I[item_batch]))

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


def test_rel_multi( users_to_test, U, I, R, purc_rel_to_test ):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE

    test_users = users_to_test
    rel_users = purc_rel_to_test
    
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        rel_batch = rel_users[start: end]
        
        item_batch = range(ITEM_NUM)
        
        # sum(u+r+i) = (u+r) dot i 
        rate_batch = np.matmul(U[user_batch] + R[rel_batch] , np.transpose(I[item_batch]))

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



def test( users_to_test, U, I):
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

    assert count == n_test_users
    pool.close()
    return result



def test_rel_multi_ui( users_to_test, U, I, RU, RI, purc_rel_u_to_test, purc_rel_i_to_test ):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))}

    pool = multiprocessing.Pool(cores)

    u_batch_size = BATCH_SIZE

    test_users = users_to_test
    rel_users = purc_rel_u_to_test
    rel_items = purc_rel_i_to_test
    
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        rel_u_batch = rel_users[start: end]

        rel_i_batch = rel_items
        item_batch = range(ITEM_NUM)
        
        # sum(u+r+i) = (u+ru) dot (i+ri) 
        rate_batch = np.matmul(U[user_batch] + RU[rel_u_batch] , np.transpose(I[item_batch] + RI[rel_i_batch] ))

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