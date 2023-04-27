#include "quadruple_optimizer.h"

QuadrupleOptimizer::QuadrupleOptimizer() {
    // pre-compute sigmoid func
    this->cached_sigmoid.resize(SIGMOID_TABLE_SIZE);
    for (int i = 0; i != SIGMOID_TABLE_SIZE + 1; i++)
    {
        double x = i * 2.0 * MAX_SIGMOID / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
        this->cached_sigmoid[i] = 1.0 / (1.0 + exp(-x));
    }
}

double QuadrupleOptimizer::fast_sigmoid(double value) {
    if (value < -MAX_SIGMOID)
    {
        return 0.0;
    }
    else if (value > MAX_SIGMOID)
    {
        return 1.0;
    }
    else
    {
        return this->cached_sigmoid[ int((value + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2) ];
    }
}

void QuadrupleOptimizer::feed_trans_bpr_loss(std::vector<double>& from_embedding, std::vector<double>& relation_embedding,
                                             std::vector<double>& to_pos_embedding, std::vector<double>& to_neg_embedding,
                                             int dimension,
                                             std::vector<double>& from_loss, std::vector<double>& relation_loss,
                                             std::vector<double>& to_pos_loss, std::vector<double>& to_neg_loss) {

    std::vector<double> source_embedding(dimension, 0.0);
    std::vector<double> target_embedding(dimension, 0.0);

    double gradient, prediction=0;

    for (int d=0; d<dimension; ++d)
    {
        source_embedding[d] = from_embedding[d] + relation_embedding[d];
        target_embedding[d] = to_pos_embedding[d] - to_neg_embedding[d];
        prediction += source_embedding[d] * target_embedding[d];
    }

    gradient = this->fast_sigmoid(-prediction);
    for (int d=0; d<dimension; ++d)
    {
        from_loss[d] += gradient * target_embedding[d];
        relation_loss[d] += gradient * target_embedding[d];
        to_pos_loss[d] += gradient * source_embedding[d];
        to_neg_loss[d] -= gradient * source_embedding[d];
    }
}

void QuadrupleOptimizer::feed_double_bpr_loss(std::vector<double>& pos_embedding, std::vector<double>& neg_embedding,
                                              std::vector<double>& to_pos_embedding, std::vector<double>& to_neg_embedding,
                                              int dimension,
                                              std::vector<double>& from_loss, std::vector<double>& relation_loss,
                                              std::vector<double>& to_pos_loss, std::vector<double>& to_neg_loss) {

    std::vector<double> source_embedding(dimension, 0.0);
    std::vector<double> target_embedding(dimension, 0.0);

    double gradient, prediction=0;

    for (int d=0; d<dimension; ++d)
    {
        source_embedding[d] = pos_embedding[d] - neg_embedding[d];
        target_embedding[d] = to_pos_embedding[d] - to_neg_embedding[d];
        prediction += source_embedding[d] * target_embedding[d];
    }

    gradient = this->fast_sigmoid(-prediction);
    for (int d=0; d<dimension; ++d)
    {
        from_loss[d] += gradient * target_embedding[d];
        relation_loss[d] -= gradient * target_embedding[d];
        to_pos_loss[d] += gradient * source_embedding[d];
        to_neg_loss[d] -= gradient * source_embedding[d];
    }
}


double QuadrupleOptimizer::feed_trans_margin_bpr_loss(std::vector<double>& from_embedding, std::vector<double>& relation_embedding,
                                                      std::vector<double>& to_pos_embedding, std::vector<double>& to_neg_embedding,
                                                      double margin, int dimension,
                                                      std::vector<double>& from_loss, std::vector<double>& relation_loss,
                                                      std::vector<double>& to_pos_loss, std::vector<double>& to_neg_loss) {

    std::vector<double> source_embedding(dimension, 0.0);
    std::vector<double> target_embedding(dimension, 0.0);

    double gradient, prediction=0;

    for (int d=0; d<dimension; ++d)
    {
        source_embedding[d] = from_embedding[d] + relation_embedding[d];
        target_embedding[d] = to_pos_embedding[d] - to_neg_embedding[d];
        prediction += source_embedding[d] * target_embedding[d];
    }
    prediction -= margin;
    //if (prediction > margin)
    //    return 0;

    gradient = this->fast_sigmoid(-prediction);
    for (int d=0; d<dimension; ++d)
    {
        from_loss[d] += gradient * target_embedding[d];
        relation_loss[d] += gradient * target_embedding[d];
        to_pos_loss[d] += gradient * source_embedding[d];
        to_neg_loss[d] -= gradient * source_embedding[d];
    }
    return gradient;
}

double QuadrupleOptimizer::feed_ll_loss(
    LookupMapper& base_s_embedding,
    LookupMapper& base_t_embedding,
    std::vector<long>& source_nodes,
    std::vector<long>& target_nodes,
    int dimension,
    std::vector<double>& source_loss,
    std::vector<double>& target_loss,
    double label
) {

    std::vector<double> source_embedding(dimension, 0.0);
    std::vector<double> target_embedding(dimension, 0.0);
    long node;
    double gradient, prediction=0;

    for (int i=0; i<source_nodes.size(); i++)
    {
        node = source_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            source_embedding[d] += base_s_embedding[node][d];
        }
    }
    for (int i=0; i<target_nodes.size(); i++)
    {
        node = target_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            target_embedding[d] += base_t_embedding[node][d];
        }
    }
    for (int d=0; d<dimension; ++d)
    {
        prediction += source_embedding[d] * target_embedding[d];
    }
    if (label > 0.0)
        gradient = this->fast_sigmoid(-prediction);
    else if (prediction > 0)
        gradient = 0.0 - prediction;
    else
        return 0.0;

    for (int d=0; d<dimension; ++d)
    {
        source_loss[d] += gradient * target_embedding[d];
        target_loss[d] += gradient * source_embedding[d];
    }
    return std::abs(gradient);
}


double QuadrupleOptimizer::feed_bpr_loss(
    LookupMapper& base_s_embedding,
    LookupMapper& base_t_embedding,
    std::vector<long>& given_nodes,
    std::vector<long>& pos_nodes,
    std::vector<long>& neg_nodes,
    int dimension,
    std::vector<double>& given_loss,
    std::vector<double>& pos_loss,
    std::vector<double>& neg_loss
) {

    std::vector<double> source_embedding(dimension, 0.0);
    std::vector<double> target_embedding(dimension, 0.0);
    long node;
    double gradient, prediction=0;

    for (int i=0; i<given_nodes.size(); i++)
    {
        node = given_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            source_embedding[d] += base_s_embedding[node][d];
        }
    }
    for (int i=0; i<pos_nodes.size(); i++)
    {
        node = pos_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            target_embedding[d] += base_t_embedding[node][d];
        }
    }
    for (int i=0; i<neg_nodes.size(); i++)
    {
        node = neg_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            target_embedding[d] -= base_t_embedding[node][d];
        }
    }
    for (int d=0; d<dimension; ++d)
    {
        prediction += source_embedding[d] * target_embedding[d];
    }
    gradient = this->fast_sigmoid(-prediction);
    for (int d=0; d<dimension; ++d)
    {
        given_loss[d] += gradient * target_embedding[d];
        pos_loss[d] += gradient * source_embedding[d];
        neg_loss[d] -= gradient * source_embedding[d];
    }
    return gradient;
}

double QuadrupleOptimizer::feed_bpr_loss(
    LookupMapper& base_s_embedding,
    LookupMapper& base_t_embedding,
    long user,
    long item,
    std::vector<long>& pos_nodes,
    std::vector<long>& neg_nodes,
    int dimension,
    std::vector<double>& user_loss,
    std::vector<double>& item_loss,
    std::vector<double>& pos_loss,
    std::vector<double>& neg_loss
) {

    std::vector<double> source_embedding(dimension, 0.0);
    std::vector<double> target_embedding(dimension, 0.0);
    long node;
    double gradient, prediction=0;

    for (int d=0; d<dimension; ++d)
    {
        source_embedding[d] += base_s_embedding[user][d];
        source_embedding[d] += base_s_embedding[item][d];
    }
    for (int i=0; i<pos_nodes.size(); i++)
    {
        node = pos_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            target_embedding[d] += base_t_embedding[node][d];
        }
    }
    for (int i=0; i<neg_nodes.size(); i++)
    {
        node = neg_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            target_embedding[d] -= base_t_embedding[node][d];
        }
    }
    for (int d=0; d<dimension; ++d)
    {
        prediction += source_embedding[d] * target_embedding[d];
    }
    gradient = this->fast_sigmoid(-prediction);
    for (int d=0; d<dimension; ++d)
    {
        user_loss[d] += gradient * target_embedding[d];
        item_loss[d] += gradient * target_embedding[d];
        pos_loss[d] += gradient * source_embedding[d];
        neg_loss[d] -= gradient * source_embedding[d];
    }
    return gradient;
}

double QuadrupleOptimizer::feed_relational_bpr_loss(
    LookupMapper& base_s_embedding,
    LookupMapper& base_t_embedding,
    LookupMapper& rel_s_embedding,
    LookupMapper& rel_t_embedding,
    std::vector<long>& given_nodes,
    std::vector<long>& pos_nodes,
    std::vector<long>& neg_nodes,
    int dimension,
    std::vector<double>& given_loss,
    std::vector<double>& pos_loss,
    std::vector<double>& neg_loss
) {

    std::vector<double> source_embedding(dimension, 0.0);
    std::vector<double> target_embedding(dimension, 0.0);
    long node;
    double gradient, prediction=0;

    for (int i=0; i<given_nodes.size(); i++)
    {
        node = given_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            source_embedding[d] += base_s_embedding[node][d];
            source_embedding[d] += rel_s_embedding[node][d];
        }
    }
    for (int i=0; i<pos_nodes.size(); i++)
    {
        node = pos_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            target_embedding[d] += base_t_embedding[node][d];
            target_embedding[d] += rel_t_embedding[node][d];
        }
    }
    for (int i=0; i<neg_nodes.size(); i++)
    {
        node = neg_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            target_embedding[d] -= base_t_embedding[node][d];
            target_embedding[d] -= rel_t_embedding[node][d];
        }
    }
    for (int d=0; d<dimension; ++d)
    {
        prediction += source_embedding[d] * target_embedding[d];
    }
    gradient = this->fast_sigmoid(-prediction);
    for (int d=0; d<dimension; ++d)
    {
        given_loss[d] += gradient * target_embedding[d];
        pos_loss[d] += gradient * source_embedding[d];
        neg_loss[d] -= gradient * source_embedding[d];
    }
    return gradient;
}


double QuadrupleOptimizer::feed_relational_ll_loss(
    LookupMapper& base_s_embedding,
    LookupMapper& base_t_embedding,
    LookupMapper& rel_s_embedding,
    LookupMapper& rel_t_embedding,
    std::vector<long>& user_nodes,
    std::vector<long>& item_nodes,
    std::vector<long>& pos_nodes,
    std::vector<long>& neg_nodes,
    int dimension,
    std::vector<double>& user_loss,
    std::vector<double>& item_loss,
    std::vector<double>& pos_loss,
    std::vector<double>& neg_loss,
    double alpha
) {

    std::vector<double> source_embedding(dimension, 0.0);
    std::vector<double> pos_embedding(dimension, 0.0);
    std::vector<double> neg_embedding(dimension, 0.0);
    long node;
    double pos_gradient, neg_gradient, pos_prediction=0, neg_prediction=0;

    for (int i=0; i<user_nodes.size(); i++)
    {
        node = user_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            source_embedding[d] += base_s_embedding[node][d];
            //source_embedding[d] += rel_s_embedding[node][d];
        }
    }
    for (int i=0; i<item_nodes.size(); i++)
    {
        node = item_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            source_embedding[d] += base_s_embedding[node][d];
            //source_embedding[d] += rel_s_embedding[node][d];
        }
    }
    for (int i=0; i<pos_nodes.size(); i++)
    {
        node = pos_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            pos_embedding[d] += base_t_embedding[node][d];
            //target_embedding[d] += rel_t_embedding[node][d];
        }
    }
    for (int i=0; i<neg_nodes.size(); i++)
    {
        node = neg_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            neg_embedding[d] -= base_t_embedding[node][d];
            //target_embedding[d] -= rel_t_embedding[node][d];
        }
    }
    for (int d=0; d<dimension; ++d)
    {
        pos_prediction += source_embedding[d] * pos_embedding[d];
        neg_prediction += source_embedding[d] * neg_embedding[d];
    }
    pos_gradient = (1.0 - this->fast_sigmoid(pos_prediction)) * alpha;
    neg_gradient = (0.0 - this->fast_sigmoid(neg_prediction)) * alpha;
    for (int d=0; d<dimension; ++d)
    {
        user_loss[d] += (pos_gradient*pos_embedding[d]) + (neg_gradient*neg_embedding[d]);
        item_loss[d] += (pos_gradient*pos_embedding[d]) + (neg_gradient*neg_embedding[d]);
        pos_loss[d] += pos_gradient * source_embedding[d];
        neg_loss[d] -= neg_gradient * source_embedding[d];
    }
    return pos_gradient+neg_gradient;
}


double QuadrupleOptimizer::feed_relational_bpr_loss(
    LookupMapper& base_s_embedding,
    LookupMapper& base_t_embedding,
    LookupMapper& rel_s_embedding,
    LookupMapper& rel_t_embedding,
    std::vector<long>& user_nodes,
    std::vector<long>& item_nodes,
    std::vector<long>& pos_nodes,
    std::vector<long>& neg_nodes,
    int dimension,
    std::vector<double>& user_loss,
    std::vector<double>& item_loss,
    std::vector<double>& pos_loss,
    std::vector<double>& neg_loss
) {

    std::vector<double> source_embedding(dimension, 0.0);
    std::vector<double> target_embedding(dimension, 0.0);
    long node;
    double gradient, prediction=0;

    for (int i=0; i<user_nodes.size(); i++)
    {
        node = user_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            source_embedding[d] += base_s_embedding[node][d];
            source_embedding[d] += rel_s_embedding[node][d];
        }
    }
    for (int i=0; i<item_nodes.size(); i++)
    {
        node = item_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            source_embedding[d] += base_s_embedding[node][d];
            source_embedding[d] += rel_s_embedding[node][d];
        }
    }
    for (int i=0; i<pos_nodes.size(); i++)
    {
        node = pos_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            target_embedding[d] += base_t_embedding[node][d];
            target_embedding[d] += rel_t_embedding[node][d];
        }
    }
    for (int i=0; i<neg_nodes.size(); i++)
    {
        node = neg_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            target_embedding[d] -= base_t_embedding[node][d];
            target_embedding[d] -= rel_t_embedding[node][d];
        }
    }
    for (int d=0; d<dimension; ++d)
    {
        prediction += source_embedding[d] * target_embedding[d];
    }
    gradient = this->fast_sigmoid(-prediction);
    for (int d=0; d<dimension; ++d)
    {
        user_loss[d] += gradient * target_embedding[d];
        item_loss[d] += gradient * target_embedding[d];
        pos_loss[d] += gradient * source_embedding[d];
        neg_loss[d] -= gradient * source_embedding[d];
    }
    return gradient;
}

// double QuadrupleOptimizer::feed_relational_bpr_loss(
//     LookupMapper& base_s_embedding,
//     LookupMapper& base_t_embedding,
//     LookupMapper& rel_s_embedding,
//     LookupMapper& rel_t_embedding,
//     std::vector<long>& user_nodes,
//     std::vector<long>& feat_nodes,
//     std::vector<long>& pos_nodes,
//     std::vector<long>& pos_feat_nodes,
//     std::vector<long>& neg_nodes,
//     std::vector<long>& neg_feat_nodes,
//     int dimension,
//     std::vector<double>& user_loss,
//     std::vector<double>& feat_loss,
//     std::vector<double>& pos_loss,
//     std::vector<double>& pos_feat_loss,
//     std::vector<double>& neg_loss,
//     std::vector<double>& neg_feat_loss)
// {
//     std::vector<double> source_embedding(dimension, 0.0);
//     std::vector<double> target_embedding(dimension, 0.0);
//     long node;
//     double gradient, prediction=0;

//     for (int i=0; i<user_nodes.size(); i++)
//     {
//         node = user_nodes[i];
//         for (int d=0; d<dimension; ++d)
//         {
//             source_embedding[d] += base_s_embedding[node][d];
//             source_embedding[d] += rel_s_embedding[node][d];
//         }
//     }
//     for (int i=0; i<feat_nodes.size(); i++)
//     {
//         node = feat_nodes[i];
//         for (int d=0; d<dimension; ++d)
//         {
//             source_embedding[d] += base_s_embedding[node][d];
//         }
//     }
//     for (int i=0; i<pos_nodes.size(); i++)
//     {
//         node = pos_nodes[i];
//         for (int d=0; d<dimension; ++d)
//         {
//             target_embedding[d] += base_t_embedding[node][d];
//             target_embedding[d] += rel_t_embedding[node][d];
//         }
//     }
//     for (int i=0; i<pos_feat_nodes.size(); i++)
//     {
//         node = pos_feat_nodes[i];
//         for (int d=0; d<dimension; ++d)
//         {
//             target_embedding[d] += base_t_embedding[node][d];
//         }
//     }
//     for (int i=0; i<neg_nodes.size(); i++)
//     {
//         node = neg_nodes[i];
//         for (int d=0; d<dimension; ++d)
//         {
//             target_embedding[d] -= base_t_embedding[node][d];
//             target_embedding[d] -= rel_t_embedding[node][d];
//         }
//     }
//     for (int i=0; i<neg_feat_nodes.size(); i++)
//     {
//         node = neg_nodes[i];
//         for (int d=0; d<dimension; ++d)
//         {
//             target_embedding[d] -= base_t_embedding[node][d];
//         }
//     }
//     for (int d=0; d<dimension; ++d)
//     {
//         prediction += source_embedding[d] * target_embedding[d];
//     }
//     gradient = this->fast_sigmoid(-prediction);
//     for (int d=0; d<dimension; ++d)
//     {
//         user_loss[d] += gradient * target_embedding[d];
//         feat_loss[d] += gradient * target_embedding[d];
//         pos_loss[d] += gradient * source_embedding[d];
//         pos_feat_loss[d] += gradient * source_embedding[d];
//         neg_loss[d] -= gradient * source_embedding[d];
//         neg_feat_loss[d] -= gradient * source_embedding[d];
//     }
//     return gradient;
// }

double QuadrupleOptimizer::feed_relational_global_bpr_loss(
    LookupMapper& base_s_embedding,
    LookupMapper& base_t_embedding,
    LookupMapper& rel_s_embedding,
    LookupMapper& rel_t_embedding,
    std::vector<long>& user_nodes,
    std::vector<long>& item_nodes,
    std::vector<long>& pos_nodes,
    std::vector<long>& neg_nodes,
    int dimension,
    std::vector<double>& user_loss,
    std::vector<double>& item_loss,
    std::vector<double>& pos_loss,
    std::vector<double>& neg_loss
) {

    std::vector<double> source_embedding(dimension, 0.0);
    std::vector<double> target_embedding(dimension, 0.0);
    long node;
    double gradient, prediction=0;

    for (int i=0; i<user_nodes.size(); i++)
    {
        node = user_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            source_embedding[d] += base_s_embedding[node][d];
            // source_embedding[d] += rel_s_embedding[node][d];
        }
    }
    for (int i=0; i<item_nodes.size(); i++)
    {
        node = item_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            source_embedding[d] += base_s_embedding[node][d];
            source_embedding[d] += rel_s_embedding[0][d];
        }
    }
    for (int i=0; i<pos_nodes.size(); i++)
    {
        node = pos_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            target_embedding[d] += base_t_embedding[node][d];
            //target_embedding[d] += rel_t_embedding[node][d];
        }
    }
    for (int i=0; i<neg_nodes.size(); i++)
    {
        node = neg_nodes[i];
        for (int d=0; d<dimension; ++d)
        {
            target_embedding[d] -= base_t_embedding[node][d];
            //target_embedding[d] -= rel_t_embedding[node][d];
        }
    }
    for (int d=0; d<dimension; ++d)
    {
        prediction += source_embedding[d] * target_embedding[d];
    }
    gradient = this->fast_sigmoid(-prediction);
    for (int d=0; d<dimension; ++d)
    {
        user_loss[d] += gradient * target_embedding[d];
        item_loss[d] += gradient * target_embedding[d];
        pos_loss[d] += gradient * source_embedding[d];
        neg_loss[d] -= gradient * source_embedding[d];
    }
    return gradient;
}