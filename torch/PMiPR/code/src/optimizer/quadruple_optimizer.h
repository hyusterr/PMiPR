#ifndef QUARDRUPLE_OPTIMIZER_H
#define QUARDRUPLE_OPTIMIZER_H
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <vector>
#include "../mapper/lookup_mapper.h"

#define SIGMOID_TABLE_SIZE 1000
#define MAX_SIGMOID 8.0

class QuadrupleOptimizer {
    private:
        void init_sigmoid();

    public:
        // constructor
        QuadrupleOptimizer();

        // variables
        std::vector<double> cached_sigmoid;

        // functions
        double fast_sigmoid(double value);

        // loss
        void feed_trans_bpr_loss(std::vector<double>& from_embedding,
                                 std::vector<double>& relation_embedding,
                                 std::vector<double>& to_pos_embedding,
                                 std::vector<double>& to_neg_embedding,
                                 int dimension,
                                 std::vector<double>& from_loss,
                                 std::vector<double>& relation_loss,
                                 std::vector<double>& to_pos_loss,
                                 std::vector<double>& to_neg_loss);

        void feed_double_bpr_loss(std::vector<double>& pos_embedding,
                                  std::vector<double>& neg_embedding,
                                  std::vector<double>& to_pos_embedding,
                                  std::vector<double>& to_neg_embedding,
                                  int dimension,
                                  std::vector<double>& from_loss,
                                  std::vector<double>& relation_loss,
                                  std::vector<double>& to_pos_loss,
                                  std::vector<double>& to_neg_loss);

        double feed_trans_margin_bpr_loss(std::vector<double>& from_embedding,
                                          std::vector<double>& relation_embedding,
                                          std::vector<double>& to_pos_embedding,
                                          std::vector<double>& to_neg_embedding,
                                          double margin,
                                          int dimension,
                                          std::vector<double>& from_loss,
                                          std::vector<double>& relation_loss,
                                          std::vector<double>& to_pos_loss,
                                          std::vector<double>& to_neg_loss);

        double feed_bpr_loss(LookupMapper& base_s_embedding,
                             LookupMapper& base_t_embedding,
                             long user,
                             long item,
                             std::vector<long>& pos_nodes,
                             std::vector<long>& neg_nodes,
                             int dimension,
                             std::vector<double>& user_loss,
                             std::vector<double>& item_loss,
                             std::vector<double>& pos_loss,
                             std::vector<double>& neg_loss);

        double feed_bpr_loss(LookupMapper& base_s_embedding,
                             LookupMapper& base_t_embedding,
                             std::vector<long>& given_nodes,
                             std::vector<long>& pos_nodes,
                             std::vector<long>& neg_nodes,
                             int dimension,
                             std::vector<double>& given_loss,
                             std::vector<double>& pos_loss,
                             std::vector<double>& neg_loss);

        double feed_ll_loss(LookupMapper& base_s_embedding,
                            LookupMapper& base_t_embedding,
                            std::vector<long>& source_nodes,
                            std::vector<long>& target_nodes,
                            int dimension,
                            std::vector<double>& source_loss,
                            std::vector<double>& target_loss,
                            double label);

        double feed_relational_bpr_loss(LookupMapper& base_s_embedding,
                                        LookupMapper& base_t_embedding,
                                        LookupMapper& rel_s_embedding,
                                        LookupMapper& rel_t_embedding,
                                        std::vector<long>& given_nodes,
                                        std::vector<long>& pos_nodes,
                                        std::vector<long>& neg_nodes,
                                        int dimension,
                                        std::vector<double>& given_loss,
                                        std::vector<double>& pos_loss,
                                        std::vector<double>& neg_loss);

        double feed_relational_bpr_loss(LookupMapper& base_s_embedding,
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
                                        std::vector<double>& neg_loss);

        double feed_relational_ll_loss(LookupMapper& base_s_embedding,
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
                                       double alpha);

        double feed_relational_bpr_loss(LookupMapper& base_s_embedding,
                                        LookupMapper& base_t_embedding,
                                        LookupMapper& rel_s_embedding,
                                        LookupMapper& rel_t_embedding,
                                        std::vector<long>& user_nodes,
                                        std::vector<long>& feat_nodes,
                                        std::vector<long>& pos_nodes,
                                        std::vector<long>& pos_feat_nodes,
                                        std::vector<long>& neg_nodes,
                                        std::vector<long>& neg_feat_nodes,
                                        int dimension,
                                        std::vector<double>& user_loss,
                                        std::vector<double>& feat_loss,
                                        std::vector<double>& pos_loss,
                                        std::vector<double>& pos_feat_loss,
                                        std::vector<double>& neg_loss,
                                        std::vector<double>& neg_feat_loss);


        double feed_relational_global_bpr_loss(LookupMapper& base_s_embedding,
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
                                                std::vector<double>& neg_loss);



};
#endif
