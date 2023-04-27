#ifndef LOOKUP_MAPPER_H
#define LOOKUP_MAPPER_H
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <iomanip>
#include "../util/file_graph.h"
#include "../sampler/vc_sampler.h"
#include <typeinfo>

typedef std::pair<std::string, std::vector<double>> Node2Embedding;

class LookupMapper {
    public:
        //variable
        int size, dimension;
        std::vector<std::vector<double>> embedding;

        // embedding function
        std::vector<double> avg_embedding(std::vector<long>& indexes);
        std::vector<double> textgcn_embedding(std::vector<long>& indexes);
        std::vector<double> meta_gcn_embedding(std::vector<long>& indexes);
        std::vector<double> meta_avg_embedding(std::vector<long>& indexes);
        std::vector<double> weighted_embedding(std::vector<long>& indexes, std::vector<std::vector<double>>& weights);

        // constructor
        LookupMapper(int size, int dimension);
        void init_by_sum(std::vector<std::vector<double>> embedding, FileGraph& graph);

        // load pretrain
        void load_pretrain(std::string file_name, n2iHash node2index, int binary_out);

        // update function
        void update(long index, std::vector<double>& loss_vector, double alpha);
        void update_with_l2(long index, std::vector<double>& loss_vector, double alpha, double lambda);
        void update_with_weights_l2(long index, std::vector<double>& loss_vector, std::vector<double>& weights, double alpha, double lambda);
        void update_with_l2(long index, std::vector<double>& loss_vector, double alpha, double lambda, int start_dim, int end_dim);

        // save function
        void save_to_file(std::vector<char*>& index2vertex,
                          std::string file_name,
                          int binary_out);
        void save_meta_avg_to_file(std::vector<char*>& index2node,
                                   int use_self,
                                   FileGraph* main_graph,
                                   int use_main,
                                   std::vector<FileGraph>& meta_graphs,
                                   int use_meta,
                                   std::vector<long> indexes,
                                   std::string file_name,
                                   int binary_out);

        // overload operator
        std::vector<double>& operator[](long index);
};

void dump_air_plus_embeddings(LookupMapper& mapper_bs,
                              std::vector<LookupMapper>& mapper_rs,
                              std::vector<LookupMapper>& mapper_rt,
                              int source_relation,
                              int source_neighbors,
                              int target_relation,
                              int target_neighbors,
                              int dimension,
                              std::vector<char*>& index2node,
                              std::vector<FileGraph>& ui_graphs,
                              std::vector<FileGraph>& iu_graphs,
                              std::vector<FileGraph>& up_graphs,
                              std::vector<FileGraph>& im_graphs,
                              std::string file_name);


void dump_air_plus_global_embeddings(LookupMapper& mapper_bs,
                                    std::vector<LookupMapper>& mapper_rs,
                                    std::vector<LookupMapper>& mapper_rt,
                                    int source_relation,
                                    int source_neighbors,
                                    int target_relation,
                                    int target_neighbors,
                                    int dimension,
                                    std::vector<char*>& index2node,
                                    std::vector<FileGraph>& ui_graphs,
                                    std::vector<FileGraph>& iu_graphs,
                                    std::vector<FileGraph>& up_graphs,
                                    std::vector<FileGraph>& im_graphs,
                                    std::string file_name);


#endif
