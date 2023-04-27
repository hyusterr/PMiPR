#include "lookup_mapper.h"
#include <msgpack.hpp>

LookupMapper::LookupMapper(int size, int dimension) {
    this->embedding.resize(size);
    this->size = size;
    this->dimension = dimension;
    for (long index=0; index<size; ++index)
    {
        this->embedding[index].resize(dimension);
        for (int d=0; d<dimension; ++d)
        {
            this->embedding[index][d] = (rand()/(double)RAND_MAX - 0.5) / dimension;
        }
    }
}

void LookupMapper::init_by_sum(std::vector<std::vector<double>> embedding, FileGraph& graph) {

    int match_counter = 0;
    std::cout << "Initialize Embeddings by Neighbors' Sum:" << std::endl;
    for (long index=0; index<this->size; ++index)
    {
        if (index >= embedding.size())
            break;

        int branch = 0;
        if (graph.index_graph.find(index)!=graph.index_graph.end())
            branch = graph.index_graph[index].size();
        if (branch == 0)
            continue;

        // weight avg.
        match_counter++;
        this->embedding[index].assign(this->dimension, 0.0);
        double weight_sum = 0.0;
        for (auto it: graph.index_graph[index])
        {
            long to_index = it.first;
            for (int dim=0; dim!=this->dimension; dim++)
            {
                this->embedding[index][dim] += embedding[to_index][dim];
            }
        }
    }
    std::cout << "\t# nodes:\t" << match_counter << std::endl;
}


void LookupMapper::load_pretrain(std::string file_name, n2iHash node2index, int binary_out) {
    long index, counter=0;
    std::string line, node, value;

    std::cout << "Load Pre-train Embeddings:" << std::endl;
    if (binary_out)
    {

        std::ifstream ifs(file_name);
        std::string buffer((std::istreambuf_iterator<char>(ifs)),
                            std::istreambuf_iterator<char>());
        msgpack::unpacker pac;
        Node2Embedding n2e;
        pac.reserve_buffer( buffer.size() );
        std::copy( buffer.begin(), buffer.end(), pac.buffer() );
        pac.buffer_consumed( buffer.size() );

        msgpack::object_handle oh;
        while ( pac.next(oh) )
        {
            oh.get().convert(n2e);
            index = node2index.search_key(strdup(n2e.first.c_str()));
            if (index != -1)
            {
                counter++;
                for (int dim=0; dim!=this->dimension; dim++)
                    this->embedding[index][dim] = n2e.second[dim];
            }
        }
        ifs.close();
    }
    else
    {
        std::ifstream embedding_file;

        embedding_file.open(file_name);
        if (!embedding_file.is_open()) {
            throw("embedding file error");
        }
        while (std::getline(embedding_file, line)) {
            std::istringstream line_stream(line);
            std::getline(line_stream, node, '\t');
            index = node2index.search_key(strdup(node.c_str()));
            if (index != -1)
            {
                counter++;
                for (int d=0; d<dimension; ++d)
                {
                    line_stream >> value;
                    this->embedding[index][d] = std::stod(value);
                }
            }
        }
    }
    std::cout << "\t# matched node:\t" << counter << std::endl;
}

void LookupMapper::update(long index, std::vector<double>& loss_vector, double alpha) {
    for (int d=0; d<this->dimension; d++)
    {
        this->embedding[index][d] += alpha*loss_vector[d];
    }
}

void LookupMapper::update_with_l2(long index, std::vector<double>& loss_vector, double alpha, double lambda) {
    for (int d=0; d<this->dimension; d++)
    {
        this->embedding[index][d] += alpha*(loss_vector[d] - lambda*embedding[index][d]);
    }
}

void LookupMapper::update_with_weights_l2(long index, std::vector<double>& loss_vector, std::vector<double>& weights, double alpha, double lambda) {
    for (int d=0; d<this->dimension; d++)
    {
        this->embedding[index][d] += alpha*(loss_vector[d] - lambda*embedding[index][d]*weights[d]);
    }
}

// void LookupMapper::update_with_l2(long index, std::vector<double>& loss_vector, double alpha, double lambda, int start_dim, int end_dim) {
//     for (int d=start_dim; d<end_dim; d++)
//     {
//         this->embedding[index][d] += alpha*(loss_vector[d] - lambda*embedding[index][d]);
//     }
// }

void LookupMapper::save_to_file(std::vector<char*>& index2node, std::string file_name, int binary_out) {
    std::cout << "Save Mapper:" << std::endl;

    if (binary_out)
    {
        FILE* binary_file = fopen(file_name.c_str(), "wb");
        msgpack::sbuffer msg_buffer;
        for (long index=0; index!=this->size; index++)
        {
            msgpack::pack(msg_buffer,
                          std::make_pair(std::string(index2node[index]), this->embedding[index]));
        }
        std::fwrite(msg_buffer.data(), msg_buffer.size(), 1, binary_file);
        fclose(binary_file);
        std::cout << "\tSave to <" << file_name << "> in binary mode" << std::endl;
    }
    else
    {
        std::ofstream embedding_file(file_name);
        if (!embedding_file)
        {
            std::cout << "\tfail to open file" << std::endl;
        }
        else
        {
            embedding_file << std::setprecision(9);
            for (long index=0; index!=this->size; index++)
            {
                embedding_file << index2node[index];
                embedding_file << "\t" << embedding[index][0];
                for (int dim=1; dim!=this->dimension; dim++)
                {
                    embedding_file << " " << this->embedding[index][dim];
                }
                embedding_file << std::endl;
            }
            std::cout << "\tSave to <" << file_name << ">" << std::endl;
        }
    }
}


void LookupMapper::save_meta_avg_to_file(std::vector<char*>& index2node, int use_self, FileGraph* main_graph, int use_main, std::vector<FileGraph>& meta_graphs, int use_meta, std::vector<long> indexes, std::string file_name, int binary_out) {

    std::vector<double> meta_embedding(this->dimension, 0.0);
    std::vector<double> gcn_embedding(this->dimension, 0.0);
    long branch, from_index, to_index;
    double weight, weight_sum, num_meta;

    FILE* binary_file;
    msgpack::sbuffer msg_buffer;
    std::ofstream embedding_file;
    if (binary_out)
    {
        binary_file = fopen(file_name.c_str(), "wb");
    }
    else
    {
        embedding_file.open(file_name);
    }

    std::cout << "Save Mapper:" << std::endl;
    for (auto from_index: indexes)
    {
        gcn_embedding.assign(this->dimension, 0.0);
        num_meta = 0.0;

        // self
        if (use_self)
        {
            for (int dim=0; dim!=this->dimension; dim++)
            {
                gcn_embedding[dim] = this->embedding[from_index][dim];
            }
            num_meta++;
        }

        // main graph
        if (use_main)
        {
            branch = 0;
            if (main_graph->index_graph.find(from_index)!=main_graph->index_graph.end())
                branch = main_graph->index_graph[from_index].size();
            if (branch > 0)
            {
                // weight avg.
                weight_sum = 0.0;
                meta_embedding.assign(this->dimension, 0.0);
                for (auto it: main_graph->index_graph[from_index])
                {
                    to_index = it.first;
                    weight = main_graph->index_graph[from_index][to_index];
                    weight_sum += weight;
                    for (int dim=0; dim!=this->dimension; dim++)
                    {
                        meta_embedding[dim] += this->embedding[to_index][dim]*weight;
                    }
                }
                for (int dim=0; dim!=this->dimension; dim++)
                {
                    gcn_embedding[dim] += meta_embedding[dim]/weight_sum;
                }
                num_meta++;
            }
        }

        // meta
        if (use_meta)
        for (int meta_i=0; meta_i<meta_graphs.size(); meta_i++)
        {
            branch = meta_graphs[meta_i].index_graph[from_index].size();
            if (branch > 0)
            {
                // weight avg.
                weight_sum = 0.0;
                meta_embedding.assign(this->dimension, 0.0);
                for (auto it: meta_graphs[meta_i].index_graph[from_index])
                {
                    to_index = it.first;
                    weight = meta_graphs[meta_i].index_graph[from_index][to_index];
                    weight_sum += weight;
                    for (int dim=0; dim!=this->dimension; dim++)
                    {
                        meta_embedding[dim] += this->embedding[to_index][dim]*weight;
                    }
                }
                for (int dim=0; dim!=this->dimension; dim++)
                {
                    gcn_embedding[dim] += meta_embedding[dim]/weight_sum;
                }
                num_meta++;
            }
        }
        if (binary_out)
        {
            msgpack::pack(msg_buffer,
                          std::make_pair(std::string(index2node[from_index]), gcn_embedding));
        }
        else
        {
            embedding_file << index2node[from_index] << "\t";
            embedding_file << gcn_embedding[0];
            for (int dim=0; dim!=this->dimension; dim++)
                embedding_file << " " << gcn_embedding[dim];
            embedding_file << std::endl;
        }
    }

    if (binary_out)
    {
        std::fwrite(msg_buffer.data(), msg_buffer.size(), 1, binary_file);
        fclose(binary_file);
    }
    std::cout << "\tSave to <" << file_name << ">" << std::endl;
}


std::vector<double>& LookupMapper::operator[](long index) {
    return this->embedding[index];
}

std::vector<double> LookupMapper::avg_embedding(std::vector<long>& indexes) {
    double size = indexes.size();
    std::vector<double> avg_embedding(this->dimension, 0.0);
    for (auto index: indexes)
        for (int d=0; d<this->dimension; d++)
            avg_embedding[d] += this->embedding[index][d];
    for (int d=0; d<this->dimension; d++)
        avg_embedding[d] /= size;
    return avg_embedding;
}

std::vector<double> LookupMapper::weighted_embedding(std::vector<long>& indexes, std::vector<std::vector<double>>& weights) {
    double size = indexes.size();
    std::vector<double> avg_embedding(this->dimension, 0.0);
    for (int i=0; i<size; i++)
        for (int d=0; d<this->dimension; d++)
            avg_embedding[d] += this->embedding[i][d]*weights[i][d];
    for (int d=0; d<this->dimension; d++)
        avg_embedding[d] /= size;
    return avg_embedding;
}


std::vector<double> LookupMapper::textgcn_embedding(std::vector<long>& indexes) {
    std::vector<double> avg_embedding(this->dimension, 0.0);
    double size = indexes.size()-1;
    if (size)
    {
        for (auto it=++indexes.begin(); it!=indexes.end(); it++)
            for (int d=0; d<this->dimension; d++)
                avg_embedding[d] += this->embedding[*(it)][d];
        for (int d=0; d<this->dimension; d++)
        {
            avg_embedding[d] = (this->embedding[indexes[0]][d] + avg_embedding[d]/size)/2.0;
        }
    }
    else
    {
        for (int d=0; d<this->dimension; d++)
            avg_embedding[d] += this->embedding[indexes[0]][d];
    }
    return avg_embedding;
}

std::vector<double> LookupMapper::meta_gcn_embedding(std::vector<long>& indexes) {
    std::vector<double> meta_embedding(this->dimension, 0.0);
    double observed=0;
    for (auto it=++indexes.begin(); it!=indexes.end(); it++)
    {
        if (*(it)!=-1)
        {
            for (int d=0; d<this->dimension; d++)
                meta_embedding[d] += this->embedding[*(it)][d];
            observed++;
        }
    }

    if (observed)
    {
        for (int d=0; d<this->dimension; d++)
        {
            meta_embedding[d] = this->embedding[indexes[0]][d] + meta_embedding[d]/observed;
        }
    }
    else
    {
        for (int d=0; d<this->dimension; d++)
            meta_embedding[d] += this->embedding[indexes[0]][d];
    }
    return meta_embedding;
}

std::vector<double> LookupMapper::meta_avg_embedding(std::vector<long>& indexes) {
    std::vector<double> meta_embedding(this->dimension, 0.0);
    double observed=0;
    for (auto it=indexes.begin(); it!=indexes.end(); it++)
    {
        if (*(it)!=-1)
        {
            for (int d=0; d<this->dimension; d++)
                meta_embedding[d] += this->embedding[*(it)][d];
            observed++;
        }
    }
    //for (int d=0; d<this->dimension; d++)
    //    meta_embedding[d] /= observed;
    return meta_embedding;
}

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
                              std::string file_name) {

    std::set<long> user_set, item_set;
    for (auto graph: ui_graphs)
    {
        for (auto node: graph.get_all_from_nodes())
            user_set.insert(node);
        for (auto node: graph.get_all_to_nodes())
            item_set.insert(node);
    }

    for (int i=0; i<mapper_rs.size(); i++)
    {
        std::cout << "Save Embeddings:" << std::endl;
        std::ofstream embedding_file(file_name + '.' + std::to_string(i));

        if (!embedding_file)
        {
            std::cout << "\tfail to open file" << std::endl;
            continue;
        }

        std::cout << "\tsave to <" << file_name + '.' + std::to_string(i) << ">" << std::endl;
        embedding_file << std::setprecision(9);
        std::vector<double> neighbor_embedding(dimension, 0.0);
        std::vector<double> fused_embedding(dimension, 0.0);
        int branch;

        // for user nodes
        for (auto index: user_set)
        {
            fused_embedding.assign(dimension, 0.0);

            if (source_neighbors)
            {
                neighbor_embedding.assign(dimension, 0.0);
                branch = 0;
                if (ui_graphs[i].index_graph.find(index)!=ui_graphs[i].index_graph.end())
                    branch = ui_graphs[i].index_graph[index].size();
                if (branch > 0)
                {
                    // weight avg.
                    double weight_sum = 0.0;
                    for (auto it: ui_graphs[i].index_graph[index])
                    {
                        long to_index = it.first;
                        double weight = it.second;
                        weight_sum += weight;
                        for (int dim=0; dim!=dimension; dim++)
                        {
                            neighbor_embedding[dim] += (mapper_bs[to_index][dim]+mapper_rs[i][to_index][dim])*weight;
                        }
                    }
                    for (int dim=0; dim!=dimension; dim++)
                    {
                        fused_embedding[dim] += neighbor_embedding[dim]/weight_sum;
                    }
                }
            }
            /*
            for (int j=0; j<up_graphs.size(); j++)
            {
                neighbor_embedding.assign(dimension, 0.0);
                branch = 0;
                if (up_graphs[j].index_graph.find(index)!=up_graphs[j].index_graph.end())
                    branch = up_graphs[j].index_graph[index].size();
                if (branch > 0)
                {
                    // weight avg.
                    double weight_sum = 0.0;
                    for (auto it: up_graphs[j].index_graph[index])
                    {
                        long to_index = it.first;
                        double weight = up_graphs[j].index_graph[index][to_index];
                        weight_sum += weight;
                        for (int dim=0; dim!=dimension; dim++)
                        {
                            neighbor_embedding[dim] += (mapper_bs[to_index][dim]+mapper_rs[i][to_index][dim])*weight;
                        }
                    }
                    for (int dim=0; dim!=dimension; dim++)
                    {
                        fused_embedding[dim] += neighbor_embedding[dim]/weight_sum;
                    }
                }
            }
            */
            embedding_file << index2node[index];
            embedding_file << "\t";

            if (source_relation)
            {
                embedding_file << mapper_bs[index][0] + mapper_rs[i][index][0] + fused_embedding[0];
                for (int d=1; d!=dimension; d++)
                {
                    embedding_file << " ";
                    embedding_file << mapper_bs[index][d] + mapper_rs[i][index][d] + fused_embedding[d];
                }
            }
            else
            {
                embedding_file << mapper_bs[index][0] + fused_embedding[0];
                for (int d=1; d!=dimension; d++)
                {
                    embedding_file << " ";
                    embedding_file << mapper_bs[index][d] + fused_embedding[d];
                }
            }
            embedding_file << std::endl;
        }

        // for item nodes
        for (auto index: item_set)
        {
            fused_embedding.assign(dimension, 0.0);

            if (target_neighbors)
            {
                neighbor_embedding.assign(dimension, 0.0);
                branch = 0;
                if (iu_graphs[i].index_graph.find(index)!=iu_graphs[i].index_graph.end())
                    branch = iu_graphs[i].index_graph[index].size();
                if (branch > 0)
                {
                    // weight avg.
                    double weight_sum = 0.0;
                    for (auto it: iu_graphs[i].index_graph[index])
                    {
                        long to_index = it.first;
                        double weight = it.second;
                        weight_sum += weight;
                        for (int dim=0; dim!=dimension; dim++)
                        {
                            neighbor_embedding[dim] += (mapper_bs[to_index][dim]+mapper_rs[i][to_index][dim])*weight;
                        }
                    }
                    for (int dim=0; dim!=dimension; dim++)
                    {
                        fused_embedding[dim] += neighbor_embedding[dim]/weight_sum;
                    }
                }
            }
            /*
            for (int j=0; j<im_graphs.size(); j++)
            {
                neighbor_embedding.assign(dimension, 0.0);
                branch = 0;
                if (im_graphs[j].index_graph.find(index)!=im_graphs[j].index_graph.end())
                    branch = im_graphs[j].index_graph[index].size();
                if (branch > 0)
                {
                    // weight avg.
                    double weight_sum = 0.0;
                    for (auto it: im_graphs[j].index_graph[index])
                    {
                        long to_index = it.first;
                        double weight = im_graphs[j].index_graph[index][to_index];
                        weight_sum += weight;
                        for (int dim=0; dim!=dimension; dim++)
                        {
                            neighbor_embedding[dim] += (mapper_bs[to_index][dim]+mapper_rs[i][to_index][dim])*weight;
                        }
                    }
                    for (int dim=0; dim!=dimension; dim++)
                    {
                        fused_embedding[dim] += neighbor_embedding[dim]/weight_sum;
                    }
                }
            }
            */

            embedding_file << index2node[index];
            embedding_file << "\t";
            if (target_relation)
            {
                embedding_file << mapper_bs[index][0] + mapper_rt[i][index][0] + fused_embedding[0];
                for (int d=1; d!=dimension; d++)
                {
                    embedding_file << " ";
                    embedding_file << mapper_bs[index][d] + mapper_rt[i][index][d] + fused_embedding[d];
                }
            }
            else
            {
                embedding_file << mapper_bs[index][0] + fused_embedding[0];
                for (int d=1; d!=dimension; d++)
                {
                    embedding_file << " ";
                    embedding_file << mapper_bs[index][d] + fused_embedding[d];
                }
            }
            embedding_file << std::endl;
        }

        for (auto graph: im_graphs)
        {
            for (auto index: graph.get_all_to_nodes())
            {
                fused_embedding.assign(dimension, 0.0);
                embedding_file << index2node[index];
                embedding_file << "\t";
                if (target_relation)
                {
                    embedding_file << mapper_bs[index][0] + mapper_rt[i][index][0] + fused_embedding[0];
                    for (int d=1; d!=dimension; d++)
                    {
                        embedding_file << " ";
                        embedding_file << mapper_bs[index][d] + mapper_rt[i][index][d] + fused_embedding[d];
                    }
                }
                else
                {
                    embedding_file << mapper_bs[index][0] + fused_embedding[0];
                    for (int d=1; d!=dimension; d++)
                    {
                        embedding_file << " ";
                        embedding_file << mapper_bs[index][d] + fused_embedding[d];
                    }
                }
                embedding_file << std::endl;
            }
        }

    }
}


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
                                    std::string file_name) {

    std::set<long> user_set, item_set;
    for (auto graph: ui_graphs)
    {
        for (auto node: graph.get_all_from_nodes())
            user_set.insert(node);
        for (auto node: graph.get_all_to_nodes())
            item_set.insert(node);
    }

    for (int i=0; i<mapper_rs.size(); i++)
    {
        std::cout << "Save Embeddings:" << std::endl;
        std::ofstream embedding_file(file_name + '.' + std::to_string(i));

        if (!embedding_file)
        {
            std::cout << "\tfail to open file" << std::endl;
            continue;
        }

        std::cout << "\tsave to <" << file_name + '.' + std::to_string(i) << ">" << std::endl;
        embedding_file << std::setprecision(9);
        std::vector<double> neighbor_embedding(dimension, 0.0);
        std::vector<double> fused_embedding(dimension, 0.0);
        int branch;

        // for user nodes
        for (auto index: user_set)
        {
            fused_embedding.assign(dimension, 0.0);

            if (source_neighbors)
            {
                neighbor_embedding.assign(dimension, 0.0);
                branch = 0;
                if (ui_graphs[i].index_graph.find(index)!=ui_graphs[i].index_graph.end())
                    branch = ui_graphs[i].index_graph[index].size();
                if (branch > 0)
                {
                    // weight avg.
                    double weight_sum = 0.0;
                    for (auto it: ui_graphs[i].index_graph[index])
                    {
                        long to_index = it.first;
                        double weight = it.second;
                        weight_sum += weight;
                        for (int dim=0; dim!=dimension; dim++)
                        {
                            neighbor_embedding[dim] += (mapper_bs[to_index][dim]+mapper_rs[i][to_index][dim])*weight;
                        }
                    }
                    for (int dim=0; dim!=dimension; dim++)
                    {
                        fused_embedding[dim] += neighbor_embedding[dim]/weight_sum;
                    }
                }
            }
            /*
            for (int j=0; j<up_graphs.size(); j++)
            {
                neighbor_embedding.assign(dimension, 0.0);
                branch = 0;
                if (up_graphs[j].index_graph.find(index)!=up_graphs[j].index_graph.end())
                    branch = up_graphs[j].index_graph[index].size();
                if (branch > 0)
                {
                    // weight avg.
                    double weight_sum = 0.0;
                    for (auto it: up_graphs[j].index_graph[index])
                    {
                        long to_index = it.first;
                        double weight = up_graphs[j].index_graph[index][to_index];
                        weight_sum += weight;
                        for (int dim=0; dim!=dimension; dim++)
                        {
                            neighbor_embedding[dim] += (mapper_bs[to_index][dim]+mapper_rs[i][to_index][dim])*weight;
                        }
                    }
                    for (int dim=0; dim!=dimension; dim++)
                    {
                        fused_embedding[dim] += neighbor_embedding[dim]/weight_sum;
                    }
                }
            }
            */
            embedding_file << index2node[index];
            embedding_file << "\t";

            if (source_relation)
            {
                embedding_file << mapper_bs[index][0] + mapper_rs[i][0][0] + fused_embedding[0];
                for (int d=1; d!=dimension; d++)
                {
                    embedding_file << " ";
                    embedding_file << mapper_bs[index][d] + mapper_rs[i][0][d] + fused_embedding[d];
                }
            }
            else
            {
                embedding_file << mapper_bs[index][0] + fused_embedding[0];
                for (int d=1; d!=dimension; d++)
                {
                    embedding_file << " ";
                    embedding_file << mapper_bs[index][d] + fused_embedding[d];
                }
            }
            embedding_file << std::endl;
        }

        // for item nodes
        for (auto index: item_set)
        {
            fused_embedding.assign(dimension, 0.0);

            if (target_neighbors)
            {
                neighbor_embedding.assign(dimension, 0.0);
                branch = 0;
                if (iu_graphs[i].index_graph.find(index)!=iu_graphs[i].index_graph.end())
                    branch = iu_graphs[i].index_graph[index].size();
                if (branch > 0)
                {
                    // weight avg.
                    double weight_sum = 0.0;
                    for (auto it: iu_graphs[i].index_graph[index])
                    {
                        long to_index = it.first;
                        double weight = it.second;
                        weight_sum += weight;
                        for (int dim=0; dim!=dimension; dim++)
                        {
                            neighbor_embedding[dim] += (mapper_bs[to_index][dim]+mapper_rs[i][0][dim])*weight;
                        }
                    }
                    for (int dim=0; dim!=dimension; dim++)
                    {
                        fused_embedding[dim] += neighbor_embedding[dim]/weight_sum;
                    }
                }
            }
            /*
            for (int j=0; j<im_graphs.size(); j++)
            {
                neighbor_embedding.assign(dimension, 0.0);
                branch = 0;
                if (im_graphs[j].index_graph.find(index)!=im_graphs[j].index_graph.end())
                    branch = im_graphs[j].index_graph[index].size();
                if (branch > 0)
                {
                    // weight avg.
                    double weight_sum = 0.0;
                    for (auto it: im_graphs[j].index_graph[index])
                    {
                        long to_index = it.first;
                        double weight = im_graphs[j].index_graph[index][to_index];
                        weight_sum += weight;
                        for (int dim=0; dim!=dimension; dim++)
                        {
                            neighbor_embedding[dim] += (mapper_bs[to_index][dim]+mapper_rs[i][to_index][dim])*weight;
                        }
                    }
                    for (int dim=0; dim!=dimension; dim++)
                    {
                        fused_embedding[dim] += neighbor_embedding[dim]/weight_sum;
                    }
                }
            }
            */

            embedding_file << index2node[index];
            embedding_file << "\t";
            if (target_relation)
            {
                embedding_file << mapper_bs[index][0] + fused_embedding[0];
                for (int d=1; d!=dimension; d++)
                {
                    embedding_file << " ";
                    embedding_file << mapper_bs[index][d] + fused_embedding[d];
                }
            }
            else
            {
                embedding_file << mapper_bs[index][0] + fused_embedding[0];
                for (int d=1; d!=dimension; d++)
                {
                    embedding_file << " ";
                    embedding_file << mapper_bs[index][d] + fused_embedding[d];
                }
            }
            embedding_file << std::endl;
        }

        for (auto graph: im_graphs)
        {
            for (auto index: graph.get_all_to_nodes())
            {
                fused_embedding.assign(dimension, 0.0);
                embedding_file << index2node[index];
                embedding_file << "\t";
                if (target_relation)
                {
                    embedding_file << mapper_bs[index][0] + fused_embedding[0];
                    for (int d=1; d!=dimension; d++)
                    {
                        embedding_file << " ";
                        embedding_file << mapper_bs[index][d] + fused_embedding[d];
                    }
                }
                else
                {
                    embedding_file << mapper_bs[index][0] + fused_embedding[0];
                    for (int d=1; d!=dimension; d++)
                    {
                        embedding_file << " ";
                        embedding_file << mapper_bs[index][d] + fused_embedding[d];
                    }
                }
                embedding_file << std::endl;
            }
        }

    }
}