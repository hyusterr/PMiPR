#define _GLIBCXX_USE_CXX11_ABI 1
#include <omp.h>
#include <sstream>
#include "../src/util/util.h"                       // arguments
#include "../src/util/file_graph.h"                 // graph
#include "../src/sampler/alias_methods.h"           // sampler
#include "../src/sampler/vc_sampler.h"              // sampler
#include "../src/mapper/lookup_mapper.h"            // mapper
#include "../src/optimizer/quadruple_optimizer.h"   // optimizer

int main(int argc, char **argv){

    // arguments
    ArgParser arg_parser(argc, argv);
    std::string train_ui_path = arg_parser.get_str("-train_ui", "", "input graph path");
    std::string train_up_path = arg_parser.get_str("-train_up", "", "input graph path");
    std::string train_im_path = arg_parser.get_str("-train_im", "", "input graph path");
    std::string save_name = arg_parser.get_str("-save", "air.embed", "path for saving mapper");
    int dimension = arg_parser.get_int("-dimension", 64, "embedding dimension");
    int num_negative = arg_parser.get_int("-num_negative", 5, "number of negative sample");
    int report_period = arg_parser.get_int("-report_period", 1, "period of progress report");
    int binary_out = arg_parser.get_int("-binary_out", 0, "whether to output embeddings in binary mode");
    double margin = arg_parser.get_double("-margin", 0.0, "margin of bpr");
    double update_times = arg_parser.get_double("-update_times", 10, "update times (*million)");
    double init_alpha = arg_parser.get_double("-init_alpha", 0.025, "init learning rate");
    double l2_reg = arg_parser.get_double("-l2_reg", 0.0025, "l2 regularization");
    int worker = arg_parser.get_int("-worker", 1, "number of worker (thread)");

    if (argc == 1) {
        return 0;
    }

    // 0. [FileGraph] read graph
    std::vector<char*> index2node;
    std::vector<FileGraph> ui_file_graph, iu_file_graph;
    std::vector<FileGraph> up_file_graph, pu_file_graph;
    std::vector<FileGraph> im_file_graph, mi_file_graph;
    std::stringstream ss;
    std::string sub_path;

    std::cout << "(UI-Graph)" << std::endl;
    ss << train_ui_path;
    while(ss.good())
    {
        getline(ss, sub_path, ',');
        std::cout << "[INFO] Load from <" << sub_path << ">" << std::endl;
        ui_file_graph.push_back(FileGraph(sub_path, 0, index2node));
        iu_file_graph.push_back(FileGraph(sub_path, -1, index2node));
    }
    ss.clear();

    if (!train_up_path.empty())
    {
        std::cout << "(UP-Graph)" << std::endl;
        ss << train_up_path;
        while(ss.good())
        {
            getline(ss, sub_path, ',');
            up_file_graph.push_back(FileGraph(sub_path, 0, index2node));
            //pu_file_graph.push_back(FileGraph(sub_path, -1, index2node));
        }
        ss.clear();
    }

    if (!train_im_path.empty())
    {
        std::cout << "(IM-Graph)" << std::endl;
        ss << train_im_path;
        while(ss.good())
        {
            getline(ss, sub_path, ',');
            im_file_graph.push_back(FileGraph(sub_path, 0, index2node));
            //mi_file_graph.push_back(FileGraph(sub_path, 0, index2node));
        }
    }

    // 1. [Sampler] determine what sampler to be used
    std::vector<VCSampler> ui_sampler, iu_sampler, up_sampler, im_sampler;
    for (auto graph: ui_file_graph)
        ui_sampler.push_back(VCSampler(&graph));
    for (auto graph: iu_file_graph)
        iu_sampler.push_back(VCSampler(&graph));
    for (auto graph: up_file_graph)
        up_sampler.push_back(VCSampler(&graph));
    for (auto graph: im_file_graph)
        im_sampler.push_back(VCSampler(&graph));

    // 1-2. Store the distriution among diffeerent ui graphs
    std::vector<double> type_dist;
    for (auto graph: ui_file_graph)
        type_dist.push_back(graph.edge_size);
    AliasMethods type_sampler;
    type_sampler.append(type_dist, 1.0);

    // 2. [Mapper] define what embedding mapper to be used
    LookupMapper mapper_bs(index2node.size(), dimension);
    //LookupMapper mapper_bt(index2node.size(), dimension);
    std::vector<LookupMapper> mapper_rs, mapper_rt;
    for (auto graph: ui_file_graph)
    {
        mapper_rs.push_back(LookupMapper(index2node.size(), dimension));
    }

    // 3. [Optimizer] claim the optimizer
    QuadrupleOptimizer optimizer;

    // 4. building the blocks [MF]
    std::cout << "Start Training:" << std::endl;
    long worker_update_times = 1000000/worker;
    Monitor monitor(update_times);

    omp_set_num_threads(worker);
    for (int u=0; u<update_times; u++)
    {
        double G = 0;
        #pragma omp parallel for
        for (int w=0; w<worker; w++)
        {
            long user, item, feat, pos, neg;
            int type, neg_type;
            std::vector<long> user_nodes, item_nodes, feat_nodes, pos_nodes, neg_nodes;
            std::vector<double> user_loss(dimension, 0.0),
                                item_loss(dimension, 0.0),
                                pos_loss(dimension, 0.0),
                                neg_loss(dimension, 0.0),
                                feat_loss(dimension, 0.0);

            double alpha = init_alpha* ( 1.0 - (double)(u)/update_times );
            double g=0;
            if (alpha < init_alpha*0.0001)
                alpha = init_alpha*0.0001;

            for (int _u=0; _u<worker_update_times; _u++)
            {
                type = type_sampler.draw();

                user = ui_sampler[type].draw_a_vertex();
                user_nodes.clear();
                user_nodes.push_back(user);

                for (int n=0; n<num_negative; n++)
                {
                    item_nodes.clear();
                    item = ui_sampler[type].draw_a_context(user);
                    item_nodes.push_back(item);

                    pos_nodes.clear();
                    neg_nodes.clear();
                    for (int i=0; i<up_sampler.size(); i++)
                    {
                        for (int k=0; k<1; k++)
                        {
                            pos = up_sampler[i].draw_a_context_safely(user);
                            if (pos==-1)
                                break;
                            neg = up_sampler[i].draw_a_context_uniformly();
                            if (neg==-1)
                                break;
                            pos_nodes.push_back(pos);
                            neg_nodes.push_back(neg);
                        }
                    }
                    for (int i=0; i<im_sampler.size(); i++)
                    {
                        for (int k=0; k<1; k++)
                        {
                            pos = im_sampler[i].draw_a_context_safely(item);
                            if (pos==-1)
                                break;
                            neg = im_sampler[i].draw_a_context_uniformly();
                            if (neg==-1)
                                break;
                            pos_nodes.push_back(pos);
                            neg_nodes.push_back(neg);
                        }
                    }
                    if (pos_nodes.size())
                    {

                    g += optimizer.feed_relational_bpr_loss(mapper_bs,
                                                            mapper_bs,
                                                            mapper_rs[type],
                                                            mapper_rs[type],
                                                            user_nodes,
                                                            item_nodes,
                                                            pos_nodes,
                                                            neg_nodes,
                                                            dimension,
                                                            user_loss,
                                                            item_loss,
                                                            pos_loss,
                                                            neg_loss);
                    for (int i=0; i<pos_nodes.size(); i++)
                    {
                        mapper_bs.update_with_l2(pos_nodes[i], pos_loss, alpha, l2_reg);
                        mapper_rs[type].update_with_l2(pos_nodes[i], pos_loss, alpha, l2_reg);
                    }
                    for (int i=0; i<neg_nodes.size(); i++)
                    {
                        mapper_bs.update_with_l2(neg_nodes[i], neg_loss, alpha, l2_reg);
                        mapper_rs[type].update_with_l2(neg_nodes[i], neg_loss, alpha, l2_reg);
                    }
                    pos_loss.assign(dimension, 0.0);
                    neg_loss.assign(dimension, 0.0);

                    for (int d=0; d!=dimension; d++)
                    {
                        user_loss[d] *= 0.1;
                        item_loss[d] *= 0.1;
                    }
                    }

                    pos_nodes.clear();
                    neg_nodes.clear();
                    neg_type = type_sampler.draw();
                    pos_nodes.push_back( ui_sampler[type].draw_a_context(user) );
                    pos_nodes.push_back( iu_sampler[type].draw_a_context(item) );
                    neg_nodes.push_back( ui_sampler[neg_type].draw_a_context_uniformly() );
                    neg_nodes.push_back( iu_sampler[neg_type].draw_a_context(neg_nodes[0]) );

                    g += optimizer.feed_relational_bpr_loss(mapper_bs,
                                                            mapper_bs,
                                                            mapper_rs[type],
                                                            mapper_rs[type],
                                                            user_nodes,
                                                            item_nodes,
                                                            pos_nodes,
                                                            neg_nodes,
                                                            dimension,
                                                            user_loss,
                                                            item_loss,
                                                            pos_loss,
                                                            neg_loss);
                    for (int i=0; i<pos_nodes.size(); i++)
                    {
                        mapper_bs.update_with_l2(pos_nodes[i], pos_loss, alpha, l2_reg);
                        mapper_rs[type].update_with_l2(pos_nodes[i], pos_loss, alpha, l2_reg);
                    }
                    for (int i=0; i<neg_nodes.size(); i++)
                    {
                        mapper_bs.update_with_l2(neg_nodes[i], neg_loss, alpha, l2_reg);
                        mapper_rs[type].update_with_l2(neg_nodes[i], neg_loss, alpha, l2_reg);
                    }
                    pos_loss.assign(dimension, 0.0);
                    neg_loss.assign(dimension, 0.0);

                    for (int i=0; i<item_nodes.size(); i++)
                    {
                        mapper_bs.update_with_l2(item_nodes[i], item_loss, alpha, l2_reg);
                        mapper_rs[type].update_with_l2(item_nodes[i], item_loss, alpha, l2_reg);
                    }
                    item_loss.assign(dimension, 0.0);
                }

                for (int i=0; i<user_nodes.size(); i++)
                {
                    mapper_bs.update_with_l2(user_nodes[i], user_loss, alpha, l2_reg);
                    mapper_rs[type].update_with_l2(user_nodes[i], user_loss, alpha, l2_reg);
                }
                user_loss.assign(dimension, 0.0);

            }
            G += g/worker_update_times;
        }
        if (u % report_period == 0)
            monitor.progress(u, G/worker);
    }
    monitor.end();

    dump_air_plus_embeddings(mapper_bs,
                             mapper_rs,
                             mapper_rs,
                             1,
                             0,
                             1,
                             0,
                             dimension,
                             index2node,
                             ui_file_graph,
                             iu_file_graph,
                             up_file_graph,
                             im_file_graph,
                             save_name);
    return 0;
}
