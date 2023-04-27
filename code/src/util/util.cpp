#include "util.h"

void load_index2step(std::unordered_map<long, int>& index2step, n2iHash& node2index, std::string path) {
    FILE *fin;
    char c_line[1000];
    unsigned long long num_lines = 0;

    std::cout << "Lines Preview (Index2Step):" << std::endl;
    fin = fopen(path.c_str(), "rb");
    while (fgets(c_line, sizeof(c_line), fin))
    {
        if (num_lines % 10000 == 0)
        {
            printf("\t# of lines:\t\t%lld%c", num_lines, 13);
        }
        ++num_lines;
    }
    printf("\t# of lines:\t\t%lld\n", num_lines);
    fclose(fin);

    std::cout << "Loading Lines (Index2Step):" << std::endl;
    fin = fopen(path.c_str(), "rb");
    char node[256];
    long node_index;
    int step;
    unsigned long long line = 0;
    for (; line != num_lines; line++)
    {
         if ( fscanf(fin, "%[^\t]\t%d\n", node, &step) != 2 )
         {
             std::cout << "\t[WARNING] skip line " << line << std::endl;
             continue;
         }
         if (line % 10000 == 0)
         {
             printf("\tProgress:\t\t%.2f %%%c", (double)(line)/(num_lines), 13);
             fflush(stdout);
         }
         node_index = node2index.search_key(node);
         if (node_index == -1)
         {
             continue;
         }
         index2step[node_index] = step;
    }
    std::cout << "\tProgress:\t\t100.00 %\r" << std::endl;
    std::cout << "\t# of matched node:\t" << index2step.size() << std::endl;
}


ArgParser::ArgParser(int argc, char** argv) {
    this->argc = argc;
    this->argv = argv;
    std::cout << "Parse Arguments:" << std::endl;
}

int ArgParser::get_int(std::string flag, int value, std::string description) {
    for (int a=1; a<this->argc; a++)
        if (!strcmp(flag.c_str(), this->argv[a]))
        {
            std::cout << "\t" << flag << " " << atoi(argv[a+1]);
            std::cout << " (" << description << ")" << std::endl;
            return atoi(argv[a+1]);
        }
    std::cout << "\t" << flag << " " << value;
    std::cout << " (" << description << ")" << std::endl;
    return value;
}

double ArgParser::get_double(std::string flag, double value, std::string description) {
    for (int a=1; a<this->argc; a++)
        if (!strcmp(flag.c_str(), this->argv[a]))
        {
            std::cout << "\t" << flag << " " << atof(argv[a+1]);
            std::cout << " (" << description << ")" << std::endl;
            return atof(argv[a+1]);
        }
    std::cout << "\t" << flag << " " << value;
    std::cout << " (" << description << ")" << std::endl;
    return value;
}

std::string ArgParser::get_str(std::string flag, std::string value, std::string description) {
    for (int a=1; a<this->argc; a++)
        if (!strcmp(flag.c_str(), this->argv[a]))
        {
            std::cout << "\t" << flag << " " << argv[a+1];
            std::cout << " (" << description << ")" << std::endl;
            return argv[a+1];
        }
    std::cout << "\t" << flag << " " << value;
    std::cout << " (" << description << ")" << std::endl;
    return value;
}


int is_directory(std::string path) {
    struct stat info;
    if( stat( path.c_str(), &info ) != 0 ) // nothing
        return -1;
    else if( info.st_mode & S_IFDIR ) // a folder
        return 1;
    return 0; // a file
}

double dot_similarity(std::vector<double>& embeddingA, std::vector<double>& embeddingB, int dimension) {
    double prediction=0;
    for (int d=0; d<dimension; d++)
    {
        prediction += embeddingA[d]*embeddingB[d];
    }
    return prediction;
}

Monitor::Monitor(unsigned long long total_step) {
    this->total_step = total_step;
}
Monitor::Monitor(double total_step) {
    this->_total_step = total_step;
}

void Monitor::progress(unsigned long long* current_step) {
    printf("\tProgress:\t%.3f %%%c", (double)*current_step/this->total_step*100.0, 13);
    fflush(stdout);
}
void Monitor::progress(int current_step, double gradient) {
    printf("\tPROG. / GRAD.:\t%.3f %% / %.3f %c", (double)current_step/this->_total_step*100.0, gradient, 13);
    fflush(stdout);
}

void Monitor::end() {
    printf("\tProgress:\t%.3f %%\n", 100.0);
    fflush(stdout);
}

void Monitor::end(double gradient) {
    printf("\tPROG. / GRAD.:\t%.3f %% / %.3f\n", 100.0, gradient);
    fflush(stdout);
}

