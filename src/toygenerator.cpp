#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <random>

#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

#include "tqdm.h"
#include "toygenerator.cuh"
#include "ta_utilities.hpp"
#include "equations.hpp"

using std::cerr;
using std::cout;
using std::endl;


/*
Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}


void check_args(int argc, char **argv){
    
    #if GOF
        if (argc != 6){
            std::cerr << "Incorrect number of arguments.\n";
            std::cerr << "Arguments: <number of bins> <background template file> <observed data file> <number of toys> <output file>\n";
            exit(EXIT_FAILURE);
        }
    #else
        if (argc != 7){
            std::cerr << "Incorrect number of arguments.\n";
            std::cerr << "Arguments: <number of bins> <background template file> <signal template file> <observed data file> <number of toys> <output file>\n";
            exit(EXIT_FAILURE);
        }
    #endif
}


int frequentist_test(int argc, char **argv){

    check_args(argc, argv);
    
    cudaEvent_t start;
    cudaEvent_t stop;
    float cpu_time_ms = -1;
    float gpu_time_ms = -1;

    #define START_TIMER() {                         \
          gpuErrchk(cudaEventCreate(&start));       \
          gpuErrchk(cudaEventCreate(&stop));        \
          gpuErrchk(cudaEventRecord(start));        \
        }

    #define STOP_RECORD_TIMER(name) {                           \
          gpuErrchk(cudaEventRecord(stop));                     \
          gpuErrchk(cudaEventSynchronize(stop));                \
          gpuErrchk(cudaEventElapsedTime(&name, start, stop));  \
          gpuErrchk(cudaEventDestroy(start));                   \
          gpuErrchk(cudaEventDestroy(stop));                    \
        }


    // ******************************************************
    //                GETTING INPUTS
    // ******************************************************
    

    #if GOF == 0
        std::string out_filename = argv[6];
        float ntoys = std::stof(argv[5]); // number of toy experiments
        std::string obs_filename = argv[4];
        std::string sig_filename = argv[3];
    #else
        std::string out_filename = argv[5];
        float ntoys = std::stof(argv[4]); // number of toy experiments
        std::string obs_filename = argv[3];
    #endif

    static const int n_bins = std::stoi(argv[1]);

    std::ifstream bkg_file(argv[2]);
    float *bkg_expected = (float*) malloc(sizeof(float) * n_bins);

    if (bkg_file.is_open())
    {
        std::cout << "Reading " << n_bins << " bins from background file " << argv[2] << std::endl;
        for (int i =0; i < n_bins; i++)
            bkg_file >> bkg_expected[i];
    }
    else
    {
        std::cerr << "Cannot open background template file, exiting\n";
        exit(EXIT_FAILURE);
    }

    std::ifstream obs_file(obs_filename);
    float *obs_data = (float*) malloc(sizeof(float) * n_bins);
    if (obs_file.is_open())
    {
        std::cout << "Reading " << n_bins << " bins from data file " << argv[2] << std::endl;
        for (int i =0; i < n_bins; i++)
            obs_file >> obs_data[i];
    }
    else
    {
        std::cerr << "Cannot open observed data file, exiting\n";
        exit(EXIT_FAILURE);
    }
    #if GOF==0
        std::ifstream sig_file(sig_filename);
        float *sig_expected = (float*) malloc(sizeof(float) * n_bins);
        if (sig_file.is_open())
        {
            std::cout << "Reading " << n_bins << " bins from signal file " << argv[2] << std::endl;
            for (int i =0; i < n_bins; i++)
                sig_file >> sig_expected[i];
        }
        else
        {
            std::cerr << "Cannot open signal template file, exiting\n";
            exit(EXIT_FAILURE);
        }

    #endif

    // ******************************************************
    //                CPU IMPLEMENTATION
    // ******************************************************
        
    START_TIMER();
    
    float q_obs = 0;

    // Compute test statistics for the observed data
    #if GOF == 0
        float denominator, numerator;
    #endif
        for (int i = 0; i < n_bins; i++)
        {
    #if GOF
            q_obs += equations::chisquare(bkg_expected[i], obs_data[i]);
    #else
            denominator = equations::log_poisson(bkg_expected[i], obs_data[i]);
            numerator = equations::log_poisson(sig_expected[i]+bkg_expected[i], obs_data[i]);
            q_obs += -2 * numerator/denominator;
    #endif
    }

    // Generate toys 
    std::default_random_engine generator;

    // Create Poisson distribution for each bin
    std::vector<std::poisson_distribution<int>> distributions;
    for (int i = 0; i < n_bins; i++)
    {
        distributions.push_back(std::poisson_distribution<int>(bkg_expected[i]));
    }
    
    // Generating toy Monte Carlo for test statistics
    int toy;
    int larger = 0;
    float *q_toys = (float*) malloc(sizeof(float) * ntoys);
    std::cout << "Generating " << ntoys << " toy experiments to obtain the test statistics distribution" << std::endl;
    tqdm bar;
    for (int experiment = 0; experiment < ntoys; experiment++)
    {
        bar.progress(experiment, ntoys);
        q_toys[experiment] = 0;
        for (int bin = 0; bin < n_bins; bin++)
        {
            toy = distributions[bin](generator);
            #if GOF
                q_toys[experiment] += equations::chisquare(bkg_expected[bin], toy);
            #else
                denominator = equations::log_poisson(bkg_expected[bin], toy);
                numerator = equations::log_poisson(sig_expected[bin]+bkg_expected[bin], toy);
                q_toys[experiment] += -2 * numerator/denominator;
            #endif
        }
        if (q_toys[experiment] > q_obs) larger++;
    }
    bar.finish();
    STOP_RECORD_TIMER(cpu_time_ms);


    // ******************************************************
    //                GPU IMPLEMENTATION
    // ******************************************************
    float *dev_bkg_expected;
    gpuErrchk(cudaMalloc((void **) &dev_bkg_expected, n_bins * sizeof(float)));
    float *dev_obs_data;
    gpuErrchk(cudaMalloc((void **) &dev_obs_data, n_bins * sizeof(float)));
    float *dev_q_toys;
    gpuErrchk(cudaMalloc((void **) &dev_q_toys, ntoys * sizeof(float)));

    #if GOF==0
        float *dev_sig_expected;
        gpuErrchk(cudaMalloc((void **) &dev_sig_expected, n_bins * sizeof(float)));
    #endif
    
    START_TIMER();
    gpuErrchk(cudaMemcpy(dev_bkg_expected, bkg_expected, 
            n_bins * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_obs_data, obs_data, 
            n_bins * sizeof(float), cudaMemcpyHostToDevice));
    #if GOF==0
    gpuErrchk(cudaMemcpy(dev_sig_expected, sig_expected, 
            n_bins * sizeof(float), cudaMemcpyHostToDevice));
    #endif
    
    gpuErrchk(cudaMemset(dev_q_toys, 0, 
                ntoys * sizeof(float)));


    STOP_RECORD_TIMER(gpu_time_ms);

    // ******************************************************
    //            PRINTOUT AND SAVE RESULTS
    // ******************************************************

    float pval = float(larger)/ntoys;
    #if GOF
        std::cout << "p-value from Goodness-of-fit test: ";
    #else
        std::cout << "p-value from Neyman-Pearson hypothesis test: ";
    #endif
    if (larger == 0)
        std::cout << "less than " << 1/ntoys << std::endl;
    else
        std::cout << pval << std::endl;

    out_filename.append(".cpu");
    std::cout << "Saving the toy experiments' test statistics to " << out_filename << std::endl;
    std::ofstream fileSave;
    fileSave.open(out_filename);
    tqdm bar2;
    for (int i = 0; i < ntoys; i++)
    {
        bar2.progress(i, ntoys);
        fileSave << q_toys[i] << "\n";
    }
    bar2.finish();
    return EXIT_SUCCESS;

}


int main(int argc, char **argv){
    TA_Utilities::select_coldest_GPU();
    //int max_time_allowed_in_seconds = 500;
    //TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);
    return frequentist_test(argc, argv);
}


