#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <iostream>
#include <random>

#include <algorithm>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "tqdm.h"
#include "toygenerator.cuh"
#include "equations.hpp"
#include "helper_cuda.h"

using std::cerr;
using std::cout;
using std::endl;

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
          CUDA_CALL(cudaEventCreate(&start));       \
          CUDA_CALL(cudaEventCreate(&stop));        \
          CUDA_CALL(cudaEventRecord(start));        \
        }

    #define STOP_RECORD_TIMER(name) {                           \
          CUDA_CALL(cudaEventRecord(stop));                     \
          CUDA_CALL(cudaEventSynchronize(stop));                \
          CUDA_CALL(cudaEventElapsedTime(&name, start, stop));  \
          CUDA_CALL(cudaEventDestroy(start));                   \
          CUDA_CALL(cudaEventDestroy(stop));                    \
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
    float *q_toys = (float*) malloc(sizeof(float) * ntoys);
    std::cout << "Generating " << ntoys << " toy experiments to obtain the test statistics distribution on CPU" << std::endl;
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
    }
    bar.finish();
    STOP_RECORD_TIMER(cpu_time_ms);


    // ******************************************************
    //                GPU IMPLEMENTATION
    // ******************************************************
    
    // Allocate space for input data on device
    float *dev_bkg_expected;
    CUDA_CALL(cudaMalloc((void **) &dev_bkg_expected, n_bins * sizeof(float)));
    float *dev_obs_data;
    CUDA_CALL(cudaMalloc((void **) &dev_obs_data, n_bins * sizeof(float)));
    float *dev_q_toys;
    CUDA_CALL(cudaMalloc((void **) &dev_q_toys, ntoys * sizeof(float)));
    curandState *devStates;
    float *host_q_toys = (float*) malloc(sizeof(float) * ntoys);


    #if GOF==0
        float *dev_sig_expected;
        CUDA_CALL(cudaMalloc((void **) &dev_sig_expected, n_bins * sizeof(float)));
    #endif
    
    // Getting CUDA attributes
    int device;
    CUDA_CALL(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, device));
    int threadsPerBlock = prop.maxThreadsPerBlock;
    int *maxGridSize = prop.maxGridSize;

    // Determine number of blocks 
    int nBlocks = ceil(ntoys/threadsPerBlock);
    int trialsPerThread = 1;
    if (nBlocks > maxGridSize[0]) 
    {
        trialsPerThread = ceil(nBlocks/maxGridSize[0]);
        nBlocks = maxGridSize[0];
    }

    // Allocate space for prng states on device
    CUDA_CALL(cudaMalloc((void **) &devStates, ntoys * sizeof(curandState)));
    
    // Generating toys
    START_TIMER();
    
    // Copy input data from host to device
    CUDA_CALL(cudaMemcpy(dev_bkg_expected, bkg_expected, 
            n_bins * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dev_obs_data, obs_data, 
            n_bins * sizeof(float), cudaMemcpyHostToDevice));
    #if GOF==0
    CUDA_CALL(cudaMemcpy(dev_sig_expected, sig_expected, 
            n_bins * sizeof(float), cudaMemcpyHostToDevice));
    #endif
    
    // Set the result arrays to zero
    CUDA_CALL(cudaMemset(dev_q_toys, 0, 
                ntoys * sizeof(float)));

    #if GOF
    cuda_call_generate_goodness_of_fit_toys(nBlocks,
                                            threadsPerBlock,
                                            dev_bkg_expected, 
                                            dev_obs_data,
                                            dev_q_toys,
                                            n_bins,
                                            ntoys,
                                            devStates,
                                            trialsPerThread);
    #else
    cuda_call_generate_neyman_pearson_toys(nBlocks,
                                           threadsPerBlock,
                                           dev_bkg_expected, 
                                           dev_sig_expected,
                                           dev_obs_data,
                                           dev_q_toys,
                                           n_bins,
                                           ntoys,
                                           devStates,
                                           trialsPerThread);
    
    #endif

    // Copy result back to host
    CUDA_CALL(cudaMemcpy(host_q_toys, dev_q_toys, ntoys * sizeof(float), cudaMemcpyDeviceToHost));

    STOP_RECORD_TIMER(gpu_time_ms);

    // ******************************************************
    //            COMPARE AND SAVE RESULTS
    // ******************************************************
    int larger_cpu = 0;
    int larger_gpu = 0;
    std::string out_cpu = out_filename;
    std::string out_gpu = out_filename;
    out_cpu.append(".cpu");
    out_gpu.append(".gpu");
    std::cout << "Saving the toy experiments' test statistics to " << out_cpu << " and " << out_gpu << std::endl;
    std::ofstream file_cpu, file_gpu;
    file_cpu.open(out_cpu);
    file_gpu.open(out_gpu);
    tqdm bar2;
    for (int i = 0; i < ntoys; i++)
    {
        bar2.progress(i, ntoys);
        file_cpu << q_toys[i] << "\n";
        file_gpu << host_q_toys[i] << "\n";
        if (q_toys[i] > q_obs) larger_cpu++;
        if (host_q_toys[i] > q_obs) larger_gpu++;
    }
    bar2.finish();

    float pval_cpu = float(larger_cpu)/ntoys;
    float pval_gpu = float(larger_gpu)/ntoys;
    #if GOF
        std::cout << "p-value from Goodness-of-fit test: ";
    #else
        std::cout << "p-value from Neyman-Pearson hypothesis test: ";
    #endif
    if (larger_cpu == 0)
        std::cout << "less than " << 1/ntoys << " (CPU), ";
    else
        std::cout << pval_cpu << " (CPU), ";
    if (larger_gpu == 0)
        std::cout << "less than " << 1/ntoys << " (GPU)\n";
    else
        std::cout << pval_gpu << " (GPU)\n";

    std::cout << "Toy-generation run time: \n";
    std::cout << "+ On CPU: " << cpu_time_ms << " ms\n";
    std::cout << "+ On GPU: " << gpu_time_ms << " ms\n";
    float speed_up = cpu_time_ms/gpu_time_ms;
    printf("Gain a %.0f-time speedup with GPU\n", speed_up);
    
    // Free memory on GPU
    cudaFree(dev_q_toys);
    cudaFree(devStates);
    cudaFree(dev_bkg_expected);
    cudaFree(dev_obs_data);

    // Free memory on host
    free(host_q_toys);
    free(bkg_expected);
    free(obs_data);

    #if GOF==0
        cudaFree(dev_sig_expected);
        free(sig_expected);
    #endif

    return EXIT_SUCCESS;

}


int main(int argc, char **argv){
    return frequentist_test(argc, argv);
}


