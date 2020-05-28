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


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void check_args(int argc, char **argv){
    
    #if GOF
        if (argc < 5){
            std::cerr << "Incorrect number of arguments.\n";
            std::cerr << "Arguments: <number of bins> <background template file> <observed data file> <number of toys> [--out <output file> (optional)] [--GPUonly <0 or 1> (optional)]\n";
            exit(EXIT_FAILURE);
        }
    #else
        if (argc < 6){
            std::cerr << "Incorrect number of arguments.\n";
            std::cerr << "Arguments: <number of bins> <background template file> <signal template file> <observed data file> <number of toys> [--out <output file> (optional)] [--GPUonly <0 or 1> (optional)]\n";
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
        float ntoys = std::stof(argv[5]); // number of toy experiments
        std::string obs_filename = argv[4];
        std::string sig_filename = argv[3];
    #else
        float ntoys = std::stof(argv[4]); // number of toy experiments
        std::string obs_filename = argv[3];
    #endif
    
    // Get output and GPU argument, if any
    std::string out_filename;
    int GPUonly = 0;

    for (int i = 1; i < argc; ++i) 
    {
        if (std::string(argv[i]) == "--out") 
        {
            if (i + 1 < argc) 
            { 
                out_filename = argv[i + 1]; 
            } 
            else 
            { 
                std::cerr << "--out option requires one [string] argument." << std::endl;
                return 1;
            }
        }
        else if (std::string(argv[i]) == "--GPUonly")
        {
            if (i+1 < argc)
            { 
                GPUonly = std::stoi(argv[i + 1]);
                std::cout << "Generating " << ntoys << " toy experiments to obtain the test statistics distribution on GPU only." << std::endl;
            }
            else
            {
                std::cerr << "--GPUonly option requires one [integer] argument." << std::endl;
                return 1;
            }
        } 
    }

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
        std::cout << "Reading " << n_bins << " bins from data file " << obs_filename << std::endl;
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
            std::cout << "Reading " << n_bins << " bins from signal file " << sig_filename << std::endl;
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
    float *q_toys;
    if (GPUonly == 0)
    {
        START_TIMER();

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
        q_toys = (float*) malloc(sizeof(float) * ntoys);
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
    }

    // ******************************************************
    //                GPU IMPLEMENTATION
    // ******************************************************
    
    // Getting CUDA attributes
    int device;
    CUDA_CALL(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CALL(cudaGetDeviceProperties(&prop, device));
    int threadsPerBlock = prop.maxThreadsPerBlock;
    int *maxGridSize = prop.maxGridSize;

    // Allocate space for input data on device
    size_t freeMem, totalMem;
    float availableFraction = 0.95; 
    
    float *dev_bkg_expected;
    CUDA_CALL(cudaMalloc((void **) &dev_bkg_expected, n_bins * sizeof(float)));
    float *dev_obs_data;
    CUDA_CALL(cudaMalloc((void **) &dev_obs_data, n_bins * sizeof(float)));
    
    #if GOF==0
        float *dev_sig_expected;
        CUDA_CALL(cudaMalloc((void **) &dev_sig_expected, n_bins * sizeof(float)));
    #endif
    
    curandState *devStates;
    float *dev_q_toys;
    float *host_q_toys = (float*) malloc(sizeof(float) * ntoys);
    memset(host_q_toys, 0, ntoys * sizeof(float));

    int batch_toys = 0, nbatches = 0;
    
    // Determine whether can generate in 1 run or multiple chunks
    CUDA_CALL(cudaMemGetInfo(&freeMem, &totalMem));
    
    if (ntoys * sizeof(float) > 0.1 * freeMem)
    {
        batch_toys = ceil(0.1 * freeMem / sizeof(float)); // Number of toys generated per batch
        nbatches = ceil(ntoys / batch_toys); // Number of batches
    }

    if (nbatches == 0) // Do everything in 1 run
    {
        CUDA_CALL(cudaMalloc((void **) &dev_q_toys, ntoys * sizeof(float)));
        
        // Set the result arrays to zero
        CUDA_CALL(cudaMemset(dev_q_toys, 0, ntoys * sizeof(float)));
        
        // Get available memory 
        CUDA_CALL(cudaMemGetInfo(&freeMem, &totalMem));
        std::cout << "Free device memory: " << freeMem/(1024*1024) << "/" << totalMem/(1024*1024) << " MB" << std::endl;
        
        size_t availableMem = availableFraction * freeMem;
        // Determine number of blocks based on number of toys requested 
        int nBlocks = ceil(ntoys/threadsPerBlock);
        if (nBlocks > maxGridSize[0]) 
        {
            nBlocks = maxGridSize[0];
        }
        
        // Determine number of blocks based on available memory to allocate for curandState
        if (availableMem < nBlocks * threadsPerBlock * sizeof(curandState))
        {
            nBlocks = floor(availableMem / (threadsPerBlock * sizeof(curandState)));
        }
        
        // Allocate space for prng states on device
        int nStates = nBlocks * threadsPerBlock;
        CUDA_CALL(cudaMalloc((void **) &devStates, nStates * sizeof(curandState)));
        
        printf("+  Using %d blocks with %d threads per block\n", nBlocks, threadsPerBlock);
        
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
        

        #if GOF
        cuda_call_generate_goodness_of_fit_toys(nBlocks,
                                                threadsPerBlock,
                                                dev_bkg_expected, 
                                                dev_obs_data,
                                                dev_q_toys,
                                                n_bins,
                                                ntoys,
                                                devStates,
                                                nStates);
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
                                               nStates);
        
        #endif
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // Copy result back to host
        CUDA_CALL(cudaMemcpy(host_q_toys, dev_q_toys, ntoys * sizeof(float), cudaMemcpyDeviceToHost));
        cudaFree(dev_q_toys);
        cudaFree(devStates);
        STOP_RECORD_TIMER(gpu_time_ms);
    }
    else // Generate toys by batch to accommodate with the device memory
    {
        std::cout << "Generating in " << nbatches << " batches\n";
        START_TIMER();
        
        // Only do this once
        CUDA_CALL(cudaMemcpy(dev_bkg_expected, bkg_expected, 
                n_bins * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(dev_obs_data, obs_data, 
                n_bins * sizeof(float), cudaMemcpyHostToDevice));
        #if GOF==0
        CUDA_CALL(cudaMemcpy(dev_sig_expected, sig_expected, 
                n_bins * sizeof(float), cudaMemcpyHostToDevice));
        #endif
            
        int toy_pointer = 0;   
        for (int batch = 0; batch < nbatches; batch++)
        {
            if ((batch+1) * batch_toys > ntoys) 
            {
                batch_toys = ntoys - batch * batch_toys;
                if (batch_toys < 1) break;
            }
            std::cout << "+ Batch " << batch+1 << " of " << nbatches << ": Generating " << batch_toys << " toys\n";

            CUDA_CALL(cudaMalloc((void **) &dev_q_toys, batch_toys * sizeof(float)));
            CUDA_CALL(cudaMemset(dev_q_toys, 0, batch_toys * sizeof(float)));

            // Get available memory 
            CUDA_CALL(cudaMemGetInfo(&freeMem, &totalMem));
            //std::cout << "Free device memory: " << freeMem/(1024*1024) << "/" << totalMem/(1024*1024) << " MB" << std::endl;
            
            size_t availableMem = availableFraction * freeMem;
            // Determine number of blocks based on number of toys requested 
            int nBlocks = ceil(batch_toys/threadsPerBlock);
            if (nBlocks > maxGridSize[0]) 
            {
                nBlocks = maxGridSize[0];
            }
            
            // Determine number of blocks based on available memory to allocate for curandState
            if (availableMem < nBlocks * threadsPerBlock * sizeof(curandState))
            {
                nBlocks = floor(availableMem / (threadsPerBlock * sizeof(curandState)));
            }
            
            // Allocate space for prng states on device
            int nStates = nBlocks * threadsPerBlock;
            CUDA_CALL(cudaMalloc((void **) &devStates, nStates * sizeof(curandState)));
            
            // Generating toys
            printf("\t-- Using %d blocks with %d threads per block\n", nBlocks, threadsPerBlock);
            

            #if GOF
            cuda_call_generate_goodness_of_fit_toys(nBlocks,
                                                    threadsPerBlock,
                                                    dev_bkg_expected, 
                                                    dev_obs_data,
                                                    dev_q_toys,
                                                    n_bins,
                                                    batch_toys,
                                                    devStates,
                                                    nStates);
            #else
            cuda_call_generate_neyman_pearson_toys(nBlocks,
                                                   threadsPerBlock,
                                                   dev_bkg_expected, 
                                                   dev_sig_expected,
                                                   dev_obs_data,
                                                   dev_q_toys,
                                                   n_bins,
                                                   batch_toys,
                                                   devStates,
                                                   nStates);
            
            #endif
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());

            // Copy result back to host
            CUDA_CALL(cudaMemcpy(host_q_toys+toy_pointer, dev_q_toys, batch_toys * sizeof(float), cudaMemcpyDeviceToHost));
            cudaFree(dev_q_toys);
            cudaFree(devStates);
            toy_pointer += batch_toys;
        }
        STOP_RECORD_TIMER(gpu_time_ms);
    }
    // ******************************************************
    //            COMPARE AND SAVE RESULTS
    // ******************************************************
    
    std::ofstream file_cpu, file_gpu;
    std::string out_cpu, out_gpu;
    
    if (GPUonly == 0)
    {
        int larger_cpu = 0;
        int larger_gpu = 0;

        std::cout << "Toy-generation run time: \n";
        std::cout << "+ On CPU: " << cpu_time_ms << " ms\n";
        std::cout << "+ On GPU: " << gpu_time_ms << " ms\n";
        float speed_up = cpu_time_ms/gpu_time_ms;
        printf("Gained a %.0f-time speedup with GPU\n", speed_up);
       
        if (!out_filename.empty()) 
        {
            out_cpu = out_filename;
            out_gpu = out_filename;
            out_cpu.append("."+std::to_string(q_obs)+".cpu");
            out_gpu.append("."+std::to_string(q_obs)+".gpu");
            std::cout << "Saving the toy experiments' test statistics to " << out_cpu << " and " << out_gpu << std::endl;
            file_cpu.open(out_cpu);
            file_gpu.open(out_gpu);
        }
        tqdm bar2;
        for (int i = 0; i < ntoys; i++)
        {
            bar2.progress(i, ntoys);
            if (!out_filename.empty()) 
            {
                file_cpu << q_toys[i] << "\n";
                file_gpu << host_q_toys[i] << "\n";
            }
            if (q_toys[i] > q_obs) larger_cpu++;
            if (host_q_toys[i] > q_obs) larger_gpu++;
        }
        bar2.finish();
        if (!out_filename.empty()) 
        {
            file_cpu.close();
            file_gpu.close();
        }
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

        // Free memory on host
        free(host_q_toys);
        free(bkg_expected);
        free(obs_data);

        #if GOF==0
            cudaFree(dev_sig_expected);
            free(sig_expected);
        #endif
    }
    else
    {
        int larger_gpu = 0;

        std::cout << "Toy-generation run time on GPU: " << gpu_time_ms << " ms\n";
        
        if (!out_filename.empty()) 
        {
            out_gpu = out_filename;
            out_gpu.append("."+std::to_string(q_obs)+".gpu");
            std::cout << "Saving the toy experiments' test statistics to " << out_gpu << std::endl;
            file_gpu.open(out_gpu);
        }
        tqdm bar2;
        for (int i = 0; i < ntoys; i++)
        {
            bar2.progress(i, ntoys);
            if (host_q_toys[i] > q_obs) larger_gpu++;
            if (!out_filename.empty()) file_gpu << host_q_toys[i] << "\n";
        }
        bar2.finish();
        if (!out_filename.empty()) file_gpu.close();
        float pval_gpu = float(larger_gpu)/ntoys;
        #if GOF
            std::cout << "p-value from Goodness-of-fit test: ";
        #else
            std::cout << "p-value from Neyman-Pearson hypothesis test: ";
        #endif
        if (larger_gpu == 0)
            std::cout << "less than " << 1/ntoys << " (GPU)\n";
        else
            std::cout << pval_gpu << " (GPU)\n";
        

    }
    // Free memory on GPU
    cudaFree(dev_bkg_expected);
    cudaFree(dev_obs_data);

    return EXIT_SUCCESS;

}


int main(int argc, char **argv){
    return frequentist_test(argc, argv);
}


