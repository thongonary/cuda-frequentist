/* CUDA toy
 * Thong Nguyen, 2020 */

#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "toygenerator.cuh"


__global__
void generate_goodness_of_fit_toys(float * dev_bkg_expected, 
                                   float * dev_obs_data,
                                   float * dev_q_toys,
                                   int n_bins,
                                   unsigned long ntoys,
                                   curandState *states,
                                   unsigned int nStates)
{
    int toy;
    unsigned long tid = threadIdx.x + blockDim.x * blockIdx.x;
    float sum_log_likelihood;
    while (tid < ntoys)
    {
        curand_init((unsigned long long)clock() + tid, 0, 0, &states[tid % nStates]);
        sum_log_likelihood = 0;
        for (int bin = 0; bin < n_bins; bin++)
        {
            toy = curand_poisson(&states[tid % nStates], dev_bkg_expected[bin]);
            sum_log_likelihood += chisquare(dev_bkg_expected[bin], toy);
        }
        dev_q_toys[tid] = sum_log_likelihood;
        tid += blockDim.x * gridDim.x;
    }
}

__global__
void count_extreme_goodness_of_fit(float * dev_bkg_expected, 
                                   float * dev_obs_data,
                                   unsigned int * dev_larger_gpu,
                                   float q_obs,
                                   int n_bins,
                                   unsigned long ntoys,
                                   curandState *states,
                                   unsigned int nStates)
{
    int toy;
    unsigned long tid = threadIdx.x + blockDim.x * blockIdx.x;
    float sum_log_likelihood;
    while (tid < ntoys)
    {
        curand_init((unsigned long long)clock() + tid, 0, 0, &states[tid % nStates]);
        sum_log_likelihood = 0;
        for (int bin = 0; bin < n_bins; bin++)
        {
            toy = curand_poisson(&states[tid % nStates], dev_bkg_expected[bin]);
            sum_log_likelihood += chisquare(dev_bkg_expected[bin], toy);
        }
        if (sum_log_likelihood > q_obs) atomicAdd(dev_larger_gpu, (unsigned int) 1);
        tid += blockDim.x * gridDim.x;
    }
}


__global__
void generate_neyman_pearson_toys(float * dev_bkg_expected, 
                                   float * dev_sig_expected,
                                   float * dev_obs_data,
                                   float * dev_q_toys,
                                   int n_bins,
                                   unsigned long ntoys,
                                   curandState *states,
                                   unsigned int nStates)
{
    int toy;
    unsigned long tid = (threadIdx.x + blockDim.x * blockIdx.x);
    float sum_log_likelihood, numerator, denominator;
    while (tid < ntoys)
    {
        curand_init((unsigned long long)clock() + tid, 0, 0, &states[tid % nStates]);
        sum_log_likelihood = 0;
        for (int bin = 0; bin < n_bins; bin++)
        {
            toy = curand_poisson(&states[tid % nStates], dev_bkg_expected[bin]);
            numerator = log_poisson(dev_bkg_expected[bin]+dev_sig_expected[bin], toy);
            denominator = log_poisson(dev_bkg_expected[bin], toy);
            sum_log_likelihood += 2 * (numerator-denominator);
        }
        dev_q_toys[tid] = sum_log_likelihood;
        tid += (blockDim.x * gridDim.x);
    }
}

__global__
void count_extreme_neyman_pearson(float * dev_bkg_expected, 
                                   float * dev_sig_expected,
                                   float * dev_obs_data,
                                   unsigned int * dev_larger_gpu,
                                   float q_obs,
                                   int n_bins,
                                   unsigned long ntoys,
                                   curandState *states,
                                   unsigned int nStates)
{
    int toy;
    unsigned long tid = (threadIdx.x + blockDim.x * blockIdx.x);
    float sum_log_likelihood, numerator, denominator;
    while (tid < ntoys)
    {
        curand_init((unsigned long long)clock() + tid, 0, 0, &states[tid % nStates]);
        sum_log_likelihood = 0;
        for (int bin = 0; bin < n_bins; bin++)
        {
            toy = curand_poisson(&states[tid % nStates], dev_bkg_expected[bin]);
            numerator = log_poisson(dev_bkg_expected[bin]+dev_sig_expected[bin], toy);
            denominator = log_poisson(dev_bkg_expected[bin], toy);
            sum_log_likelihood += 2 * (numerator-denominator);
        }
        if (sum_log_likelihood > q_obs) atomicAdd(dev_larger_gpu, (unsigned int) 1);
        tid += (blockDim.x * gridDim.x);
    }
}

void cuda_call_generate_goodness_of_fit_toys(unsigned int nBlocks,
                                            int threadsPerBlock,
                                            float * dev_bkg_expected, 
                                            float * dev_obs_data,
                                            float * dev_q_toys,
                                            int n_bins,
                                            unsigned long ntoys,
                                            curandState * devStates,
                                            unsigned int nStates)
{
    generate_goodness_of_fit_toys<<<nBlocks, threadsPerBlock>>>(dev_bkg_expected, 
                                                               dev_obs_data,
                                                               dev_q_toys,
                                                               n_bins,
                                                               ntoys,
                                                               devStates,
                                                               nStates);

}

void cuda_call_count_extreme_goodness_of_fit(unsigned int nBlocks,
                                            int threadsPerBlock,
                                            float * dev_bkg_expected, 
                                            float * dev_obs_data,
                                            unsigned int * dev_larger_gpu,
                                            float q_obs,
                                            int n_bins,
                                            unsigned long ntoys,
                                            curandState * devStates,
                                            unsigned int nStates)
{
    count_extreme_goodness_of_fit<<<nBlocks, threadsPerBlock>>>(dev_bkg_expected, 
                                                               dev_obs_data,
                                                               dev_larger_gpu,
                                                               q_obs,
                                                               n_bins,
                                                               ntoys,
                                                               devStates,
                                                               nStates);

}
    
void cuda_call_generate_neyman_pearson_toys(unsigned int nBlocks,
                                           int threadsPerBlock,
                                           float * dev_bkg_expected, 
                                           float * dev_sig_expected,
                                           float * dev_obs_data,
                                           float * dev_q_toys,
                                           int n_bins,
                                           unsigned long ntoys,
                                           curandState * devStates,
                                           unsigned int nStates)
{
    generate_neyman_pearson_toys<<<nBlocks, threadsPerBlock>>>(dev_bkg_expected, 
                                                               dev_sig_expected,
                                                               dev_obs_data,
                                                               dev_q_toys,
                                                               n_bins,
                                                               ntoys,
                                                               devStates,
                                                               nStates);
}
    
void cuda_call_count_extreme_neyman_pearson(unsigned int nBlocks,
                                           int threadsPerBlock,
                                           float * dev_bkg_expected, 
                                           float * dev_sig_expected,
                                           float * dev_obs_data,
                                           unsigned int * dev_larger_gpu,
                                           float q_obs,
                                           int n_bins,
                                           unsigned long ntoys,
                                           curandState * devStates,
                                           unsigned int nStates)
{
    count_extreme_neyman_pearson<<<nBlocks, threadsPerBlock>>>(dev_bkg_expected, 
                                                               dev_sig_expected,
                                                               dev_obs_data,
                                                               dev_larger_gpu,
                                                               q_obs,
                                                               n_bins,
                                                               ntoys,
                                                               devStates,
                                                               nStates);
}
