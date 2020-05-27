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
                                   int ntoys,
                                   curandState *states,
                                   int trialsPerThread)
{
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int toy;
    float sum_log_likelihood;
    curand_init(42, tid, 0, &states[tid]); // Initialize CURAND
    for (int trial = 0; trial < trialsPerThread; trial++)
    {
        if ((trial+1) * tid < ntoys)
            {
            sum_log_likelihood = 0;
            for (int bin = 0; bin < n_bins; bin++)
            {
                toy = curand_poisson(&states[tid], dev_bkg_expected[bin]);
                sum_log_likelihood += chisquare(dev_bkg_expected[bin], toy);
            }
            dev_q_toys[tid * (trial+1)] = sum_log_likelihood;
        }
    }
}

__global__
void generate_neyman_pearson_toys(float * dev_bkg_expected, 
                                   float * dev_sig_expected,
                                   float * dev_obs_data,
                                   float * dev_q_toys,
                                   int n_bins,
                                   int ntoys,
                                   curandState *states,
                                   int trialsPerThread)
{
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int toy;
    float sum_log_likelihood, numerator, denominator;
    curand_init(42, tid, 0, &states[tid]); // Initialize CURAND
    for (int trial = 0; trial < trialsPerThread; trial++)
    {
        if ((trial+1) * tid < ntoys)
        {
            sum_log_likelihood = 0;
            for (int bin = 0; bin < n_bins; bin++)
            {
                toy = curand_poisson(&states[tid], dev_bkg_expected[bin]);
                numerator = log_poisson(dev_bkg_expected[bin]+dev_sig_expected[bin], toy);
                denominator = log_poisson(dev_bkg_expected[bin], toy);
                sum_log_likelihood += -2 * numerator/denominator;
            }
            dev_q_toys[tid * (trial+1)] = sum_log_likelihood;
        }
    }
}

void cuda_call_generate_goodness_of_fit_toys(int nBlocks,
                                            int threadsPerBlock,
                                            float * dev_bkg_expected, 
                                            float * dev_obs_data,
                                            float * dev_q_toys,
                                            int n_bins,
                                            int ntoys,
                                            curandState * devStates,
                                            int trialsPerThread)
{
    generate_goodness_of_fit_toys<<<nBlocks, threadsPerBlock>>>(dev_bkg_expected, 
                                                               dev_obs_data,
                                                               dev_q_toys,
                                                               n_bins,
                                                               ntoys,
                                                               devStates,
                                                               trialsPerThread);

}
    
void cuda_call_generate_neyman_pearson_toys(int nBlocks,
                                           int threadsPerBlock,
                                           float * dev_bkg_expected, 
                                           float * dev_sig_expected,
                                           float * dev_obs_data,
                                           float * dev_q_toys,
                                           int n_bins,
                                           int ntoys,
                                           curandState * devStates,
                                           int trialsPerThread)
{
    generate_neyman_pearson_toys<<<nBlocks, threadsPerBlock>>>(dev_bkg_expected, 
                                                               dev_sig_expected,
                                                               dev_obs_data,
                                                               dev_q_toys,
                                                               n_bins,
                                                               ntoys,
                                                               devStates,
                                                               trialsPerThread);
}
    

