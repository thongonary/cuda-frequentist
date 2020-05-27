/* CUDA toy
 * Thong Nguyen, 2020 */

#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "toygenerator.cuh"


__global__ void setup_kernel(curandState *state, int ntoys, int trialsPerThread)
{
    int id;
    for (int trial = 0; trial < trialsPerThread; trial++)
    {
        id = threadIdx.x + blockIdx .x * blockDim.x;
        while ((trial+1) * id < ntoys)
        {
            curand_init((unsigned long long)clock() + id * (trial + 1), 0, 0, &state[id * (trial+1)]);
            id += blockDim.x * gridDim.x;
        }
    }
}

__global__
void generate_goodness_of_fit_toys(float * dev_bkg_expected, 
                                   float * dev_obs_data,
                                   float * dev_q_toys,
                                   int n_bins,
                                   int ntoys,
                                   curandState *states,
                                   int trialsPerThread)
{
    int toy;
    unsigned int tid;
    float sum_log_likelihood;
    curandState localState;
    for (int trial = 0; trial < trialsPerThread; trial++)
    {
        tid = threadIdx.x + blockDim.x * blockIdx.x;
        while ((trial+1) * tid < ntoys)
        {
            localState = states[tid*(trial+1)];
            sum_log_likelihood = 0;
            for (int bin = 0; bin < n_bins; bin++)
            {
                toy = curand_poisson(&localState, dev_bkg_expected[bin]);
                sum_log_likelihood += chisquare(dev_bkg_expected[bin], toy);
            }
            dev_q_toys[tid * (trial+1)] = sum_log_likelihood;
            states[tid*(trial+1)] = localState;
            tid += blockDim.x * gridDim.x;
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
    int toy;
    unsigned int tid;
    float sum_log_likelihood, numerator, denominator;
    curandState localState;
    for (int trial = 0; trial < trialsPerThread; trial++)
    {
        tid = threadIdx.x + blockDim.x * blockIdx.x;
        while ((trial+1) * tid < ntoys)
        {
            localState = states[tid*(trial+1)];
            sum_log_likelihood = 0;
            for (int bin = 0; bin < n_bins; bin++)
            {
                toy = curand_poisson(&localState, dev_bkg_expected[bin]);
                numerator = log_poisson(dev_bkg_expected[bin]+dev_sig_expected[bin], toy);
                denominator = log_poisson(dev_bkg_expected[bin], toy);
                sum_log_likelihood += -2 * numerator/denominator;
            }
            dev_q_toys[tid * (trial+1)] = sum_log_likelihood;
            states[tid*(trial+1)] = localState;
            tid += blockDim.x * gridDim.x;
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
    printf("Doing the same on GPU using %d blocks and %d threads per block\n", nBlocks, threadsPerBlock);
    setup_kernel<<<nBlocks, threadsPerBlock>>>(devStates, ntoys, trialsPerThread);
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
    printf("Doing the same on GPU using %d blocks and %d threads per block\n", nBlocks, threadsPerBlock);
    setup_kernel<<<nBlocks, threadsPerBlock>>>(devStates, ntoys, trialsPerThread);
    generate_neyman_pearson_toys<<<nBlocks, threadsPerBlock>>>(dev_bkg_expected, 
                                                               dev_sig_expected,
                                                               dev_obs_data,
                                                               dev_q_toys,
                                                               n_bins,
                                                               ntoys,
                                                               devStates,
                                                               trialsPerThread);
}
    

