/* CUDA toy generator
 * Thong Nguyen, 2020 */


#ifndef CUDA_TOY_GENERATOR_CUH
#define CUDA_TOY_GENERATOR_CUH

#define PI 3.14159265358979

// Stirling approxmination of log(x!)
__inline__ __device__
float stirling_approx(int x)
{
    return (x + 0.5) * __logf(x) - x + 0.5 * __logf(2*PI); 
}

// Computation of x!, use only when x < 20
__inline__ __device__
unsigned long long factorial(int x)
{
    unsigned long long result = 1;
    for (int i = 1; i <= x; i++)
        result *= i;

    return result;
}

__inline__ __device__
float log_factorial_approx(int a)
{
    // Use Stirling approximation for larger number
    if (a > 20) return stirling_approx(a);
    else return __logf(factorial(a));
}

__inline__ __device__
float log_poisson(float mu, int k)
{
    // Log likelihood of poisson distribution with mean mu
    return - mu - log_factorial_approx(k) + k * __logf(mu);
}

__inline__ __device__
float chisquare(float exp, int obs)
{
    return -2 * (exp - obs + __logf(obs/exp));
}

__global__
void generate_goodness_of_fit_toys(float * dev_bkg_expected, 
                                   float * dev_obs_data,
                                   float * dev_q_toys,
                                   int n_bins,
                                   unsigned long ntoys,
                                   curandState *states,
                                   unsigned int nStates);

__global__
void count_extreme_goodness_of_fit(float * dev_bkg_expected, 
                                   float * dev_obs_data,
                                   unsigned int * dev_larger_gpu,
                                   float q_obs,
                                   int n_bins,
                                   unsigned long ntoys,
                                   curandState *states,
                                   unsigned int nStates);

__global__
void generate_neyman_pearson_toys(float * dev_bkg_expected, 
                                   float * dev_sig_expected,
                                   float * dev_obs_data,
                                   float * dev_q_toys,
                                   int n_bins,
                                   unsigned long ntoys,
                                   curandState *states,
                                   unsigned int nStates);

__global__
void count_extreme_neyman_pearson(float * dev_bkg_expected, 
                                   float * dev_sig_expected,
                                   float * dev_obs_data,
                                   unsigned int * dev_larger_gpu,
                                   float q_obs,
                                   int n_bins,
                                   unsigned long ntoys,
                                   curandState *states,
                                   unsigned int nStates);

void cuda_call_generate_goodness_of_fit_toys(unsigned int nBlocks,
                                            int threadsPerBlock,
                                            float * dev_bkg_expected, 
                                            float * dev_obs_data,
                                            float * dev_q_toys,
                                            int n_bins,
                                            unsigned long ntoys,
                                            curandState * devStates,
                                            unsigned int nStates);

void cuda_call_count_extreme_goodness_of_fit(unsigned int nBlocks,
                                            int threadsPerBlock,
                                            float * dev_bkg_expected, 
                                            float * dev_obs_data,
                                            unsigned int * dev_larger_gpu,
                                            float q_obs,
                                            int n_bins,
                                            unsigned long ntoys,
                                            curandState * devStates,
                                            unsigned int nStates);

void cuda_call_generate_neyman_pearson_toys(unsigned int nBlocks,
                                           int threadsPerBlock,
                                           float * dev_bkg_expected, 
                                           float * dev_sig_expected,
                                           float * dev_obs_data,
                                           float * dev_q_toys,
                                           int n_bins,
                                           unsigned long ntoys,
                                           curandState * devStates,
                                           unsigned int nStates);

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
                                           unsigned int nStates);

#endif
