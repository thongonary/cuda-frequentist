#pragma once

namespace equations
{
    // Stirling's approximation of log(x!)
    float stirling_approx(int x);

    // x!
    unsigned long long factorial(int x);

    // log(x!)
    float log_factorial_approx(int a);

    // Log poisson likelihood
    float log_poisson(float mu, int k);

    // Improved chisquare test statistics using saturated model
    float chisquare(float exp, int obs);

}
