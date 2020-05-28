#include <cmath>
#include "equations.hpp"

namespace equations
{
    const float PI = 3.14159265358979;

    // Stirling approxmination of log(x!)
    float stirling_approx(int x)
    {
        return (x + 0.5) * log(x) - x + 0.5 * log(2*PI); 
    }

    // Computation of x!, use only when x < 20
    unsigned long long factorial(int x)
    {
        unsigned long long result = 1;
        for (int i = 1; i <= x; i++)
            result *= i;

        return result;
    }

    float log_factorial_approx(int a)
    {
        // Use Stirling approximation for larger number
        if (a > 20) return stirling_approx(a);
        else return log(factorial(a));
    }

    float log_poisson(float mu, int k)
    {
        // Log likelihood of poisson distribution with mean mu
        return - mu - log_factorial_approx(k) + k * log(mu);
    }

    float chisquare(float exp, int obs)
    {
        // http://cousins.web.cern.ch/cousins/ongoodness6march2016.pdf
        return -2 * (exp - obs + log(obs/exp));
    }
}
