#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <gmp.h>
using namespace std;

int main(int argc, char *argv[])
{

    mpz_t num;
    mpz_t result;
    mpz_t mod;

    int exponent = 100000;

    mpz_init(num);
    mpz_init(result);
    mpz_init(mod);

    mpz_set_str(num, "875932476593476593247650374765237496523475234652347659234652349562394786583648327492367", 10);
    mpz_set_ui(result, 1);
    mpz_set_ui(mod, 20);

    int thread_nums = atoi(argv[1]);
     double start = omp_get_wtime();

#pragma omp parallel num_threads(thread_nums)
    {
        int chunk = exponent / omp_get_num_threads();
        int chunkRemainder = exponent % omp_get_num_threads();

        int start = chunk * omp_get_thread_num();
        int end = start + chunk;

        if (omp_get_thread_num() == omp_get_num_threads() - 1)
        {
            end = end + chunkRemainder;
        }

        mpz_t localResult;
        mpz_init(localResult);
        mpz_set_ui(localResult, 1);

        #pragma omp for
        for (int i = start; i < end; i++)
        {
            mpz_mul(localResult, localResult, num);
            mpz_mod(localResult, localResult, mod);
        }

#pragma omp critical
        mpz_mul(result, result, localResult);
        mpz_mod(result, result, mod);
    }

    double end = omp_get_wtime();

    double elapsed_time = end - start;
    cout << "Time taken: " << elapsed_time << " seconds" << endl;
    return 0;
}