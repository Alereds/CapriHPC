#include <omp.h>
#include <stdio.h>
#include <fstream>
#include <limits>

void random_vector(double* v, int n)
{
	for (int i = 0; i < n; ++i) {
		v[i] = (double)rand() / RAND_MAX;
	}
}

int main(int argc, char** argv)
{
	std::size_t n = 50000000; // these will contain the dimension of the matrix
	double sum = 0;
	double* v;        // this will contain the 'linearized matrix'
// enter the parallel region
#pragma omp parallel 
	{
#pragma omp single
		{
			int master_num = omp_get_thread_num();
			printf("Master is thread number %d\n", master_num);
			v = new double[n];
			random_vector(v, n);
			printf("Loaded\n");

		} // MASTER ENDS

		int thread_num = omp_get_thread_num();
		printf("Thread %d started\n", thread_num);


		// here all threads execute the code
		int N = (int)(n);
#pragma omp for reduction(+: sum)
		for (int i = 0; i < N; i++)
		{
			sum += v[i];
		}
		// we don't want to see as many outputs as there are threads
		// also we don't want all threads to free the resources
#pragma omp barrier // blocchiamo prima di mostrare la somma!
#pragma omp single
		{
			printf("Average value %f\n", sum / (double)N);
			// never forget to delete / free resources...
			delete[] v;
		}
	}
	// an alternative is to put delete and output here (outside the
	// parallel region), but also corresponding declarations should
	// have been put before starting the parallel region.

	return 0;
}

