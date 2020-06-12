#include <omp.h>
#include <stdio.h>

int
main(int argc, char** argv)
{
  int n = 16;
  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    int my_num = omp_get_thread_num();
    printf("Hello OMP I am %d, this is loop #%d\n", my_num, i);
  }
  return 0;
}
