#include <omp.h>
#include <iostream> // Uses C++, for fun

using namespace std;

int
main(int argc, char** argv) {
  std::cout << "Parallel Region Example\n";
  #pragma omp parallel
  {
    int total = omp_get_num_threads();
    int num = omp_get_thread_num();
    // this way of using 'cout' could make a mess
    std::cout << "I'm in a Parallel Region "
    	      << num << "/" << total << "\n";
    
    cout << output;
  }
  return 0;
}
