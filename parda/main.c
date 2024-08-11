#include "parda.h"
#ifdef enable_mpi
#include "parda_mpi.h"
#endif
#ifdef enable_omp
#include "parda_omp.h"
#endif
#include "process_args.h"
#include "seperate.h"

int main(int argc, char **argv) {
  process_args(argc, argv);
  if (is_seperate == 1) {
    parda_seperate_file(inputFileName, threads, lines);
  } else if (is_omp == 0 && is_mpi == 0) {
    DEBUG(printf("This is seq stackdist\n");)
    classical_tree_based_stackdist(inputFileName, lines);
  } else if (is_omp == 1 && is_mpi == 0) {
    DEBUG(printf("This is omp stackdist\n");)
#ifdef enable_omp
    parda_omp_stackdist(inputFileName, lines, threads);
#else
    printf("openmp is not enabled, try to define enable_omp and add OMP "
           "variable in Makefile\n");
    abort();
#endif
  } else if (is_omp == 0 && is_mpi == 1) {
    DEBUG(printf("This is mpistackdist\n");)
#ifdef enable_mpi
    parda_mpi_stackdist(inputFileName, lines, threads, argc, argv);
#else
    printf("mpi is not enabled, try to define enable_omp and add MPI variable "
           "in Makefile\n");
    abort();
#endif
  } else if (is_omp == 1 && is_mpi == 1) {
    DEBUG(printf("This is hybrid stackdist\n");)
#if defined(enable_omp) && defined(enable_mpi)
    parda_hybrid_stackdist(inputFileName, lines, threads, argc, argv);
#else
    printf("hybridis not enabled, try to define enable_omp and enable_mpi and "
           "add MPI and OMP variable in Makefile\n");
    abort();
#endif
  }
  return 0;
}
