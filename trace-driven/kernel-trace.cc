#include "kernel-trace.h"

kernel_trace_t::kernel_trace_t() {
  kernel_name = "Empty";
  shmem_base_addr = 0;
  local_base_addr = 0;
  binary_verion = 0;
  trace_verion = 0;
}