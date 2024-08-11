#include <fstream>
#include <string>

#include "../common/common_def.h"

#ifndef KERNEL_TRACE_H
#define KERNEL_TRACE_H

struct kernel_trace_t {
  kernel_trace_t();

  std::string kernel_name;
  unsigned kernel_id;
  unsigned grid_dim_x;
  unsigned grid_dim_y;
  unsigned grid_dim_z;
  unsigned tb_dim_x;
  unsigned tb_dim_y;
  unsigned tb_dim_z;
  unsigned shmem;
  unsigned nregs;
  unsigned long cuda_stream_id;
  unsigned binary_verion;
  unsigned enable_lineinfo;
  unsigned trace_verion;
  std::string nvbit_verion;
  unsigned long long shmem_base_addr;
  unsigned long long local_base_addr;

#ifdef ENABLE_SAMPLING_POINT
  unsigned sampling_point;
#endif

  std::ifstream *ifs;
};

#endif
