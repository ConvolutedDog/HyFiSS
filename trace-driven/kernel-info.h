#include <cstdio>
#include <string>
#include <vector>

#include "../common/vector_types.h"
#include "../trace-parser/trace-parser.h"
#include "mem-access.h"

#ifndef KERNEL_INFO_H
#define KERNEL_INFO_H

class kernel_info_t {
public:
  kernel_info_t(dim3 gridDim, dim3 blockDim);
  ~kernel_info_t(){};

  size_t num_blocks() const {
    return m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;
  }

  size_t threads_per_cta() const {
    return m_block_dim.x * m_block_dim.y * m_block_dim.z;
  }

  dim3 get_grid_dim() const { return m_grid_dim; }

  dim3 get_cta_dim() const { return m_block_dim; }

  unsigned get_uid() const { return m_uid; }

  unsigned m_uid;

  dim3 m_grid_dim;
  dim3 m_block_dim;
};

class trace_kernel_info_t : public kernel_info_t {
public:
  trace_kernel_info_t(dim3 gridDim, dim3 blockDim, trace_parser *parser,

                      kernel_trace_t *kernel_trace_info);
  ~trace_kernel_info_t() { delete m_kernel_trace_info; };
  std::vector<std::vector<inst_trace_t> *>
  get_next_threadblock_traces(std::string kernel_name, unsigned kernel_id,
                              unsigned num_warps_per_thread_block);
  std::vector<mem_instn> &
  get_one_kernel_one_threadblock_traces(unsigned kernel_id, unsigned block_id);

  unsigned long get_cuda_stream_id() {
    return m_kernel_trace_info->cuda_stream_id;
  }

  kernel_trace_t *get_trace_info() { return m_kernel_trace_info; }

private:
  const std::unordered_map<std::string, OpcodeChar> *OpcodeMap;
  trace_parser *m_parser;
  kernel_trace_t *m_kernel_trace_info;
};

#endif