#include "kernel-info.h"

kernel_info_t::kernel_info_t(dim3 gridDim, dim3 blockDim) {

  m_grid_dim = gridDim;

  m_block_dim = blockDim;

  m_uid = kernel_info_m_next_uid++;
}

trace_kernel_info_t::trace_kernel_info_t(dim3 gridDim, dim3 blockDim,
                                         trace_parser *parser,

                                         kernel_trace_t *kernel_trace_info)
    : kernel_info_t(gridDim, blockDim) {
  m_parser = parser;
  m_kernel_trace_info = kernel_trace_info;

  if (kernel_trace_info->binary_verion == AMPERE_RTX_BINART_VERSION ||
      kernel_trace_info->binary_verion == AMPERE_A100_BINART_VERSION)
    OpcodeMap = &Ampere_OpcodeMap;
  else if (kernel_trace_info->binary_verion == VOLTA_BINART_VERSION)
    OpcodeMap = &Volta_OpcodeMap;
  else if (kernel_trace_info->binary_verion == PASCAL_TITANX_BINART_VERSION ||
           kernel_trace_info->binary_verion == PASCAL_P100_BINART_VERSION)
    OpcodeMap = &Pascal_OpcodeMap;
  else if (kernel_trace_info->binary_verion == KEPLER_BINART_VERSION)
    OpcodeMap = &Kepler_OpcodeMap;
  else if (kernel_trace_info->binary_verion == TURING_BINART_VERSION)
    OpcodeMap = &Turing_OpcodeMap;
  else {
    printf("unsupported binary version: %d\n",
           kernel_trace_info->binary_verion);
    fflush(stdout);
    exit(0);
  }
}

std::vector<mem_instn> &
trace_kernel_info_t::get_one_kernel_one_threadblock_traces(unsigned kernel_id,
                                                           unsigned block_id) {
  return m_parser->get_one_kernel_one_threadblcok_mem_instns(kernel_id,
                                                             block_id);
}

std::vector<std::vector<inst_trace_t> *>
trace_kernel_info_t::get_next_threadblock_traces(
    std::string kernel_name, unsigned kernel_id,
    unsigned num_warps_per_thread_block) {
  return m_parser->get_next_threadblock_traces(
      m_kernel_trace_info->trace_verion, m_kernel_trace_info->enable_lineinfo,
      m_kernel_trace_info->ifs, kernel_name, kernel_id,
      num_warps_per_thread_block);
}