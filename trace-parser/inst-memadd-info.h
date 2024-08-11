#include <bitset>

#include "../common/common_def.h"
#include "../common/vector_types.h"

#ifndef INST_MEMADD_INFO_H
#define INST_MEMADD_INFO_H

class inst_memadd_info_t {
public:
  uint64_t addrs[WARP_SIZE];
  int32_t width = 0;
  bool empty = true;

  void base_stride_decompress(unsigned long long base_address, int stride,
                              const std::bitset<WARP_SIZE> &mask);
  void base_delta_decompress(unsigned long long base_address,
                             const std::vector<long long> &deltas,
                             const std::bitset<WARP_SIZE> &mask);
};

#endif