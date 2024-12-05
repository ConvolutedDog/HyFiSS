#include <bitset>
#include <fstream>
#include <iostream>

#ifndef COMMON_DEF_H
#define COMMON_DEF_H

#define USE_BOOST
#define gpgpu_concurrent_kernel_sm false

#define ENABLE_SAMPLING_POINT

#define DUMP_TIME_SUMMARY

#define WARP_SIZE 32
#define MAX_DST 1
#define MAX_SRC 4

#define MAX_WARP_PER_SHADER 64

#define MAX_INPUT_VALUES 24
#define MAX_OUTPUT_VALUES 8

#define MAX_REG_OPERANDS 32

enum command_type {
  kernel_launch = 1,
  cpu_gpu_mem_copy,
  gpu_cpu_mem_copy,
};

enum address_space { GLOBAL_MEM = 1, SHARED_MEM, LOCAL_MEM, TEX_MEM };

enum address_scope {
  L1_CACHE = 1,
  L2_CACHE,
  SYS_MEM,
};

enum address_format { list_all = 0, base_stride = 1, base_delta = 2 };

const unsigned MAX_WARP_SIZE = 32;
typedef std::bitset<MAX_WARP_SIZE> active_mask_t;

const unsigned MAX_ACCESSES_PER_INSN_PER_THREAD = 8;

typedef unsigned long long new_addr_type;

const unsigned MAX_MEMORY_ACCESS_SIZE = 128;
typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;

const unsigned SECTOR_CHUNCK_SIZE = 4;
const unsigned SECTOR_SIZE = 32;
typedef std::bitset<SECTOR_CHUNCK_SIZE> mem_access_sector_mask_t;

enum _memory_op_t { no_memory_op = 0, memory_load, memory_store };

enum mem_operation_t { NOT_TEX, TEX };
typedef enum mem_operation_t mem_operation;

#define MEM_ACCESS_TYPE_TUP_DEF                                                \
  MA_TUP_BEGIN(mem_access_type)                                                \
  MA_TUP(GLOBAL_ACC_R), MA_TUP(LOCAL_ACC_R), MA_TUP(CONST_ACC_R),              \
      MA_TUP(TEXTURE_ACC_R), MA_TUP(GLOBAL_ACC_W), MA_TUP(LOCAL_ACC_W),        \
      MA_TUP(L1_WRBK_ACC), MA_TUP(L2_WRBK_ACC), MA_TUP(INST_ACC_R),            \
      MA_TUP(L1_WR_ALLOC_R), MA_TUP(L2_WR_ALLOC_R),                            \
      MA_TUP(NUM_MEM_ACCESS_TYPE) MA_TUP_END(mem_access_type)

#define MA_TUP_BEGIN(X) enum X {
#define MA_TUP(X) X
#define MA_TUP_END(X)                                                          \
  }                                                                            \
  ;
enum mem_access_type {
  GLOBAL_ACC_R,
  LOCAL_ACC_R,
  CONST_ACC_R,
  TEXTURE_ACC_R,
  GLOBAL_ACC_W,
  LOCAL_ACC_W,
  L1_WRBK_ACC,
  L2_WRBK_ACC,
  INST_ACC_R,
  L1_WR_ALLOC_R,
  L2_WR_ALLOC_R,
  NUM_MEM_ACCESS_TYPE
};
#undef MA_TUP_BEGIN
#undef MA_TUP
#undef MA_TUP_END

enum _memory_space_t {
  undefined_space = 0,
  reg_space,
  local_space,
  shared_space,
  sstarr_space,
  param_space_unclassified,
  param_space_kernel,
  param_space_local,
  const_space,
  tex_space,
  surf_space,
  global_space,
  generic_space,
  instruction_space
};

enum cache_operator_type {
  CACHE_UNDEFINED,

  CACHE_ALL,
  CACHE_LAST_USE,
  CACHE_VOLATILE,
  CACHE_L1,

  CACHE_STREAMING,
  CACHE_GLOBAL,

  CACHE_WRITE_BACK,
  CACHE_WRITE_THROUGH
};

#define MAX_REG_OPERANDS 32

#define MAX_KERNELS_NUM 300

#ifdef USE_BOOST

#include <boost/mpi.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#endif

#ifdef USE_BOOST
void simple_mpi_test(int argc, char **argv);
#endif

extern int kernel_info_m_next_uid;

extern unsigned long long GLOBAL_HEAP_START;

extern unsigned long long SHARED_MEM_SIZE_MAX;

extern unsigned long long LOCAL_MEM_SIZE_MAX;

extern unsigned MAX_STREAMING_MULTIPROCESSORS;

extern unsigned MAX_THREAD_PER_SM;

extern unsigned MAX_WARP_PER_SM;
extern unsigned long long TOTAL_LOCAL_MEM_PER_SM;
extern unsigned long long TOTAL_SHARED_MEM;
extern unsigned long long TOTAL_LOCAL_MEM;
extern unsigned long long SHARED_GENERIC_START;
extern unsigned long long LOCAL_GENERIC_START;
extern unsigned long long STATIC_ALLOC_LIMIT;

#endif
