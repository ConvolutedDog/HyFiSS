#include <assert.h>
#include <bitset>
#include <list>
#include <map>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "../ISA-Def/ampere_opcode.h"
#include "../ISA-Def/kepler_opcode.h"
#include "../ISA-Def/pascal_opcode.h"
#include "../ISA-Def/trace_opcode.h"
#include "../ISA-Def/turing_opcode.h"
#include "../ISA-Def/volta_opcode.h"

#include "../common/common_def.h"
#include "../common/vector_types.h"
#include "../trace-driven/kernel-trace.h"
#include "../trace-driven/trace-warp-inst.h"
#include "inst-memadd-info.h"
#include "inst-trace.h"
#include "memory-space.h"

#include "../trace-driven/inst-stt.h"

#ifndef TRACE_PARSER_H
#define TRACE_PARSER_H

enum config_type { APP_CONFIG, INSTN_CONFIG, ISSUE_CONFIG, CONFIGS_TYPE_NUM };

class app_config {
public:
  app_config() {
    m_valid = false;
    kernels_num = 1;
  }
  void init(std::string config_path, bool PRINT_LOG);

  unsigned get_kernels_num() { return kernels_num; }

  std::vector<int> *get_kernels_grid_size() { return &kernel_grid_size; }
  int get_kernel_grid_size(int kernel_id) {
    return kernel_grid_size[kernel_id];
  }

  std::vector<std::string> *get_kernels_name() { return &kernel_name; }
  std::string get_kernel_name(int kernel_id) { return kernel_name[kernel_id]; }

  std::vector<int> *get_app_kernels_id() { return &app_kernels_id; }
  int get_app_kernel_id(int kernel_id) { return app_kernels_id[kernel_id]; }

  std::vector<int> *get_kernels_grid_dim_x() { return &kernel_grid_dim_x; }
  int get_kernel_grid_dim_x(int kernel_id) {
    return kernel_grid_dim_x[kernel_id];
  }
  std::vector<int> *get_kernels_grid_dim_y() { return &kernel_grid_dim_y; }
  int get_kernel_grid_dim_y(int kernel_id) {
    return kernel_grid_dim_y[kernel_id];
  }
  std::vector<int> *get_kernels_grid_dim_z() { return &kernel_grid_dim_z; }
  int get_kernel_grid_dim_z(int kernel_id) {
    return kernel_grid_dim_z[kernel_id];
  }

  int get_num_global_warps(int kernel_id) {
    return (
        int)((get_kernel_grid_dim_x(kernel_id) *
              get_kernel_grid_dim_y(kernel_id) *
              get_kernel_grid_dim_z(kernel_id)) *
             ((get_kernel_tb_dim_x(kernel_id) * get_kernel_tb_dim_y(kernel_id) *
                   get_kernel_tb_dim_z(kernel_id) +
               WARP_SIZE - 1) /
              WARP_SIZE));
  }

  int get_num_warp_per_block(int kernel_id) {
    return (int)((get_kernel_tb_dim_x(kernel_id) *
                      get_kernel_tb_dim_y(kernel_id) *
                      get_kernel_tb_dim_z(kernel_id) +
                  WARP_SIZE - 1) /
                 WARP_SIZE);
  }

  std::vector<int> *get_kernels_tb_dim_x() { return &kernel_tb_dim_x; }
  int get_kernel_tb_dim_x(int kernel_id) { return kernel_tb_dim_x[kernel_id]; }
  std::vector<int> *get_kernels_tb_dim_y() { return &kernel_tb_dim_y; }
  int get_kernel_tb_dim_y(int kernel_id) { return kernel_tb_dim_y[kernel_id]; }
  std::vector<int> *get_kernels_tb_dim_z() { return &kernel_tb_dim_z; }
  int get_kernel_tb_dim_z(int kernel_id) { return kernel_tb_dim_z[kernel_id]; }

  std::vector<int> *get_kernels_num_registers() {
    return &kernel_num_registers;
  }
  int get_kernel_num_registers(int kernel_id) {
    return kernel_num_registers[kernel_id];
  }

  std::vector<int> *get_kernels_shared_mem_bytes() {
    return &kernel_shared_mem_bytes;
  }
  int get_kernel_shared_mem_bytes(int kernel_id) {
    return kernel_shared_mem_bytes[kernel_id];
  }

  std::vector<int> *get_kernels_block_size() { return &kernel_block_size; }
  int get_kernel_block_size(int kernel_id) {
    return kernel_block_size[kernel_id];
  }

  std::vector<int> *get_kernels_cuda_stream_id() {
    return &kernel_cuda_stream_id;
  }
  int get_kernel_cuda_stream_id(int kernel_id) {
    return kernel_cuda_stream_id[kernel_id];
  }

#ifdef ENABLE_SAMPLING_POINT
  std::vector<int> *get_kernels_sampling_point() {
    return &kernel_sampling_point;
  }
  int get_kernel_sampling_point(int kernel_id) {
    return kernel_sampling_point[kernel_id];
  }
#endif

  std::vector<unsigned long long> *get_kernels_shmem_base_addr() {
    return &kernel_shmem_base_addr;
  }
  unsigned long long get_kernel_shmem_base_addr(int kernel_id) {
    return kernel_shmem_base_addr[kernel_id];
  }

  std::vector<unsigned long long> *get_kernels_local_base_addr() {
    return &kernel_local_base_addr;
  }
  unsigned long long get_kernel_local_base_addr(int kernel_id) {
    return kernel_local_base_addr[kernel_id];
  }
  int get_concurrentKernels() { return concurrentKernels; }

private:
  bool m_valid;
  int concurrentKernels = 0;
  std::string app_kernels_id_string;
  std::vector<int> app_kernels_id;
  int kernels_num;

  std::vector<std::string> kernel_name;
  std::vector<int> kernel_num_registers;
  std::vector<int> kernel_shared_mem_bytes;
  std::vector<int> kernel_grid_size;
  std::vector<int> kernel_block_size;
  std::vector<int> kernel_cuda_stream_id;

  std::vector<int> kernel_grid_dim_x;
  std::vector<int> kernel_grid_dim_y;
  std::vector<int> kernel_grid_dim_z;
  std::vector<int> kernel_tb_dim_x;
  std::vector<int> kernel_tb_dim_y;
  std::vector<int> kernel_tb_dim_z;
  std::vector<unsigned long long> kernel_shmem_base_addr;
  std::vector<unsigned long long> kernel_local_base_addr;

#ifdef ENABLE_SAMPLING_POINT
  std::vector<int> kernel_sampling_point;
#endif
};

class instn_info_t {
public:
  instn_info_t() {}
  instn_info_t(unsigned kernel_id, unsigned pc, std::string instn_str)
    : kernel_id(kernel_id), pc(pc), instn_str(instn_str) {}

private:
  unsigned kernel_id;
  unsigned pc;
  std::string instn_str;
};

class instn_config {
public:
  instn_config() { m_valid = false; }
  instn_config(hw_config *hw_cfg) {
    m_valid = false;
    this->hw_cfg = hw_cfg;
  }
  ~instn_config() {
    for (auto iter = instn_info_vector.begin(); 
         iter != instn_info_vector.end();
         ++iter) {
      delete iter->second;
    }
  }
  void init(std::string config_path, bool PRINT_LOG);

  std::map<std::pair<int, int>, _inst_trace_t *> *get_instn_info_vector() {
    return &instn_info_vector;
  }

  int get_instn_latency(int kernel_id, int pc) {
    auto iter = instn_info_vector.find(std::make_pair(kernel_id, pc));
    if (iter != instn_info_vector.end()) {
      return iter->second->latency;
    } else {
      return -1;
    }
  }

private:
  bool m_valid;
  hw_config *hw_cfg;

  std::map<std::pair<int, int>, _inst_trace_t *> instn_info_vector;
};

struct block_info_t {
  block_info_t() {}
  block_info_t(const unsigned kernel_id,
               const unsigned block_id,
               const unsigned long long time_stamp,
               const unsigned sm_id)
    : kernel_id(kernel_id), block_id(block_id),
      time_stamp(time_stamp), sm_id(sm_id) {}

  unsigned kernel_id;
  unsigned block_id;
  unsigned long long time_stamp;
  unsigned sm_id;

/// TODO: May not need this.
#ifdef USE_BOOST
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar &kernel_id;
    ar &block_id;
    ar &time_stamp;
    ar &sm_id;
  }
#endif
};

class issue_config {
public:
  issue_config() {
    m_valid = false;
    trace_issued_sms_num = 1;
  }
  void init(const std::string config_path, bool dump_log);

  std::vector<std::vector<block_info_t>> *get_trace_issued_sm_id_blocks() {
    return &trace_issued_sm_id_blocks;
  }
  std::vector<block_info_t> *get_trace_issued_one_sm_blocks(unsigned sm_id) {
    for (auto iter = trace_issued_sm_id_blocks.begin();
         iter != trace_issued_sm_id_blocks.end(); ++iter)
      if ((*iter)[0].sm_id == sm_id)
        return &(*iter);
    return NULL;
  }

  block_info_t *get_trace_issued_one_sm_one_block(unsigned sm_id,
                                                  unsigned block_id) {
    for (auto iter = trace_issued_sm_id_blocks.begin();
         iter != trace_issued_sm_id_blocks.end(); ++iter)
      if ((*iter)[0].sm_id == sm_id)
        for (auto iter2 = iter->begin(); iter2 != iter->end(); ++iter2)
          if (iter2->block_id == block_id)
            return &(*iter2);

    return NULL;
  }

  std::vector<std::pair<int, int>> get_kernel_block_by_smid(int smid) {
    std::vector<std::pair<int, int>> result;
    //  Each element in `trace_issued_sm_id_blocks` of type `vector<...>`
    // is a vector of type `vector<block_info_t>`.
    for (auto iter = trace_issued_sm_id_blocks.begin();
         iter != trace_issued_sm_id_blocks.end(); ++iter) {
      if ((*iter)[0].sm_id == (unsigned)smid) {
        for (auto iter2 = iter->begin(); iter2 != iter->end(); ++iter2) {
          result.push_back(std::make_pair(iter2->kernel_id, iter2->block_id));
        }
      }
    }
    return result;
  }

  /// @brief 
  /// @param smid 
  /// @return 
  std::vector<std::pair<int, int>> get_kernel_block_of_all_sms() {
    std::vector<std::pair<int, int>> result;
    for (auto iter = trace_issued_sm_id_blocks.begin();
         iter != trace_issued_sm_id_blocks.end(); ++iter) {

      for (auto iter2 = iter->begin(); iter2 != iter->end(); ++iter2) {

        bool is_in_result = false;
        for (auto iter3 = result.begin(); iter3 != result.end(); ++iter3) {
          if (((unsigned)iter3->first == iter2->kernel_id) &&
              ((unsigned)iter3->second == iter2->block_id)) {
            is_in_result = true;
            break;
          }
        }
        if (!is_in_result) {
          result.push_back(std::make_pair(iter2->kernel_id, iter2->block_id));
        }
      }
    }
    return result;
  }

  inline int get_trace_issued_sms_num() const { return trace_issued_sms_num; }

  int get_sm_id_of_one_block(unsigned kernel_id, unsigned block_id);
  int get_sm_id_of_one_block_fast(unsigned kernel_id, unsigned block_id);

  std::vector<int> get_trace_issued_sms_vector() {
    return trace_issued_sms_vector;
  }

  inline int serialNum2Index(const int serial_num) const {
    return trace_issued_sms_vector[serial_num];
  }

private:
  bool m_valid;
  int trace_issued_sms_num;

  std::map<std::pair<unsigned, unsigned>, int> trace_issued_sm_id_blocks_map;

  std::vector<block_info_t>
  parse_blocks_info(const std::string &blocks_info_str);
  std::vector<int> trace_issued_sms_vector = {};

  std::vector<std::string> trace_issued_sm_id_blocks_str;
  std::vector<std::vector<block_info_t>> trace_issued_sm_id_blocks;
};

enum mem_instn_type {
  UNKOWN_TYPE = 0,
  RED,
  ATOM,
  LDG,
  STG,
  LDS,
  STS,
  LDL,
  STL,
  num_mem_instn_types,
};

struct mem_instn {
  mem_instn() {}
  mem_instn(unsigned _pc, unsigned long long _addr_start1, unsigned _time_stamp,
            int addr_groups, unsigned long long _addr_start2, unsigned _mask,
            std::string _opcode) {

    pc = _pc;
    time_stamp = _time_stamp;
    mask = _mask;
    std::bitset<32> active_mask(mask);
    opcode = _opcode;
    for (unsigned i = 0; i < 32; i++)
      if (active_mask.test(i))
        addr.push_back(_addr_start1 + i * 8);
    if (addr_groups == 2)
      for (unsigned i = 0; i < 32; i++)
        if (active_mask.test(i))
          addr.push_back(_addr_start2 + i * 8);
    valid = true;
    mem_access_type = has_mem_instn_type();

    distance.resize(addr.size());
    distance_L2.resize(addr.size());
    miss.resize(addr.size());
  }
  mem_instn(unsigned _pc, unsigned long long _addr_start1, unsigned _time_stamp,
            int addr_groups, unsigned long long _addr_start2, unsigned _mask,
            std::string _opcode, std::vector<long long> *_stride_num) {
    pc = _pc;
    time_stamp = _time_stamp;
    mask = _mask;
    std::bitset<32> active_mask(mask);
    opcode = _opcode;
    valid = true;
    mem_access_type = has_mem_instn_type();

    unsigned long long last_addr;
    for (unsigned i = 0; i < 32; i++) {
      if (i == 0) {
        last_addr = _addr_start1;
      } else {
        last_addr += (*_stride_num)[i - 1];
      }
      if (active_mask.test(i))
        addr.push_back(last_addr);
    }
    if (addr_groups == 2) {
      for (unsigned i = 0; i < 32; i++) {
        if (i == 0) {
          last_addr = _addr_start2;
        } else {
          last_addr += (*_stride_num)[31 + i - 1];
        }
        if (active_mask.test(i))
          addr.push_back(last_addr);
      }
    }

    distance.resize(addr.size());
    distance_L2.resize(addr.size());
    miss.resize(addr.size());
  }

  unsigned pc;
  std::vector<unsigned long long> addr;
  unsigned time_stamp;
  bool valid = false;
  unsigned mask;
  std::string opcode;
  enum mem_instn_type mem_access_type;

  std::vector<int> distance;
  std::vector<int> distance_L2;
  std::vector<bool> miss;

  enum mem_instn_type has_mem_instn_type() {
    if (opcode.find("RED") != std::string::npos)
      return RED;
    else if (opcode.find("ATOM") != std::string::npos)
      return ATOM;
    else if (opcode.find("LDG") != std::string::npos)
      return LDG;
    else if (opcode.find("STG") != std::string::npos)
      return STG;
    else if (opcode.find("LDS") != std::string::npos)
      return LDS;
    else if (opcode.find("STS") != std::string::npos)
      return STS;
    else if (opcode.find("LDL") != std::string::npos)
      return LDL;
    else if (opcode.find("STL") != std::string::npos)
      return STL;
    else
      return UNKOWN_TYPE;
  }

#ifdef USE_BOOST
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar &pc;
    ar &addr;
    ar &time_stamp;
    ar &mask;
    ar &opcode;
    ar &valid;
    ar &mem_access_type;
  }
#endif
};

struct compute_instn {
  compute_instn() {}
  compute_instn(unsigned _kernel_id, unsigned _pc, unsigned _mask,
                unsigned _gwarp_id) {
    kernel_id = _kernel_id;
    pc = _pc;
    mask = _mask;
    std::bitset<32> active_mask(mask);
    gwarp_id = _gwarp_id;

    inst_trace = NULL;

    valid = true;
  }
  compute_instn(unsigned _kernel_id, unsigned _pc, unsigned _mask,
                unsigned _gwarp_id, _inst_trace_t *_inst_trace) {
    kernel_id = _kernel_id;
    pc = _pc;
    mask = _mask;
    std::bitset<32> active_mask(mask);
    gwarp_id = _gwarp_id;

    inst_trace = _inst_trace;
    inst_trace->mask = _mask;

    valid = true;
  }
  ~compute_instn() {}
  compute_instn(unsigned _kernel_id, unsigned _pc, unsigned _mask,
                unsigned _gwarp_id, _inst_trace_t *_inst_trace,
                trace_warp_inst_t *_trace_warp_inst) {
    kernel_id = _kernel_id;
    pc = _pc;
    mask = _mask;
    std::bitset<32> active_mask(mask);
    gwarp_id = _gwarp_id;

    inst_trace = _inst_trace;
    inst_trace->mask = _mask;

    trace_warp_inst = trace_warp_inst_t();

    trace_warp_inst.parse_from_trace_struct(_inst_trace, &Volta_OpcodeMap,
                                            gwarp_id);

    inst_stt_ptr = inst_stt();

    valid = true;
  }

  bool valid = false;
  unsigned kernel_id, pc;
  unsigned mask;

  unsigned gwarp_id;

  _inst_trace_t *inst_trace;

  trace_warp_inst_t trace_warp_inst;

  inst_stt inst_stt_ptr;

#ifdef USE_BOOST
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar &pc;
    ar &kernel_id;

    ar &mask;
    ar &gwarp_id;
    ar &valid;
  }
#endif
};

class trace_parser {
public:
  trace_parser(const char *input_configs_filepath);
  trace_parser(const char *input_configs_filepath, hw_config *hw_cfg) {
    configs_filepath = input_configs_filepath;
    this->hw_cfg = hw_cfg;
    m_valid = true;
  }

  void parse_configs_file(bool PRINT_LOG);
  void process_configs_file(std::string config_path, int config_type,
                            bool PRINT_LOG);
  void judge_concurrent_issue();

  void read_mem_instns(bool dump_log, std::vector<std::pair<int, int>> *x);
  void process_mem_instns(std::string mem_instns_filepath, bool PRINT_LOG,
                          std::vector<std::pair<int, int>> *x);

  void read_compute_instns(bool PRINT_LOG, std::vector<std::pair<int, int>> *x);
  void process_compute_instns(std::string compute_instns_dir, bool PRINT_LOG,
                              std::vector<std::pair<int, int>> *x);
  void process_compute_instns_fast(std::string compute_instns_dir,
                                   bool PRINT_LOG,
                                   std::vector<std::pair<int, int>> *x);

  kernel_trace_t *parse_kernel_info(const std::string &kerneltraces_filepath);
  kernel_trace_t *parse_kernel_info(int kernel_id, bool PRINT_LOG);

  void parse_memcpy_info(const std::string &memcpy_command, size_t &add,
                         size_t &count);

  std::vector<std::vector<inst_trace_t> *>
  get_next_threadblock_traces(unsigned trace_version, unsigned enable_lineinfo,
                              std::ifstream *ifs, std::string kernel_name,
                              unsigned kernel_id,
                              unsigned num_warps_per_thread_block);

  void kernel_finalizer(kernel_trace_t *trace_info);

  app_config *get_appcfg() { return &appcfg; }
  instn_config *get_instncfg() { return &instncfg; }
  issue_config *get_issuecfg() { return &issuecfg; }

  std::vector<std::vector<std::vector<mem_instn>>> &get_mem_instns() {
    return mem_instns;
  }
  std::vector<std::vector<mem_instn>> &
  get_one_kernel_mem_instns(int kernel_id) {
    return mem_instns[kernel_id];
  }
  std::vector<mem_instn> &
  get_one_kernel_one_threadblcok_mem_instns(int kernel_id, int block_id) {
    return mem_instns[kernel_id][block_id];
  }

  std::vector<std::vector<int>> &get_concurrent_kernels() {
    return concurrent_kernels;
  }

  compute_instn *get_one_kernel_one_warp_one_instn(int kernel_id, int warp_id,
                                                   int next_instn_id) {
    return &conpute_instns[kernel_id][warp_id][next_instn_id];
  }

  unsigned get_one_kernel_one_warp_one_instn_max_size(int kernel_id,
                                                      int warp_id) {
    return conpute_instns[kernel_id][warp_id].size();
  }

  mem_instn *get_one_kernel_one_block_one_uid_mem_instn(int kernel_id,
                                                        int gwarp_id, int uid) {
    unsigned block_id =
        (unsigned)(gwarp_id / appcfg.get_num_warp_per_block(kernel_id));
    return &mem_instns[kernel_id][block_id][uid];
  }

  unsigned get_one_kernel_one_warp_instn_count(int kernel_id, int warp_id) {
    return conpute_instns[kernel_id][warp_id].size();
  }

  unsigned get_one_kernel_one_warp_instn_size(int kernel_id, int warp_id) {
    return conpute_instns[kernel_id][warp_id].size();
  }

  bool get_m_valid() { return m_valid; }

  unsigned get_the_least_sm_id_of_all_blocks() {

    std::vector<int> trace_issued_sms = issuecfg.get_trace_issued_sms_vector();
    unsigned least_sm_id = trace_issued_sms[0];
    for (auto iter = trace_issued_sms.begin(); iter != trace_issued_sms.end();
         ++iter) {
      if ((unsigned)(*iter) < least_sm_id) {
        least_sm_id = *iter;
      }
    }
    return least_sm_id;
  }

private:
  std::string configs_filepath;
  std::string app_config_path;
  std::string instn_config_path;
  std::string issue_config_path;

  std::string mem_instns_dir;
  std::string compute_instns_dir;

  app_config appcfg;
  instn_config instncfg;
  issue_config issuecfg;
  hw_config *hw_cfg;

  /// Kernel Index -> Block Index -> Mem Instn Indx.
  std::vector<std::vector<std::vector<mem_instn>>> mem_instns;

  std::vector<std::vector<std::vector<compute_instn>>> conpute_instns;

  std::vector<std::vector<int>> concurrent_kernels;

  bool m_valid = false;
};

#endif
