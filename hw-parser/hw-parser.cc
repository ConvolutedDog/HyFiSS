#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>

#include "hw-parser.h"

std::vector<unsigned> hw_config::parse_value(std::string value) {
  std::vector<unsigned> result;
  std::stringstream ss(value);
  std::string token;
  while (std::getline(ss, token, ',')) {
    result.push_back(std::stoi(token));
  }
  return result;
}

std::vector<unsigned> hw_config::parse_value_spec_unit(std::string value) {
  std::vector<unsigned> result;
  std::stringstream ss(value);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (token == "BRA")
      result.push_back(BRA_name);
    else if (token == "TEX")
      result.push_back(TEX_name);
    else if (token == "TENSOR")
      result.push_back(TENSOR_name);
    else {
      result.push_back(std::stoi(token));
    }
  }

  return result;
}

std::string change_to_name(unsigned name) {
  switch (name) {
  case BRA_name:
    return "BRA";
  case TEX_name:
    return "TEX";
  case TENSOR_name:
    return "TENSOR";
  default:
    return "UNKNOWN";
  }
}

void hw_config::init(const std::string config_file) {
  std::stringstream ss;

  std::ifstream inputFile;
  inputFile.open(config_file);
  if (!inputFile.good()) {
    fprintf(stderr, "\n\nOptionParser ** ERROR: Cannot open config file '%s'\n",
            config_file.c_str());
    exit(1);
  }

  while (inputFile.good()) {
    std::string line;
    getline(inputFile, line);
    size_t commentStart = line.find_first_of("#");
    if (commentStart != line.npos)
      continue;
    commentStart = line.find_first_of("-");
    if (commentStart == line.npos)
      continue;

    commentStart = line.find_first_of(" ");

    std::string entry = line.substr(0, commentStart);
    std::string value = line.substr(commentStart + 1);

    configs_str[entry] = value;
  }
  inputFile.close();

  for (auto iter = configs_str.begin(); iter != configs_str.end(); iter++) {
    std::string entry = iter->first;
    std::string value = iter->second;
    if (entry == "-gpgpu_stack_size_limit") {
      stack_size_limit = std::stoi(value);
    } else if (entry == "-gpgpu_heap_size_limit") {
      heap_size_limit = std::stoi(value);
    } else if (entry == "-gpgpu_kernel_launch_latency") {
      kernel_launch_latency = std::stoi(value);
    } else if (entry == "-gpgpu_thread_block_launch_latency") {
      thread_block_launch_latency = std::stoi(value);
    } else if (entry == "-gpgpu_max_concurrent_kernel") {
      max_concurent_kernel = std::stoi(value);
    } else if (entry == "-gpgpu_num_clusters") {
      num_clusters = std::stoi(value);
    } else if (entry == "-gpgpu_num_sms_per_cluster") {
      num_sms_per_cluster = std::stoi(value);
    } else if (entry == "-gpgpu_num_memory_controllers") {
      num_memory_controllers = std::stoi(value);
    } else if (entry == "-gpgpu_num_sub_partition_per_memory_channel") {
      num_sub_partition_per_memory_channel = std::stoi(value);
    } else if (entry == "-gpgpu_core_clock_mhz") {
      core_clock_mhz = std::stoi(value) MHZ;
    } else if (entry == "-gpgpu_icnt_clock_mhz") {
      icnt_clock_mhz = std::stoi(value) MHZ;
    } else if (entry == "-gpgpu_l2d_clock_mhz") {
      l2d_clock_mhz = std::stoi(value) MHZ;
    } else if (entry == "-gpgpu_dram_clock_mhz") {
      dram_clock_mhz = std::stoi(value) MHZ;
    } else if (entry == "-gpgpu_max_registers_per_sm") {
      max_registers_per_sm = std::stoi(value);
    } else if (entry == "-gpgpu_max_registers_per_cta") {
      max_registers_per_cta = std::stoi(value);
    } else if (entry == "-gpgpu_max_threads_per_sm") {
      max_threads_per_sm = std::stoi(value);
    } else if (entry == "-gpgpu_warp_size") {
      warp_size = std::stoi(value);
    } else if (entry == "-gpgpu_max_ctas_per_sm") {
      max_ctas_per_sm = std::stoi(value);
    } else if (entry == "-gpgpu_ID_OC_SP_pipeline_width") {
      ID_OC_SP_pipeline_width = std::stoi(value);
    } else if (entry == "-gpgpu_ID_OC_DP_pipeline_width") {
      ID_OC_DP_pipeline_width = std::stoi(value);
    } else if (entry == "-gpgpu_ID_OC_INT_pipeline_width") {
      ID_OC_INT_pipeline_width = std::stoi(value);
    } else if (entry == "-gpgpu_ID_OC_SFU_pipeline_width") {
      ID_OC_SFU_pipeline_width = std::stoi(value);
    } else if (entry == "-gpgpu_ID_OC_MEM_pipeline_width") {
      ID_OC_MEM_pipeline_width = std::stoi(value);
    } else if (entry == "-gpgpu_OC_EX_SP_pipeline_width") {
      OC_EX_SP_pipeline_width = std::stoi(value);
    } else if (entry == "-gpgpu_OC_EX_DP_pipeline_width") {
      OC_EX_DP_pipeline_width = std::stoi(value);
    } else if (entry == "-gpgpu_OC_EX_INT_pipeline_width") {
      OC_EX_INT_pipeline_width = std::stoi(value);
    } else if (entry == "-gpgpu_OC_EX_SFU_pipeline_width") {
      OC_EX_SFU_pipeline_width = std::stoi(value);
    } else if (entry == "-gpgpu_OC_EX_MEM_pipeline_width") {
      OC_EX_MEM_pipeline_width = std::stoi(value);
    } else if (entry == "-gpgpu_EX_WB_pipeline_width") {
      EX_WB_pipeline_width = std::stoi(value);
    } else if (entry == "-gpgpu_ID_OC_TENSOR_CORE_pipeline_width") {
      ID_OC_TENSOR_CORE_pipeline_width = std::stoi(value);
    } else if (entry == "-gpgpu_OC_EX_TENSOR_CORE_pipeline_width") {
      OC_EX_TENSOR_CORE_pipeline_width = std::stoi(value);
    } else if (entry == "-gpgpu_num_sp_units") {
      num_sp_units = std::stoi(value);
    } else if (entry == "-gpgpu_num_sfu_units") {
      num_sfu_units = std::stoi(value);
    } else if (entry == "-gpgpu_num_dp_units") {
      num_dp_units = std::stoi(value);
    } else if (entry == "-gpgpu_num_int_units") {
      num_int_units = std::stoi(value);
    } else if (entry == "-gpgpu_num_tensor_core_units") {
      num_tensor_core_units = std::stoi(value);
    } else if (entry == "-gpgpu_num_mem_units") {
      num_mem_units = std::stoi(value);
    } else if (entry == "-gpgpu_opcode_latency_int") {
      opcode_latency_int = parse_value(value);
    } else if (entry == "-gpgpu_opcode_latency_fp") {
      opcode_latency_fp = parse_value(value);
    } else if (entry == "-gpgpu_opcode_latency_dp") {
      opcode_latency_dp = parse_value(value);
    } else if (entry == "-gpgpu_opcode_latency_sfu") {
      opcode_latency_sfu = std::stoi(value);
    } else if (entry == "-gpgpu_opcode_latency_tensor_core") {
      opcode_latency_tensor_core = std::stoi(value);
    } else if (entry == "-gpgpu_opcode_initiation_interval_int") {
      opcode_initiation_interval_int = parse_value(value);
    } else if (entry == "-gpgpu_opcode_initiation_interval_fp") {
      opcode_initiation_interval_fp = parse_value(value);
    } else if (entry == "-gpgpu_opcode_initiation_interval_dp") {
      opcode_initiation_interval_dp = parse_value(value);
    } else if (entry == "-gpgpu_opcode_initiation_interval_sfu") {
      opcode_initiation_interval_sfu = std::stoi(value);
    } else if (entry == "-gpgpu_opcode_initiation_interval_tensor_core") {
      opcode_initiation_interval_tensor_core = std::stoi(value);
    } else if (entry == "-gpgpu_sub_core_model") {
      sub_core_model = (bool)std::stoi(value);
    } else if (entry == "-gpgpu_operand_collector_num_units_gen") {
      operand_collector_num_units_gen = std::stoi(value);
    } else if (entry == "-gpgpu_operand_collector_num_in_ports_gen") {
      operand_collector_num_in_ports_gen = std::stoi(value);
    } else if (entry == "-gpgpu_operand_collector_num_out_ports_gen") {
      operand_collector_num_out_ports_gen = std::stoi(value);
    } else if (entry == "-gpgpu_num_reg_banks") {
      num_reg_banks = std::stoi(value);
    } else if (entry == "-gpgpu_reg_file_port_throughput") {
      reg_file_port_throughput = std::stoi(value);
    } else if (entry == "-gpgpu_shmem_num_banks") {
      shmem_num_banks = std::stoi(value);
    } else if (entry == "-gpgpu_shmem_limited_broadcast") {
      shmem_limited_broadcast = std::stoi(value);
    } else if (entry == "-gpgpu_shmem_warp_parts") {
      shmem_warp_parts = std::stoi(value);
    } else if (entry == "-gpgpu_coalesce_arch") {
      coalesce_arch = std::stoi(value);
    } else if (entry == "-gpgpu_num_sched_per_sm") {
      num_sched_per_sm = std::stoi(value);
    } else if (entry == "-gpgpu_inst_fetch_throughput") {
      inst_fetch_throughput = std::stoi(value);
    } else if (entry == "-gpgpu_max_insn_issue_per_warp") {
      max_insn_issue_per_warp = std::stoi(value);
    } else if (entry == "-gpgpu_dual_issue_diff_exec_units") {
      dual_issue_diff_exec_units = (bool)std::stoi(value);
    } else if (entry == "-gpgpu_unified_l1d_size") {
      unified_l1d_size = std::stoi(value);
    } else if (entry == "-gpgpu_l1d_cache_banks") {
      l1d_cache_banks = std::stoi(value);
    } else if (entry == "-gpgpu_l1d_cache_sets") {
      l1d_cache_sets = std::stoi(value);
    } else if (entry == "-gpgpu_l1d_cache_block_size") {
      l1d_cache_block_size = std::stoi(value);
    } else if (entry == "-gpgpu_l1d_cache_associative") {
      l1d_cache_associative = std::stoi(value);
    } else if (entry == "-gpgpu_l1d_latency") {
      l1d_latency = std::stoi(value);
    } else if (entry == "-gpgpu_l1_cache_line_size_for_reuse_distance") {
      l1_cache_line_size_for_reuse_distance = std::stoi(value);
    } else if (entry == "-gpgpu_l2_cache_line_size_for_reuse_distance") {
      l2_cache_line_size_for_reuse_distance = std::stoi(value);
    } else if (entry == "-gpgpu_dram_mem_access_latency") {
      dram_mem_access_latency = std::stoi(value);
    } else if (entry == "-gpgpu_l1_cache_access_latency") {
      l1_access_latency = std::stoi(value);
    } else if (entry == "-gpgpu_l2_cache_access_latency") {
      l2_access_latency = std::stoi(value);
    } else if (entry == "-gpgpu_const_mem_access_latency") {
      const_mem_access_latency = std::stoi(value);
    } else if (entry == "-gpgpu_shmem_size_per_sm") {
      shmem_size_per_sm = std::stoi(value);
    } else if (entry == "-gpgpu_shmem_size_per_cta") {
      shmem_size_per_cta = std::stoi(value);
    } else if (entry == "-gpgpu_smem_allocation_size") {
      smem_allocation_size = std::stoi(value);
    } else if (entry == "-gpgpu_register_allocation_size") {
      register_allocation_size = std::stoi(value);
    } else if (entry == "-gpgpu_shmem_latency") {
      shmem_latency = std::stoi(value);
    } else if (entry == "-gpgpu_l2d_size_per_sub_partition") {
      l2d_size_per_sub_partition = std::stoi(value);
    } else if (entry == "-gpgpu_l2d_cache_sets") {
      l2d_cache_sets = std::stoi(value);
    } else if (entry == "-gpgpu_l2d_cache_block_size") {
      l2d_cache_block_size = std::stoi(value);
    } else if (entry == "-gpgpu_l2d_cache_associative") {
      l2d_cache_associative = std::stoi(value);
    } else if (entry == "-gpgpu_dram_partition_queues_icnt_to_l2") {
      dram_partition_queues_icnt_to_l2 = std::stoi(value);
    } else if (entry == "-gpgpu_dram_partition_queues_l2_to_dram") {
      dram_partition_queues_l2_to_dram = std::stoi(value);
    } else if (entry == "-gpgpu_dram_partition_queues_dram_to_l2") {
      dram_partition_queues_dram_to_l2 = std::stoi(value);
    } else if (entry == "-gpgpu_dram_partition_queues_l2_to_icnt") {
      dram_partition_queues_l2_to_icnt = std::stoi(value);
    } else if (entry == "-gpgpu_num_pkts_cluster_ejection_buffer") {
      num_pkts_cluster_ejection_buffer = std::stoi(value);
    } else if (entry == "-gpgpu_icnt_in_buffer_limit") {
      icnt_in_buffer_limit = std::stoi(value);
    } else if (entry == "-gpgpu_icnt_out_buffer_limit") {
      icnt_out_buffer_limit = std::stoi(value);
    } else if (entry == "-gpgpu_icnt_subnets") {
      icnt_subnets = std::stoi(value);
    } else if (entry == "-gpgpu_icnt_flit_size") {
      icnt_flit_size = std::stoi(value);
    } else if (entry == "-gpgpu_dram_latency") {
      dram_latency = std::stoi(value);
    } else if (entry == "-gpgpu_trace_opcode_latency_initiation_int") {
      auto result = parse_value(value);
      for (unsigned i = 0; i < 2; i++) {
        trace_opcode_latency_initiation_int.push_back(result[i]);
      }

    } else if (entry == "-gpgpu_trace_opcode_latency_initiation_sp") {
      trace_opcode_latency_initiation_sp = parse_value(value);
    } else if (entry == "-gpgpu_trace_opcode_latency_initiation_dp") {
      trace_opcode_latency_initiation_dp = parse_value(value);
    } else if (entry == "-gpgpu_trace_opcode_latency_initiation_sfu") {
      trace_opcode_latency_initiation_sfu = parse_value(value);
    } else if (entry == "-gpgpu_trace_opcode_latency_initiation_tensor") {
      trace_opcode_latency_initiation_tensor = parse_value(value);
    } else if (entry == "-gpgpu_trace_opcode_latency_initiation_spec_op_1") {
      trace_opcode_latency_initiation_spec_op_1 = parse_value(value);
    } else if (entry == "-gpgpu_trace_opcode_latency_initiation_spec_op_2") {
      trace_opcode_latency_initiation_spec_op_2 = parse_value(value);
    } else if (entry == "-gpgpu_trace_opcode_latency_initiation_spec_op_3") {
      trace_opcode_latency_initiation_spec_op_3 = parse_value(value);
    } else if (entry == "-gpgpu_specialized_unit_1") {

      auto result = parse_value_spec_unit(value);
      m_specialized_unit_1_enabled = (bool)result[0];
      num_specialized_unit_1_units = result[1];
      m_specialized_unit_1_max_latency = result[2];
      ID_OC_specialized_unit_1_pipeline_width = result[3];
      OC_EX_specialized_unit_1_pipeline_width = result[4];
      m_specialized_unit_1_name = change_to_name(result[5]);
      if (m_specialized_unit_1_enabled)
        m_specialized_unit_size += 1;
    } else if (entry == "-gpgpu_specialized_unit_2") {
      auto result = parse_value_spec_unit(value);
      m_specialized_unit_2_enabled = (bool)result[0];
      num_specialized_unit_2_units = result[1];
      m_specialized_unit_2_max_latency = result[2];
      ID_OC_specialized_unit_2_pipeline_width = result[3];
      OC_EX_specialized_unit_2_pipeline_width = result[4];
      m_specialized_unit_2_name = change_to_name(result[5]);
      if (m_specialized_unit_2_enabled)
        m_specialized_unit_size += 1;
    } else if (entry == "-gpgpu_specialized_unit_3") {
      auto result = parse_value_spec_unit(value);
      m_specialized_unit_3_enabled = (bool)result[0];
      num_specialized_unit_3_units = result[1];
      m_specialized_unit_3_max_latency = result[2];
      ID_OC_specialized_unit_3_pipeline_width = result[3];
      OC_EX_specialized_unit_3_pipeline_width = result[4];
      m_specialized_unit_3_name = change_to_name(result[5]);
      if (m_specialized_unit_3_enabled)
        m_specialized_unit_size += 1;
    } else {
      std::cout << "Unknown hardware option: " << entry << std::endl;
    }
  }

  max_warps_per_sm = (unsigned)(max_threads_per_sm / warp_size);
  bank_warp_shift = (unsigned)(int)(log(warp_size + 0.5) / log(2.0));

  m_valid = true;
}

#ifdef HW_PARSER_MOUDLE_TEST
int main() {
#else
int hw_parser_test() {
#endif

  std::cout << "hw-parser module test ..." << std::endl;
  const std::string config_file = "../DEV-Def/QV100.config";
  hw_config hw_cfg(config_file);

  std::cout << "stack_size_limit: " << hw_cfg.get_stack_size_limit()
            << std::endl;
  std::cout << "heap_size_limit: " << hw_cfg.get_heap_size_limit() << std::endl;
  std::cout << "kernel_launch_latency: " << hw_cfg.get_kernel_launch_latency()
            << std::endl;
  std::cout << "thread_block_launch_latency: "
            << hw_cfg.get_thread_block_launch_latency() << std::endl;
  std::cout << "max_concurent_kernel: " << hw_cfg.get_max_concurent_kernel()
            << std::endl;
  std::cout << "num_clusters: " << hw_cfg.get_num_clusters() << std::endl;
  std::cout << "num_sms_per_cluster: " << hw_cfg.get_num_sms_per_cluster()
            << std::endl;
  std::cout << "num_memory_controllers: " << hw_cfg.get_num_memory_controllers()
            << std::endl;
  std::cout << "num_sub_partition_per_memory_channel: "
            << hw_cfg.get_num_sub_partition_per_memory_channel() << std::endl;
  std::cout << "core_clock_mhz: " << hw_cfg.get_core_clock_mhz() << std::endl;
  std::cout << "icnt_clock_mhz: " << hw_cfg.get_icnt_clock_mhz() << std::endl;
  std::cout << "l2d_clock_mhz: " << hw_cfg.get_l2d_clock_mhz() << std::endl;
  std::cout << "dram_clock_mhz: " << hw_cfg.get_dram_clock_mhz() << std::endl;
  std::cout << "max_registers_per_sm: " << hw_cfg.get_max_registers_per_sm()
            << std::endl;
  std::cout << "max_registers_per_cta: " << hw_cfg.get_max_registers_per_cta()
            << std::endl;
  std::cout << "max_threads_per_sm: " << hw_cfg.get_max_threads_per_sm()
            << std::endl;
  std::cout << "warp_size: " << hw_cfg.get_warp_size() << std::endl;
  std::cout << "max_ctas_per_sm: " << hw_cfg.get_max_ctas_per_sm() << std::endl;
  std::cout << "ID_OC_SP_pipeline_width: "
            << hw_cfg.get_ID_OC_SP_pipeline_width() << std::endl;
  std::cout << "ID_OC_DP_pipeline_width: "
            << hw_cfg.get_ID_OC_DP_pipeline_width() << std::endl;
  std::cout << "ID_OC_INT_pipeline_width: "
            << hw_cfg.get_ID_OC_INT_pipeline_width() << std::endl;
  std::cout << "ID_OC_SFU_pipeline_width: "
            << hw_cfg.get_ID_OC_SFU_pipeline_width() << std::endl;
  std::cout << "ID_OC_MEM_pipeline_width: "
            << hw_cfg.get_ID_OC_MEM_pipeline_width() << std::endl;
  std::cout << "OC_EX_SP_pipeline_width: "
            << hw_cfg.get_OC_EX_SP_pipeline_width() << std::endl;
  std::cout << "OC_EX_DP_pipeline_width: "
            << hw_cfg.get_OC_EX_DP_pipeline_width() << std::endl;
  std::cout << "OC_EX_INT_pipeline_width: "
            << hw_cfg.get_OC_EX_INT_pipeline_width() << std::endl;
  std::cout << "OC_EX_SFU_pipeline_width: "
            << hw_cfg.get_OC_EX_SFU_pipeline_width() << std::endl;
  std::cout << "OC_EX_MEM_pipeline_width: "
            << hw_cfg.get_OC_EX_MEM_pipeline_width() << std::endl;
  std::cout << "EX_WB_pipeline_width: " << hw_cfg.get_EX_WB_pipeline_width()
            << std::endl;
  std::cout << "ID_OC_TENSOR_CORE_pipeline_width: "
            << hw_cfg.get_ID_OC_TENSOR_CORE_pipeline_width() << std::endl;
  std::cout << "OC_EX_TENSOR_CORE_pipeline_width: "
            << hw_cfg.get_OC_EX_TENSOR_CORE_pipeline_width() << std::endl;
  std::cout << "num_sp_units: " << hw_cfg.get_num_sp_units() << std::endl;
  std::cout << "num_sfu_units: " << hw_cfg.get_num_sfu_units() << std::endl;
  std::cout << "num_dp_units: " << hw_cfg.get_num_dp_units() << std::endl;
  std::cout << "num_int_units: " << hw_cfg.get_num_int_units() << std::endl;
  std::cout << "num_tensor_core_units: " << hw_cfg.get_num_tensor_core_units()
            << std::endl;

  std::cout << "opcode_latency_int_ADD: " << hw_cfg.get_opcode_latency_int(ADD)
            << std::endl;
  std::cout << "opcode_latency_int_MAX: " << hw_cfg.get_opcode_latency_int(MAX)
            << std::endl;
  std::cout << "opcode_latency_int_MUL: " << hw_cfg.get_opcode_latency_int(MUL)
            << std::endl;
  std::cout << "opcode_latency_int_MAD: " << hw_cfg.get_opcode_latency_int(MAD)
            << std::endl;
  std::cout << "opcode_latency_int_DIV: " << hw_cfg.get_opcode_latency_int(DIV)
            << std::endl;
  std::cout << "opcode_latency_int_SHFL: "
            << hw_cfg.get_opcode_latency_int(SHFL) << std::endl;

  std::cout << "opcode_latency_fp_ADD: " << hw_cfg.get_opcode_latency_fp(ADD)
            << std::endl;
  std::cout << "opcode_latency_fp_MAX: " << hw_cfg.get_opcode_latency_fp(MAX)
            << std::endl;
  std::cout << "opcode_latency_fp_MUL: " << hw_cfg.get_opcode_latency_fp(MUL)
            << std::endl;
  std::cout << "opcode_latency_fp_MAD: " << hw_cfg.get_opcode_latency_fp(MAD)
            << std::endl;
  std::cout << "opcode_latency_fp_DIV: " << hw_cfg.get_opcode_latency_fp(DIV)
            << std::endl;

  std::cout << "opcode_latency_dp_ADD: " << hw_cfg.get_opcode_latency_dp(ADD)
            << std::endl;
  std::cout << "opcode_latency_dp_MAX: " << hw_cfg.get_opcode_latency_dp(MAX)
            << std::endl;
  std::cout << "opcode_latency_dp_MUL: " << hw_cfg.get_opcode_latency_dp(MUL)
            << std::endl;
  std::cout << "opcode_latency_dp_MAD: " << hw_cfg.get_opcode_latency_dp(MAD)
            << std::endl;
  std::cout << "opcode_latency_dp_DIV: " << hw_cfg.get_opcode_latency_dp(DIV)
            << std::endl;

  std::cout << "opcode_latency_sfu: " << hw_cfg.get_opcode_latency_sfu()
            << std::endl;

  std::cout << "opcode_latency_tensor_core: "
            << hw_cfg.get_opcode_latency_tensor_core() << std::endl;

  std::cout << "opcode_initiation_interval_int_ADD: "
            << hw_cfg.get_opcode_initiation_interval_int(ADD) << std::endl;
  std::cout << "opcode_initiation_interval_int_MAX: "
            << hw_cfg.get_opcode_initiation_interval_int(MAX) << std::endl;
  std::cout << "opcode_initiation_interval_int_MUL: "
            << hw_cfg.get_opcode_initiation_interval_int(MUL) << std::endl;
  std::cout << "opcode_initiation_interval_int_MAD: "
            << hw_cfg.get_opcode_initiation_interval_int(MAD) << std::endl;
  std::cout << "opcode_initiation_interval_int_DIV: "
            << hw_cfg.get_opcode_initiation_interval_int(DIV) << std::endl;
  std::cout << "opcode_initiation_interval_int_SHFL: "
            << hw_cfg.get_opcode_initiation_interval_int(SHFL) << std::endl;

  std::cout << "opcode_initiation_interval_fp_ADD: "
            << hw_cfg.get_opcode_initiation_interval_fp(ADD) << std::endl;
  std::cout << "opcode_initiation_interval_fp_MAX: "
            << hw_cfg.get_opcode_initiation_interval_fp(MAX) << std::endl;
  std::cout << "opcode_initiation_interval_fp_MUL: "
            << hw_cfg.get_opcode_initiation_interval_fp(MUL) << std::endl;
  std::cout << "opcode_initiation_interval_fp_MAD: "
            << hw_cfg.get_opcode_initiation_interval_fp(MAD) << std::endl;
  std::cout << "opcode_initiation_interval_fp_DIV: "
            << hw_cfg.get_opcode_initiation_interval_fp(DIV) << std::endl;

  std::cout << "opcode_initiation_interval_dp_ADD: "
            << hw_cfg.get_opcode_initiation_interval_dp(ADD) << std::endl;
  std::cout << "opcode_initiation_interval_dp_MAX: "
            << hw_cfg.get_opcode_initiation_interval_dp(MAX) << std::endl;
  std::cout << "opcode_initiation_interval_dp_MUL: "
            << hw_cfg.get_opcode_initiation_interval_dp(MUL) << std::endl;
  std::cout << "opcode_initiation_interval_dp_MAD: "
            << hw_cfg.get_opcode_initiation_interval_dp(MAD) << std::endl;
  std::cout << "opcode_initiation_interval_dp_DIV: "
            << hw_cfg.get_opcode_initiation_interval_dp(DIV) << std::endl;

  std::cout << "opcode_initiation_interval_sfu: "
            << hw_cfg.get_opcode_initiation_interval_sfu() << std::endl;

  std::cout << "opcode_initiation_interval_tensor_core: "
            << hw_cfg.get_opcode_initiation_interval_tensor_core() << std::endl;
  std::cout << "sub_core_model: " << hw_cfg.get_sub_core_model() << std::endl;
  std::cout << "operand_collector_num_units_gen: "
            << hw_cfg.get_operand_collector_num_units_gen() << std::endl;
  std::cout << "operand_collector_num_in_ports_gen: "
            << hw_cfg.get_operand_collector_num_in_ports_gen() << std::endl;
  std::cout << "operand_collector_num_out_ports_gen: "
            << hw_cfg.get_operand_collector_num_out_ports_gen() << std::endl;
  std::cout << "num_reg_banks: " << hw_cfg.get_num_reg_banks() << std::endl;
  std::cout << "reg_file_port_throughput: "
            << hw_cfg.get_reg_file_port_throughput() << std::endl;
  std::cout << "shmem_num_banks: " << hw_cfg.get_shmem_num_banks() << std::endl;
  std::cout << "shmem_limited_broadcast: "
            << hw_cfg.get_shmem_limited_broadcast() << std::endl;
  std::cout << "shmem_warp_parts: " << hw_cfg.get_shmem_warp_parts()
            << std::endl;
  std::cout << "coalesce_arch: " << hw_cfg.get_coalesce_arch() << std::endl;
  std::cout << "num_sched_per_sm: " << hw_cfg.get_num_sched_per_sm()
            << std::endl;
  std::cout << "max_insn_issue_per_warp: "
            << hw_cfg.get_max_insn_issue_per_warp() << std::endl;
  std::cout << "dual_issue_diff_exec_units: "
            << hw_cfg.get_dual_issue_diff_exec_units() << std::endl;
  std::cout << "unified_l1d_size: " << hw_cfg.get_unified_l1d_size()
            << std::endl;
  std::cout << "l1d_cache_banks: " << hw_cfg.get_l1d_cache_banks() << std::endl;
  std::cout << "l1d_cache_sets: " << hw_cfg.get_l1d_cache_sets() << std::endl;
  std::cout << "l1d_cache_block_size: " << hw_cfg.get_l1d_cache_block_size()
            << std::endl;
  std::cout << "l1_cache_line_size_for_reuse_distance: " << hw_cfg.get_l1_cache_line_size_for_reuse_distance()
            << std::endl;
  std::cout << "l2_cache_line_size_for_reuse_distance: " << hw_cfg.get_l2_cache_line_size_for_reuse_distance()
            << std::endl;
  std::cout << "dram_mem_access_latency: " << hw_cfg.get_dram_mem_access_latency() << std::endl;
  std::cout << "l1_access_latency: " << hw_cfg.get_l1_access_latency() << std::endl;
  std::cout << "l2_access_latency: " << hw_cfg.get_l2_access_latency() << std::endl;
  std::cout << "const_mem_access_latency: " << hw_cfg.get_const_mem_access_latency() << std::endl;
  std::cout << "l1d_cache_associative: " << hw_cfg.get_l1d_cache_associative()
            << std::endl;
  std::cout << "l1d_latency: " << hw_cfg.get_l1d_latency() << std::endl;
  std::cout << "shmem_size_per_sm: " << hw_cfg.get_shmem_size_per_sm()
            << std::endl;
  std::cout << "shmem_size_per_cta: " << hw_cfg.get_shmem_size_per_cta()
            << std::endl;
  std::cout << "shmem_latency: " << hw_cfg.get_shmem_latency() << std::endl;
  std::cout << "smem_allocation_size: " << hw_cfg.get_smem_allocation_size() << std::endl;
  std::cout << "register_allocation_size: " << hw_cfg.get_register_allocation_size() << std::endl;
  std::cout << "l2d_size_per_sub_partition: "
            << hw_cfg.get_l2d_size_per_sub_partition() << std::endl;
  std::cout << "l2d_cache_sets: " << hw_cfg.get_l2d_cache_sets() << std::endl;
  std::cout << "l2d_cache_block_size: " << hw_cfg.get_l2d_cache_block_size()
            << std::endl;
  std::cout << "l2d_cache_associative: " << hw_cfg.get_l2d_cache_associative()
            << std::endl;
  std::cout << "dram_partition_queues_icnt_to_l2: "
            << hw_cfg.get_dram_partition_queues_icnt_to_l2() << std::endl;
  std::cout << "dram_partition_queues_l2_to_dram: "
            << hw_cfg.get_dram_partition_queues_l2_to_dram() << std::endl;
  std::cout << "dram_partition_queues_dram_to_l2: "
            << hw_cfg.get_dram_partition_queues_dram_to_l2() << std::endl;
  std::cout << "dram_partition_queues_l2_to_icnt: "
            << hw_cfg.get_dram_partition_queues_l2_to_icnt() << std::endl;
  std::cout << "num_pkts_cluster_ejection_buffer: "
            << hw_cfg.get_num_pkts_cluster_ejection_buffer() << std::endl;
  std::cout << "icnt_in_buffer_limit: " << hw_cfg.get_icnt_in_buffer_limit()
            << std::endl;
  std::cout << "icnt_out_buffer_limit: " << hw_cfg.get_icnt_out_buffer_limit()
            << std::endl;
  std::cout << "icnt_subnets: " << hw_cfg.get_icnt_subnets() << std::endl;
  std::cout << "icnt_flit_size: " << hw_cfg.get_icnt_flit_size() << std::endl;
  std::cout << "dram_latency: " << hw_cfg.get_dram_latency() << std::endl;
  std::cout << "hw-parser module test done." << std::endl;

  std::cout << "trace_opcode_latency_initiation_int: "
            << hw_cfg.get_opcode_latency_initiation_int(0) << " "
            << hw_cfg.get_opcode_latency_initiation_int(1) << std::endl;
  std::cout << "trace_opcode_latency_initiation_sp: "
            << hw_cfg.get_opcode_latency_initiation_sp(0) << " "
            << hw_cfg.get_opcode_latency_initiation_sp(1) << std::endl;
  std::cout << "trace_opcode_latency_initiation_dp: "
            << hw_cfg.get_opcode_latency_initiation_dp(0) << " "
            << hw_cfg.get_opcode_latency_initiation_dp(1) << std::endl;
  std::cout << "trace_opcode_latency_initiation_sfu: "
            << hw_cfg.get_opcode_latency_initiation_sfu(0) << " "
            << hw_cfg.get_opcode_latency_initiation_sfu(1) << std::endl;
  std::cout << "trace_opcode_latency_initiation_tensor_core: "
            << hw_cfg.get_opcode_latency_initiation_tensor_core(0) << " "
            << hw_cfg.get_opcode_latency_initiation_tensor_core(1) << std::endl;
  std::cout << "trace_opcode_latency_initiation_spec_op_1: "
            << hw_cfg.get_opcode_latency_initiation_spec_op_1(0) << " "
            << hw_cfg.get_opcode_latency_initiation_spec_op_1(1) << std::endl;
  std::cout << "trace_opcode_latency_initiation_spec_op_2: "
            << hw_cfg.get_opcode_latency_initiation_spec_op_2(0) << " "
            << hw_cfg.get_opcode_latency_initiation_spec_op_2(1) << std::endl;
  std::cout << "trace_opcode_latency_initiation_spec_op_3: "
            << hw_cfg.get_opcode_latency_initiation_spec_op_3(0) << " "
            << hw_cfg.get_opcode_latency_initiation_spec_op_3(1) << std::endl;
  std::cout << "m_specialized_unit_size: " << hw_cfg.get_specialized_unit_size()
            << std::endl;

  return 0;
}
