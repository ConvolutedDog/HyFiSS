#include "../trace-parser/trace-parser.h"
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <chrono>
#include <sys/stat.h>
#include <unistd.h>

#include "IBuffer.h"

#include "../hw-parser/hw-parser.h"
#include "../trace-driven/entry.h"
#include "../trace-driven/register-set.h"
#include "RegBankAlloc.h"
#include "Scoreboard.h"

#include "OperandCollector.h"

#include "PipelineUnit.h"

#ifndef PRIVATESM_H
#define PRIVATESM_H

#define PRED_NUM_OFFSET 65536
#define MAX_ALU_LATENCY 1024

#define WARP_SIZE 32

#define PRINT_STALLS_DISTRIBUTION 0

enum exec_unit_type_t {
  NONE = 0,
  SP = 1,
  SFU = 2,
  LDST = 3,
  DP = 4,
  INT = 5,
  TENSOR = 6,
  SPECIALIZED = 7
};

struct stage_instns_identifier {
  stage_instns_identifier(unsigned _kid, unsigned _pc, unsigned _wid,
                          unsigned _uid) {
    kid = _kid;
    pc = _pc;
    wid = _wid;
    uid = _uid;
  };
  unsigned kid;
  unsigned pc;
  unsigned wid;
  unsigned uid;
};

#include <fstream>
#include <string>

class stat_collector {
public:
  stat_collector(hw_config* hw_cfg, unsigned kernel_id);
  void set_Unified_L1_cache_hit_rate(float value, unsigned smid) {
    Unified_L1_cache_hit_rate[smid] = value;
  }
  float get_Unified_L1_cache_hit_rate(unsigned smid) {
    return Unified_L1_cache_hit_rate[smid];
  }
  void set_L2_cache_hit_rate(float value) { L2_cache_hit_rate = value; }
  float get_L2_cache_hit_rate() { return L2_cache_hit_rate; }
  void set_L2_cache_requests(unsigned value) { L2_cache_requests = value; }

  void set_Unified_L1_cache_requests(unsigned value, unsigned smid) {
    Unified_L1_cache_requests[smid] = value;
  }

  void print_Unified_L1_cache_hit_rate() {
    for (unsigned i = 0; i < Unified_L1_cache_hit_rate.size(); i++) {
      std::cout << "SM[" << i << "] = " << Unified_L1_cache_hit_rate[i]
                << std::endl;
    }
  }

  unsigned get_warp_size() { return warp_size; }
  unsigned get_smem_allocation_size() { return smem_allocation_size; }
  unsigned get_max_registers_per_SM() { return max_registers_per_SM; }
  unsigned get_max_registers_per_block() { return max_registers_per_block; }
  unsigned get_register_allocation_size() { return register_allocation_size; }
  unsigned get_max_active_blocks_per_SM() { return max_active_blocks_per_SM; }
  unsigned get_max_active_threads_per_SM() { return max_active_threads_per_SM; }
  unsigned get_shared_mem_size() { return shared_mem_size; }
  unsigned get_total_num_workloads() { return total_num_workloads; }
  unsigned get_active_SMs() { return active_SMs; }
  unsigned get_m_num_sm() { return m_num_sm; }
  unsigned get_allocated_active_warps_per_block() {
    return allocated_active_warps_per_block;
  }

  void set_warp_size(unsigned value) { warp_size = value; }
  void set_smem_allocation_size(unsigned value) {
    smem_allocation_size = value;
  }
  void set_max_registers_per_SM(unsigned value) {
    max_registers_per_SM = value;
  }
  void set_max_registers_per_block(unsigned value) {
    max_registers_per_block = value;
  }
  void set_register_allocation_size(unsigned value) {
    register_allocation_size = value;
  }
  void set_max_active_blocks_per_SM(unsigned value) {
    max_active_blocks_per_SM = value;
  }
  void set_max_active_threads_per_SM(unsigned value) {
    max_active_threads_per_SM = value;
  }
  void set_shared_mem_size(unsigned value) { shared_mem_size = value; }
  void set_total_num_workloads(unsigned value) { total_num_workloads = value; }
  void set_active_SMs(unsigned value) { active_SMs = value; }
  void set_m_num_sm(unsigned value) { m_num_sm = value; }
  void set_allocated_active_warps_per_block(unsigned value) {
    allocated_active_warps_per_block = value;
  }

  unsigned get_Thread_block_limit_SM() { return Thread_block_limit_SM; }
  unsigned get_Thread_block_limit_registers() {
    return Thread_block_limit_registers;
  }
  unsigned get_Thread_block_limit_shared_memory() {
    return Thread_block_limit_shared_memory;
  }
  unsigned get_Thread_block_limit_warps() { return Thread_block_limit_warps; }
  unsigned get_Theoretical_max_active_warps_per_SM() {
    return Theoretical_max_active_warps_per_SM;
  }
  float get_Theoretical_occupancy() { return Theoretical_occupancy; }

  void set_Thread_block_limit_SM(unsigned value) {
    Thread_block_limit_SM = value;
  }
  void set_Thread_block_limit_registers(unsigned value) {
    Thread_block_limit_registers = value;
  }
  void set_Thread_block_limit_shared_memory(unsigned value) {
    Thread_block_limit_shared_memory = value;
  }
  void set_Thread_block_limit_warps(unsigned value) {
    Thread_block_limit_warps = value;
  }
  void set_Theoretical_max_active_warps_per_SM(unsigned value) {
    Theoretical_max_active_warps_per_SM = value;
  }
  void set_Theoretical_occupancy(float value) { Theoretical_occupancy = value; }

  unsigned get_allocated_active_blocks_per_SM() {
    return allocated_active_blocks_per_SM;
  }
  void set_allocated_active_blocks_per_SM(unsigned value) {
    allocated_active_blocks_per_SM = value;
  }

  void set_GEMM_read_requests(unsigned value, unsigned smid) {
    GMEM_read_requests[smid] = value;
  }
  unsigned get_GEMM_read_requests(unsigned smid) {
    return GMEM_read_requests[smid];
  }
  void set_GEMM_write_requests(unsigned value, unsigned smid) {
    GMEM_write_requests[smid] = value;
  }
  unsigned get_GEMM_write_requests(unsigned smid) {
    return GMEM_write_requests[smid];
  }
  void set_GEMM_total_requests(unsigned value, unsigned smid) {
    GMEM_total_requests[smid] = value;
  }
  unsigned get_GEMM_total_requests(unsigned smid) {
    return GMEM_total_requests[smid];
  }
  void set_GEMM_read_transactions(unsigned value, unsigned smid) {
    GMEM_read_transactions[smid] = value;
  }
  unsigned get_GEMM_read_transactions(unsigned smid) {
    return GMEM_read_transactions[smid];
  }
  void set_GEMM_write_transactions(unsigned value, unsigned smid) {
    GMEM_write_transactions[smid] = value;
  }
  unsigned get_GEMM_write_transactions(unsigned smid) {
    return GMEM_write_transactions[smid];
  }
  void set_GEMM_total_transactions(unsigned value, unsigned smid) {
    GMEM_total_transactions[smid] = value;
  }
  unsigned get_GEMM_total_transactions(unsigned smid) {
    return GMEM_total_transactions[smid];
  }
  void set_Number_of_read_transactions_per_read_requests(float value,
                                                         unsigned smid) {
    Number_of_read_transactions_per_read_requests[smid] = value;
  }
  float get_Number_of_read_transactions_per_read_requests(unsigned smid) {
    return Number_of_read_transactions_per_read_requests[smid];
  }
  void set_Number_of_write_transactions_per_write_requests(float value,
                                                           unsigned smid) {
    Number_of_write_transactions_per_write_requests[smid] = value;
  }
  float get_Number_of_write_transactions_per_write_requests(unsigned smid) {
    return Number_of_write_transactions_per_write_requests[smid];
  }

  void set_Total_number_of_global_atomic_requests(unsigned value,
                                                  unsigned smid) {
    Total_number_of_global_atomic_requests[smid] = value;
  }
  unsigned get_Total_number_of_global_atomic_requests(unsigned smid) {
    return Total_number_of_global_atomic_requests[smid];
  }
  void set_Total_number_of_global_reduction_requests(unsigned value,
                                                     unsigned smid) {
    Total_number_of_global_reduction_requests[smid] = value;
  }
  unsigned get_Total_number_of_global_reduction_requests(unsigned smid) {
    return Total_number_of_global_reduction_requests[smid];
  }
  void set_Global_memory_atomic_and_reduction_transactions(unsigned value,
                                                           unsigned smid) {
    Global_memory_atomic_and_reduction_transactions[smid] = value;
  }
  unsigned get_Global_memory_atomic_and_reduction_transactions(unsigned smid) {
    return Global_memory_atomic_and_reduction_transactions[smid];
  }

  void set_L2_read_transactions(unsigned value, unsigned smid) {
    L2_read_transactions[smid] = value;
  }
  unsigned get_L2_read_transactions(unsigned smid) {
    return L2_read_transactions[smid];
  }
  void set_L2_write_transactions(unsigned value, unsigned smid) {
    L2_write_transactions[smid] = value;
  }
  unsigned get_L2_write_transactions(unsigned smid) {
    return L2_write_transactions[smid];
  }
  void set_L2_total_transactions(unsigned value, unsigned smid) {
    L2_total_transactions[smid] = value;
  }
  unsigned get_L2_total_transactions(unsigned smid) {
    return L2_total_transactions[smid];
  }

  void set_DRAM_total_transactions(unsigned value) {
    DRAM_total_transactions = value;
  }
  unsigned get_DRAM_total_transactions() { return DRAM_total_transactions; }

  void set_GPU_active_cycles(unsigned value, unsigned smid) {
    GPU_active_cycles[smid] = value;
  }
  unsigned get_GPU_active_cycles(unsigned smid) {
    return GPU_active_cycles[smid];
  }
  void set_SM_active_cycles(unsigned value, unsigned smid) {
    SM_active_cycles[smid] = value;
  }
  unsigned get_SM_active_cycles(unsigned smid) {
    return SM_active_cycles[smid];
  }

  void set_Warp_instructions_executed(unsigned value, unsigned smid) {
    Warp_instructions_executed[smid] = value;
  }
  unsigned get_Warp_instructions_executed(unsigned smid) {
    return Warp_instructions_executed[smid];
  }
  void set_Instructions_executed_per_clock_cycle_IPC(float value,
                                                     unsigned smid) {
    Instructions_executed_per_clock_cycle_IPC[smid] = value;
  }
  float get_Instructions_executed_per_clock_cycle_IPC(unsigned smid) {
    return Instructions_executed_per_clock_cycle_IPC[smid];
  }
  void set_Total_instructions_executed_per_seconds(float value, unsigned smid) {
    Total_instructions_executed_per_seconds[smid] = value;
  }
  float get_Total_instructions_executed_per_seconds(unsigned smid) {
    return Total_instructions_executed_per_seconds[smid];
  }

  void set_Kernel_execution_time(unsigned value, unsigned smid) {
    Kernel_execution_time[smid] = value;
  }
  unsigned get_Kernel_execution_time(unsigned smid) {
    return Kernel_execution_time[smid];
  }

  void set_Simulation_time_memory_model(float value, unsigned smid) {
    Simulation_time_memory_model[smid] = value;
  }
  float get_Simulation_time_memory_model(unsigned smid) {
    return Simulation_time_memory_model[smid];
  }
  void set_Simulation_time_compute_model(float value, unsigned smid) {
    Simulation_time_compute_model[smid] = value;
  }
  float get_Simulation_time_compute_model(unsigned smid) {
    return Simulation_time_compute_model[smid];
  }

  unsigned get_kernel_id() { return kernel_id; }
  void set_kernel_id(unsigned value) { kernel_id = value; }

  void set_Achieved_active_warps_per_SM(float value, unsigned smid) {
    Achieved_active_warps_per_SM[smid] = value;
  }
  float get_Achieved_active_warps_per_SM(unsigned smid) {
    return Achieved_active_warps_per_SM[smid];
  }
  void set_Achieved_occupancy(float value, unsigned smid) {
    Achieved_occupancy[smid] = value;
  }
  float get_Achieved_occupancy(unsigned smid) {
    return Achieved_occupancy[smid];
  }

  void set_Compute_Structural_Stall(unsigned value, unsigned smid) {
    Compute_Structural_Stall[smid] = value;
  }
  void increment_Compute_Structural_Stall(unsigned smid) {
    Compute_Structural_Stall[smid]++;
  }
  unsigned get_Compute_Structural_Stall(unsigned smid) {
    return Compute_Structural_Stall[smid];
  }
  void set_Compute_Data_Stall(unsigned value, unsigned smid) {
    Compute_Data_Stall[smid] = value;
  }
  void increment_Compute_Data_Stall(unsigned smid) {
    Compute_Data_Stall[smid]++;
  }
  unsigned get_Compute_Data_Stall(unsigned smid) {
    return Compute_Data_Stall[smid];
  }
  void set_Memory_Structural_Stall(unsigned value, unsigned smid) {
    Memory_Structural_Stall[smid] = value;
  }
  void increment_Memory_Structural_Stall(unsigned smid) {
    Memory_Structural_Stall[smid]++;
  }
  unsigned get_Memory_Structural_Stall(unsigned smid) {
    return Memory_Structural_Stall[smid];
  }
  void set_Memory_Data_Stall(unsigned value, unsigned smid) {
    Memory_Data_Stall[smid] = value;
  }
  void increment_Memory_Data_Stall(unsigned smid) { Memory_Data_Stall[smid]++; }
  unsigned get_Memory_Data_Stall(unsigned smid) {
    return Memory_Data_Stall[smid];
  }
  void set_Synchronization_Stall(unsigned value, unsigned smid) {
    Synchronization_Stall[smid] = value;
  }
  void increment_Synchronization_Stall(unsigned smid) {
    Synchronization_Stall[smid]++;
  }
  unsigned get_Synchronization_Stall(unsigned smid) {
    return Synchronization_Stall[smid];
  }
  void set_Control_Stall(unsigned value, unsigned smid) {
    Control_Stall[smid] = value;
  }
  void increment_Control_Stall(unsigned smid) { Control_Stall[smid]++; }
  unsigned get_Control_Stall(unsigned smid) { return Control_Stall[smid]; }
  void set_Idle_Stall(unsigned value, unsigned smid) {
    Idle_Stall[smid] = value;
  }
  void increment_Idle_Stall(unsigned smid) { Idle_Stall[smid]++; }
  unsigned get_Idle_Stall(unsigned smid) { return Idle_Stall[smid]; }
  void set_No_Stall(unsigned value, unsigned smid) { No_Stall[smid] = value; }
  void increment_No_Stall(unsigned smid) { No_Stall[smid]++; }
  unsigned get_No_Stall(unsigned smid) { return No_Stall[smid]; }
  void set_Other_Stall(unsigned value, unsigned smid) {
    Other_Stall[smid] = value;
  }
  void increment_Other_Stall(unsigned smid) { Other_Stall[smid]++; }
  unsigned get_Other_Stall(unsigned smid) { return Other_Stall[smid]; }

  void set_At_least_four_instns_issued(bool value) {
    At_least_four_instns_issued = value;
  }
  bool get_At_least_four_instns_issued() { return At_least_four_instns_issued; }
  void set_At_least_one_Compute_Structural_Stall_found(bool value) {
    At_least_one_Compute_Structural_Stall_found = value;
  }
  bool get_At_least_one_Compute_Structural_Stall_found() {
    return At_least_one_Compute_Structural_Stall_found;
  }
  void set_At_least_one_Compute_Data_Stall_found(bool value) {
    At_least_one_Compute_Data_Stall_found = value;
  }
  bool get_At_least_one_Compute_Data_Stall_found() {
    return At_least_one_Compute_Data_Stall_found;
  }
  void set_At_least_one_Memory_Structural_Stall_found(bool value) {
    At_least_one_Memory_Structural_Stall_found = value;
  }
  bool get_At_least_one_Memory_Structural_Stall_found() {
    return At_least_one_Memory_Structural_Stall_found;
  }
  void set_At_least_one_Memory_Data_Stall_found(bool value) {
    At_least_one_Memory_Data_Stall_found = value;
  }
  bool get_At_least_one_Memory_Data_Stall_found() {
    return At_least_one_Memory_Data_Stall_found;
  }
  void set_At_least_one_Synchronization_Stall_found(bool value) {
    At_least_one_Synchronization_Stall_found = value;
  }
  bool get_At_least_one_Synchronization_Stall_found() {
    return At_least_one_Synchronization_Stall_found;
  }
  void set_At_least_one_Control_Stall_found(bool value) {
    At_least_one_Control_Stall_found = value;
  }
  bool get_At_least_one_Control_Stall_found() {
    return At_least_one_Control_Stall_found;
  }
  void set_At_least_one_Idle_Stall_found(bool value) {
    At_least_one_Idle_Stall_found = value;
  }
  bool get_At_least_one_Idle_Stall_found() {
    return At_least_one_Idle_Stall_found;
  }
  void set_At_least_one_No_Stall_found(bool value) {
    At_least_one_No_Stall_found = value;
  }
  bool get_At_least_one_No_Stall_found() { return At_least_one_No_Stall_found; }

  void set_num_Issue_Compute_Structural_out_has_no_free_slot(unsigned value,
                                                             unsigned smid) {
    num_Issue_Compute_Structural_out_has_no_free_slot[smid] = value;
  }
  void
  increment_num_Issue_Compute_Structural_out_has_no_free_slot(unsigned smid) {
    num_Issue_Compute_Structural_out_has_no_free_slot[smid]++;
  }
  void
  increment_num_Issue_Compute_Structural_out_has_no_free_slot(unsigned value,
                                                              unsigned smid) {
    num_Issue_Compute_Structural_out_has_no_free_slot[smid] += value;
  }
  unsigned
  get_num_Issue_Compute_Structural_out_has_no_free_slot(unsigned smid) {
    return num_Issue_Compute_Structural_out_has_no_free_slot[smid];
  }
  void set_num_Issue_Memory_Structural_out_has_no_free_slot(unsigned value,
                                                            unsigned smid) {
    num_Issue_Memory_Structural_out_has_no_free_slot[smid] = value;
  }
  void
  increment_num_Issue_Memory_Structural_out_has_no_free_slot(unsigned smid) {
    num_Issue_Memory_Structural_out_has_no_free_slot[smid]++;
  }
  void
  increment_num_Issue_Memory_Structural_out_has_no_free_slot(unsigned value,
                                                             unsigned smid) {
    num_Issue_Memory_Structural_out_has_no_free_slot[smid] += value;
  }
  unsigned get_num_Issue_Memory_Structural_out_has_no_free_slot(unsigned smid) {
    return num_Issue_Memory_Structural_out_has_no_free_slot[smid];
  }
  void
  set_num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute(
      unsigned value, unsigned smid) {
    num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute
        [smid] = value;
  }
  void
  increment_num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute(
      unsigned smid) {
    num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute
        [smid]++;
  }
  void
  increment_num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute(
      unsigned value, unsigned smid) {
    num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute
        [smid] += value;
  }
  unsigned
  get_num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute(
      unsigned smid) {
    return num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute
        [smid];
  }
  void set_num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory(
      unsigned value, unsigned smid) {
    num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory[smid] =
        value;
  }
  void
  increment_num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory(
      unsigned smid) {
    num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory
        [smid]++;
  }
  void
  increment_num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory(
      unsigned value, unsigned smid) {
    num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory
        [smid] += value;
  }
  unsigned
  get_num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory(
      unsigned smid) {
    return num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory
        [smid];
  }
  void set_num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency(
      unsigned value, unsigned smid) {
    num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency[smid] =
        value;
  }
  void
  increment_num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency(
      unsigned smid) {
    num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency[smid]++;
  }
  void
  increment_num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency(
      unsigned value, unsigned smid) {
    num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency[smid] +=
        value;
  }
  unsigned
  get_num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency(
      unsigned smid) {
    return num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency
        [smid];
  }
  void set_num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency(
      unsigned value, unsigned smid) {
    num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency[smid] =
        value;
  }
  void
  increment_num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency(
      unsigned smid) {
    num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency[smid]++;
  }
  void
  increment_num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency(
      unsigned value, unsigned smid) {
    num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency[smid] +=
        value;
  }
  unsigned get_num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency(
      unsigned smid) {
    return num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency
        [smid];
  }
  void set_num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty(
      unsigned value, unsigned smid) {
    num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty[smid] =
        value;
  }
  void
  increment_num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty(
      unsigned smid) {
    num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty[smid]++;
  }
  void
  increment_num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty(
      unsigned value, unsigned smid) {
    num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty[smid] +=
        value;
  }
  unsigned get_num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty(
      unsigned smid) {
    return num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty
        [smid];
  }
  void set_num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty(
      unsigned value, unsigned smid) {
    num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty[smid] =
        value;
  }
  void
  increment_num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty(
      unsigned smid) {
    num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty[smid]++;
  }
  void
  increment_num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty(
      unsigned value, unsigned smid) {
    num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty[smid] +=
        value;
  }
  unsigned get_num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty(
      unsigned smid) {
    return num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty
        [smid];
  }
  void
  set_num_Writeback_Compute_Structural_bank_of_reg_is_not_idle(unsigned value,
                                                               unsigned smid) {
    num_Writeback_Compute_Structural_bank_of_reg_is_not_idle[smid] = value;
  }
  void increment_num_Writeback_Compute_Structural_bank_of_reg_is_not_idle(
      unsigned smid) {
    num_Writeback_Compute_Structural_bank_of_reg_is_not_idle[smid]++;
  }
  void increment_num_Writeback_Compute_Structural_bank_of_reg_is_not_idle(
      unsigned value, unsigned smid) {
    num_Writeback_Compute_Structural_bank_of_reg_is_not_idle[smid] += value;
  }
  unsigned
  get_num_Writeback_Compute_Structural_bank_of_reg_is_not_idle(unsigned smid) {
    return num_Writeback_Compute_Structural_bank_of_reg_is_not_idle[smid];
  }
  void
  set_num_Writeback_Memory_Structural_bank_of_reg_is_not_idle(unsigned value,
                                                              unsigned smid) {
    num_Writeback_Memory_Structural_bank_of_reg_is_not_idle[smid] = value;
  }
  void increment_num_Writeback_Memory_Structural_bank_of_reg_is_not_idle(
      unsigned smid) {
    num_Writeback_Memory_Structural_bank_of_reg_is_not_idle[smid]++;
  }
  void increment_num_Writeback_Memory_Structural_bank_of_reg_is_not_idle(
      unsigned value, unsigned smid) {
    num_Writeback_Memory_Structural_bank_of_reg_is_not_idle[smid] += value;
  }
  unsigned
  get_num_Writeback_Memory_Structural_bank_of_reg_is_not_idle(unsigned smid) {
    return num_Writeback_Memory_Structural_bank_of_reg_is_not_idle[smid];
  }
  void
  set_num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated(
      unsigned value, unsigned smid) {
    num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated
        [smid] = value;
  }
  void
  increment_num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated(
      unsigned smid) {
    num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated
        [smid]++;
  }
  void
  increment_num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated(
      unsigned value, unsigned smid) {
    num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated
        [smid] += value;
  }
  unsigned
  get_num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated(
      unsigned smid) {
    return num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated
        [smid];
  }
  void
  set_num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated(
      unsigned value, unsigned smid) {
    num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated
        [smid] = value;
  }
  void
  increment_num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated(
      unsigned smid) {
    num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated
        [smid]++;
  }
  void
  increment_num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated(
      unsigned value, unsigned smid) {
    num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated
        [smid] += value;
  }
  unsigned
  get_num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated(
      unsigned smid) {
    return num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated
        [smid];
  }
  void
  set_num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu(
      unsigned value, unsigned smid) {
    num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu
        [smid] = value;
  }
  void
  increment_num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu(
      unsigned smid) {
    num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu
        [smid]++;
  }
  void
  increment_num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu(
      unsigned value, unsigned smid) {
    num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu
        [smid] += value;
  }
  unsigned
  get_num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu(
      unsigned smid) {
    return num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu
        [smid];
  }
  void
  set_num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu(
      unsigned value, unsigned smid) {
    num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu
        [smid] = value;
  }
  void
  increment_num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu(
      unsigned smid) {
    num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu
        [smid]++;
  }
  void
  increment_num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu(
      unsigned value, unsigned smid) {
    num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu
        [smid] += value;
  }
  unsigned
  get_num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu(
      unsigned smid) {
    return num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu
        [smid];
  }
  void set_num_Execute_Memory_Structural_icnt_injection_buffer_is_full(
      unsigned value, unsigned smid) {
    num_Execute_Memory_Structural_icnt_injection_buffer_is_full[smid] = value;
  }
  void increment_num_Execute_Memory_Structural_icnt_injection_buffer_is_full(
      unsigned smid) {
    num_Execute_Memory_Structural_icnt_injection_buffer_is_full[smid]++;
  }
  void increment_num_Execute_Memory_Structural_icnt_injection_buffer_is_full(
      unsigned value, unsigned smid) {
    num_Execute_Memory_Structural_icnt_injection_buffer_is_full[smid] += value;
  }
  unsigned get_num_Execute_Memory_Structural_icnt_injection_buffer_is_full(
      unsigned smid) {
    return num_Execute_Memory_Structural_icnt_injection_buffer_is_full[smid];
  }
  void set_num_Issue_Compute_Data_scoreboard(unsigned value, unsigned smid) {
    num_Issue_Compute_Data_scoreboard[smid] = value;
  }
  void increment_num_Issue_Compute_Data_scoreboard(unsigned smid) {
    num_Issue_Compute_Data_scoreboard[smid]++;
  }
  void increment_num_Issue_Compute_Data_scoreboard(unsigned value,
                                                   unsigned smid) {
    num_Issue_Compute_Data_scoreboard[smid] += value;
  }
  unsigned get_num_Issue_Compute_Data_scoreboard(unsigned smid) {
    return num_Issue_Compute_Data_scoreboard[smid];
  }
  void set_num_Issue_Memory_Data_scoreboard(unsigned value, unsigned smid) {
    num_Issue_Memory_Data_scoreboard[smid] = value;
  }
  void increment_num_Issue_Memory_Data_scoreboard(unsigned smid) {
    num_Issue_Memory_Data_scoreboard[smid]++;
  }
  void increment_num_Issue_Memory_Data_scoreboard(unsigned value,
                                                  unsigned smid) {
    num_Issue_Memory_Data_scoreboard[smid] += value;
  }
  unsigned get_num_Issue_Memory_Data_scoreboard(unsigned smid) {
    return num_Issue_Memory_Data_scoreboard[smid];
  }
  void set_num_Execute_Memory_Data_L1(unsigned value, unsigned smid) {
    num_Execute_Memory_Data_L1[smid] = value;
  }
  void increment_num_Execute_Memory_Data_L1(unsigned smid) {
    num_Execute_Memory_Data_L1[smid]++;
  }
  void increment_num_Execute_Memory_Data_L1(unsigned value, unsigned smid) {
    num_Execute_Memory_Data_L1[smid] += value;
  }
  unsigned get_num_Execute_Memory_Data_L1(unsigned smid) {
    return num_Execute_Memory_Data_L1[smid];
  }
  void set_num_Execute_Memory_Data_L2(unsigned value, unsigned smid) {
    num_Execute_Memory_Data_L2[smid] = value;
  }
  void increment_num_Execute_Memory_Data_L2(unsigned smid) {
    num_Execute_Memory_Data_L2[smid]++;
  }
  void increment_num_Execute_Memory_Data_L2(unsigned value, unsigned smid) {
    num_Execute_Memory_Data_L2[smid] += value;
  }
  unsigned get_num_Execute_Memory_Data_L2(unsigned smid) {
    return num_Execute_Memory_Data_L2[smid];
  }
  void set_num_Execute_Memory_Data_Main_Memory(unsigned value, unsigned smid) {
    num_Execute_Memory_Data_Main_Memory[smid] = value;
  }
  void increment_num_Execute_Memory_Data_Main_Memory(unsigned smid) {
    num_Execute_Memory_Data_Main_Memory[smid]++;
  }
  void increment_num_Execute_Memory_Data_Main_Memory(unsigned value,
                                                     unsigned smid) {
    num_Execute_Memory_Data_Main_Memory[smid] += value;
  }
  unsigned get_num_Execute_Memory_Data_Main_Memory(unsigned smid) {
    return num_Execute_Memory_Data_Main_Memory[smid];
  }

  void increment_SP_UNIT_execute_clks_sum(unsigned smid,
                                          unsigned long long value) {
    SP_UNIT_execute_clks_sum[smid] += value;
  }
  void increment_SFU_UNIT_execute_clks_sum(unsigned smid,
                                           unsigned long long value) {
    SFU_UNIT_execute_clks_sum[smid] += value;
  }
  void increment_INT_UNIT_execute_clks_sum(unsigned smid,
                                           unsigned long long value) {
    INT_UNIT_execute_clks_sum[smid] += value;
  }
  void increment_DP_UNIT_execute_clks_sum(unsigned smid,
                                          unsigned long long value) {
    DP_UNIT_execute_clks_sum[smid] += value;
  }
  void increment_TENSOR_CORE_UNIT_execute_clks_sum(unsigned smid,
                                                   unsigned long long value) {
    TENSOR_CORE_UNIT_execute_clks_sum[smid] += value;
  }
  void increment_LDST_UNIT_execute_clks_sum(unsigned smid,
                                            unsigned long long value) {
    LDST_UNIT_execute_clks_sum[smid] += value;
  }
  void increment_SPEC_UNIT_1_execute_clks_sum(unsigned smid,
                                              unsigned long long value) {
    SPEC_UNIT_1_execute_clks_sum[smid] += value;
  }
  void increment_SPEC_UNIT_2_execute_clks_sum(unsigned smid,
                                              unsigned long long value) {
    SPEC_UNIT_2_execute_clks_sum[smid] += value;
  }
  void increment_SPEC_UNIT_3_execute_clks_sum(unsigned smid,
                                              unsigned long long value) {
    SPEC_UNIT_3_execute_clks_sum[smid] += value;
  }
  void increment_Other_UNIT_execute_clks_sum(unsigned smid,
                                             unsigned long long value) {
    Other_UNIT_execute_clks_sum[smid] += value;
  }

  void increment_SP_UNIT_Instns_num(unsigned smid) {
    SP_UNIT_Instns_num[smid]++;
  }
  void increment_SFU_UNIT_Instns_num(unsigned smid) {
    SFU_UNIT_Instns_num[smid]++;
  }
  void increment_INT_UNIT_Instns_num(unsigned smid) {
    INT_UNIT_Instns_num[smid]++;
  }
  void increment_DP_UNIT_Instns_num(unsigned smid) {
    DP_UNIT_Instns_num[smid]++;
  }
  void increment_TENSOR_CORE_UNIT_Instns_num(unsigned smid) {
    TENSOR_CORE_UNIT_Instns_num[smid]++;
  }
  void increment_LDST_UNIT_Instns_num(unsigned smid) {
    LDST_UNIT_Instns_num[smid]++;
  }
  void increment_SPEC_UNIT_1_Instns_num(unsigned smid) {
    SPEC_UNIT_1_Instns_num[smid]++;
  }
  void increment_SPEC_UNIT_2_Instns_num(unsigned smid) {
    SPEC_UNIT_2_Instns_num[smid]++;
  }
  void increment_SPEC_UNIT_3_Instns_num(unsigned smid) {
    SPEC_UNIT_3_Instns_num[smid]++;
  }
  void increment_Other_UNIT_Instns_num(unsigned smid) {
    Other_UNIT_Instns_num[smid]++;
  }

  void dump_output(const std::string &path, unsigned rank);

private:
  unsigned kernel_id;

  unsigned Thread_block_limit_SM;
  unsigned Thread_block_limit_registers;
  unsigned Thread_block_limit_shared_memory;
  unsigned Thread_block_limit_warps;
  unsigned Theoretical_max_active_warps_per_SM;
  float Theoretical_occupancy;

  std::vector<float> Achieved_active_warps_per_SM;
  std::vector<float> Achieved_occupancy;
  std::vector<float> Unified_L1_cache_hit_rate;
  std::vector<unsigned> Unified_L1_cache_requests;
  std::vector<float> Unified_L1_cache_hit_rate_for_read_transactions;
  float L2_cache_hit_rate;
  unsigned L2_cache_requests;
  std::vector<unsigned> GMEM_read_requests;
  std::vector<unsigned> GMEM_write_requests;
  std::vector<unsigned> GMEM_total_requests;
  std::vector<unsigned> GMEM_read_transactions;
  std::vector<unsigned> GMEM_write_transactions;
  std::vector<unsigned> GMEM_total_transactions;
  std::vector<float> Number_of_read_transactions_per_read_requests;
  std::vector<float> Number_of_write_transactions_per_write_requests;

  std::vector<unsigned> L2_read_transactions;
  std::vector<unsigned> L2_write_transactions;
  std::vector<unsigned> L2_total_transactions;
  unsigned DRAM_total_transactions;

  std::vector<unsigned> Total_number_of_global_atomic_requests;
  std::vector<unsigned> Total_number_of_global_reduction_requests;
  std::vector<unsigned> Global_memory_atomic_and_reduction_transactions;

  std::vector<unsigned> GPU_active_cycles;
  std::vector<unsigned> SM_active_cycles;
  std::vector<unsigned> Warp_instructions_executed;
  std::vector<float> Instructions_executed_per_clock_cycle_IPC;
  std::vector<float> Total_instructions_executed_per_seconds;
  std::vector<unsigned> Kernel_execution_time;

  std::vector<float> Simulation_time_memory_model;
  std::vector<float> Simulation_time_compute_model;

  unsigned m_num_sm;
  unsigned warp_size;
  unsigned smem_allocation_size;
  unsigned max_registers_per_SM;
  unsigned max_registers_per_block;
  unsigned register_allocation_size;
  unsigned max_active_blocks_per_SM;
  unsigned max_active_threads_per_SM;

  unsigned shared_mem_size;

  unsigned total_num_workloads;
  unsigned active_SMs;
  unsigned allocated_active_warps_per_block;
  unsigned allocated_active_blocks_per_SM;

  bool At_least_four_instns_issued;

  std::vector<unsigned> Compute_Structural_Stall;
  bool At_least_one_Compute_Structural_Stall_found;
  unsigned num_At_least_one_Compute_Structural_Stall_found;

  std::vector<unsigned> Compute_Data_Stall;
  bool At_least_one_Compute_Data_Stall_found;
  unsigned num_At_least_one_Compute_Data_Stall_found;

  std::vector<unsigned> Memory_Structural_Stall;
  bool At_least_one_Memory_Structural_Stall_found;
  unsigned num_At_least_one_Memory_Structural_Stall_found;

  std::vector<unsigned> Memory_Data_Stall;
  bool At_least_one_Memory_Data_Stall_found;
  unsigned num_At_least_one_Memory_Data_Stall_found;

  std::vector<unsigned> Synchronization_Stall;
  bool At_least_one_Synchronization_Stall_found;
  unsigned num_At_least_one_Synchronization_Stall_found;

  std::vector<unsigned> Control_Stall;
  bool At_least_one_Control_Stall_found;
  unsigned num_At_least_one_Control_Stall_found;

  std::vector<unsigned> Idle_Stall;
  bool At_least_one_Idle_Stall_found;
  unsigned num_At_least_one_Idle_Stall_found;

  std::vector<unsigned> No_Stall;
  bool At_least_one_No_Stall_found;
  unsigned num_At_least_one_No_Stall_found;

  std::vector<unsigned> Other_Stall;

  std::vector<unsigned> num_Issue_Compute_Structural_out_has_no_free_slot;
  std::vector<unsigned> num_Issue_Memory_Structural_out_has_no_free_slot;
  std::vector<unsigned>
      num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute;
  std::vector<unsigned>
      num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory;
  std::vector<unsigned>
      num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency;
  std::vector<unsigned>
      num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency;
  std::vector<unsigned>
      num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty;
  std::vector<unsigned>
      num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty;
  std::vector<unsigned>
      num_Writeback_Compute_Structural_bank_of_reg_is_not_idle;
  std::vector<unsigned> num_Writeback_Memory_Structural_bank_of_reg_is_not_idle;
  std::vector<unsigned>
      num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated;
  std::vector<unsigned>
      num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated;
  std::vector<unsigned>
      num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu;
  std::vector<unsigned>
      num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu;
  std::vector<unsigned>
      num_Execute_Memory_Structural_icnt_injection_buffer_is_full;
  std::vector<unsigned> num_Issue_Compute_Data_scoreboard;
  std::vector<unsigned> num_Issue_Memory_Data_scoreboard;
  std::vector<unsigned> num_Execute_Memory_Data_L1;
  std::vector<unsigned> num_Execute_Memory_Data_L2;
  std::vector<unsigned> num_Execute_Memory_Data_Main_Memory;

  std::vector<unsigned long long> SP_UNIT_execute_clks_sum;
  std::vector<unsigned long long> SFU_UNIT_execute_clks_sum;
  std::vector<unsigned long long> INT_UNIT_execute_clks_sum;
  std::vector<unsigned long long> DP_UNIT_execute_clks_sum;
  std::vector<unsigned long long> TENSOR_CORE_UNIT_execute_clks_sum;
  std::vector<unsigned long long> LDST_UNIT_execute_clks_sum;
  std::vector<unsigned long long> SPEC_UNIT_1_execute_clks_sum;
  std::vector<unsigned long long> SPEC_UNIT_2_execute_clks_sum;
  std::vector<unsigned long long> SPEC_UNIT_3_execute_clks_sum;
  std::vector<unsigned long long> Other_UNIT_execute_clks_sum;

  std::vector<unsigned long long> SP_UNIT_Instns_num;
  std::vector<unsigned long long> SFU_UNIT_Instns_num;
  std::vector<unsigned long long> INT_UNIT_Instns_num;
  std::vector<unsigned long long> DP_UNIT_Instns_num;
  std::vector<unsigned long long> TENSOR_CORE_UNIT_Instns_num;
  std::vector<unsigned long long> LDST_UNIT_Instns_num;
  std::vector<unsigned long long> SPEC_UNIT_1_Instns_num;
  std::vector<unsigned long long> SPEC_UNIT_2_Instns_num;
  std::vector<unsigned long long> SPEC_UNIT_3_Instns_num;
  std::vector<unsigned long long> Other_UNIT_Instns_num;
};

class PrivateSM {
public:
  PrivateSM(const unsigned smid, trace_parser *tracer, hw_config *hw_cfg);
  ~PrivateSM();
  void run(const unsigned KERNEL_EVALUATION, const unsigned MEM_ACCESS_LATENCY,
           stat_collector *stat_coll);

  bool get_active() { return active; }
  unsigned long long get_cycle() { return m_cycle; }
  void set_cycle(unsigned long long value) { m_cycle = value; }
  void increment_cycle(unsigned long long value) { m_cycle += value; }

  bool is_active() { return active; }
  bool check_active();

  unsigned get_num_warps_per_sm(unsigned kernel_id);

  unsigned get_num_warp_instns_executed() { return num_warp_instns_executed; }

  hw_config *get_hw_cfg() { return m_hw_cfg; }
  trace_parser *get_tracer() { return tracer; }

  int register_bank(int regnum, int wid, unsigned sched_id);

  void parse_blocks_per_kernel();

  std::vector<unsigned> get_blocks_per_kernel(unsigned kernel_id);

  std::map<unsigned, std::vector<unsigned>> *get_blocks_per_kernel();

  unsigned get_inst_fetch_throughput();
  unsigned get_reg_file_port_throughput();

  void issue_warp(register_set &pipe_reg_set, ibuffer_entry entry,
                  unsigned sch_id);

  regBankAlloc *get_reg_bank_allocator() {
    return m_reg_bank_allocator;
  }

  unsigned test_result_bus(unsigned latency) {
    for (unsigned i = 0; i < num_result_bus; i++) {
      if (!m_result_bus[i]->test(latency)) {
        return i;
      }
    }
    return -1;
  }

  unsigned get_active_cycles() { return active_cycles; }
  unsigned long long get_active_warps_id_size_sum() {
    return active_warps_id_size_sum;
  }

  app_config *get_appcfg() { return appcfg; }
  std::vector<std::pair<int, int>> *get_kernel_block_pair() {
    return &kernel_block_pair;
  }
  std::vector<unsigned> *get_num_warps_per_sm() { return &m_num_warps_per_sm; }
  unsigned get_num_scheds() { return num_scheds; }

  unsigned get_num_m_warp_active_status() const {
    unsigned num_active_warps = 0;
    for (unsigned i = 0; i < m_warp_active_status.size(); i++) {
      for (unsigned j = 0; j < m_warp_active_status[i].size(); j++) {
        if (m_warp_active_status[i][j]) {
          num_active_warps++;
        }
      }
    }
    return num_active_warps;
  }
  // unsigned get_num_m_warp_active_status() const {
  //   return std::accumulate(m_warp_active_status.begin(), m_warp_active_status.end(), 0u,
  //                          [](unsigned sum, const std::vector<bool>& row) {
  //                            return sum + std::count(row.begin(), row.end(), true);
  //                          });
  // }

  unsigned get_num_m_warp_active_status(unsigned index) {
    unsigned num_active_warps = 0;
    for (unsigned j = 0; j < m_warp_active_status[index].size(); j++) {
      if (m_warp_active_status[index][j]) {
        num_active_warps++;
      }
    }
    return num_active_warps;
  }

  template <unsigned pos>
  void set_clk_record(unsigned kid, unsigned wid, unsigned uid,
                      unsigned value) {
    const std::tuple<unsigned, unsigned, unsigned> key(kid, wid, uid);
    auto it = clk_record.find(key);

    if (it == clk_record.end()) {
      std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>
          new_value(0, 0, 0, 0, 0, 0);
      std::get<pos>(new_value) = value;
      clk_record[key] = new_value;
    } else {
      std::get<pos>(it->second) = value;
    }
  }
  template <unsigned pos>
  unsigned get_clk_record(unsigned kid, unsigned wid, unsigned uid) {
    std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>
        &clk_record_value = clk_record[std::make_tuple(kid, wid, uid)];
    return std::get<pos>(clk_record_value);
  }
  bool judge_can_issue_of_bar(unsigned gwid, unsigned kid) { // 0704
    // calculate the block id
    unsigned block_id = (int)(gwid / appcfg->get_num_warp_per_block(kid));
    // find if block_id is in block_is_in_bar
    bool is_in_block_is_in_bar = (block_is_in_bar.find(block_id) != block_is_in_bar.end());
    if (is_in_block_is_in_bar) {
      unsigned already_in_bar_num = block_is_in_bar[block_id].size();
      bool all_true = true;
      bool all_false = true;
      
      for (unsigned _ = 0; _ < already_in_bar_num; _++) {
        if (block_is_in_bar[block_id][_] == false) {
          all_true = false;
        }
        if (block_is_in_bar[block_id][_] == true) {
          all_false = false;
        }
      }
      
      if ((int)already_in_bar_num < appcfg->get_num_warp_per_block(kid) - 1 && already_in_bar_num > 0 && all_true) {
        block_is_in_bar[block_id].push_back(true);
        return false;
      } else if ((int)already_in_bar_num == appcfg->get_num_warp_per_block(kid) - 1 && all_true) {
        for (unsigned _ = 0; _ < already_in_bar_num; _++) {
          block_is_in_bar[block_id][_] = false;
        }
        return true;
      } else if ((int)already_in_bar_num <= appcfg->get_num_warp_per_block(kid) - 1 && all_false) {
        // pop one from block_is_in_bar[block_id]
        block_is_in_bar[block_id].pop_back();
        if (block_is_in_bar[block_id].size() == 0) {
          block_is_in_bar.erase(block_id);
        }
        return true;
      } else {
        block_is_in_bar.erase(block_id);
        return true;
      }
    } else {
      block_is_in_bar[block_id] = {true};
      return false;
    }

    return true;
  }

  /// Since I initially wanted to support multi-kernel concurrent execution
  /// in various situations in the code, I read all kernel thread block
  /// emission information into main memory, but then put this plan on hold
  /// for the time being. This is to set all other kernels that is not
  /// currently being simulated as "do not need to be simulated".
  void setKernelsNotNeedToBeSimulated(const unsigned &evaluatedKernel) {
    unsigned index = 0;
    for (auto it_kernel_block_pair = kernel_block_pair.begin();
         it_kernel_block_pair != kernel_block_pair.end();
         ++it_kernel_block_pair, ++index) {
      if (it_kernel_block_pair->first - 1 != evaluatedKernel) {
        m_thread_block_has_executed_status[index] = true;
      }
    }
  }

  /// Here we simulate the thread block emitting scheduler to emit a new
  /// thread block to the SM, and then what we need to do is to set the
  /// active status to true for all warps in this new emitted thread block.
  void simCTAEmittingSceduler() {
    unsigned index = 0;
    unsigned total_active_warps = get_num_m_warp_active_status();
    for (auto it_kernel_block_pair = kernel_block_pair.begin();
        it_kernel_block_pair != kernel_block_pair.end();
        ++it_kernel_block_pair, ++index) {
      if (m_thread_block_has_executed_status[index]) continue;

      unsigned kid = it_kernel_block_pair->first - 1;
      unsigned warps_per_block = appcfg->get_num_warp_per_block(kid);

      // Here we simulate the thread block emitting scheduler to emit
      // a new thread block to the SM, and then what we need to do is
      // to set the active status to true for all warps in this new
      // emitted thread block.
      /// TODO: Here, we only considered that will the total number
      /// of warps for all thread blocks exceed the maximum number of
      /// warps supported by a single SM.
      if (total_active_warps + warps_per_block <=
          m_hw_cfg->get_max_warps_per_sm()) {
        for (unsigned wid = 0; wid < warps_per_block; ++wid) {
          if (m_warp_active_status[index][wid] == false) {
            m_warp_active_status[index][wid] = true;
            ++total_active_warps;
            m_thread_block_has_executed_status[index] = true;
          }
        }
      }
    }
  }

private:
  unsigned m_smid;
  unsigned long long m_cycle;
  bool active;

  unsigned long long active_cycles;
  unsigned long long active_warps_id_size_sum;

  unsigned num_banks;
  unsigned bank_warp_shift;
  unsigned num_scheds;
  bool sub_core_model;
  unsigned banks_per_sched;
  unsigned inst_fetch_throughput;
  unsigned reg_file_port_throughput;

  unsigned num_warp_instns_executed;

  std::vector<unsigned> m_num_warps_per_sm;

  std::vector<unsigned> m_num_blocks_per_kernel;

  unsigned all_warps_num;

  unsigned warps_per_sched;

  std::vector<std::pair<int, int>> kernel_block_pair;
  std::map<unsigned, std::vector<unsigned>> blocks_per_kernel;

  trace_parser *tracer;
  issue_config *issuecfg;
  app_config *appcfg;
  instn_config *instncfg;

  IBuffer *m_ibuffer;
  inst_fetch_buffer_entry *m_inst_fetch_buffer;
  inst_fetch_buffer_entry *m_inst_fetch_buffer_copy;

  unsigned total_pipeline_stages;
  std::vector<register_set> m_pipeline_reg;
  std::vector<register_set *> m_specilized_dispatch_reg;
  register_set *m_sp_out;
  register_set *m_dp_out;
  register_set *m_sfu_out;
  register_set *m_int_out;
  register_set *m_tensor_core_out;
  std::vector<register_set *> m_spec_cores_out;
  register_set *m_mem_out;

  Scoreboard *m_scoreboard;

  std::map<int, int> last_fetch_warp_id;
  int distance_last_fetch_kid;
  int last_issue_sched_id;

  std::map<std::pair<int, int>, int> last_issue_warp_ids;

  std::vector<int> last_issue_block_index_per_sched;

  regBankAlloc *m_reg_bank_allocator;

  opndcoll_rfu_t *m_operand_collector;

  std::map<curr_instn_id_per_warp_entry, unsigned> curr_instn_id_per_warp;

  std::vector<stage_instns_identifier> fetch_stage_instns;

  std::vector<stage_instns_identifier> writeback_stage_instns;

  std::vector<stage_instns_identifier> warp_exit_stage_instns;

  hw_config *m_hw_cfg;

  std::vector<pipelined_simd_unit *> m_fu;
  std::vector<unsigned> m_dispatch_port;
  std::vector<unsigned> m_issue_port;

  unsigned num_result_bus;
  std::vector<std::bitset<MAX_ALU_LATENCY> *> m_result_bus;

  std::vector<std::vector<bool>> m_warp_active_status;
  std::vector<bool> m_thread_block_has_executed_status;

  unsigned last_check_block_id_index_idx = 0;

  std::map<std::pair<unsigned, unsigned>, unsigned>
      kernel_id_block_id_last_fetch_wid;

  std::map<
      std::tuple<unsigned, unsigned, unsigned>,
      std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>>
      clk_record;

  std::map<unsigned, std::vector<bool>> block_is_in_bar; // 0704 
};

#endif
