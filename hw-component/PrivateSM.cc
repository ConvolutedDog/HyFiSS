#include "PrivateSM.h"

stat_collector::stat_collector(hw_config* hw_cfg, unsigned kernel_id) {
  this->kernel_id = kernel_id;

  m_num_sm = hw_cfg->get_num_sms_per_cluster() * hw_cfg->get_num_clusters();
  warp_size = WARP_SIZE;
  smem_allocation_size = hw_cfg->get_smem_allocation_size();
  max_registers_per_SM = hw_cfg->get_max_registers_per_sm();
  max_registers_per_block = hw_cfg->get_max_registers_per_cta();
  register_allocation_size = hw_cfg->get_register_allocation_size();
  max_active_blocks_per_SM = hw_cfg->get_max_ctas_per_sm();
  max_active_threads_per_SM = hw_cfg->get_max_threads_per_sm();
  shared_mem_size = hw_cfg->get_l2d_size_per_sub_partition() * 1024;

  active_SMs = 0;

  Thread_block_limit_SM = hw_cfg->get_max_ctas_per_sm();
  Thread_block_limit_registers = 0;
  Thread_block_limit_shared_memory = 0;
  Thread_block_limit_warps = 0;
  Theoretical_max_active_warps_per_SM = 0;
  Theoretical_occupancy = 0.;

  Achieved_active_warps_per_SM.resize(m_num_sm, 0.);
  Achieved_occupancy.resize(m_num_sm, 0.);
  Unified_L1_cache_hit_rate.resize(m_num_sm, 0.);
  Unified_L1_cache_requests.resize(m_num_sm, 0);
  Unified_L1_cache_hit_rate_for_read_transactions.resize(m_num_sm, 0.);
  L2_cache_hit_rate = 0.;
  L2_cache_requests = 0;
  GMEM_read_requests.resize(m_num_sm, 0);
  GMEM_write_requests.resize(m_num_sm, 0);
  GMEM_total_requests.resize(m_num_sm, 0);
  GMEM_read_transactions.resize(m_num_sm, 0);
  GMEM_write_transactions.resize(m_num_sm, 0);
  GMEM_total_transactions.resize(m_num_sm, 0);
  Number_of_read_transactions_per_read_requests.resize(m_num_sm, 0);
  Number_of_write_transactions_per_write_requests.resize(m_num_sm, 0);
  L2_read_transactions.resize(m_num_sm, 0);
  L2_write_transactions.resize(m_num_sm, 0);
  L2_total_transactions.resize(m_num_sm, 0);
  DRAM_total_transactions = 0;
  Total_number_of_global_atomic_requests.resize(m_num_sm, 0);
  Total_number_of_global_reduction_requests.resize(m_num_sm, 0);
  Global_memory_atomic_and_reduction_transactions.resize(m_num_sm, 0);
  GPU_active_cycles.resize(m_num_sm, 0);
  SM_active_cycles.resize(m_num_sm, 0);
  Warp_instructions_executed.resize(m_num_sm, 0);
  Instructions_executed_per_clock_cycle_IPC.resize(m_num_sm, 0.);
  Total_instructions_executed_per_seconds.resize(m_num_sm, 0.);
  Kernel_execution_time.resize(m_num_sm, 0);
  Simulation_time_memory_model.resize(m_num_sm, 0);
  Simulation_time_compute_model.resize(m_num_sm, 0);

  Compute_Structural_Stall.resize(m_num_sm, 0);
  Compute_Data_Stall.resize(m_num_sm, 0);
  Memory_Structural_Stall.resize(m_num_sm, 0);
  Memory_Data_Stall.resize(m_num_sm, 0);
  Synchronization_Stall.resize(m_num_sm, 0);
  Control_Stall.resize(m_num_sm, 0);
  Idle_Stall.resize(m_num_sm, 0);
  No_Stall.resize(m_num_sm, 0);
  Other_Stall.resize(m_num_sm, 0);

  At_least_four_instns_issued = false;
  At_least_one_Compute_Structural_Stall_found = false;
  At_least_one_Compute_Data_Stall_found = false;
  At_least_one_Memory_Structural_Stall_found = false;
  At_least_one_Memory_Data_Stall_found = false;
  At_least_one_Synchronization_Stall_found = false;
  At_least_one_Control_Stall_found = false;
  At_least_one_Idle_Stall_found = false;
  At_least_one_No_Stall_found = false;

  num_Issue_Compute_Structural_out_has_no_free_slot.resize(m_num_sm, 0);
  num_Issue_Memory_Structural_out_has_no_free_slot.resize(m_num_sm, 0);
  num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute.resize(
      m_num_sm, 0);
  num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory.resize(
      m_num_sm, 0);
  num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency.resize(
      m_num_sm, 0);
  num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency.resize(
      m_num_sm, 0);
  num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty.resize(
      m_num_sm, 0);
  num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty.resize(
      m_num_sm, 0);
  num_Writeback_Compute_Structural_bank_of_reg_is_not_idle.resize(m_num_sm, 0);
  num_Writeback_Memory_Structural_bank_of_reg_is_not_idle.resize(m_num_sm, 0);
  num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated.resize(
      m_num_sm, 0);
  num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated.resize(
      m_num_sm, 0);
  num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu
      .resize(m_num_sm, 0);
  num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu
      .resize(m_num_sm, 0);
  num_Execute_Memory_Structural_icnt_injection_buffer_is_full.resize(m_num_sm,
                                                                     0);
  num_Issue_Compute_Data_scoreboard.resize(m_num_sm, 0);
  num_Issue_Memory_Data_scoreboard.resize(m_num_sm, 0);
  num_Execute_Memory_Data_L1.resize(m_num_sm, 0);
  num_Execute_Memory_Data_L2.resize(m_num_sm, 0);
  num_Execute_Memory_Data_Main_Memory.resize(m_num_sm, 0);

  SP_UNIT_execute_clks_sum.resize(m_num_sm, 0);
  SFU_UNIT_execute_clks_sum.resize(m_num_sm, 0);
  INT_UNIT_execute_clks_sum.resize(m_num_sm, 0);
  DP_UNIT_execute_clks_sum.resize(m_num_sm, 0);
  TENSOR_CORE_UNIT_execute_clks_sum.resize(m_num_sm, 0);
  LDST_UNIT_execute_clks_sum.resize(m_num_sm, 0);
  SPEC_UNIT_1_execute_clks_sum.resize(m_num_sm, 0);
  SPEC_UNIT_2_execute_clks_sum.resize(m_num_sm, 0);
  SPEC_UNIT_3_execute_clks_sum.resize(m_num_sm, 0);
  Other_UNIT_execute_clks_sum.resize(m_num_sm, 0);

  SP_UNIT_Instns_num.resize(m_num_sm, 0);
  SFU_UNIT_Instns_num.resize(m_num_sm, 0);
  INT_UNIT_Instns_num.resize(m_num_sm, 0);
  DP_UNIT_Instns_num.resize(m_num_sm, 0);
  TENSOR_CORE_UNIT_Instns_num.resize(m_num_sm, 0);
  LDST_UNIT_Instns_num.resize(m_num_sm, 0);
  SPEC_UNIT_1_Instns_num.resize(m_num_sm, 0);
  SPEC_UNIT_2_Instns_num.resize(m_num_sm, 0);
  SPEC_UNIT_3_Instns_num.resize(m_num_sm, 0);
  Other_UNIT_Instns_num.resize(m_num_sm, 0);
}

bool create_directory_if_not_exists(const std::string &dir) {
  struct stat sb;
  if (stat(dir.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
    return true;
  } else {

    if (mkdir(dir.c_str(), 0777) == -1) {
      std::cerr << "Error creating directory." << std::endl;
      return false;
    }
    return true;
  }
}

void remove_file_if_exists(const std::string &filepath) {

  if (access(filepath.c_str(), F_OK) != -1) {

    remove(filepath.c_str());
  }
}

void stat_collector::dump_output(const std::string &path, unsigned rank) {

  std::string full_dir = path + std::string("/../outputs");
  std::string full_path = path + std::string("/../outputs/kernel-") +
                          std::to_string(get_kernel_id()) +
                          std::string("-rank-") + std::to_string(rank) +
                          std::string(".temp.txt");
  if (!create_directory_if_not_exists(full_dir)) {
    std::cout << "Error when creating directory" << full_dir << std::endl;
    exit(0);
  }

  remove_file_if_exists(full_path);

  std::ofstream file(full_path);
  if (file.is_open()) {
    file << "From rank: " << rank << std::endl;

    file << "Thread_block_limit_SM = " << Thread_block_limit_SM << std::endl;
    file << "Thread_block_limit_registers = " << Thread_block_limit_registers
         << std::endl;
    file << "Thread_block_limit_shared_memory = "
         << Thread_block_limit_shared_memory << std::endl;
    file << "Thread_block_limit_warps = " << Thread_block_limit_warps
         << std::endl;
    file << "Theoretical_max_active_warps_per_SM = "
         << Theoretical_max_active_warps_per_SM << std::endl;
    file << "Theoretical_occupancy = " << Theoretical_occupancy << std::endl;

    file << std::endl;

    file << "Unified_L1_cache_hit_rate[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Unified_L1_cache_hit_rate[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Unified_L1_cache_requests[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Unified_L1_cache_requests[sm_id] << " ";
      ;
    }
    file << std::endl;

    file << std::endl;

    file << "L2_cache_hit_rate = " << L2_cache_hit_rate << std::endl;
    file << "L2_cache_requests = " << L2_cache_requests << std::endl;

    file << std::endl;

    file << "GMEM_read_requests[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << GMEM_read_requests[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "GMEM_write_requests[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << GMEM_write_requests[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "GMEM_total_requests[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << GMEM_total_requests[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "GMEM_read_transactions[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << GMEM_read_transactions[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "GMEM_write_transactions[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << GMEM_write_transactions[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "GMEM_total_transactions[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << GMEM_total_transactions[sm_id] << " ";
      ;
    }
    file << std::endl;

    file << std::endl;

    file << "Number_of_read_transactions_per_read_requests[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Number_of_read_transactions_per_read_requests[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Number_of_write_transactions_per_write_requests[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Number_of_write_transactions_per_write_requests[sm_id] << " ";
      ;
    }
    file << std::endl;

    file << std::endl;

    file << "L2_read_transactions[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << L2_read_transactions[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "L2_write_transactions[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << L2_write_transactions[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "L2_total_transactions[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << L2_total_transactions[sm_id] << " ";
      ;
    }

    file << std::endl;
    file << std::endl;
    file << "DRAM_total_transactions = " << DRAM_total_transactions;
    file << std::endl;
    file << std::endl;
    file << "Total_number_of_global_atomic_requests[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Total_number_of_global_atomic_requests[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Total_number_of_global_reduction_requests[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Total_number_of_global_reduction_requests[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Global_memory_atomic_and_reduction_transactions[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Global_memory_atomic_and_reduction_transactions[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << std::endl;
    file << "Achieved_active_warps_per_SM[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Achieved_active_warps_per_SM[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Achieved_occupancy[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Achieved_occupancy[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << std::endl;
    file << "GPU_active_cycles[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << GPU_active_cycles[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "SM_active_cycles[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << SM_active_cycles[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << std::endl;
    file << "Warp_instructions_executed[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Warp_instructions_executed[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Instructions_executed_per_clock_cycle_IPC[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Instructions_executed_per_clock_cycle_IPC[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Total_instructions_executed_per_seconds[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Total_instructions_executed_per_seconds[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << std::endl;
    file << "Kernel_execution_time[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Kernel_execution_time[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << std::endl;

    file << "Simulation_time_memory_model[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Simulation_time_memory_model[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Simulation_time_compute_model[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Simulation_time_compute_model[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << std::endl;
    file << "Compute_Structural_Stall[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Compute_Structural_Stall[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Compute_Data_Stall[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Compute_Data_Stall[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Memory_Structural_Stall[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Memory_Structural_Stall[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Memory_Data_Stall[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Memory_Data_Stall[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Synchronization_Stall[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Synchronization_Stall[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Control_Stall[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Control_Stall[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Idle_Stall[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Idle_Stall[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "No_Stall[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << No_Stall[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Other_Stall[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Other_Stall[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << std::endl;
    file << "num_Issue_Compute_Structural_out_has_no_free_slot[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << num_Issue_Compute_Structural_out_has_no_free_slot[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "num_Issue_Memory_Structural_out_has_no_free_slot[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << num_Issue_Memory_Structural_out_has_no_free_slot[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_"
            "compute[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file
          << num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute
                 [sm_id]
          << " ";
      ;
    }
    file << std::endl;
    file << "num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_"
            "memory[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file
          << num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory
                 [sm_id]
          << " ";
      ;
    }
    file << std::endl;
    file << "num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency["
            "]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency
                  [sm_id]
           << " ";
      ;
    }
    file << std::endl;
    file << "num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency[]"
            ": ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency
                  [sm_id]
           << " ";
      ;
    }
    file << std::endl;
    file << "num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty[]"
            ": ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty
                  [sm_id]
           << " ";
      ;
    }
    file << std::endl;
    file << "num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty[]:"
            " ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty
                  [sm_id]
           << " ";
      ;
    }
    file << std::endl;
    file << "num_Writeback_Compute_Structural_bank_of_reg_is_not_idle[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << num_Writeback_Compute_Structural_bank_of_reg_is_not_idle[sm_id]
           << " ";
      ;
    }
    file << std::endl;
    file << "num_Writeback_Memory_Structural_bank_of_reg_is_not_idle[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << num_Writeback_Memory_Structural_bank_of_reg_is_not_idle[sm_id]
           << " ";
      ;
    }
    file << std::endl;
    file << "num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_"
            "allocated[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file
          << num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated
                 [sm_id]
          << " ";
      ;
    }
    file << std::endl;
    file << "num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_"
            "allocated[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file
          << num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated
                 [sm_id]
          << " ";
      ;
    }
    file << std::endl;
    file << "num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_"
            "fails_as_not_found_free_cu[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file
          << num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu
                 [sm_id]
          << " ";
      ;
    }
    file << std::endl;
    file << "num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_"
            "as_not_found_free_cu[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file
          << num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu
                 [sm_id]
          << " ";
      ;
    }
    file << std::endl;
    file << "num_Execute_Memory_Structural_icnt_injection_buffer_is_full[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << num_Execute_Memory_Structural_icnt_injection_buffer_is_full[sm_id]
           << " ";
      ;
    }
    file << std::endl;
    file << "num_Issue_Compute_Data_scoreboard[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << num_Issue_Compute_Data_scoreboard[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "num_Issue_Memory_Data_scoreboard[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << num_Issue_Memory_Data_scoreboard[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "num_Execute_Memory_Data_L1[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << num_Execute_Memory_Data_L1[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "num_Execute_Memory_Data_L2[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << num_Execute_Memory_Data_L2[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "num_Execute_Memory_Data_Main_Memory[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << num_Execute_Memory_Data_Main_Memory[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << std::endl;
    file << "SP_UNIT_execute_clks_sum[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << SP_UNIT_execute_clks_sum[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "SFU_UNIT_execute_clks_sum[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << SFU_UNIT_execute_clks_sum[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "INT_UNIT_execute_clks_sum[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << INT_UNIT_execute_clks_sum[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "DP_UNIT_execute_clks_sum[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << DP_UNIT_execute_clks_sum[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "TENSOR_CORE_UNIT_execute_clks_sum[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << TENSOR_CORE_UNIT_execute_clks_sum[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "LDST_UNIT_execute_clks_sum[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << LDST_UNIT_execute_clks_sum[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "SPEC_UNIT_1_execute_clks_sum[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << SPEC_UNIT_1_execute_clks_sum[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "SPEC_UNIT_2_execute_clks_sum[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << SPEC_UNIT_2_execute_clks_sum[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "SPEC_UNIT_3_execute_clks_sum[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << SPEC_UNIT_3_execute_clks_sum[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Other_UNIT_execute_clks_sum[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Other_UNIT_execute_clks_sum[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << std::endl;

    file << "SP_UNIT_Instns_num[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << SP_UNIT_Instns_num[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "SFU_UNIT_Instns_num[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << SFU_UNIT_Instns_num[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "INT_UNIT_Instns_num[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << INT_UNIT_Instns_num[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "DP_UNIT_Instns_num[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << DP_UNIT_Instns_num[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "TENSOR_CORE_UNIT_Instns_num[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << TENSOR_CORE_UNIT_Instns_num[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "LDST_UNIT_Instns_num[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << LDST_UNIT_Instns_num[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "SPEC_UNIT_1_Instns_num[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << SPEC_UNIT_1_Instns_num[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "SPEC_UNIT_2_Instns_num[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << SPEC_UNIT_2_Instns_num[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "SPEC_UNIT_3_Instns_num[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << SPEC_UNIT_3_Instns_num[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << "Other_UNIT_Instns_num[]: ";
    for (unsigned sm_id = 0; sm_id < m_num_sm; sm_id++) {
      file << Other_UNIT_Instns_num[sm_id] << " ";
      ;
    }
    file << std::endl;
    file << std::endl;

    file.close();
  }
}

bool operator<(const curr_instn_id_per_warp_entry &lhs,
               const curr_instn_id_per_warp_entry &rhs) {
  if (lhs.kid != rhs.kid) {
    return lhs.kid < rhs.kid;
  } else {
    if (lhs.block_id != rhs.block_id) {
      return lhs.block_id < rhs.block_id;
    } else {
      return lhs.warp_id < rhs.warp_id;
    }
  }
}

PrivateSM::PrivateSM(const unsigned smid, trace_parser *tracer,
                     hw_config *hw_cfg) {

  m_smid = smid;
  m_cycle = 0;
  active = true;
  m_active_warps = 0;
  max_warps_init = 0;

  num_warp_instns_executed = 0;

  active_cycles = 0;
  active_warps_id_size_sum = 0;

  m_hw_cfg = hw_cfg;

  this->tracer = tracer;
  issuecfg = this->tracer->get_issuecfg();
  appcfg = this->tracer->get_appcfg();
  instncfg = this->tracer->get_instncfg();

  kernel_block_pair = issuecfg->get_kernel_block_by_smid(m_smid);

  m_num_warps_per_sm.resize(appcfg->get_kernels_num(), 0);

  m_warp_active_status.reserve(kernel_block_pair.size());
  for (auto it = kernel_block_pair.begin(); it != kernel_block_pair.end();
       it++) {
    unsigned kid = it->first - 1;
    unsigned _warps_per_block = appcfg->get_num_warp_per_block(kid);
    m_warp_active_status.push_back(std::vector<bool>(_warps_per_block, false));
  }
  m_thread_block_has_executed_status.reserve(kernel_block_pair.size());
  for (auto it = kernel_block_pair.begin(); it != kernel_block_pair.end();
       it++) {
    m_thread_block_has_executed_status.push_back(false);
  }

  for (auto it = kernel_block_pair.begin(); it != kernel_block_pair.end();
       it++) {
    unsigned kid = it->first - 1;
    unsigned _warps_per_block = appcfg->get_num_warp_per_block(kid);
    m_num_warps_per_sm[kid] += _warps_per_block;
  }

  for (auto it = kernel_block_pair.begin(); it != kernel_block_pair.end();
       it++) {
    unsigned kid = it->first - 1;
    unsigned block_id = it->second;
    kernel_id_block_id_last_fetch_wid[{kid, block_id}] = 0;
  }

  m_num_blocks_per_kernel.resize(appcfg->get_kernels_num(), 0);

  for (auto it = kernel_block_pair.begin(); it != kernel_block_pair.end();
       it++) {
    unsigned kid = it->first - 1;
    unsigned block_id = it->second;

    m_num_blocks_per_kernel[kid] += 1;

    unsigned _warps_per_block = appcfg->get_num_warp_per_block(kid);
    for (unsigned _i = 0; _i < _warps_per_block; _i++) {

      curr_instn_id_per_warp_entry _entry =
          curr_instn_id_per_warp_entry(kid, block_id, _i);
      curr_instn_id_per_warp[_entry] = 0;
    }
  }

  for (unsigned i = 0; i < m_num_warps_per_sm.size(); i++) {
    last_fetch_warp_id[i] = 0;
  }

  all_warps_num =
      std::accumulate(m_num_warps_per_sm.begin(), m_num_warps_per_sm.end(), 0);

  m_ibuffer = new IBuffer(m_smid, all_warps_num);

  distance_last_fetch_kid = 0;
  last_issue_sched_id = 0;

  m_scoreboard = new Scoreboard(m_smid, all_warps_num);

  m_inst_fetch_buffer = new inst_fetch_buffer_entry();
  m_inst_fetch_buffer_copy = new inst_fetch_buffer_entry();

  num_banks = hw_cfg->get_num_reg_banks();
  bank_warp_shift = hw_cfg->get_bank_warp_shift();
  num_scheds = hw_cfg->get_num_sched_per_sm();
  sub_core_model = hw_cfg->get_sub_core_model();
  banks_per_sched = (unsigned)(num_banks / num_scheds);
  inst_fetch_throughput = hw_cfg->get_inst_fetch_throughput();
  reg_file_port_throughput = hw_cfg->get_reg_file_port_throughput();

  warps_per_sched = (unsigned)(all_warps_num / num_scheds);

  last_issue_block_index_per_sched.resize(num_scheds, 0);

  m_reg_bank_allocator = new regBankAlloc(
      m_smid, num_banks, num_scheds, bank_warp_shift, banks_per_sched);

  parse_blocks_per_kernel();

  total_pipeline_stages =
      N_PIPELINE_STAGES + hw_cfg->get_specialized_unit_size() * 2;
  m_pipeline_reg.reserve(total_pipeline_stages);

  for (unsigned j = 0; j < N_PIPELINE_STAGES; j++) {

    m_pipeline_reg.push_back(register_set(
        hw_cfg->get_pipe_widths(static_cast<pipeline_stage_name_t>(j)),
        std::string(hw_cfg->get_pipeline_stage_name_decode(
            static_cast<pipeline_stage_name_t>(j))),
        hw_cfg));
  }

  for (unsigned j = 0; j < hw_cfg->get_specialized_unit_size(); j++) {

    m_pipeline_reg.push_back(
        register_set(hw_cfg->get_pipe_widths_ID_OC_spec_unit(j),
                     std::string(std::string("ID_OC_") +
                                 hw_cfg->get_m_specialized_unit_name(j)),
                     hw_cfg));

    m_specilized_dispatch_reg.push_back(
        &m_pipeline_reg[m_pipeline_reg.size() - 1]);
  }

  for (unsigned j = 0; j < hw_cfg->get_specialized_unit_size(); j++) {

    m_pipeline_reg.push_back(
        register_set(hw_cfg->get_pipe_widths_OC_EX_spec_unit(j),
                     std::string(std::string("OC_EX_") +
                                 hw_cfg->get_m_specialized_unit_name(j)),
                     hw_cfg));
  }

  m_sp_out = &m_pipeline_reg[ID_OC_SP];
  m_dp_out = &m_pipeline_reg[ID_OC_DP];
  m_sfu_out = &m_pipeline_reg[ID_OC_SFU];
  m_int_out = &m_pipeline_reg[ID_OC_INT];
  m_tensor_core_out = &m_pipeline_reg[ID_OC_TENSOR_CORE];

  for (unsigned j = 0; j < m_specilized_dispatch_reg.size(); j++) {
    m_spec_cores_out.push_back(m_specilized_dispatch_reg[j]);
  }
  m_mem_out = &m_pipeline_reg[ID_OC_MEM];

  enum { SP_CUS, DP_CUS, SFU_CUS, TENSOR_CORE_CUS, INT_CUS, MEM_CUS, GEN_CUS };

  opndcoll_rfu_t::port_vector_t in_ports;
  opndcoll_rfu_t::port_vector_t out_ports;
  opndcoll_rfu_t::uint_vector_t cu_sets;

  m_operand_collector =
      new opndcoll_rfu_t(m_hw_cfg, m_reg_bank_allocator, this->tracer);

  m_operand_collector->add_cu_set(
      GEN_CUS, hw_cfg->get_operand_collector_num_units_gen(),
      hw_cfg->get_operand_collector_num_out_ports_gen());

  for (unsigned i = 0; i < hw_cfg->get_operand_collector_num_in_ports_gen();
       i++) {
    in_ports.push_back(&m_pipeline_reg[ID_OC_SP]);
    in_ports.push_back(&m_pipeline_reg[ID_OC_SFU]);
    in_ports.push_back(&m_pipeline_reg[ID_OC_MEM]);
    out_ports.push_back(&m_pipeline_reg[OC_EX_SP]);
    out_ports.push_back(&m_pipeline_reg[OC_EX_SFU]);
    out_ports.push_back(&m_pipeline_reg[OC_EX_MEM]);
    if (1) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_TENSOR_CORE]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_TENSOR_CORE]);
    }
    if (hw_cfg->get_num_dp_units() > 0) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_DP]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_DP]);
    }
    if (hw_cfg->get_num_int_units() > 0) {
      in_ports.push_back(&m_pipeline_reg[ID_OC_INT]);
      out_ports.push_back(&m_pipeline_reg[OC_EX_INT]);
    }
    if (hw_cfg->get_specialized_unit_size() > 0) {
      for (unsigned j = 0; j < hw_cfg->get_specialized_unit_size(); ++j) {
        in_ports.push_back(&m_pipeline_reg[N_PIPELINE_STAGES + 0 + j]);
        out_ports.push_back(&m_pipeline_reg[N_PIPELINE_STAGES + 3 + j]);
      }
    }
    cu_sets.push_back((unsigned)GEN_CUS);
    m_operand_collector->add_port(in_ports, out_ports, cu_sets);
    in_ports.clear(), out_ports.clear(), cu_sets.clear();
  }

  m_operand_collector->init(m_hw_cfg, m_reg_bank_allocator, this->tracer);

  for (unsigned k = 0; k < m_hw_cfg->get_num_sp_units(); k++) {
    m_fu.push_back(
        new sp_unit(&m_pipeline_reg[EX_WB], k, m_hw_cfg, this->tracer));
    m_dispatch_port.push_back(ID_OC_SP);
    m_issue_port.push_back(OC_EX_SP);
  }

  for (unsigned k = 0; k < m_hw_cfg->get_num_sfu_units(); k++) {
    m_fu.push_back(new sfu(&m_pipeline_reg[EX_WB], k, m_hw_cfg, this->tracer));
    m_dispatch_port.push_back(ID_OC_SFU);
    m_issue_port.push_back(OC_EX_SFU);
  }

  for (unsigned k = 0; k < m_hw_cfg->get_num_int_units(); k++) {
    m_fu.push_back(
        new int_unit(&m_pipeline_reg[EX_WB], k, m_hw_cfg, this->tracer));
    m_dispatch_port.push_back(ID_OC_INT);
    m_issue_port.push_back(OC_EX_INT);
  }

  for (unsigned k = 0; k < m_hw_cfg->get_num_dp_units(); k++) {
    m_fu.push_back(
        new dp_unit(&m_pipeline_reg[EX_WB], k, m_hw_cfg, this->tracer));
    m_dispatch_port.push_back(ID_OC_DP);
    m_issue_port.push_back(OC_EX_DP);
  }

  for (unsigned k = 0; k < m_hw_cfg->get_num_tensor_core_units(); k++) {
    m_fu.push_back(
        new tensor_core(&m_pipeline_reg[EX_WB], k, m_hw_cfg, this->tracer));
    m_dispatch_port.push_back(ID_OC_TENSOR_CORE);
    m_issue_port.push_back(OC_EX_TENSOR_CORE);
  }

  for (unsigned k = 0; k < m_hw_cfg->get_num_mem_units(); k++) {
    m_fu.push_back(
        new mem_unit(&m_pipeline_reg[EX_WB], k, m_hw_cfg, this->tracer));
    m_dispatch_port.push_back(ID_OC_MEM);
    m_issue_port.push_back(OC_EX_MEM);
  }

  for (unsigned k = 0; k < m_hw_cfg->get_specialized_unit_size(); k++) {
    m_fu.push_back(new specialized_unit(&m_pipeline_reg[EX_WB], k, m_hw_cfg,
                                        this->tracer, k));
    m_dispatch_port.push_back(N_PIPELINE_STAGES + 0 + k);
    m_issue_port.push_back(N_PIPELINE_STAGES + 3 + k);
  }

  num_result_bus =
      hw_cfg->get_pipe_widths(static_cast<pipeline_stage_name_t>(EX_WB));
  for (unsigned _ = 0; _ < num_result_bus; _++) {
    m_result_bus.push_back(new std::bitset<MAX_ALU_LATENCY>());
  }
}

int PrivateSM::register_bank(int regnum, int wid, unsigned sched_id) {
  int bank = regnum;

  if (bank_warp_shift)
    bank += wid;

  if (sub_core_model) {
    unsigned bank_num = (bank % banks_per_sched) + (sched_id * banks_per_sched);
    assert(bank_num < num_banks);
    return bank_num;
  } else
    return bank % num_banks;
}

bool PrivateSM::check_active() { return false; }

PrivateSM::~PrivateSM() {
  delete m_ibuffer;
  delete m_inst_fetch_buffer;
  delete m_inst_fetch_buffer_copy;
  delete m_scoreboard;
  delete m_reg_bank_allocator;
  delete m_operand_collector;

  m_sp_out->release_register_set();
  m_dp_out->release_register_set();
  m_sfu_out->release_register_set();
  m_int_out->release_register_set();
  m_tensor_core_out->release_register_set();
  m_mem_out->release_register_set();
  for (auto ptr : m_specilized_dispatch_reg) {
    ptr->release_register_set();
  }
  m_specilized_dispatch_reg.clear();
  m_spec_cores_out.clear();

  for (auto ptr : m_pipeline_reg) {
    ptr.release_register_set();
  }

  for (auto ptr : m_fu) {
    delete ptr;
  }

  for (auto ptr : m_result_bus) {
    delete ptr;
  }
}

unsigned PrivateSM::get_num_warps_per_sm(unsigned kernel_id) {
  return m_num_warps_per_sm[kernel_id];
}

void PrivateSM::parse_blocks_per_kernel() {
  for (unsigned i = 0; i < kernel_block_pair.size(); i++) {
    if (blocks_per_kernel.find(kernel_block_pair[i].first) ==
        blocks_per_kernel.end()) {
      blocks_per_kernel[kernel_block_pair[i].first] =
          std::vector<unsigned>(1, kernel_block_pair[i].second);
    } else {
      blocks_per_kernel[kernel_block_pair[i].first].push_back(
          kernel_block_pair[i].second);
    }
  }
}

std::vector<unsigned> PrivateSM::get_blocks_per_kernel(unsigned kernel_id) {
  return blocks_per_kernel[kernel_id];
}

std::map<unsigned, std::vector<unsigned>> *PrivateSM::get_blocks_per_kernel() {
  return &blocks_per_kernel;
}

unsigned PrivateSM::get_inst_fetch_throughput() {
  return inst_fetch_throughput;
}
unsigned PrivateSM::get_reg_file_port_throughput() {
  return reg_file_port_throughput;
}

void PrivateSM::issue_warp(register_set &pipe_reg_set, ibuffer_entry entry,
                           unsigned sch_id) {

  inst_fetch_buffer_entry *tmp = new inst_fetch_buffer_entry();
  tmp->kid = entry.kid;
  tmp->pc = entry.pc;
  tmp->uid = entry.uid;
  tmp->wid = entry.wid;
  tmp->m_valid = true;

  pipe_reg_set.move_in(m_hw_cfg->get_sub_core_model(), sch_id, tmp);

  delete tmp;
  tmp = nullptr;
}

void insert_into_active_warps_id(std::vector<unsigned> *active_warps_id,
                                 unsigned wid) {

  if (std::find(active_warps_id->begin(), active_warps_id->end(), wid) ==
      active_warps_id->end()) {
    active_warps_id->push_back(wid);
  }
}

void PrivateSM::run(unsigned KERNEL_EVALUATION, unsigned MEM_ACCESS_LATENCY,
                    stat_collector *stat_coll) {

  m_cycle++;

  if (_CALIBRATION_LOG_) {
    std::cout << "#: m_cycle: " << m_cycle << std::endl;
  }

  if (m_cycle > 2000000) {
    std::cout << "EXIT BY TOO MAX CYCLE !!!" << std::endl;
    exit(0);
  }

  bool active_during_this_cycle = false;

  std::vector<unsigned> active_warps_id;

  bool flag_Issue_Compute_Structural_out_has_no_free_slot = false;
  bool flag_Issue_Memory_Structural_out_has_no_free_slot = false;
  bool flag_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute =
      false;
  bool flag_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory =
      false;
  bool flag_Execute_Compute_Structural_result_bus_has_no_slot_for_latency =
      false;
  bool flag_Execute_Memory_Structural_result_bus_has_no_slot_for_latency =
      false;
  bool flag_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty =
      false;
  bool flag_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty = false;
  bool flag_Writeback_Compute_Structural_bank_of_reg_is_not_idle = false;
  bool flag_Writeback_Memory_Structural_bank_of_reg_is_not_idle = false;
  bool flag_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated =
      false;
  bool flag_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated =
      false;
  bool
      flag_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu =
          false;
  bool
      flag_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu =
          false;
  bool flag_Execute_Memory_Structural_icnt_injection_buffer_is_full = false;
  bool flag_Issue_Compute_Data_scoreboard = false;
  bool flag_Issue_Memory_Data_scoreboard = false;
  bool flag_Execute_Memory_Data_L1 = false;
  bool flag_Execute_Memory_Data_L2 = false;
  bool flag_Execute_Memory_Data_Main_Memory = false;

  if (m_cycle == 1) {
    for (auto it_kernel_block_pair_1 = kernel_block_pair.begin();
         it_kernel_block_pair_1 != kernel_block_pair.end();
         it_kernel_block_pair_1++) {
      if ((unsigned)(it_kernel_block_pair_1->first) - 1 != KERNEL_EVALUATION) {
        unsigned _index_ =
            std::distance(kernel_block_pair.begin(), it_kernel_block_pair_1);
        m_thread_block_has_executed_status[_index_] = true;
      }
    }
  }

  for (auto it_kernel_block_pair_2 = kernel_block_pair.begin();
       it_kernel_block_pair_2 != kernel_block_pair.end();
       it_kernel_block_pair_2++) {
    if ((unsigned)(it_kernel_block_pair_2->first) - 1 != KERNEL_EVALUATION)
      continue;

    unsigned _index_ =
        std::distance(kernel_block_pair.begin(), it_kernel_block_pair_2);

    if (m_thread_block_has_executed_status[_index_] == true)
      continue;

    unsigned _kid_ = it_kernel_block_pair_2->first - 1;
    unsigned _warps_per_block_ = appcfg->get_num_warp_per_block(_kid_);

    if (get_num_m_warp_active_status() + _warps_per_block_ <=
        m_hw_cfg->get_max_warps_per_sm()) {

      for (unsigned _wid_ = 0; _wid_ < _warps_per_block_; _wid_++) {

        if (m_warp_active_status[_index_][_wid_] == false) {
          m_warp_active_status[_index_][_wid_] = true;
          m_thread_block_has_executed_status[_index_] = true;
          m_active_warps++;
          max_warps_init++;
        }
      }
    }
  }

  inst_fetch_buffer_entry **preg = m_pipeline_reg[EX_WB].get_ready();
  inst_fetch_buffer_entry *pipe_reg = (preg == NULL) ? NULL : *preg;
  std::vector<inst_fetch_buffer_entry> except_regs;
  while (preg and pipe_reg->m_valid) {

    unsigned _kid = pipe_reg->kid;
    unsigned _pc = pipe_reg->pc;

    unsigned _wid = pipe_reg->wid;
    unsigned _uid = pipe_reg->uid;

    unsigned _warps_per_block = appcfg->get_num_warp_per_block(_kid);

    unsigned _block_id = (unsigned)(_wid / _warps_per_block);

    unsigned _gwarp_id_start = _warps_per_block * _block_id;

    auto _compute_instn =
        tracer->get_one_kernel_one_warp_one_instn(_kid, _wid, _uid);
    auto _trace_warp_inst = _compute_instn->trace_warp_inst;
    unsigned dst_reg_num = _trace_warp_inst.get_outcount();

    std::vector<int> need_write_back_regs_num;

    for (unsigned i = 0; i < dst_reg_num; i++) {
      int dst_reg_id = _trace_warp_inst.get_arch_reg_dst(i);

      if (dst_reg_id >= 0) {
        auto local_wid = (unsigned)(_wid % _warps_per_block);
        auto sched_id = (unsigned)(local_wid % num_scheds);

        auto bank_id = register_bank(dst_reg_id, local_wid, sched_id);

        if (m_reg_bank_allocator->getBankState(bank_id) == FREE) {

          m_reg_bank_allocator->setBankState(bank_id, ON_WRITING);

          _trace_warp_inst.set_arch_reg_dst(i, -1);

          insert_into_active_warps_id(&active_warps_id, _wid);
        } else {

          flag_Writeback_Compute_Structural_bank_of_reg_is_not_idle = true;
        }
      }
    }

    bool all_write_back = true;
    for (unsigned i = 0; i < dst_reg_num; i++) {
      if (_trace_warp_inst.get_arch_reg_dst(i) != -1) {
        all_write_back = false;
        break;
      }
    }

    if (all_write_back) {

      if (_CALIBRATION_LOG_) {
        std::cout << "    Write back: (" << _kid << ", " << _wid << ", " << _uid
                  << ", " << _pc << ")" << std::endl;
      }
      set_clk_record<5>(_kid, _wid, _uid, m_cycle);

      num_warp_instns_executed++;

      pipe_reg->m_valid = false;

      if (_trace_warp_inst.get_opcode() == OP_EXIT &&
          tracer->get_one_kernel_one_warp_instn_count(_kid, _wid) == _uid + 1) {

        unsigned _index = 0;
        for (auto _it_kernel_block_pair = kernel_block_pair.begin();
             _it_kernel_block_pair != kernel_block_pair.end();
             _it_kernel_block_pair++) {
          if ((unsigned)(_it_kernel_block_pair->first) - 1 != KERNEL_EVALUATION)
            continue;
          if ((unsigned)(_it_kernel_block_pair->first) - 1 == _kid &&
              (unsigned)(_it_kernel_block_pair->second) == _block_id) {
            _index =
                std::distance(kernel_block_pair.begin(), _it_kernel_block_pair);
            break;
          }
        }

        m_warp_active_status[_index][_wid - _gwarp_id_start] = false;
        m_active_warps--;
        insert_into_active_warps_id(&active_warps_id, _wid);
      }

    } else {
      except_regs.push_back(*pipe_reg);

      insert_into_active_warps_id(&active_warps_id, _wid);
    }

    unsigned _kid_block_id_count = 0;
    for (auto _it_kernel_block_pair = kernel_block_pair.begin();
         _it_kernel_block_pair != kernel_block_pair.end();
         _it_kernel_block_pair++) {
      if ((unsigned)(_it_kernel_block_pair->first) - 1 != KERNEL_EVALUATION)
        continue;
      if ((unsigned)(_it_kernel_block_pair->first) - 1 == _kid) {
        if ((unsigned)(_it_kernel_block_pair->second) < _block_id) {
          _kid_block_id_count++;
        }
      }
    }

    auto global_all_kernels_warp_id =
        (unsigned)(_wid % _warps_per_block) +
        _kid_block_id_count * _warps_per_block +
        std::accumulate(m_num_warps_per_sm.begin(),
                        m_num_warps_per_sm.begin() + _kid, 0);

    if (all_write_back) {

      _inst_trace_t *tmp_inst_trace = _compute_instn->inst_trace;
      for (unsigned i = 0; i < tmp_inst_trace->reg_srcs_num; i++) {
        need_write_back_regs_num.push_back(tmp_inst_trace->reg_src[i]);
      }
      for (unsigned i = 0; i < tmp_inst_trace->reg_dsts_num; i++) {
        if (tmp_inst_trace->reg_dest_is_pred[i]) {
          need_write_back_regs_num.push_back(tmp_inst_trace->reg_dest[i] +
                                             PRED_NUM_OFFSET);
        } else {
          need_write_back_regs_num.push_back(tmp_inst_trace->reg_dest[i]);
        }
      }
      auto pred = _trace_warp_inst.get_pred();
      need_write_back_regs_num.push_back((pred < 0) ? pred
                                                    : pred + PRED_NUM_OFFSET);

      for (auto regnum : need_write_back_regs_num) {

        m_scoreboard->releaseRegister(global_all_kernels_warp_id, regnum);
        insert_into_active_warps_id(&active_warps_id,
                                    global_all_kernels_warp_id);
      }
    }

    preg = m_pipeline_reg[EX_WB].get_ready(&except_regs);
    pipe_reg = (preg == NULL) ? NULL : *preg;
    active_during_this_cycle = true;
  }

  for (unsigned i = 0; i < num_result_bus; i++) {
    *(m_result_bus[i]) >>= 1;
  }

  for (unsigned n = 0; n < m_fu.size(); n++) {
    for (unsigned _ = 0; _ < m_fu[n]->clock_multiplier(); _++) {

      std::vector<unsigned> returned_wids = m_fu[n]->cycle(
          tracer, m_scoreboard, appcfg, &kernel_block_pair, &m_num_warps_per_sm,
          KERNEL_EVALUATION, num_scheds, m_reg_bank_allocator,
          &flag_Writeback_Memory_Structural_bank_of_reg_is_not_idle,
          &clk_record, m_cycle);
      for (auto wid : returned_wids) {
        insert_into_active_warps_id(&active_warps_id, wid);
        active_during_this_cycle = true;
      }
    }
  }

  std::vector<opndcoll_rfu_t::input_port_t> *m_in_ports_ptr =
      m_operand_collector->get_m_in_ports();

  for (unsigned p = 0; p < m_in_ports_ptr->size(); p++) {

    opndcoll_rfu_t::input_port_t &inp = (*m_in_ports_ptr)[p];

    for (unsigned i = 0; i < inp.m_out.size(); i++) {
      if ((*inp.m_out[i]).has_ready()) {

        std::vector<unsigned> ready_reg_ids =
            (*inp.m_out[i]).get_ready_reg_ids();
        for (unsigned j = 0; j < ready_reg_ids.size(); j++) {
          unsigned reg_id = ready_reg_ids[j];
          unsigned _kid = (*inp.m_out[i]).get_kid(reg_id);
          unsigned _wid = (*inp.m_out[i]).get_wid(reg_id);
          unsigned _uid = (*inp.m_out[i]).get_uid(reg_id);

          compute_instn *tmp =
              tracer->get_one_kernel_one_warp_one_instn(_kid, _wid, _uid);
          _inst_trace_t *tmp_inst_trace = tmp->inst_trace;

          unsigned offset_fu = 0;
          bool schedule_wb_now = false;
          int resbus = -1;

          switch (tmp_inst_trace->get_func_unit()) {

          case SP_UNIT:

            (*inp.m_out[i]).set_latency(tmp_inst_trace->get_latency(), reg_id);

            (*inp.m_out[i])
                .set_initial_interval(tmp_inst_trace->get_initiation_interval(),
                                      reg_id);

            offset_fu = 0;
            for (unsigned _ = 0; _ < m_hw_cfg->get_num_sp_units(); _++) {
              if (m_fu[offset_fu + _]->can_issue(
                      tmp_inst_trace->get_latency())) {
                schedule_wb_now = !m_fu[offset_fu + _]->stallable();
                resbus = test_result_bus(tmp_inst_trace->get_latency());

                insert_into_active_warps_id(&active_warps_id, _wid);
                active_during_this_cycle = true;
                if (schedule_wb_now && (resbus != -1)) {
                  m_result_bus[resbus]->set(tmp_inst_trace->get_latency());
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else if (!schedule_wb_now) {
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else {

                  flag_Execute_Compute_Structural_result_bus_has_no_slot_for_latency =
                      true;
                }

              } else {
                flag_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty =
                    true;
              }
            }
            break;
          case DP_UNIT:

            (*inp.m_out[i]).set_latency(tmp_inst_trace->get_latency(), reg_id);
            (*inp.m_out[i])
                .set_initial_interval(tmp_inst_trace->get_initiation_interval(),
                                      reg_id);

            offset_fu = m_hw_cfg->get_num_sp_units() +
                        m_hw_cfg->get_num_sfu_units() +
                        m_hw_cfg->get_num_int_units();
            for (unsigned _ = 0; _ < m_hw_cfg->get_num_dp_units(); _++) {
              if (m_fu[offset_fu + _]->can_issue(
                      tmp_inst_trace->get_latency())) {
                schedule_wb_now = !m_fu[offset_fu + _]->stallable();
                resbus = test_result_bus(tmp_inst_trace->get_latency());

                insert_into_active_warps_id(&active_warps_id, _wid);
                active_during_this_cycle = true;
                if (schedule_wb_now && (resbus != -1)) {
                  m_result_bus[resbus]->set(tmp_inst_trace->get_latency());
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else if (!schedule_wb_now) {
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else {

                  flag_Execute_Compute_Structural_result_bus_has_no_slot_for_latency =
                      true;
                }

              } else {
                flag_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty =
                    true;
              }
            }
            break;
          case SFU_UNIT:

            (*inp.m_out[i]).set_latency(tmp_inst_trace->get_latency(), reg_id);
            (*inp.m_out[i])
                .set_initial_interval(tmp_inst_trace->get_initiation_interval(),
                                      reg_id);

            offset_fu = m_hw_cfg->get_num_sp_units();
            for (unsigned _ = 0; _ < m_hw_cfg->get_num_sfu_units(); _++) {
              if (m_fu[offset_fu + _]->can_issue(
                      tmp_inst_trace->get_latency())) {
                schedule_wb_now = !m_fu[offset_fu + _]->stallable();
                resbus = test_result_bus(tmp_inst_trace->get_latency());

                insert_into_active_warps_id(&active_warps_id, _wid);
                active_during_this_cycle = true;
                if (schedule_wb_now && (resbus != -1)) {
                  m_result_bus[resbus]->set(tmp_inst_trace->get_latency());
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else if (!schedule_wb_now) {
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else {

                  flag_Execute_Compute_Structural_result_bus_has_no_slot_for_latency =
                      true;
                }

              } else {
                flag_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty =
                    true;
              }
            }
            break;
          case TENSOR_CORE_UNIT:

            (*inp.m_out[i]).set_latency(tmp_inst_trace->get_latency(), reg_id);
            (*inp.m_out[i])
                .set_initial_interval(tmp_inst_trace->get_initiation_interval(),
                                      reg_id);

            offset_fu =
                m_hw_cfg->get_num_sp_units() + m_hw_cfg->get_num_sfu_units() +
                m_hw_cfg->get_num_int_units() + m_hw_cfg->get_num_dp_units();
            for (unsigned _ = 0; _ < m_hw_cfg->get_num_tensor_core_units();
                 _++) {
              if (m_fu[offset_fu + _]->can_issue(
                      tmp_inst_trace->get_latency())) {
                schedule_wb_now = !m_fu[offset_fu + _]->stallable();
                resbus = test_result_bus(tmp_inst_trace->get_latency());

                insert_into_active_warps_id(&active_warps_id, _wid);
                active_during_this_cycle = true;
                if (schedule_wb_now && (resbus != -1)) {
                  m_result_bus[resbus]->set(tmp_inst_trace->get_latency());
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else if (!schedule_wb_now) {
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else {

                  flag_Execute_Compute_Structural_result_bus_has_no_slot_for_latency =
                      true;
                }

              } else {
                flag_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty =
                    true;
              }
            }
            break;
          case INT_UNIT:

            (*inp.m_out[i]).set_latency(tmp_inst_trace->get_latency(), reg_id);
            (*inp.m_out[i])
                .set_initial_interval(tmp_inst_trace->get_initiation_interval(),
                                      reg_id);

            offset_fu =
                m_hw_cfg->get_num_sp_units() + m_hw_cfg->get_num_sfu_units();
            for (unsigned _ = 0; _ < m_hw_cfg->get_num_int_units(); _++) {
              if (m_fu[offset_fu + _]->can_issue(
                      tmp_inst_trace->get_latency())) {
                schedule_wb_now = !m_fu[offset_fu + _]->stallable();
                resbus = test_result_bus(tmp_inst_trace->get_latency());

                insert_into_active_warps_id(&active_warps_id, _wid);
                active_during_this_cycle = true;
                if (schedule_wb_now && (resbus != -1)) {
                  m_result_bus[resbus]->set(tmp_inst_trace->get_latency());
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else if (!schedule_wb_now) {
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else {

                  flag_Execute_Compute_Structural_result_bus_has_no_slot_for_latency =
                      true;
                }

              } else {

                flag_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty =
                    true;
              }
            }
            break;
          case LDST_UNIT: {
            std::vector<std::string> opcode_tokens =
                tmp_inst_trace->get_opcode_tokens();

            (*inp.m_out[i]).set_latency(m_hw_cfg->get_l1_access_latency(), reg_id);
            for (const auto &token : opcode_tokens) {
              if (token.find("CONSTANT") != std::string::npos) {
                (*inp.m_out[i]).set_latency(m_hw_cfg->get_const_mem_access_latency(), reg_id);
              } else if (token.find("LDC") != std::string::npos) {
                (*inp.m_out[i]).set_latency(m_hw_cfg->get_const_mem_access_latency(), reg_id);
              } else if (token.find("LDS") != std::string::npos) {
                (*inp.m_out[i]).set_latency(m_hw_cfg->get_l1_access_latency(), reg_id);
              } else if (token.find("STRONG") != std::string::npos) {
                (*inp.m_out[i]).set_latency(MEM_ACCESS_LATENCY, reg_id);
              } else if (token.find("LDL") != std::string::npos) {
                (*inp.m_out[i]).set_latency(m_hw_cfg->get_dram_mem_access_latency(), reg_id);
              } else if (token.find("LD.") != std::string::npos) {
                (*inp.m_out[i]).set_latency(m_hw_cfg->get_l1_access_latency(), reg_id);
              } else if (token.find("STL") != std::string::npos) {
                (*inp.m_out[i]).set_latency(m_hw_cfg->get_dram_mem_access_latency(), reg_id);
              } else if (token.find("STS") != std::string::npos) {
                (*inp.m_out[i]).set_latency(m_hw_cfg->get_l1_access_latency(), reg_id);
              } else if (token.find("STG") != std::string::npos) {
                (*inp.m_out[i]).set_latency(MEM_ACCESS_LATENCY, reg_id);
              }
            }

            (*inp.m_out[i])
                .set_initial_interval(tmp_inst_trace->get_initiation_interval(),
                                      reg_id);

            offset_fu =
                m_hw_cfg->get_num_sp_units() + m_hw_cfg->get_num_sfu_units() +
                m_hw_cfg->get_num_int_units() + m_hw_cfg->get_num_dp_units() +
                m_hw_cfg->get_num_tensor_core_units();
            for (unsigned _ = 0; _ < m_hw_cfg->get_num_mem_units(); _++) {

              if (m_fu[offset_fu + _]->can_issue(MEM_ACCESS_LATENCY)) {
                schedule_wb_now = !m_fu[offset_fu + _]->stallable();

                resbus = test_result_bus(MEM_ACCESS_LATENCY);

                insert_into_active_warps_id(&active_warps_id, _wid);
                active_during_this_cycle = true;
                if (schedule_wb_now && (resbus != -1)) {

                  m_result_bus[resbus]->set(MEM_ACCESS_LATENCY);
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else if (!schedule_wb_now) {
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else {

                  flag_Execute_Memory_Structural_icnt_injection_buffer_is_full =
                      true;
                }

              } else {
                flag_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty =
                    true;
              }
            }

            break;
          }
          case SPEC_UNIT_1:

            (*inp.m_out[i]).set_latency(tmp_inst_trace->get_latency(), reg_id);
            (*inp.m_out[i])
                .set_initial_interval(tmp_inst_trace->get_initiation_interval(),
                                      reg_id);

            offset_fu =
                m_hw_cfg->get_num_sp_units() + m_hw_cfg->get_num_sfu_units() +
                m_hw_cfg->get_num_int_units() + m_hw_cfg->get_num_dp_units() +
                m_hw_cfg->get_num_tensor_core_units() +
                m_hw_cfg->get_num_mem_units();
            for (unsigned _ = 0; _ < 1; _++) {
              if (m_fu[offset_fu + _]->can_issue(
                      tmp_inst_trace->get_latency())) {
                schedule_wb_now = !m_fu[offset_fu + _]->stallable();
                resbus = test_result_bus(tmp_inst_trace->get_latency());

                insert_into_active_warps_id(&active_warps_id, _wid);
                active_during_this_cycle = true;
                if (schedule_wb_now && (resbus != -1)) {
                  m_result_bus[resbus]->set(tmp_inst_trace->get_latency());
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else if (!schedule_wb_now) {
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else {

                  flag_Execute_Compute_Structural_result_bus_has_no_slot_for_latency =
                      true;
                }

              } else {
                flag_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty =
                    true;
              }
            }
            break;
          case SPEC_UNIT_2:

            (*inp.m_out[i]).set_latency(tmp_inst_trace->get_latency(), reg_id);
            (*inp.m_out[i])
                .set_initial_interval(tmp_inst_trace->get_initiation_interval(),
                                      reg_id);

            offset_fu =
                m_hw_cfg->get_num_sp_units() + m_hw_cfg->get_num_sfu_units() +
                m_hw_cfg->get_num_int_units() + m_hw_cfg->get_num_dp_units() +
                m_hw_cfg->get_num_tensor_core_units() +
                m_hw_cfg->get_num_mem_units();
            for (unsigned _ = 1; _ < 2; _++) {
              if (m_fu[offset_fu + _]->can_issue(
                      tmp_inst_trace->get_latency())) {
                schedule_wb_now = !m_fu[offset_fu + _]->stallable();
                resbus = test_result_bus(tmp_inst_trace->get_latency());

                insert_into_active_warps_id(&active_warps_id, _wid);
                active_during_this_cycle = true;
                if (schedule_wb_now && (resbus != -1)) {
                  m_result_bus[resbus]->set(tmp_inst_trace->get_latency());
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else if (!schedule_wb_now) {
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else {

                  flag_Execute_Compute_Structural_result_bus_has_no_slot_for_latency =
                      true;
                }

              } else {
                flag_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty =
                    true;
              }
            }
            break;
          case SPEC_UNIT_3:

            (*inp.m_out[i]).set_latency(tmp_inst_trace->get_latency(), reg_id);
            (*inp.m_out[i])
                .set_initial_interval(tmp_inst_trace->get_initiation_interval(),
                                      reg_id);

            offset_fu =
                m_hw_cfg->get_num_sp_units() + m_hw_cfg->get_num_sfu_units() +
                m_hw_cfg->get_num_int_units() + m_hw_cfg->get_num_dp_units() +
                m_hw_cfg->get_num_tensor_core_units() +
                m_hw_cfg->get_num_mem_units();
            for (unsigned _ = 2; _ < 3; _++) {
              if (m_fu[offset_fu + _]->can_issue(
                      tmp_inst_trace->get_latency())) {
                schedule_wb_now = !m_fu[offset_fu + _]->stallable();
                resbus = test_result_bus(tmp_inst_trace->get_latency());

                insert_into_active_warps_id(&active_warps_id, _wid);
                active_during_this_cycle = true;
                if (schedule_wb_now && (resbus != -1)) {
                  m_result_bus[resbus]->set(tmp_inst_trace->get_latency());
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else if (!schedule_wb_now) {
                  m_fu[offset_fu + _]->issue((*inp.m_out[i]), reg_id);

                  break;
                } else {

                  flag_Execute_Compute_Structural_result_bus_has_no_slot_for_latency =
                      true;
                }

              } else {
                flag_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty =
                    true;
              }
            }
            break;
          default:
            std::cout << "Error: tmp_inst_trace->get_func_unit(): "
                      << tmp_inst_trace->get_func_unit() << std::endl;
            assert(false);
          }
        }
      }
    }
  }

  for (unsigned _iter = 0; _iter < get_reg_file_port_throughput(); _iter++) {

    m_operand_collector->step(
        &flag_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
        &flag_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated,
        &flag_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated,
        &flag_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
        tracer, &clk_record, m_cycle);
  }

  unsigned max_issue_per_warp = m_hw_cfg->get_max_insn_issue_per_warp();

  unsigned total_issued_instn_num = 0;
  for (unsigned _sched_id = 0; _sched_id < num_scheds; _sched_id++) {
    auto sched_id = (last_issue_sched_id + _sched_id) % num_scheds;

    for (unsigned i = 0; i < kernel_block_pair.size(); i++) {

      auto it_kernel_block_pair =
          kernel_block_pair.begin() +
          (last_issue_block_index_per_sched[sched_id] + i) %
              kernel_block_pair.size();

      if ((unsigned)(it_kernel_block_pair->first) - 1 != KERNEL_EVALUATION)
        continue;

      unsigned _kid = it_kernel_block_pair->first - 1;
      unsigned _block_id = it_kernel_block_pair->second;
      unsigned _warps_per_block = appcfg->get_num_warp_per_block(_kid);
      unsigned _gwarp_id_start = _warps_per_block * _block_id;
      unsigned _gwarp_id_end = _gwarp_id_start + _warps_per_block - 1;

      unsigned last_issue_warp_id;

      auto find_last_issue_warp_id =
          last_issue_warp_ids.find(std::make_pair(_kid, _block_id));

      if (find_last_issue_warp_id == last_issue_warp_ids.end()) {
        last_issue_warp_ids[std::make_pair(_kid, _block_id)] = 0;
        last_issue_warp_id = 0;
      } else {
        last_issue_warp_id = find_last_issue_warp_id->second;
      }

      for (auto gwid = _gwarp_id_start; (gwid <= _gwarp_id_end);
           gwid++) {

        auto wid =
            (last_issue_warp_id + gwid) % _warps_per_block + _gwarp_id_start;

        unsigned _idx_wid_in_SM = wid - _gwarp_id_start;
        for (auto _ = kernel_block_pair.begin(); _ != kernel_block_pair.end();
             _++) {
          if ((unsigned)(_->first) - 1 != KERNEL_EVALUATION)
            continue;
          if (((unsigned)(_->first) - 1 < _kid) ||
              ((unsigned)(_->first) - 1 == _kid &&
               (unsigned)(_->second) < _block_id))
            _idx_wid_in_SM += appcfg->get_num_warp_per_block(_->first - 1);
        }
        if (_idx_wid_in_SM % num_scheds != sched_id) {

          continue;
        }

        unsigned _kid_block_id_count = 0;
        for (auto _it_kernel_block_pair = kernel_block_pair.begin();
             _it_kernel_block_pair != kernel_block_pair.end();
             _it_kernel_block_pair++) {
          if ((unsigned)(_it_kernel_block_pair->first) - 1 != KERNEL_EVALUATION)
            continue;
          if ((unsigned)(_it_kernel_block_pair->first) - 1 == _kid) {
            if ((unsigned)(_it_kernel_block_pair->second) < _block_id) {
              _kid_block_id_count++;
            }
          }
        }

        auto global_all_kernels_warp_id =
            (unsigned)(wid % _warps_per_block) +
            _kid_block_id_count * _warps_per_block +
            std::accumulate(m_num_warps_per_sm.begin(),
                            m_num_warps_per_sm.begin() + _kid, 0);

        unsigned issued_num = 0;
        unsigned checked_num = 0;

        exec_unit_type_t previous_issued_inst_exec_type =
            exec_unit_type_t::NONE;

        while ((issued_num < max_issue_per_warp) &&
               (checked_num <= issued_num)) {
          bool warp_inst_issued = false;

          std::vector<int> regnums;
          int pred;
          int ar1;
          int ar2;

          if (m_ibuffer->is_not_empty(global_all_kernels_warp_id)) {

            ibuffer_entry entry = m_ibuffer->front(global_all_kernels_warp_id);

            unsigned _fetch_instn_id = entry.uid;
            unsigned _pc = entry.pc;
            unsigned _gwid = entry.wid;
            unsigned _kid = entry.kid;

            compute_instn *tmp = tracer->get_one_kernel_one_warp_one_instn(
                _kid, _gwid, _fetch_instn_id);
            _inst_trace_t *tmp_inst_trace = tmp->inst_trace;
            trace_warp_inst_t *tmp_trace_warp_inst = &(tmp->trace_warp_inst);

            pred = tmp_trace_warp_inst->get_pred();
            ar1 = tmp_trace_warp_inst->get_ar1();
            ar2 = tmp_trace_warp_inst->get_ar2();

            if (tmp_trace_warp_inst->get_op() == MEMORY_BARRIER_OP || 
                tmp_trace_warp_inst->get_op() == BARRIER_OP) {
              bool can_issue_of_bar = judge_can_issue_of_bar(_gwid, _kid);
              if (!can_issue_of_bar) {
                stat_coll->set_At_least_one_Synchronization_Stall_found(true);
                checked_num++;
                continue;
              }
            }

            for (unsigned i = 0; i < tmp_inst_trace->reg_srcs_num; i++) {
              regnums.push_back(tmp_inst_trace->reg_src[i]);
            }
            for (unsigned i = 0; i < tmp_inst_trace->reg_dsts_num; i++) {
              if (tmp_inst_trace->reg_dest_is_pred[i])
                regnums.push_back(tmp_inst_trace->reg_dest[i] +
                                  PRED_NUM_OFFSET);
              else
                regnums.push_back(tmp_inst_trace->reg_dest[i]);
            }

            bool check_is_scoreboard_collision = false;
            check_is_scoreboard_collision = m_scoreboard->checkCollision(
                global_all_kernels_warp_id, regnums,

                (pred < 0) ? pred : pred + PRED_NUM_OFFSET, ar1, ar2);

            auto fu = tmp_inst_trace->get_func_unit();

            if (check_is_scoreboard_collision) {
              if (fu == LDST_UNIT) {
                stat_coll->set_At_least_one_Memory_Data_Stall_found(true);
                flag_Issue_Memory_Data_scoreboard = true;
              } else {
                stat_coll->set_At_least_one_Compute_Data_Stall_found(true);
                flag_Issue_Compute_Data_scoreboard = true;
              }
            }

            if (tmp_trace_warp_inst->get_opcode() == OP_EXIT &&
                tracer->get_one_kernel_one_warp_instn_count(_kid, _gwid) ==
                    _fetch_instn_id + 1 &&
                m_scoreboard->regs_size(global_all_kernels_warp_id) > 0) {
              check_is_scoreboard_collision = true;
            }

            if (check_is_scoreboard_collision) {
              checked_num++;
              continue;
            }

            bool sp_pipe_avail = false;
            bool sfu_pipe_avail = false;
            bool int_pipe_avail = false;
            bool dp_pipe_avail = false;
            bool tensor_core_pipe_avail = false;
            bool ldst_pipe_avail = false;
            bool spec_1_pipe_avail = false;
            bool spec_2_pipe_avail = false;
            bool spec_3_pipe_avail = false;

            switch (fu) {
            case NON_UNIT:
              assert(0);
              break;
            case SP_UNIT: {
              bool _has_free_slot_sp =
                  m_sp_out->has_free(m_hw_cfg->get_sub_core_model(), sched_id);
              if (!_has_free_slot_sp) {
                flag_Issue_Compute_Structural_out_has_no_free_slot = true;
              }
              bool pre =
                  (previous_issued_inst_exec_type != exec_unit_type_t::SP);
              if (!pre) {
                flag_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute =
                    true;
              }
              sp_pipe_avail =
                  (m_hw_cfg->get_num_sp_units() > 0) && _has_free_slot_sp &&
                  (!m_hw_cfg->get_dual_issue_diff_exec_units() || pre);

              if (sp_pipe_avail) {
                warp_inst_issued = true;
                issued_num++;
                issue_warp(*m_sp_out, entry, sched_id);
                previous_issued_inst_exec_type = exec_unit_type_t::SP;

              } else {
                stat_coll->set_At_least_one_Compute_Structural_Stall_found(
                    true);
              }
              break;
            }
            case SFU_UNIT: {
              bool _has_free_slot_sfu =
                  m_sfu_out->has_free(m_hw_cfg->get_sub_core_model(), sched_id);
              if (!_has_free_slot_sfu) {
                flag_Issue_Compute_Structural_out_has_no_free_slot = true;
              }
              bool pre =
                  (previous_issued_inst_exec_type != exec_unit_type_t::SFU);
              if (!pre) {
                flag_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute =
                    true;
              }
              sfu_pipe_avail =
                  (m_hw_cfg->get_num_sfu_units() > 0) && _has_free_slot_sfu &&
                  (!m_hw_cfg->get_dual_issue_diff_exec_units() || pre);

              if (sfu_pipe_avail) {

                warp_inst_issued = true;
                issued_num++;
                issue_warp(*m_sfu_out, entry, sched_id);
                previous_issued_inst_exec_type = exec_unit_type_t::SFU;

              } else {
                stat_coll->set_At_least_one_Compute_Structural_Stall_found(
                    true);
              }
              break;
            }
            case INT_UNIT: {
              bool _has_free_slot_int =
                  m_int_out->has_free(m_hw_cfg->get_sub_core_model(), sched_id);
              if (!_has_free_slot_int) {
                flag_Issue_Compute_Structural_out_has_no_free_slot = true;
              }
              bool pre =
                  (previous_issued_inst_exec_type != exec_unit_type_t::INT);
              if (!pre) {
                flag_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute =
                    true;
              }
              int_pipe_avail =
                  (m_hw_cfg->get_num_int_units() > 0) && _has_free_slot_int &&
                  (!m_hw_cfg->get_dual_issue_diff_exec_units() || pre);

              if (int_pipe_avail) {

                warp_inst_issued = true;
                issued_num++;
                issue_warp(*m_int_out, entry, sched_id);
                previous_issued_inst_exec_type = exec_unit_type_t::INT;

              } else {
                stat_coll->set_At_least_one_Compute_Structural_Stall_found(
                    true);
              }
              break;
            }
            case DP_UNIT: {
              bool _has_free_slot_dp =
                  m_dp_out->has_free(m_hw_cfg->get_sub_core_model(), sched_id);
              if (!_has_free_slot_dp) {
                flag_Issue_Compute_Structural_out_has_no_free_slot = true;
              }
              bool pre =
                  (previous_issued_inst_exec_type != exec_unit_type_t::DP);
              if (!pre) {
                flag_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute =
                    true;
              }
              dp_pipe_avail =
                  (m_hw_cfg->get_num_dp_units() > 0) && _has_free_slot_dp &&
                  (!m_hw_cfg->get_dual_issue_diff_exec_units() || pre);

              if (dp_pipe_avail) {

                warp_inst_issued = true;
                issued_num++;
                issue_warp(*m_dp_out, entry, sched_id);
                previous_issued_inst_exec_type = exec_unit_type_t::DP;

              } else {
                stat_coll->set_At_least_one_Compute_Structural_Stall_found(
                    true);
              }
              break;
            }
            case TENSOR_CORE_UNIT: {
              bool _has_free_slot_tc = m_tensor_core_out->has_free(
                  m_hw_cfg->get_sub_core_model(), sched_id);
              if (!_has_free_slot_tc) {
                flag_Issue_Compute_Structural_out_has_no_free_slot = true;
              }
              bool pre =
                  (previous_issued_inst_exec_type != exec_unit_type_t::TENSOR);
              if (!pre) {
                flag_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute =
                    true;
              }
              tensor_core_pipe_avail =
                  (m_hw_cfg->get_num_tensor_core_units() > 0) &&
                  _has_free_slot_tc &&
                  (!m_hw_cfg->get_dual_issue_diff_exec_units() || pre);

              if (tensor_core_pipe_avail) {

                warp_inst_issued = true;
                issued_num++;
                issue_warp(*m_tensor_core_out, entry, sched_id);
                previous_issued_inst_exec_type = exec_unit_type_t::TENSOR;

              } else {
                stat_coll->set_At_least_one_Compute_Structural_Stall_found(
                    true);
              }
              break;
            }
            case LDST_UNIT: {
              bool _has_free_slot_mem =
                  m_mem_out->has_free(m_hw_cfg->get_sub_core_model(), sched_id);
              if (!_has_free_slot_mem) {
                flag_Issue_Memory_Structural_out_has_no_free_slot = true;
              }
              bool pre =
                  (previous_issued_inst_exec_type != exec_unit_type_t::LDST);
              if (!pre) {
                flag_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory =
                    true;
              }
              ldst_pipe_avail =
                  (m_hw_cfg->get_num_mem_units() > 0) && _has_free_slot_mem &&
                  (!m_hw_cfg->get_dual_issue_diff_exec_units() || pre);

              if (ldst_pipe_avail) {

                warp_inst_issued = true;
                issued_num++;
                issue_warp(*m_mem_out, entry, sched_id);
                previous_issued_inst_exec_type = exec_unit_type_t::LDST;

              } else {
                stat_coll->set_At_least_one_Memory_Structural_Stall_found(true);
              }
              break;
            }
            case SPEC_UNIT_1: {
              bool _has_free_slot_spec1 = m_spec_cores_out[0]->has_free(
                  m_hw_cfg->get_sub_core_model(), sched_id);
              if (!_has_free_slot_spec1) {
                flag_Issue_Compute_Structural_out_has_no_free_slot = true;
              }
              bool pre = (previous_issued_inst_exec_type !=
                          exec_unit_type_t::SPECIALIZED);
              if (!pre) {
                flag_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute =
                    true;
              }
              spec_1_pipe_avail =
                  (m_hw_cfg->get_specialized_unit_1_enabled()) &&
                  _has_free_slot_spec1 &&
                  (!m_hw_cfg->get_dual_issue_diff_exec_units() || pre);

              if (spec_1_pipe_avail) {

                warp_inst_issued = true;
                issued_num++;
                issue_warp(*m_spec_cores_out[0], entry, sched_id);
                previous_issued_inst_exec_type = exec_unit_type_t::SPECIALIZED;

              } else {
                stat_coll->set_At_least_one_Compute_Structural_Stall_found(
                    true);
              }
              break;
            }
            case SPEC_UNIT_2: {
              bool _has_free_slot_spec2 = m_spec_cores_out[1]->has_free(
                  m_hw_cfg->get_sub_core_model(), sched_id);
              if (!_has_free_slot_spec2) {
                flag_Issue_Compute_Structural_out_has_no_free_slot = true;
              }
              bool pre = (previous_issued_inst_exec_type !=
                          exec_unit_type_t::SPECIALIZED);
              if (!pre) {
                flag_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute =
                    true;
              }
              spec_2_pipe_avail =
                  (m_hw_cfg->get_specialized_unit_2_enabled()) &&
                  _has_free_slot_spec2 &&
                  (!m_hw_cfg->get_dual_issue_diff_exec_units() || pre);

              if (spec_2_pipe_avail) {

                warp_inst_issued = true;
                issued_num++;
                issue_warp(*m_spec_cores_out[1], entry, sched_id);
                previous_issued_inst_exec_type = exec_unit_type_t::SPECIALIZED;

              } else {
                stat_coll->set_At_least_one_Compute_Structural_Stall_found(
                    true);
              }
              break;
            }
            case SPEC_UNIT_3: {
              bool _has_free_slot_spec3 = m_spec_cores_out[2]->has_free(
                  m_hw_cfg->get_sub_core_model(), sched_id);
              if (!_has_free_slot_spec3) {
                flag_Issue_Compute_Structural_out_has_no_free_slot = true;
              }
              bool pre = (previous_issued_inst_exec_type !=
                          exec_unit_type_t::SPECIALIZED);
              if (!pre) {
                flag_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute =
                    true;
              }
              spec_3_pipe_avail =
                  (m_hw_cfg->get_specialized_unit_3_enabled()) &&
                  _has_free_slot_spec3 &&
                  (!m_hw_cfg->get_dual_issue_diff_exec_units() || pre);

              if (spec_3_pipe_avail) {

                warp_inst_issued = true;
                issued_num++;
                issue_warp(*m_spec_cores_out[2], entry, sched_id);
                previous_issued_inst_exec_type = exec_unit_type_t::SPECIALIZED;

              } else {
                stat_coll->set_At_least_one_Compute_Structural_Stall_found(
                    true);
              }
              break;
            }
            default:
              assert(0);
            }

            if (false) {
              std::cout << "=================================================="
                        << std::endl;
              for (auto it = m_pipeline_reg.begin(); it != m_pipeline_reg.end();
                   it++) {
                std::cout << "  ||name: " << it->get_name() << std::endl;
                std::cout << "    has_ready: " << it->has_ready() << std::endl;
                std::cout << "    has_free: " << it->has_free() << std::endl;
              }

              std::cout << "  ||name: " << m_sp_out->get_name() << std::endl;
              std::cout << "    has_ready: " << m_sp_out->has_ready()
                        << std::endl;
              std::cout << "    has_free: " << m_sp_out->has_free()
                        << std::endl;

              std::cout << "  ||name: " << m_dp_out->get_name() << std::endl;
              std::cout << "    has_ready: " << m_dp_out->has_ready()
                        << std::endl;
              std::cout << "    has_free: " << m_dp_out->has_free()
                        << std::endl;

              std::cout << "  ||name: " << m_sfu_out->get_name() << std::endl;
              std::cout << "    has_ready: " << m_sfu_out->has_ready()
                        << std::endl;
              std::cout << "    has_free: " << m_sfu_out->has_free()
                        << std::endl;

              std::cout << "  ||name: " << m_int_out->get_name() << std::endl;
              std::cout << "    has_ready: " << m_int_out->has_ready()
                        << std::endl;
              std::cout << "    has_free: " << m_int_out->has_free()
                        << std::endl;

              std::cout << "  ||name: " << m_tensor_core_out->get_name()
                        << std::endl;
              std::cout << "    has_ready: " << m_tensor_core_out->has_ready()
                        << std::endl;
              std::cout << "    has_free: " << m_tensor_core_out->has_free()
                        << std::endl;

              std::cout << "  ||name: " << m_mem_out->get_name() << std::endl;
              std::cout << "    has_ready: " << m_mem_out->has_ready()
                        << std::endl;
              std::cout << "    has_free: " << m_mem_out->has_free()
                        << std::endl;

              std::cout << "  ||name: " << m_spec_cores_out[0]->get_name()
                        << std::endl;
              std::cout << "    has_ready: " << m_spec_cores_out[0]->has_ready()
                        << std::endl;
              std::cout << "    has_free: " << m_spec_cores_out[0]->has_free()
                        << std::endl;

              std::cout << "  ||name: " << m_spec_cores_out[1]->get_name()
                        << std::endl;
              std::cout << "    has_ready: " << m_spec_cores_out[1]->has_ready()
                        << std::endl;
              std::cout << "    has_free: " << m_spec_cores_out[1]->has_free()
                        << std::endl;

              std::cout << "  ||name: " << m_spec_cores_out[2]->get_name()
                        << std::endl;
              std::cout << "    has_ready: " << m_spec_cores_out[2]->has_ready()
                        << std::endl;
              std::cout << "    has_free: " << m_spec_cores_out[2]->has_free()
                        << std::endl;

              std::cout << "=================================================="
                        << std::endl;
            }

            if (warp_inst_issued) {
              total_issued_instn_num++;
              if (_CALIBRATION_LOG_) {
                std::cout << "    ISSUE: (" << _kid << ", " << _gwid << ", "
                          << _fetch_instn_id << ", " << _pc << ")" << std::endl;
              }
              set_clk_record<2>(_kid, _gwid, _fetch_instn_id, m_cycle);
            }

          } else {
            stat_coll->set_At_least_one_Idle_Stall_found(true);
          }

          if (warp_inst_issued) {
            active_during_this_cycle = true;
            insert_into_active_warps_id(&active_warps_id,
                                        global_all_kernels_warp_id);
            m_ibuffer->pop_front(global_all_kernels_warp_id);

            regnums.push_back((pred < 0) ? pred : pred + PRED_NUM_OFFSET);

            regnums.push_back(ar1);
            regnums.push_back(ar2);

            /// TODO: Use `std::unordered_set` to replace `std::vector<int> regnums`.
            m_scoreboard->reserveRegisters(global_all_kernels_warp_id, regnums,
                                           false);
          }

          checked_num++;
        }
      }
      last_issue_warp_ids[std::make_pair(_kid, _block_id)] =
          (last_issue_warp_ids[std::make_pair(_kid, _block_id)] + 1) %
          _warps_per_block;
    }
    last_issue_block_index_per_sched[sched_id] =
        (last_issue_block_index_per_sched[sched_id] + 1) %
        kernel_block_pair.size();
  }

  last_issue_sched_id = (last_issue_sched_id + 1) % num_scheds;

  if (total_issued_instn_num >= num_scheds) {
    stat_coll->set_At_least_one_No_Stall_found(true);
  }

  for (unsigned _ = 0; _ < get_inst_fetch_throughput(); _++) {

    if (m_inst_fetch_buffer->m_valid) {
      auto _pc = m_inst_fetch_buffer->pc;

      auto _wid = m_inst_fetch_buffer->wid;
      auto _kid = m_inst_fetch_buffer->kid;
      auto _uid = m_inst_fetch_buffer->uid;

      unsigned __pc = 0;
      unsigned __wid = 0;
      unsigned __kid = 0;
      unsigned __uid = 0;

      if (m_inst_fetch_buffer_copy->m_valid) {
        __pc = m_inst_fetch_buffer_copy->pc;
        __wid = m_inst_fetch_buffer_copy->wid;
        __kid = m_inst_fetch_buffer_copy->kid;
        __uid = m_inst_fetch_buffer_copy->uid;
      }

      unsigned _warps_per_block = appcfg->get_num_warp_per_block(_kid);

      unsigned _block_id = (unsigned)(_wid / _warps_per_block);

      unsigned _kid_block_id_count = 0;
      for (auto _it_kernel_block_pair = kernel_block_pair.begin();
           _it_kernel_block_pair != kernel_block_pair.end();
           _it_kernel_block_pair++) {
        if ((unsigned)(_it_kernel_block_pair->first) - 1 != KERNEL_EVALUATION)
          continue;
        if ((unsigned)(_it_kernel_block_pair->first) - 1 == _kid) {
          if ((unsigned)(_it_kernel_block_pair->second) < _block_id) {
            _kid_block_id_count++;
          }
        }
      }

      auto global_all_kernels_warp_id =
          (unsigned)(_wid % _warps_per_block) +
          _kid_block_id_count * _warps_per_block +
          std::accumulate(m_num_warps_per_sm.begin(),
                          m_num_warps_per_sm.begin() + _kid, 0);

      if (m_ibuffer->has_free_slot(global_all_kernels_warp_id)) {
        auto _entry = ibuffer_entry(_pc, _wid, _kid, _uid);
        m_ibuffer->push_back(global_all_kernels_warp_id, _entry);
        m_inst_fetch_buffer->m_valid = false;
        active_during_this_cycle = true;
        insert_into_active_warps_id(&active_warps_id,
                                    global_all_kernels_warp_id);
        if (_CALIBRATION_LOG_) {
          std::cout << "    DECODE: (" << _entry.kid << ", " << _entry.wid
                    << ", " << _entry.uid << ", " << _entry.pc << ")"
                    << std::endl;
        }
        set_clk_record<1>(_entry.kid, _entry.wid, _entry.uid, m_cycle);

        if (m_inst_fetch_buffer_copy->m_valid) {
          ibuffer_entry __entry = ibuffer_entry(__pc, __wid, __kid, __uid);
          m_ibuffer->push_back(global_all_kernels_warp_id, __entry);
          m_inst_fetch_buffer_copy->m_valid = false;
          if (_CALIBRATION_LOG_) {
            std::cout << "    DECODE: (" << __entry.kid << ", " << __entry.wid
                      << ", " << __entry.uid << ", " << __entry.pc << ")"
                      << std::endl;
          }
          set_clk_record<1>(__entry.kid, __entry.wid, __entry.uid, m_cycle);
        }
      }
    }

    if (m_inst_fetch_buffer->m_valid) {
      continue;
    }

    unsigned all_blocks_in_this_sm = 0;
    int first_block_in_this_sm = -1;
    for (auto it_kernel_block_pair_ = kernel_block_pair.begin();
         it_kernel_block_pair_ != kernel_block_pair.end();
         it_kernel_block_pair_++) {

      if (((unsigned)(it_kernel_block_pair_->first) - 1) == KERNEL_EVALUATION) {
        all_blocks_in_this_sm += 1;

        if (first_block_in_this_sm == -1) {
          first_block_in_this_sm =
              std::distance(kernel_block_pair.begin(), it_kernel_block_pair_);
        }
      }
    }

    if (true) {

      std::vector<std::pair<int, int>> kernel_block_pair_need_to_check;
      for (auto it_kernel_block_pair = kernel_block_pair.begin();
           it_kernel_block_pair != kernel_block_pair.end();
           it_kernel_block_pair++) {
        if ((unsigned)(it_kernel_block_pair->first) - 1 != KERNEL_EVALUATION)
          continue;
        unsigned _index =
            std::distance(kernel_block_pair.begin(), it_kernel_block_pair);
        if (m_thread_block_has_executed_status[_index] == true &&
            get_num_m_warp_active_status(_index) > 0) {
          kernel_block_pair_need_to_check.push_back(*it_kernel_block_pair);
        }
      }

      for (unsigned check_block_id_index_idx = 0;
           check_block_id_index_idx < kernel_block_pair_need_to_check.size();
           check_block_id_index_idx++) {
        if (m_inst_fetch_buffer->m_valid)
          break;

        unsigned check_block_id =
            (check_block_id_index_idx + last_check_block_id_index_idx) %
            kernel_block_pair_need_to_check.size();

        unsigned _kid =
            kernel_block_pair_need_to_check[check_block_id].first - 1;
        unsigned _block_id =
            kernel_block_pair_need_to_check[check_block_id].second;
        unsigned _warps_per_block = appcfg->get_num_warp_per_block(_kid);
        unsigned _gwarp_id_start = _warps_per_block * _block_id;
        unsigned _gwarp_id_end = _gwarp_id_start + _warps_per_block - 1;

        for (auto gwid = _gwarp_id_start; gwid <= _gwarp_id_end; gwid++) {
          unsigned wid =
              (gwid +
               kernel_id_block_id_last_fetch_wid[{_kid, check_block_id}]) %
                  _warps_per_block +
              _gwarp_id_start;

          unsigned _index = 0;
          for (auto it_kernel_block_pair = kernel_block_pair.begin();
               it_kernel_block_pair != kernel_block_pair.end();
               it_kernel_block_pair++) {
            if (((unsigned)(it_kernel_block_pair->first) - 1 == _kid) &&
                ((unsigned)(it_kernel_block_pair->second) == _block_id)) {
              _index = std::distance(kernel_block_pair.begin(),
                                     it_kernel_block_pair);
            }
          }

          unsigned _w_id_ = (unsigned)(gwid % _warps_per_block);
          if (!(m_thread_block_has_executed_status[_index] == true &&
                m_warp_active_status[_index][_w_id_]))
            continue;

          while (!m_inst_fetch_buffer->m_valid) {
            curr_instn_id_per_warp_entry _entry = curr_instn_id_per_warp_entry(
                _kid, _block_id, wid - _gwarp_id_start);
            unsigned fetch_instn_id = curr_instn_id_per_warp[_entry];

            unsigned one_warp_instn_size =
                tracer->get_one_kernel_one_warp_instn_size(_kid, wid);

            if (one_warp_instn_size <= fetch_instn_id) {
              unsigned _wid_1 = wid - _gwarp_id_start;
              m_warp_active_status[_index][_wid_1] = false;
              break;
            }

            compute_instn *tmp = tracer->get_one_kernel_one_warp_one_instn(
                _kid, wid, fetch_instn_id);

            _inst_trace_t *tmp_inst_trace = tmp->inst_trace;

            if (!tmp_inst_trace->m_valid)
              break;

            m_inst_fetch_buffer->pc = tmp_inst_trace->m_pc;

            m_inst_fetch_buffer->wid = wid;
            m_inst_fetch_buffer->kid = _kid;
            m_inst_fetch_buffer->uid = fetch_instn_id;
            m_inst_fetch_buffer->m_valid = true;

            active_during_this_cycle = true;
            insert_into_active_warps_id(&active_warps_id, wid);

            if (_CALIBRATION_LOG_) {
              std::cout << "    FETCH: (" << _kid << ", " << wid << ", "
                        << fetch_instn_id << ", " << tmp_inst_trace->m_pc << ")"
                        << std::endl;
            }
            set_clk_record<0>(_kid, wid, fetch_instn_id, m_cycle);

            curr_instn_id_per_warp[_entry] += 2;

            if (fetch_instn_id + 1 <
                tracer->get_one_kernel_one_warp_one_instn_max_size(_kid, wid)) {
              tmp = tracer->get_one_kernel_one_warp_one_instn(
                  _kid, wid, fetch_instn_id + 1);
              tmp_inst_trace = tmp->inst_trace;
              if (!tmp_inst_trace->m_valid)
                break;
              m_inst_fetch_buffer_copy->pc = tmp_inst_trace->m_pc;
              m_inst_fetch_buffer_copy->wid = wid;
              m_inst_fetch_buffer_copy->kid = _kid;
              m_inst_fetch_buffer_copy->uid = fetch_instn_id + 1;
              m_inst_fetch_buffer_copy->m_valid = true;
            } else {
              m_inst_fetch_buffer_copy->pc = tmp_inst_trace->m_pc;
              m_inst_fetch_buffer_copy->wid = wid;
              m_inst_fetch_buffer_copy->kid = _kid;
              m_inst_fetch_buffer_copy->uid = fetch_instn_id;
              m_inst_fetch_buffer_copy->m_valid = false;
            }

            if (_CALIBRATION_LOG_) {
              std::cout << "    FETCH: (" << _kid << ", " << wid << ", "
                        << fetch_instn_id + 1 << ", " << tmp_inst_trace->m_pc
                        << ")" << std::endl;
            }
            set_clk_record<0>(_kid, wid, fetch_instn_id + 1, m_cycle);
          }
        }
        kernel_id_block_id_last_fetch_wid[{_kid, check_block_id}] =
            (kernel_id_block_id_last_fetch_wid[{_kid, check_block_id}] + 1) %
            _warps_per_block;
      }

      last_check_block_id_index_idx++;
    }
  }

  for (unsigned i = 0; i < num_banks; i++) {
    if (m_reg_bank_allocator->getBankState(i) == ON_WRITING ||
        m_reg_bank_allocator->getBankState(i) == ON_READING) {
      m_reg_bank_allocator->releaseBankState(i);
    }
  }

  bool all_warps_finished = true;

  for (auto it_kernel_block_pair = kernel_block_pair.begin();
       it_kernel_block_pair != kernel_block_pair.end();
       it_kernel_block_pair++) {
    if ((unsigned)(it_kernel_block_pair->first) - 1 != KERNEL_EVALUATION)
      continue;

    unsigned _index =
        std::distance(kernel_block_pair.begin(), it_kernel_block_pair);
    unsigned _kid = it_kernel_block_pair->first - 1;
    unsigned _warps_per_block = appcfg->get_num_warp_per_block(_kid);

    for (unsigned _w_id_ = 0; _w_id_ < _warps_per_block; _w_id_++) {

      if ((m_thread_block_has_executed_status[_index] == true &&
           m_warp_active_status[_index][_w_id_]) ||
          (m_thread_block_has_executed_status[_index] == false)) {
        all_warps_finished = false;
        break;
      }
    }

    if (!all_warps_finished)
      break;
  }

  if (all_warps_finished) {

    active = false;
  }

  if (active_during_this_cycle) {
    active_cycles++;
  }

  active_warps_id_size_sum += get_num_m_warp_active_status();

  if (stat_coll->get_At_least_one_No_Stall_found()) {
    stat_coll->increment_No_Stall(m_smid);
  } else if (stat_coll->get_At_least_one_Memory_Structural_Stall_found()) {
    stat_coll->increment_Memory_Structural_Stall(m_smid);

    stat_coll->increment_num_Issue_Memory_Structural_out_has_no_free_slot(
        (unsigned)flag_Issue_Memory_Structural_out_has_no_free_slot, m_smid);
    stat_coll
        ->increment_num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory(
            (unsigned)
                flag_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory,
            m_smid);
    stat_coll
        ->increment_num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency(
            (unsigned)
                flag_Execute_Memory_Structural_result_bus_has_no_slot_for_latency,
            m_smid);
    stat_coll
        ->increment_num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty(
            (unsigned)
                flag_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty,
            m_smid);
    stat_coll
        ->increment_num_Writeback_Memory_Structural_bank_of_reg_is_not_idle(
            (unsigned)flag_Writeback_Memory_Structural_bank_of_reg_is_not_idle,
            m_smid);
    stat_coll
        ->increment_num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated(
            (unsigned)
                flag_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated,
            m_smid);
    stat_coll
        ->increment_num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu(
            (unsigned)
                flag_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
            m_smid);
    stat_coll
        ->increment_num_Execute_Memory_Structural_icnt_injection_buffer_is_full(
            (unsigned)
                flag_Execute_Memory_Structural_icnt_injection_buffer_is_full,
            m_smid);

  } else if (stat_coll->get_At_least_one_Memory_Data_Stall_found()) {
    stat_coll->increment_Memory_Data_Stall(m_smid);

    stat_coll->increment_num_Issue_Memory_Data_scoreboard(
        (unsigned)flag_Issue_Memory_Data_scoreboard, m_smid);
    stat_coll->increment_num_Execute_Memory_Data_L1(
        (unsigned)flag_Execute_Memory_Data_L1, m_smid);
    stat_coll->increment_num_Execute_Memory_Data_L2(
        (unsigned)flag_Execute_Memory_Data_L2, m_smid);
    stat_coll->increment_num_Execute_Memory_Data_Main_Memory(
        (unsigned)flag_Execute_Memory_Data_Main_Memory, m_smid);

  } else if (stat_coll->get_At_least_one_Synchronization_Stall_found()) {
    stat_coll->increment_Synchronization_Stall(m_smid);
  } else if (stat_coll->get_At_least_one_Compute_Structural_Stall_found()) {
    stat_coll->increment_Compute_Structural_Stall(m_smid);

    stat_coll->increment_num_Issue_Compute_Structural_out_has_no_free_slot(
        (unsigned)flag_Issue_Compute_Structural_out_has_no_free_slot, m_smid);
    stat_coll
        ->increment_num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute(
            (unsigned)
                flag_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute,
            m_smid);
    stat_coll
        ->increment_num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency(
            (unsigned)
                flag_Execute_Compute_Structural_result_bus_has_no_slot_for_latency,
            m_smid);
    stat_coll
        ->increment_num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty(
            (unsigned)
                flag_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty,
            m_smid);
    stat_coll
        ->increment_num_Writeback_Compute_Structural_bank_of_reg_is_not_idle(
            (unsigned)flag_Writeback_Compute_Structural_bank_of_reg_is_not_idle,
            m_smid);
    stat_coll
        ->increment_num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated(
            (unsigned)
                flag_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated,
            m_smid);
    stat_coll
        ->increment_num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu(
            (unsigned)
                flag_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
            m_smid);

  } else if (stat_coll->get_At_least_one_Compute_Data_Stall_found()) {
    stat_coll->increment_Compute_Data_Stall(m_smid);

    stat_coll->increment_num_Issue_Compute_Data_scoreboard(
        (unsigned)flag_Issue_Compute_Data_scoreboard, m_smid);

  } else if (stat_coll->get_At_least_one_Control_Stall_found()) {
    stat_coll->increment_Control_Stall(m_smid);
  } else if (stat_coll->get_At_least_one_Idle_Stall_found()) {
    stat_coll->increment_Idle_Stall(m_smid);
  } else {
    stat_coll->increment_Other_Stall(m_smid);
  }

  stat_coll->set_At_least_four_instns_issued(false);
  stat_coll->set_At_least_one_Compute_Structural_Stall_found(false);
  stat_coll->set_At_least_one_Compute_Data_Stall_found(false);
  stat_coll->set_At_least_one_Memory_Structural_Stall_found(false);
  stat_coll->set_At_least_one_Memory_Data_Stall_found(false);
  stat_coll->set_At_least_one_Synchronization_Stall_found(false);
  stat_coll->set_At_least_one_Control_Stall_found(false);
  stat_coll->set_At_least_one_Idle_Stall_found(false);
  stat_coll->set_At_least_one_No_Stall_found(false);

  if (all_warps_finished && PRINT_STALLS_DISTRIBUTION) {
    float all_total_stalls =
        (float)(stat_coll->get_Compute_Structural_Stall(m_smid) +
                stat_coll->get_Compute_Data_Stall(m_smid) +
                stat_coll->get_Memory_Structural_Stall(m_smid) +
                stat_coll->get_Memory_Data_Stall(m_smid) +
                stat_coll->get_Synchronization_Stall(m_smid) +
                stat_coll->get_Control_Stall(m_smid) +
                stat_coll->get_Idle_Stall(m_smid) +
                stat_coll->get_Other_Stall(m_smid) +
                stat_coll->get_No_Stall(m_smid));
    std::cout << "  Stalls Distribution:" << std::endl;
    std::cout << "    Compute Structural Stall: "
              << (float)stat_coll->get_Compute_Structural_Stall(m_smid) /
                     all_total_stalls
              << std::endl;
    std::cout << "    Compute Data Stall: "
              << (float)stat_coll->get_Compute_Data_Stall(m_smid) /
                     all_total_stalls
              << std::endl;
    std::cout << "    Memory Structural Stall: "
              << (float)stat_coll->get_Memory_Structural_Stall(m_smid) /
                     all_total_stalls
              << std::endl;
    std::cout << "    Memory Data Stall: "
              << (float)stat_coll->get_Memory_Data_Stall(m_smid) /
                     all_total_stalls
              << std::endl;
    std::cout << "    Synchronization Stall: "
              << (float)stat_coll->get_Synchronization_Stall(m_smid) /
                     all_total_stalls
              << std::endl;
    std::cout << "    Control Stall: "
              << (float)stat_coll->get_Control_Stall(m_smid) / all_total_stalls
              << std::endl;
    std::cout << "    Idle Stall: "
              << (float)stat_coll->get_Idle_Stall(m_smid) / all_total_stalls
              << std::endl;
    std::cout << "    Other Stall: "
              << (float)stat_coll->get_Other_Stall(m_smid) / all_total_stalls
              << std::endl;
    std::cout << "    No Stall: "
              << (float)stat_coll->get_No_Stall(m_smid) / all_total_stalls
              << std::endl;
  }
}
