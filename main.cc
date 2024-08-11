#include <algorithm>
#include <cxxabi.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <time.h>
#include <vector>

#include "./ISA-Def/trace_opcode.h"
#include "./trace-parser/trace-parser.h"

#include "./ISA-Def/accelwattch_component_mapping.h"
#include "./ISA-Def/ampere_opcode.h"
#include "./ISA-Def/kepler_opcode.h"
#include "./ISA-Def/pascal_opcode.h"
#include "./ISA-Def/trace_opcode.h"
#include "./ISA-Def/turing_opcode.h"
#include "./ISA-Def/volta_opcode.h"
#include "./common/vector_types.h"
#include "./trace-driven/kernel-info.h"
#include "./trace-driven/mem-access.h"
#include "./trace-driven/trace-warp-inst.h"

#include "./common/CLI/CLI.hpp"
#include "./hw-parser/hw-parser.h"
#include "./parda/parda.h"

#include "../hw-component/PrivateSM.h"

#include <chrono>
#include <cmath>
#include <ctime>

float ceil(float x, float s) { return s * std::ceil(x / s); }

float floor(float x, float s) { return s * std::floor(x / s); }

trace_kernel_info_t *create_kernel_info(kernel_trace_t *kernel_trace_info,
                                        trace_parser *parser) {
  dim3 gridDim(kernel_trace_info->grid_dim_x, kernel_trace_info->grid_dim_y,
               kernel_trace_info->grid_dim_z);
  dim3 blockDim(kernel_trace_info->tb_dim_x, kernel_trace_info->tb_dim_y,
                kernel_trace_info->tb_dim_z);
  trace_kernel_info_t *kernel_info =
      new trace_kernel_info_t(gridDim, blockDim, parser, kernel_trace_info);

  return kernel_info;
}

bool compare_stamp(const mem_instn a, const mem_instn b) {

  return a.time_stamp < b.time_stamp;
}

void print_SM_traces(std::vector<mem_instn> *traces) {
  for (auto mem_ins : *traces) {
    std::cout << std::setw(18) << std::right << std::hex << mem_ins.pc << " ";
    std::cout << std::hex << mem_ins.time_stamp << " ";
    std::cout << std::hex << mem_ins.addr[0] << std::endl;
  }
}

std::string func_unit_name_to_string(FUNC_UNITS_NAME unit) {
  switch (unit) {
  case NON_UNIT:
    return "NON_UNIT";
  case SP_UNIT:
    return "SP";
  case SFU_UNIT:
    return "SFU";
  case INT_UNIT:
    return "INT";
  case DP_UNIT:
    return "DP";
  case TENSOR_CORE_UNIT:
    return "TENSOR_CORE";
  case LDST_UNIT:
    return "LDST";
  case SPEC_UNIT_1:
    return "SPEC_1";
  case SPEC_UNIT_2:
    return "SPEC_2";
  case SPEC_UNIT_3:
    return "SPEC_3";
  default:
    return "Others";
  }
}

int getIthKey(std::map<int, std::vector<mem_instn>> *SM_traces_ptr, int i) {
  auto it = (*SM_traces_ptr).begin();
  std::advance(it, i);

  return it->first;
}

#ifdef USE_BOOST

void private_L1_cache_stack_distance_evaluate_boost_no_concurrent(
    int argc, char **argv,
    std::vector<std::map<int, std::vector<mem_instn>>> *SM_traces_all_passes,
    std::map<std::tuple<int, int, unsigned long long>, std::map<unsigned, bool>>
        *mem_instn_distance_overflow_flag,
    int _tmp_print_, std::string configs_dir, bool dump_histogram,
    stat_collector *stat_coll, hw_config *hw_cfg, unsigned KERNEL_EVALUATION,
    std::vector<unsigned> *MEM_ACCESS_LATENCY) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;

  const int pass_num =
      int((hw_cfg->get_num_sms() + world.size() - 1) / world.size());

  unsigned l1_cache_line_size = hw_cfg->get_l1_cache_line_size_for_reuse_distance();
  unsigned l1_cache_size = hw_cfg->get_unified_l1d_size() * 1024 - 
                           hw_cfg->get_shmem_size_per_sm();
  // unsigned l1_cache_associativity = hw_cfg->get_l1d_cache_associative();
  unsigned l1_cache_blocks = l1_cache_size / l1_cache_line_size;
  unsigned l2_cache_line_size = hw_cfg->get_l2_cache_line_size_for_reuse_distance();
  unsigned l2_cache_size = hw_cfg->get_l2d_size_per_sub_partition() * 1024 * 
                           hw_cfg->get_num_memory_controllers() * 
                           hw_cfg->get_num_sub_partition_per_memory_channel();
  // unsigned l2_cache_associativity = hw_cfg->get_l2d_cache_associative();
  unsigned l2_cache_blocks = l2_cache_size / l2_cache_line_size;

  float L1_hit_rate = 0.0;

  for (int pass = 0; pass < pass_num; pass++) {
    unsigned curr_process_idx_rank = world.rank() + pass * world.size();
    unsigned curr_process_idx;
    if (curr_process_idx_rank < hw_cfg->get_num_sms()) {
      curr_process_idx = curr_process_idx_rank;
    } else
      continue;

    HKEY input;
    long tim;
    program_data_t pdt_c;
    program_data_t *pdt;
    FILE *file;
    std::string parda_histogram_filepath;

    for (unsigned kid = 0; kid < (*SM_traces_all_passes).size(); kid++) {
      if ((unsigned)KERNEL_EVALUATION != kid)
        continue;

      unsigned miss_num_all_acc = 0;
      unsigned num_all_acc = 0;

      tim = 0;
      pdt_c = parda_init();

      std::vector<std::vector<unsigned long long>> L1_miss_instns;

      unsigned LDG_requests = 0;
      unsigned LDG_transactions = 0;
      unsigned STG_requests = 0;
      unsigned STG_transactions = 0;

      unsigned Global_atomic_requests = 0;
      unsigned Global_reduction_requests = 0;
      unsigned Global_atomic_and_reduction_transactions = 0;

      unsigned L2_read_transactions = 0;
      unsigned L2_write_transactions = 0;
      unsigned L2_total_transactions = 0;

      for (auto mem_ins : (*SM_traces_all_passes)[kid][curr_process_idx]) {

        std::map<unsigned, bool> distance_overflow_flag_vector;
        std::vector<unsigned long long> have_got_line_addr;

        L1_miss_instns.push_back(std::vector<unsigned long long>());

        if (mem_ins.has_mem_instn_type() == LDG ||
            mem_ins.has_mem_instn_type() == STG)

          for (unsigned j = 0; j < (mem_ins.addr).size(); j++) {
            unsigned long long cache_line_addr =
                mem_ins.addr[j] >> int(log2(l1_cache_line_size));

            if (std::find(have_got_line_addr.begin(), have_got_line_addr.end(),
                          cache_line_addr) != have_got_line_addr.end()) {
              mem_ins.distance[j] = 0;
              mem_ins.miss[j] = false;

              distance_overflow_flag_vector[mem_ins.addr[j]] = false;

            } else {

              sprintf(input, "0x%llx", cache_line_addr);

              mem_ins.distance[j] =
                  process_one_access_and_get_distance(input, &pdt_c, tim);
              if (curr_process_idx == 0 && kid == 0)
                if ((cache_line_addr & 3) == 0)
                  ;

              if (mem_ins.distance[j] > (int)l1_cache_blocks) {

                miss_num_all_acc++;
                mem_ins.miss[j] = true;
                L1_miss_instns.back().push_back(mem_ins.addr[j]);

                distance_overflow_flag_vector[mem_ins.addr[j]] = true;

                if (mem_ins.has_mem_instn_type() == LDG) {
                  L2_read_transactions += 1;
                  L2_total_transactions += 1;
                }
                if (mem_ins.has_mem_instn_type() == STG) {
                  L2_write_transactions += 1;
                  L2_total_transactions += 1;
                }
              } else {
                L1_miss_instns.back().push_back(mem_ins.addr[j]);
                distance_overflow_flag_vector[mem_ins.addr[j]] = false;
              }

              num_all_acc++;

              mem_instn_distance_overflow_flag->insert(std::make_pair(
                  std::make_tuple(kid, curr_process_idx, mem_ins.pc),
                  distance_overflow_flag_vector));

              tim++;
              have_got_line_addr.push_back(cache_line_addr);
            }
          }

        if (mem_ins.has_mem_instn_type() == LDG) {
          LDG_requests++;
          LDG_transactions += have_got_line_addr.size();
        }
        if (mem_ins.has_mem_instn_type() == STG) {
          STG_requests++;
          STG_transactions += have_got_line_addr.size();
        }
        if (mem_ins.has_mem_instn_type() == ATOM) {
          Global_atomic_requests++;
          Global_atomic_and_reduction_transactions += have_got_line_addr.size();
        }
        if (mem_ins.has_mem_instn_type() == RED) {
          Global_reduction_requests++;
          Global_atomic_and_reduction_transactions += have_got_line_addr.size();
        }
      }

      stat_coll->set_L2_read_transactions(L2_read_transactions,
                                          curr_process_idx);
      stat_coll->set_L2_write_transactions(L2_write_transactions,
                                           curr_process_idx);
      stat_coll->set_L2_total_transactions(L2_total_transactions,
                                           curr_process_idx);

      stat_coll->set_GEMM_read_requests(LDG_requests, curr_process_idx);
      stat_coll->set_GEMM_write_requests(STG_requests, curr_process_idx);
      stat_coll->set_GEMM_total_requests(LDG_requests + STG_requests,
                                         curr_process_idx);
      stat_coll->set_GEMM_read_transactions(LDG_transactions, curr_process_idx);
      stat_coll->set_GEMM_write_transactions(STG_transactions,
                                             curr_process_idx);
      stat_coll->set_GEMM_total_transactions(
          LDG_transactions + STG_transactions, curr_process_idx);
      stat_coll->set_Number_of_read_transactions_per_read_requests(
          (float)((float)LDG_transactions / (float)LDG_requests),
          curr_process_idx);
      stat_coll->set_Number_of_write_transactions_per_write_requests(
          (float)((float)STG_transactions / (float)STG_requests),
          curr_process_idx);

      stat_coll->set_Total_number_of_global_atomic_requests(
          Global_atomic_requests, curr_process_idx);
      stat_coll->set_Total_number_of_global_reduction_requests(
          Global_reduction_requests, curr_process_idx);
      stat_coll->set_Global_memory_atomic_and_reduction_transactions(
          Global_atomic_and_reduction_transactions, curr_process_idx);

      pdt = &pdt_c;
      pdt->histogram[B_INF] += narray_get_len(pdt->ga);

      if (dump_histogram) {
        if (configs_dir.back() == '/')
          parda_histogram_filepath =
              configs_dir + "../kernel_" + std::to_string(kid) + "_SM_" +
              std::to_string(curr_process_idx) + ".histogram";
        else
          parda_histogram_filepath =
              configs_dir + "/" + "../kernel_" + std::to_string(kid) + "_SM_" +
              std::to_string(curr_process_idx) + ".histogram";

        file = fopen(parda_histogram_filepath.c_str(), "w");

        if (file != NULL) {
          L1_hit_rate = parda_fprintf_histogram_r(pdt->histogram, file, true);
          fclose(file);
        } else {
          L1_hit_rate = parda_fprintf_histogram_r(pdt->histogram, NULL, false);
        }
      } else {
        L1_hit_rate = parda_fprintf_histogram_r(pdt->histogram, NULL, false);
      }

      stat_coll->set_Unified_L1_cache_hit_rate(L1_hit_rate, curr_process_idx);
      stat_coll->set_Unified_L1_cache_requests(num_all_acc, curr_process_idx);

      parda_free(pdt);
    }
  }

  float L2_hit_rate = 0.0;

  if (world.rank() == 0) {
    unsigned DRAM_total_transactions = 0;

    for (unsigned kid = 0; kid < (*SM_traces_all_passes).size(); kid++) {
      if ((unsigned)KERNEL_EVALUATION != kid)
        continue;
      unsigned max_instn_size = 0;
      for (unsigned sm_id = 0; sm_id < (unsigned)(hw_cfg->get_num_sms());
           sm_id++) {

        if ((*SM_traces_all_passes)[kid][sm_id].size() > max_instn_size) {
          max_instn_size = (*SM_traces_all_passes)[kid][sm_id].size();
        }
      }

      HKEY input;
      long tim;
      program_data_t pdt_c;

      tim = 0;
      pdt_c = parda_init();

      unsigned l2_miss_num_all_acc = 0;
      unsigned l2_num_all_acc = 0;

      for (unsigned instn_index = 0; instn_index < max_instn_size;
           instn_index++) {
        for (unsigned sm_id = 0; sm_id < (unsigned)(hw_cfg->get_num_sms());
             sm_id++) {

          if (instn_index < (*SM_traces_all_passes)[kid][sm_id].size()) {
            auto mem_ins = (*SM_traces_all_passes)[kid][sm_id][instn_index];

            if (!((*SM_traces_all_passes)[kid][sm_id][instn_index]
                          .has_mem_instn_type() == LDG ||
                  (*SM_traces_all_passes)[kid][sm_id][instn_index]
                          .has_mem_instn_type() == STG))
              continue;

            std::vector<unsigned long long> have_got_line_addr;
            std::vector<unsigned long long> L1_have_got_line_addr;

            for (unsigned j = 0; j < (mem_ins.addr).size(); j++) {

              unsigned long long L1_cache_line_addr =
                  mem_ins.addr[j] >> int(log2(l1_cache_line_size));
              unsigned long long cache_line_addr =
                  mem_ins.addr[j] >> int(log2(l2_cache_line_size));

              if (std::find(L1_have_got_line_addr.begin(),
                            L1_have_got_line_addr.end(), L1_cache_line_addr) !=
                  L1_have_got_line_addr.end()) {
                ;
              } else {
                if (std::find(have_got_line_addr.begin(),
                              have_got_line_addr.end(),
                              cache_line_addr) != have_got_line_addr.end()) {
                  l2_num_all_acc++;
                } else {
                  sprintf(input, "0x%llx", cache_line_addr);
                  mem_ins.distance_L2[j] =
                      process_one_access_and_get_distance(input, &pdt_c, tim);

                  if (mem_ins.distance_L2[j] > (int)l2_cache_blocks) {
                    l2_miss_num_all_acc++;
                    if (mem_ins.has_mem_instn_type() == LDG ||
                        mem_ins.has_mem_instn_type() == STG) {
                      DRAM_total_transactions++;
                    }
                  }
                  l2_num_all_acc++;
                  tim++;
                  have_got_line_addr.push_back(cache_line_addr);
                }
              }
            }
          }
        }
      }

      L2_hit_rate =
          (float)(((float)l2_num_all_acc - (float)l2_miss_num_all_acc) /
                  (float)l2_num_all_acc);

      program_data_t *pdt = &pdt_c;
      pdt->histogram[B_INF] += narray_get_len(pdt->ga);
      L2_hit_rate = parda_fprintf_histogram_r(pdt->histogram, NULL, false);

      stat_coll->set_L2_cache_hit_rate(L2_hit_rate);
      stat_coll->set_L2_cache_requests(l2_num_all_acc);

      stat_coll->set_DRAM_total_transactions(DRAM_total_transactions);

      parda_free(pdt);
    }
  }

  boost::mpi::broadcast(world, L2_hit_rate, 0);

  world.barrier();

  unsigned dram_mem_access = hw_cfg->get_dram_mem_access_latency();
  unsigned l1_cache_access = hw_cfg->get_l1_access_latency();
  unsigned l2_cache_access = hw_cfg->get_l2_access_latency();

  unsigned l1_cache_access_latency = l1_cache_access;
  unsigned l2_cache_access_latency = l2_cache_access;
  unsigned l2_cache_from_l1_access_latency =
      l2_cache_access_latency - l1_cache_access_latency;
  unsigned dram_mem_access_latency = dram_mem_access;
  unsigned l2_cache_from_dram_access_latency = dram_mem_access_latency -
                                               l2_cache_access_latency -
                                               l1_cache_access_latency;

  for (int pass = 0; pass < pass_num; pass++) {
    unsigned curr_process_idx_rank = world.rank() + pass * world.size();
    unsigned curr_process_idx;
    if (curr_process_idx_rank < hw_cfg->get_num_sms()) {
      curr_process_idx = curr_process_idx_rank;
    } else
      continue;

    float _L1_hit_rate =
        stat_coll->get_Unified_L1_cache_hit_rate(curr_process_idx);

    (*MEM_ACCESS_LATENCY)[curr_process_idx] =
        _L1_hit_rate * l1_cache_access_latency +
        (1 - _L1_hit_rate) *
            (L2_hit_rate * l2_cache_from_l1_access_latency +
             (1 - L2_hit_rate) * l2_cache_from_dram_access_latency);
  }
}

#endif

int main(int argc, char **argv) {

  /** Usage of this simulator:
   *    mpirun -np [num of processes] 
   *    ./gpu-simulator.x 
   *    --configs /path/to/application/configs 
   *    --kernel_id [kernel id you want to evalueate] 
   *    --config_file ./DEV-Def/QV100.config 
  */

#ifdef USE_BOOST
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
#endif

  CLI::App app{"GPU SIMULATOR."};

  std::string configs;
  bool sort = false;
  bool dump_histogram = false;

  unsigned KERNEL_EVALUATION = 0;

  std::string hw_config_file = "./DEV-Def/QV100.config";

  app.add_option("--configs", configs,
                 "The configs path, which is generated from our NVBit tool, "
                 "e.g., \"./traces/vectoradd/configs\"");
  // app.add_option("--sort", sort,
  //                "Simulate the order in which instructions are issued based
  //                on their " "timestamps");
  // app.add_option("--dump_histogram", dump_histogram,
  //                "Dump the histogram of the private L1 cache hit "
  //                "rate");
  app.add_option("--config_file", hw_config_file,
                 "The config file, e.g., \"../DEV-Def/QV100.config\"");
  app.add_option("--kernel_id", KERNEL_EVALUATION,
                 "The kernel id that you want to simulate");

  CLI11_PARSE(app, argc, argv);

  int passnum_concurrent_issue_to_sm = 1;

  hw_config hw_cfg(hw_config_file);

  trace_parser tracer(configs.c_str(), &hw_cfg);

  tracer.parse_configs_file(false);

  std::vector<int> need_to_read_mem_instns_sms;

  const int pass_num =
      int((hw_cfg.get_num_sms() + world.size() - 1) / world.size());
  for (int _pass = 0; _pass < pass_num; _pass++) {
    unsigned curr_process_idx_rank = world.rank() + _pass * world.size();

    unsigned curr_process_idx = curr_process_idx_rank;
    if (curr_process_idx < hw_cfg.get_num_sms())
      need_to_read_mem_instns_sms.push_back(curr_process_idx);
  }

  // statistics collector 
  stat_collector stat_coll(&hw_cfg, KERNEL_EVALUATION);

  std::vector<std::pair<int, int>> need_to_read_mem_instns_kernel_block_pair;

  for (auto sm_id : need_to_read_mem_instns_sms) {

    std::vector<std::pair<int, int>> result;
    if (world.rank() == 0)
      result = tracer.get_issuecfg()->get_kernel_block_by_smid_0(sm_id);
    else
      result = tracer.get_issuecfg()->get_kernel_block_by_smid(sm_id);
    for (auto pair : result) {

      bool flag = false;
      for (auto x : need_to_read_mem_instns_kernel_block_pair) {
        if ((x.first == pair.first && x.second == pair.second)) {
          flag = true;
          break;
        }
      }

      if ((unsigned)(pair.first) != (KERNEL_EVALUATION + 1))
        flag = true;

      if (!flag)
        need_to_read_mem_instns_kernel_block_pair.push_back(pair);
    }
  }

  tracer.read_mem_instns(false, &need_to_read_mem_instns_kernel_block_pair);

  auto issuecfg = tracer.get_issuecfg();

  app_config *appcfg = tracer.get_appcfg();

  stat_coll.set_total_num_workloads(
      appcfg->get_kernel_grid_dim_x((int)KERNEL_EVALUATION) *
      appcfg->get_kernel_grid_dim_y((int)KERNEL_EVALUATION) *
      appcfg->get_kernel_grid_dim_z((int)KERNEL_EVALUATION));

  stat_coll.set_active_SMs(
      std::min(stat_coll.get_m_num_sm(), stat_coll.get_active_SMs()));

  stat_coll.set_allocated_active_warps_per_block(
      (unsigned)(ceil(appcfg->get_kernel_block_size((int)KERNEL_EVALUATION) /
                          stat_coll.get_warp_size(),
                      1)));

  if (stat_coll.get_allocated_active_warps_per_block() == 0)
    stat_coll.set_allocated_active_warps_per_block(1);

  stat_coll.set_Thread_block_limit_warps(std::min(
      stat_coll.get_max_active_blocks_per_SM(),
      (unsigned)floor(stat_coll.get_max_active_threads_per_SM() /
                          stat_coll.get_warp_size() /
                          stat_coll.get_allocated_active_warps_per_block(),
                      1)));

  if (appcfg->get_kernel_num_registers((int)KERNEL_EVALUATION) == 0) {
    stat_coll.set_Thread_block_limit_registers(
        stat_coll.get_max_active_blocks_per_SM());
  } else {

    unsigned allocated_regs_per_warp = (unsigned)(ceil(
        appcfg->get_kernel_num_registers((int)KERNEL_EVALUATION) *
            stat_coll.get_warp_size(),
        stat_coll.get_register_allocation_size()));

    unsigned allocated_regs_per_SM = (unsigned)(floor(
        stat_coll.get_max_registers_per_block() / allocated_regs_per_warp,
        hw_cfg.get_num_sched_per_sm()));

    stat_coll.set_Thread_block_limit_registers(
        floor(allocated_regs_per_SM /
                  stat_coll.get_allocated_active_warps_per_block(),
              1) *
        floor(stat_coll.get_max_registers_per_SM() /
                  stat_coll.get_max_registers_per_block(),
              1));
  }

  if (appcfg->get_kernel_shared_mem_bytes((int)KERNEL_EVALUATION) == 0) {

    stat_coll.set_Thread_block_limit_shared_memory(
        stat_coll.get_max_active_blocks_per_SM());
  } else {

    float smem_per_block =
        ceil(appcfg->get_kernel_shared_mem_bytes((int)KERNEL_EVALUATION),
             stat_coll.get_smem_allocation_size());

    stat_coll.set_Thread_block_limit_shared_memory(
        floor(stat_coll.get_shared_mem_size() / smem_per_block, 1));
  }

  stat_coll.set_allocated_active_blocks_per_SM(
      std::min(std::min(stat_coll.get_Thread_block_limit_warps(),
                        stat_coll.get_Thread_block_limit_registers()),
               stat_coll.get_Thread_block_limit_shared_memory()));
  unsigned th_active_blocks = stat_coll.get_allocated_active_blocks_per_SM();

  stat_coll.set_Theoretical_max_active_warps_per_SM(
      th_active_blocks * stat_coll.get_allocated_active_warps_per_block());

  stat_coll.set_Theoretical_occupancy((unsigned)(ceil(
      (float)stat_coll.get_Theoretical_max_active_warps_per_SM() /
          (float)(stat_coll.get_max_active_threads_per_SM() /
                  stat_coll.get_warp_size()) *
          100.,
      1)));

  passnum_concurrent_issue_to_sm = int(
      (tracer.get_appcfg()->get_kernels_num() +
       (gpgpu_concurrent_kernel_sm ? hw_cfg.get_max_concurent_kernel()
                                   : 1) -
       1) /
      (gpgpu_concurrent_kernel_sm ? hw_cfg.get_max_concurent_kernel()
                                  : 1));

  std::vector<std::map<int, std::vector<mem_instn>>> SM_traces_all_passes;

  SM_traces_all_passes.resize(passnum_concurrent_issue_to_sm);

  for (int pass = 0; pass < passnum_concurrent_issue_to_sm; pass++) {
    if (pass != (int)KERNEL_EVALUATION)
      continue;

    std::vector<trace_kernel_info_t *> single_pass_kernels_info;

    if (pass == passnum_concurrent_issue_to_sm - 1) {
      single_pass_kernels_info.reserve(
          gpgpu_concurrent_kernel_sm
              ? tracer.get_appcfg()->get_kernels_num() -
                    hw_cfg.get_max_concurent_kernel() * pass
              : 1);
    } else if (pass == 0) {
      single_pass_kernels_info.reserve(
          gpgpu_concurrent_kernel_sm
              ? std::min(tracer.get_appcfg()->get_kernels_num(),
                         hw_cfg.get_max_concurent_kernel())
              : 1);
    } else {
      single_pass_kernels_info.reserve(
          gpgpu_concurrent_kernel_sm
              ? hw_cfg.get_max_concurent_kernel()
              : 1);
    }

    unsigned start_kernel_id =
        pass * (gpgpu_concurrent_kernel_sm
                    ? hw_cfg.get_max_concurent_kernel()
                    : 1);
    unsigned end_kernel_id =
        (pass + 1) * (gpgpu_concurrent_kernel_sm
                          ? hw_cfg.get_max_concurent_kernel()
                          : 1) -
        1;

    for (unsigned kid = start_kernel_id;
         kid <=
         std::min(end_kernel_id, tracer.get_appcfg()->get_kernels_num() - 1);
         kid++) {
      kernel_trace_t *kernel_trace_info = tracer.parse_kernel_info(kid, false);
      trace_kernel_info_t *kernel_info =
          create_kernel_info(kernel_trace_info, &tracer);
      single_pass_kernels_info.push_back(kernel_info);
    }

    std::map<int, std::vector<mem_instn>> *SM_traces =
        &SM_traces_all_passes[pass];

    int pass_num =
        int((hw_cfg.get_num_sms() + world.size() - 1) / world.size());

    if (world.rank() == 0)
      pass_num = hw_cfg.get_num_sms();

    for (int _pass = 0; _pass < pass_num; _pass++) {

      unsigned curr_process_idx_rank = world.rank() + _pass * world.size();

      unsigned curr_process_idx = curr_process_idx_rank;
      if (world.rank() == 0)
        curr_process_idx = _pass;
      if (curr_process_idx < hw_cfg.get_num_sms())
        for (auto k : single_pass_kernels_info) {

          unsigned num_threadblocks_current_kernel =
              k->get_trace_info()->grid_dim_x *
              k->get_trace_info()->grid_dim_y * k->get_trace_info()->grid_dim_z;

          std::vector<std::vector<mem_instn>> threadblock_traces;
          threadblock_traces.resize(num_threadblocks_current_kernel);

          unsigned kernel_id = k->get_trace_info()->kernel_id - 1;

          for (unsigned i = 0; i < num_threadblocks_current_kernel; i++) {

            unsigned sm_id = issuecfg->get_sm_id_of_one_block_fast(
                unsigned(kernel_id + 1), unsigned(i));
            if (sm_id == curr_process_idx) {

              threadblock_traces[i] = k->get_one_kernel_one_threadblock_traces(
                  k->get_trace_info()->kernel_id - 1, i);

              (*SM_traces)[sm_id].insert((*SM_traces)[sm_id].end(),
                                         threadblock_traces[i].begin(),
                                         threadblock_traces[i].end());
            }
          }
        }
    }

    for (auto iter : (*SM_traces)) {
      if (sort)
        std::sort(iter.second.begin(), iter.second.end(), compare_stamp);
    }

    for (auto k : single_pass_kernels_info) {
      delete k;
    }
    single_pass_kernels_info.clear();
  }

#ifdef USE_BOOST

  for (int i = 0; i < passnum_concurrent_issue_to_sm; i++) {
  }

  std::map<std::tuple<int, int, unsigned long long>, std::map<unsigned, bool>>
      mem_instn_distance_overflow_flag;

  std::vector<unsigned> MEM_ACCESS_LATENCY;
  MEM_ACCESS_LATENCY.resize(hw_cfg.get_num_sms());

  auto start_memory_timer = std::chrono::system_clock::now();
  private_L1_cache_stack_distance_evaluate_boost_no_concurrent(
      argc, argv, &SM_traces_all_passes, &mem_instn_distance_overflow_flag,
      false, configs, dump_histogram, &stat_coll, &hw_cfg, KERNEL_EVALUATION,
      &MEM_ACCESS_LATENCY);
  auto end_memory_timer = std::chrono::system_clock::now();
  auto duration_memory_timer =
      std::chrono::duration_cast<std::chrono::microseconds>(end_memory_timer -
                                                            start_memory_timer);
  auto cost_memory_timer =
      (double)(double(duration_memory_timer.count()) *
               (double)(std::chrono::microseconds::period::num) /
               (double)(std::chrono::microseconds::period::den));
  stat_coll.set_Simulation_time_memory_model(cost_memory_timer, world.rank());

#endif

  if ((unsigned)(tracer.get_the_least_sm_id_of_all_blocks() % world.size()) ==
      (unsigned)world.rank())
    tracer.read_compute_instns(false,
                               &need_to_read_mem_instns_kernel_block_pair);

  auto start_compute_timer = std::chrono::system_clock::now();

  for (int _pass = 0; _pass < pass_num; _pass++) {
    int curr_process_idx_rank = world.rank() + _pass * world.size();

    unsigned smid = curr_process_idx_rank;

#ifdef ENABLE_SAMPLING_POINT
    if (smid == (unsigned)tracer.get_appcfg()->get_kernel_sampling_point(KERNEL_EVALUATION)) {
      std::cout << "Tracer's Sampling Point: SM-" << smid << " ..." << std::endl;
#else
    if (smid == tracer.get_the_least_sm_id_of_all_blocks()) {
      std::cout << "Default Sampling Point: SM-" << smid << " ..." << std::endl;
#endif
      std::time_t now = time(0);
      char *dt = ctime(&now);
      std::cout << "\nCurrent Time: " << dt << std::endl;

      std::cout << "Simulator # Rank-" << world.rank() << ", processing SM-"
                << smid << std::endl;
      PrivateSM private_sm = PrivateSM(smid, &tracer, &hw_cfg);

      unsigned thread_blocks_num_in_this_sm = 0;

      for (auto pair : *(private_sm.get_blocks_per_kernel())) {
        unsigned kid = pair.first;
        std::vector<unsigned> block_ids = pair.second;

        if (kid - 1 == (unsigned)KERNEL_EVALUATION) {
          // for (unsigned block_id : block_ids) {
          //   thread_blocks_num_in_this_sm++;
          // }
          for (unsigned i = 0; i < block_ids.size(); i++)
            thread_blocks_num_in_this_sm++;
        }
      }
      std::cout << " ...run START... " << std::endl;

      while (private_sm.get_active()) {
        private_sm.run(KERNEL_EVALUATION, MEM_ACCESS_LATENCY[smid], &stat_coll);
      }

      std::vector<std::pair<int, int>> *kernel_block_pair =
          private_sm.get_kernel_block_pair();
      for (auto it_kernel_block_pair = kernel_block_pair->begin();
           it_kernel_block_pair != kernel_block_pair->end();
           it_kernel_block_pair++) {
        if ((unsigned)(it_kernel_block_pair->first) - 1 != KERNEL_EVALUATION)
          continue;
        unsigned _kid = it_kernel_block_pair->first - 1;
        unsigned _block_id = it_kernel_block_pair->second;
        unsigned _warps_per_block = appcfg->get_num_warp_per_block(_kid);
        unsigned _gwarp_id_start = _warps_per_block * _block_id;
        unsigned _gwarp_id_end = _gwarp_id_start + _warps_per_block - 1;
        for (auto gwid = _gwarp_id_start; gwid <= _gwarp_id_end; gwid++) {
          unsigned one_warp_instn_size =
              tracer.get_one_kernel_one_warp_instn_size(_kid, gwid);
          for (unsigned i = 0; i < one_warp_instn_size; i++) {

            compute_instn *tmp =
                tracer.get_one_kernel_one_warp_one_instn(_kid, gwid, i);
            _inst_trace_t *tmp_inst_trace = tmp->inst_trace;
            auto fu = tmp_inst_trace->get_func_unit();

            unsigned latency_from_issue_to_writeback = 0;
            if (private_sm.get_clk_record<4>(_kid, gwid, i) == 0 &&
                private_sm.get_clk_record<5>(_kid, gwid, i) == 0)
              continue;
            else {
              if (private_sm.get_clk_record<5>(_kid, gwid, i) == 0) {
                latency_from_issue_to_writeback =
                    private_sm.get_clk_record<4>(_kid, gwid, i) -
                    private_sm.get_clk_record<0>(_kid, gwid, i);
              } else {
                latency_from_issue_to_writeback =
                    private_sm.get_clk_record<5>(_kid, gwid, i) -
                    private_sm.get_clk_record<0>(_kid, gwid, i);
              }

              if (latency_from_issue_to_writeback <= 0) {
                latency_from_issue_to_writeback =
                    private_sm.get_clk_record<3>(_kid, gwid, i) -
                    private_sm.get_clk_record<0>(_kid, gwid, i);
              }
            }
            switch (fu) {
            case NON_UNIT:
              stat_coll.increment_Other_UNIT_execute_clks_sum(
                  smid, latency_from_issue_to_writeback);
              stat_coll.increment_Other_UNIT_Instns_num(smid);
              break;
            case SP_UNIT:
              stat_coll.increment_SP_UNIT_execute_clks_sum(
                  smid, latency_from_issue_to_writeback);
              stat_coll.increment_SP_UNIT_Instns_num(smid);
              break;
            case SFU_UNIT:
              stat_coll.increment_SFU_UNIT_execute_clks_sum(
                  smid, latency_from_issue_to_writeback);
              stat_coll.increment_SFU_UNIT_Instns_num(smid);
              break;
            case INT_UNIT:
              stat_coll.increment_INT_UNIT_execute_clks_sum(
                  smid, latency_from_issue_to_writeback);
              stat_coll.increment_INT_UNIT_Instns_num(smid);
              break;
            case DP_UNIT:
              stat_coll.increment_DP_UNIT_execute_clks_sum(
                  smid, latency_from_issue_to_writeback);
              stat_coll.increment_DP_UNIT_Instns_num(smid);
              break;
            case TENSOR_CORE_UNIT:
              stat_coll.increment_TENSOR_CORE_UNIT_execute_clks_sum(
                  smid, latency_from_issue_to_writeback);
              stat_coll.increment_TENSOR_CORE_UNIT_Instns_num(smid);
              break;
            case LDST_UNIT:
              stat_coll.increment_LDST_UNIT_execute_clks_sum(
                  smid, latency_from_issue_to_writeback);
              stat_coll.increment_LDST_UNIT_Instns_num(smid);
              break;
            case SPEC_UNIT_1:
              stat_coll.increment_SPEC_UNIT_1_execute_clks_sum(
                  smid, latency_from_issue_to_writeback);
              stat_coll.increment_SPEC_UNIT_1_Instns_num(smid);
              break;
            case SPEC_UNIT_2:
              stat_coll.increment_SPEC_UNIT_2_execute_clks_sum(
                  smid, latency_from_issue_to_writeback);
              stat_coll.increment_SPEC_UNIT_2_Instns_num(smid);
              break;
            case SPEC_UNIT_3:
              stat_coll.increment_SPEC_UNIT_3_execute_clks_sum(
                  smid, latency_from_issue_to_writeback);
              stat_coll.increment_SPEC_UNIT_3_Instns_num(smid);
              break;
            default:
              stat_coll.increment_Other_UNIT_execute_clks_sum(
                  smid, latency_from_issue_to_writeback);
              stat_coll.increment_Other_UNIT_Instns_num(smid);
              break;
            }
          }
        }
      }

      float achieved_occupancy =
          (float)private_sm.get_active_warps_id_size_sum() /
          (float)private_sm.get_cycle() /
          (float)stat_coll.get_Theoretical_occupancy();

      stat_coll.set_Achieved_occupancy(achieved_occupancy, smid);

      float achieved_active_warps_per_SM =
          (float)((float)achieved_occupancy) *
          (float)stat_coll.get_Theoretical_max_active_warps_per_SM();

      stat_coll.set_Achieved_active_warps_per_SM(achieved_active_warps_per_SM,
                                                 smid);

      unsigned thread_block_launch_cycles = 700;
      unsigned kernel_launch_cycles = 3500;
      private_sm.set_cycle(private_sm.get_cycle() +
                           thread_blocks_num_in_this_sm *
                               thread_block_launch_cycles / th_active_blocks);
      private_sm.set_cycle(private_sm.get_cycle() + kernel_launch_cycles);

      std::cout << " ...run EXIT... " << std::endl;
      stat_coll.set_GPU_active_cycles(private_sm.get_cycle(), smid);
      stat_coll.set_SM_active_cycles(private_sm.get_cycle(), smid);
      stat_coll.set_Warp_instructions_executed(
          private_sm.get_num_warp_instns_executed(), smid);
      stat_coll.set_Instructions_executed_per_clock_cycle_IPC(
          (float)private_sm.get_num_warp_instns_executed() /
              (float)private_sm.get_cycle(),
          smid);
      stat_coll.set_Total_instructions_executed_per_seconds(
          (float)((float)private_sm.get_num_warp_instns_executed() / 1e6) /
              (float)((float)private_sm.get_cycle() /
                      (float)hw_cfg.get_core_clock_mhz()),
          smid);
      stat_coll.set_Kernel_execution_time(
          (float)((float)private_sm.get_cycle() /
                  (float)hw_cfg.get_core_clock_mhz() * 1e9),
          smid);

      unsigned dram_mem_access = hw_cfg.get_dram_mem_access_latency();
      unsigned l1_cache_access = hw_cfg.get_l1_access_latency();
      unsigned l2_cache_access = hw_cfg.get_l2_access_latency();

      stat_coll.increment_num_Execute_Memory_Data_L1(
          stat_coll.get_GEMM_total_requests(smid) *
              stat_coll.get_Unified_L1_cache_hit_rate(smid) * l1_cache_access,
          smid);
      stat_coll.increment_num_Execute_Memory_Data_L2(
          stat_coll.get_GEMM_total_requests(smid) *
              (1.0 - stat_coll.get_Unified_L1_cache_hit_rate(smid)) *
              l2_cache_access,
          smid);
      stat_coll.increment_num_Execute_Memory_Data_Main_Memory(
          stat_coll.get_GEMM_total_requests(smid) *
              (1.0 - stat_coll.get_Unified_L1_cache_hit_rate(smid)) *
              (1.0 - stat_coll.get_L2_cache_hit_rate()) * dram_mem_access,
          smid);
    }
  }

  auto end_compute_timer = std::chrono::system_clock::now();
  auto duration_compute_timer =
      std::chrono::duration_cast<std::chrono::microseconds>(
          end_compute_timer - start_compute_timer);
  auto cost_compute_timer =
      (double)(double(duration_compute_timer.count()) *
               (double)(std::chrono::microseconds::period::num) /
               (double)(std::chrono::microseconds::period::den));

  stat_coll.set_Simulation_time_compute_model(cost_compute_timer, world.rank());

  world.barrier();
  if (world.rank() == 0)
    stat_coll.dump_output(configs, world.rank());
  world.barrier();
  if (world.rank() != 0)
    stat_coll.dump_output(configs, world.rank());

  fflush(stdout);

  return 0;
}
