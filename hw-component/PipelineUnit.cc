#include "PipelineUnit.h"

pipelined_simd_unit::pipelined_simd_unit(register_set *result_port,
                                         unsigned max_latency,
                                         unsigned issue_reg_id,
                                         hw_config *hw_cfg,
                                         trace_parser *tracer) {
  m_pipeline_depth = max_latency;
  m_pipeline_reg.reserve(m_pipeline_depth);
  for (unsigned i = 0; i < m_pipeline_depth; i++) {
    m_pipeline_reg.push_back(new inst_fetch_buffer_entry());
  }
  m_result_port = result_port;
  m_issue_reg_id = issue_reg_id;
  m_dispatch_reg = new inst_fetch_buffer_entry();
  m_hw_cfg = hw_cfg;
  m_tracer = tracer;
  occupied.reset();
  active_insts_in_pipeline = 0;
}

pipelined_simd_unit::~pipelined_simd_unit() {
  for (unsigned i = 0; i < m_pipeline_depth; i++) {
    delete m_pipeline_reg[i];
  }
  delete m_dispatch_reg;
}

bool pipelined_simd_unit::can_issue(unsigned latency) const {

  return !m_dispatch_reg->m_valid && !occupied.test(latency);
}

std::vector<unsigned> pipelined_simd_unit::cycle(
    trace_parser *tracer, Scoreboard *m_scoreboard, app_config *appcfg,
    std::vector<std::pair<int, int>> *kernel_block_pair,
    std::vector<unsigned> *m_num_warps_per_sm, unsigned KERNEL_EVALUATION,
    unsigned num_scheds, regBankAlloc *m_reg_bank_allocator,
    bool *flag_Writeback_Memory_Structural_bank_of_reg_is_not_idle,
    std::map<std::tuple<unsigned, unsigned, unsigned>,
             std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned,
                        unsigned>> *clk_record,
    unsigned m_cycle) {

  std::vector<unsigned> need_to_return_wids;

  if (m_pipeline_reg[0]->m_valid && (m_result_port->get_free() != NULL)) {
    if (_CALIBRATION_LOG_) {
      std::cout << "    Execute: (" << m_pipeline_reg[0]->kid << ", "
                << m_pipeline_reg[0]->wid << ", " << m_pipeline_reg[0]->uid
                << ", " << m_pipeline_reg[0]->pc << ")" << std::endl;
    }
    set_clk_record<4>(clk_record, m_pipeline_reg[0]->kid,
                      m_pipeline_reg[0]->wid, m_pipeline_reg[0]->uid, m_cycle);

    if (m_result_port->get_free() != NULL) {
      need_to_return_wids.push_back(m_pipeline_reg[0]->wid);

      m_result_port->move_in(m_pipeline_reg[0]);
      assert(active_insts_in_pipeline > 0);
      active_insts_in_pipeline--;
    }
  }

  if (active_insts_in_pipeline) {
    for (unsigned stage = 0; stage < m_pipeline_depth - 1; stage++) {
      if (!m_pipeline_reg[stage]->m_valid) {
        need_to_return_wids.push_back(m_pipeline_reg[stage + 1]->wid);
        m_pipeline_reg[stage]->m_valid = m_pipeline_reg[stage + 1]->m_valid;
        m_pipeline_reg[stage]->pc = m_pipeline_reg[stage + 1]->pc;
        m_pipeline_reg[stage]->wid = m_pipeline_reg[stage + 1]->wid;
        m_pipeline_reg[stage]->kid = m_pipeline_reg[stage + 1]->kid;
        m_pipeline_reg[stage]->uid = m_pipeline_reg[stage + 1]->uid;
        m_pipeline_reg[stage]->latency = m_pipeline_reg[stage + 1]->latency;
        m_pipeline_reg[stage]->initial_interval =
            m_pipeline_reg[stage + 1]->initial_interval;
        m_pipeline_reg[stage + 1]->m_valid = false;
      }
    }
  }

  if (m_dispatch_reg->m_valid) {

    if (m_dispatch_reg->initial_interval_dec_counter == 1) {
      int start_stage =
          m_dispatch_reg->latency - m_dispatch_reg->initial_interval;
      if (start_stage < 0)
        start_stage = 0;
      if ((unsigned)start_stage >= m_pipeline_depth)
        start_stage = m_pipeline_depth - 1;

      assert(start_stage >= 0 && (unsigned)start_stage < m_pipeline_depth);

      if (m_pipeline_reg[start_stage]->m_valid == false) {
        need_to_return_wids.push_back(m_dispatch_reg->wid);

        m_dispatch_reg->m_valid = false;
        active_insts_in_pipeline++;
        m_pipeline_reg[start_stage]->m_valid = true;
        m_pipeline_reg[start_stage]->pc = m_dispatch_reg->pc;
        m_pipeline_reg[start_stage]->wid = m_dispatch_reg->wid;
        m_pipeline_reg[start_stage]->kid = m_dispatch_reg->kid;
        m_pipeline_reg[start_stage]->uid = m_dispatch_reg->uid;
        m_pipeline_reg[start_stage]->latency = m_dispatch_reg->latency;
        m_pipeline_reg[start_stage]->initial_interval =
            m_dispatch_reg->initial_interval;
      }
    } else {
      m_dispatch_reg->initial_interval_dec_counter--;
    }
  }

  occupied >>= 1;

  return need_to_return_wids;
}

void pipelined_simd_unit::issue(register_set &source_reg) {
  bool partition_issue =
      m_hw_cfg->get_sub_core_model() && this->is_issue_partitioned();
  inst_fetch_buffer_entry **ready_reg =
      source_reg.get_ready(partition_issue, m_issue_reg_id);

  if (ready_reg != NULL) {
    source_reg.move_out_to(partition_issue, this->get_issue_reg_id(),
                           m_dispatch_reg);
    occupied.set(m_dispatch_reg->latency);
  }
}

void pipelined_simd_unit::issue(register_set &source_reg, unsigned reg_id) {
  bool partition_issue =
      m_hw_cfg->get_sub_core_model() && this->is_issue_partitioned();
  inst_fetch_buffer_entry **ready_reg =
      source_reg.get_ready(partition_issue, reg_id);

  if (ready_reg != NULL) {
    source_reg.move_out_to(partition_issue, reg_id, m_dispatch_reg);
    occupied.set(m_dispatch_reg->latency);
  }
}

bool sfu::can_issue(const inst_fetch_buffer_entry &inst) const {
  unsigned _fetch_instn_id = inst.uid;
  unsigned _gwid = inst.wid;
  unsigned _kid = inst.kid;
  compute_instn *tmp =
      m_tracer->get_one_kernel_one_warp_one_instn(_kid, _gwid, _fetch_instn_id);
  _inst_trace_t *tmp_inst_trace = tmp->inst_trace;
  if (tmp_inst_trace->get_func_unit() == SFU_UNIT) {
    unsigned latency = tmp_inst_trace->get_latency();
    return pipelined_simd_unit::can_issue(latency);
  } else {
    return false;
  }
}

void sfu::issue(register_set &source_reg) {

  pipelined_simd_unit::issue(source_reg);
}

void sfu::issue(register_set &source_reg, unsigned reg_id) {
  pipelined_simd_unit::issue(source_reg, reg_id);
}

bool dp_unit::can_issue(const inst_fetch_buffer_entry &inst) const {
  unsigned _fetch_instn_id = inst.uid;
  unsigned _gwid = inst.wid;
  unsigned _kid = inst.kid;
  compute_instn *tmp =
      m_tracer->get_one_kernel_one_warp_one_instn(_kid, _gwid, _fetch_instn_id);
  _inst_trace_t *tmp_inst_trace = tmp->inst_trace;
  if (tmp_inst_trace->get_func_unit() == DP_UNIT) {
    unsigned latency = tmp_inst_trace->get_latency();
    return pipelined_simd_unit::can_issue(latency);
  } else {
    return false;
  }
}

void dp_unit::issue(register_set &source_reg) {

  pipelined_simd_unit::issue(source_reg);
}

void dp_unit::issue(register_set &source_reg, unsigned reg_id) {
  pipelined_simd_unit::issue(source_reg, reg_id);
}

bool sp_unit::can_issue(const inst_fetch_buffer_entry &inst) const {
  unsigned _fetch_instn_id = inst.uid;
  unsigned _gwid = inst.wid;
  unsigned _kid = inst.kid;
  compute_instn *tmp =
      m_tracer->get_one_kernel_one_warp_one_instn(_kid, _gwid, _fetch_instn_id);
  _inst_trace_t *tmp_inst_trace = tmp->inst_trace;
  if (tmp_inst_trace->get_func_unit() == SP_UNIT) {
    unsigned latency = tmp_inst_trace->get_latency();
    return pipelined_simd_unit::can_issue(latency);
  } else {
    return false;
  }
}

void sp_unit::issue(register_set &source_reg) {

  pipelined_simd_unit::issue(source_reg);
}

void sp_unit::issue(register_set &source_reg, unsigned reg_id) {
  pipelined_simd_unit::issue(source_reg, reg_id);
}

bool tensor_core::can_issue(const inst_fetch_buffer_entry &inst) const {
  unsigned _fetch_instn_id = inst.uid;
  unsigned _gwid = inst.wid;
  unsigned _kid = inst.kid;
  compute_instn *tmp =
      m_tracer->get_one_kernel_one_warp_one_instn(_kid, _gwid, _fetch_instn_id);
  _inst_trace_t *tmp_inst_trace = tmp->inst_trace;
  if (tmp_inst_trace->get_func_unit() == TENSOR_CORE_UNIT) {
    unsigned latency = tmp_inst_trace->get_latency();
    return pipelined_simd_unit::can_issue(latency);
  } else {
    return false;
  }
}

void tensor_core::issue(register_set &source_reg) {

  pipelined_simd_unit::issue(source_reg);
}

void tensor_core::issue(register_set &source_reg, unsigned reg_id) {
  pipelined_simd_unit::issue(source_reg, reg_id);
}

bool int_unit::can_issue(const inst_fetch_buffer_entry &inst) const {
  unsigned _fetch_instn_id = inst.uid;
  unsigned _gwid = inst.wid;
  unsigned _kid = inst.kid;
  compute_instn *tmp =
      m_tracer->get_one_kernel_one_warp_one_instn(_kid, _gwid, _fetch_instn_id);
  _inst_trace_t *tmp_inst_trace = tmp->inst_trace;
  if (tmp_inst_trace->get_func_unit() == INT_UNIT) {
    unsigned latency = tmp_inst_trace->get_latency();
    return pipelined_simd_unit::can_issue(latency);
  } else {
    return false;
  }
}

void int_unit::issue(register_set &source_reg) {

  pipelined_simd_unit::issue(source_reg);
}

void int_unit::issue(register_set &source_reg, unsigned reg_id) {
  pipelined_simd_unit::issue(source_reg, reg_id);
}

bool specialized_unit::can_issue(const inst_fetch_buffer_entry &inst) const {
  unsigned _fetch_instn_id = inst.uid;
  unsigned _gwid = inst.wid;
  unsigned _kid = inst.kid;
  compute_instn *tmp =
      m_tracer->get_one_kernel_one_warp_one_instn(_kid, _gwid, _fetch_instn_id);
  _inst_trace_t *tmp_inst_trace = tmp->inst_trace;

  bool op_condition = false;

  switch (m_index) {
  case 0:
    op_condition = (tmp_inst_trace->get_func_unit() == SPEC_UNIT_1);
    break;
  case 1:
    op_condition = (tmp_inst_trace->get_func_unit() == SPEC_UNIT_2);
    break;
  case 2:
    op_condition = (tmp_inst_trace->get_func_unit() == SPEC_UNIT_3);
    break;
  default:
    return false;
    break;
  }

  if (op_condition) {
    unsigned latency = tmp_inst_trace->get_latency();
    return pipelined_simd_unit::can_issue(latency);
  } else {
    return false;
  }
}

void specialized_unit::issue(register_set &source_reg) {

  pipelined_simd_unit::issue(source_reg);
}

void specialized_unit::issue(register_set &source_reg, unsigned reg_id) {
  pipelined_simd_unit::issue(source_reg, reg_id);
}

bool mem_unit::can_issue(const inst_fetch_buffer_entry &inst) const {
  unsigned _fetch_instn_id = inst.uid;
  unsigned _gwid = inst.wid;
  unsigned _kid = inst.kid;
  compute_instn *tmp =
      m_tracer->get_one_kernel_one_warp_one_instn(_kid, _gwid, _fetch_instn_id);
  _inst_trace_t *tmp_inst_trace = tmp->inst_trace;
  if (tmp_inst_trace->get_func_unit() == LDST_UNIT) {
    unsigned latency = tmp_inst_trace->get_latency();
    return pipelined_simd_unit::can_issue(latency);
  } else {
    return false;
  }
}

void mem_unit::issue(register_set &source_reg) {

  pipelined_simd_unit::issue(source_reg);
}

void mem_unit::issue(register_set &source_reg, unsigned reg_id) {
  pipelined_simd_unit::issue(source_reg, reg_id);
}

std::vector<unsigned> mem_unit::cycle(
    trace_parser *tracer, Scoreboard *m_scoreboard, app_config *appcfg,
    std::vector<std::pair<int, int>> *kernel_block_pair,
    std::vector<unsigned> *m_num_warps_per_sm, unsigned KERNEL_EVALUATION,
    unsigned num_scheds, regBankAlloc *m_reg_bank_allocator,
    bool *flag_Writeback_Memory_Structural_bank_of_reg_is_not_idle,
    std::map<std::tuple<unsigned, unsigned, unsigned>,
             std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned,
                        unsigned>> *clk_record,
    unsigned m_cycle) {
  std::vector<unsigned> need_to_return_wids;
  if (m_pipeline_reg[0]->m_valid) {
    if (_CALIBRATION_LOG_) {
      std::cout << "    Execute: (" << m_pipeline_reg[0]->kid << ", "
                << m_pipeline_reg[0]->wid << ", " << m_pipeline_reg[0]->uid
                << ", " << m_pipeline_reg[0]->pc << ")" << std::endl;
    }
    set_clk_record<4>(clk_record, m_pipeline_reg[0]->kid,
                      m_pipeline_reg[0]->wid, m_pipeline_reg[0]->uid, m_cycle);

    auto _compute_instn = tracer->get_one_kernel_one_warp_one_instn(
        m_pipeline_reg[0]->kid, m_pipeline_reg[0]->wid, m_pipeline_reg[0]->uid);
    auto _trace_warp_inst = _compute_instn->trace_warp_inst;
    std::vector<int> need_write_back_regs_num;

    unsigned _warps_per_block =
        appcfg->get_num_warp_per_block(m_pipeline_reg[0]->kid);
    unsigned dst_reg_num = _trace_warp_inst.get_outcount();
    for (unsigned i = 0; i < dst_reg_num; i++) {
      int dst_reg_id = _trace_warp_inst.get_arch_reg_dst(i);

      if (dst_reg_id >= 0) {
        auto local_wid = (unsigned)(m_pipeline_reg[0]->wid % _warps_per_block);
        auto sched_id = (unsigned)(local_wid % num_scheds);

        auto bank_id = m_reg_bank_allocator->register_bank(dst_reg_id,
                                                           local_wid, sched_id);

        if (m_reg_bank_allocator->getBankState(bank_id) == FREE) {

          m_reg_bank_allocator->setBankState(bank_id, ON_WRITING);

          _trace_warp_inst.set_arch_reg_dst(i, -1);

        } else {

          *flag_Writeback_Memory_Structural_bank_of_reg_is_not_idle = true;
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
      m_pipeline_reg[0]->m_valid = false;
      active_insts_in_pipeline--;

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

      unsigned _warps_per_block =
          appcfg->get_num_warp_per_block(m_pipeline_reg[0]->kid);

      unsigned _block_id =
          (unsigned)(m_pipeline_reg[0]->wid / _warps_per_block);
      unsigned _kid_block_id_count = 0;
      for (auto _it_kernel_block_pair = kernel_block_pair->begin();
           _it_kernel_block_pair != kernel_block_pair->end();
           _it_kernel_block_pair++) {
        if ((unsigned)(_it_kernel_block_pair->first) - 1 != KERNEL_EVALUATION)
          continue;
        if ((unsigned)(_it_kernel_block_pair->first) - 1 ==
            m_pipeline_reg[0]->kid) {
          if ((unsigned)(_it_kernel_block_pair->second) < _block_id) {
            _kid_block_id_count++;
          }
        }
      }

      auto global_all_kernels_warp_id =
          (unsigned)(m_pipeline_reg[0]->wid % _warps_per_block) +
          _kid_block_id_count * _warps_per_block +
          std::accumulate(m_num_warps_per_sm->begin(),
                          m_num_warps_per_sm->begin() + m_pipeline_reg[0]->kid,
                          0);

      for (auto regnum : need_write_back_regs_num) {

        m_scoreboard->releaseRegisters(global_all_kernels_warp_id, regnum);
      }
    }
  }

  if (active_insts_in_pipeline) {
    for (unsigned stage = 0; stage < m_pipeline_depth - 1; stage++) {
      if (!m_pipeline_reg[stage]->m_valid) {
        m_pipeline_reg[stage]->m_valid = m_pipeline_reg[stage + 1]->m_valid;
        m_pipeline_reg[stage]->pc = m_pipeline_reg[stage + 1]->pc;
        m_pipeline_reg[stage]->wid = m_pipeline_reg[stage + 1]->wid;
        m_pipeline_reg[stage]->kid = m_pipeline_reg[stage + 1]->kid;
        m_pipeline_reg[stage]->uid = m_pipeline_reg[stage + 1]->uid;
        m_pipeline_reg[stage]->latency = m_pipeline_reg[stage + 1]->latency;
        m_pipeline_reg[stage]->initial_interval =
            m_pipeline_reg[stage + 1]->initial_interval;
        m_pipeline_reg[stage + 1]->m_valid = false;
      }
    }
  }

  if (m_dispatch_reg->m_valid) {

    if (m_dispatch_reg->initial_interval_dec_counter == 1) {
      int start_stage =
          m_dispatch_reg->latency - m_dispatch_reg->initial_interval;
      if (start_stage < 0)
        start_stage = 0;

      assert(start_stage >= 0 && (unsigned)start_stage < m_pipeline_depth);

      if (m_pipeline_reg[start_stage]->m_valid == false) {
        m_dispatch_reg->m_valid = false;
        active_insts_in_pipeline++;
        m_pipeline_reg[start_stage]->m_valid = true;
        m_pipeline_reg[start_stage]->pc = m_dispatch_reg->pc;
        m_pipeline_reg[start_stage]->wid = m_dispatch_reg->wid;
        m_pipeline_reg[start_stage]->kid = m_dispatch_reg->kid;
        m_pipeline_reg[start_stage]->uid = m_dispatch_reg->uid;
        m_pipeline_reg[start_stage]->latency = m_dispatch_reg->latency;
        m_pipeline_reg[start_stage]->initial_interval =
            m_dispatch_reg->initial_interval;
      }
    } else {
      m_dispatch_reg->initial_interval_dec_counter--;
    }
  }

  occupied >>= 1;

  return need_to_return_wids;
}