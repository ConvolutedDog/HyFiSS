#include "OperandCollector.h"

unsigned register_bank_opcoll(unsigned regnum, unsigned wid, unsigned num_banks,
                              bool bank_warp_shift, bool sub_core_model,
                              unsigned banks_per_sched, unsigned sched_id) {
  unsigned bank = regnum;

  if (bank_warp_shift)
    bank += wid;

  if (sub_core_model) {
    unsigned bank_num = (bank % banks_per_sched) + (sched_id * banks_per_sched);
    assert(bank_num < num_banks);
    return bank_num;
  } else
    return bank % num_banks;
}

void opndcoll_rfu_t::add_cu_set(unsigned set_id, unsigned num_cu,
                                unsigned num_dispatch) {
  m_cus[set_id].reserve(num_cu);

  for (unsigned i = 0; i < num_cu; i++) {

    m_cus[set_id].emplace_back(m_hw_cfg, m_tracer);
    m_cu.push_back(&m_cus[set_id].back());
  }

  for (unsigned i = 0; i < num_dispatch; i++) {
    m_dispatch_units.push_back(dispatch_unit_t(&m_cus[set_id]));
  }
}

void opndcoll_rfu_t::add_port(port_vector_t &input, port_vector_t &output,
                              const uint_vector_t cu_sets) {

  m_in_ports.push_back(input_port_t(input, output, cu_sets));
}

void opndcoll_rfu_t::init(hw_config *hw_cfg,
                          RegisterBankAllocator *reg_bank_allocator,
                          trace_parser *tracer) {

  unsigned num_banks = m_hw_cfg->get_num_reg_banks();

  m_arbiter = arbiter_t(m_reg_bank_allocator);

  m_arbiter.init(m_cu.size(), num_banks);

  m_num_banks = num_banks;
  m_bank_warp_shift = 0;
  m_warp_size = m_hw_cfg->get_warp_size();
  m_bank_warp_shift = (unsigned)(int)(log(m_warp_size + 0.5) / log(2.0));
  assert((m_bank_warp_shift == 5) || (m_warp_size != 32));

  sub_core_model = m_hw_cfg->get_sub_core_model();
  m_num_warp_scheds = m_hw_cfg->get_num_sched_per_sm();
  unsigned reg_id = -1;
  if (sub_core_model) {
    assert(num_banks % m_num_warp_scheds == 0);
    assert(m_num_warp_scheds <= m_cu.size() &&
           m_cu.size() % m_num_warp_scheds == 0);
  }
  m_num_banks_per_sched = num_banks / m_num_warp_scheds;

  for (unsigned j = 0; j < m_cu.size(); j++) {
    if (sub_core_model) {
      unsigned cusPerSched = m_cu.size() / m_num_warp_scheds;
      reg_id = j / cusPerSched;
    }
    m_cu[j]->init(j, num_banks, m_bank_warp_shift, m_hw_cfg, this,
                  sub_core_model, reg_id, m_num_banks_per_sched);
  }

  for (unsigned j = 0; j < m_dispatch_units.size(); j++) {
    m_dispatch_units[j].init(sub_core_model, m_num_warp_scheds);
  }

  m_initialized = true;
}

void opndcoll_rfu_t::dispatch_ready_cu(
    std::map<std::tuple<unsigned, unsigned, unsigned>,
             std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned,
                        unsigned>> *clk_record,
    unsigned cycle) {

  for (unsigned p = 0; p < m_dispatch_units.size(); ++p) {
    dispatch_unit_t &du = m_dispatch_units[p];
    collector_unit_t *cu = du.find_ready();
    if (cu) {
      cu->dispatch(clk_record, cycle);

    } else {
    }
  }
}

void opndcoll_rfu_t::allocate_cu(
    unsigned port_num,
    bool *
        flag_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
    bool *
        flag_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated,
    bool
        *flag_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated,
    bool *
        flag_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
    trace_parser *tracer) {
  input_port_t &inp = m_in_ports[port_num];
  for (unsigned i = 0; i < inp.m_in.size(); i++) {

    if ((*inp.m_in[i]).has_ready()) {
      bool have_found_free_cu = false;
      for (unsigned j = 0; j < inp.m_cu_sets.size(); j++) {
        std::vector<collector_unit_t> &cu_set = m_cus[inp.m_cu_sets[j]];
        bool allocated = false;
        unsigned cuLowerBound = 0;
        unsigned cuUpperBound = cu_set.size();
        unsigned schd_id;
        unsigned reg_id = 0;
        if (sub_core_model) {

          reg_id = (*inp.m_in[i]).get_ready_reg_id();
          schd_id = (*inp.m_in[i]).get_schd_id(reg_id);
          assert(cu_set.size() % m_num_warp_scheds == 0 &&
                 cu_set.size() >= m_num_warp_scheds);
          unsigned cusPerSched = cu_set.size() / m_num_warp_scheds;
          cuLowerBound = schd_id * cusPerSched;
          cuUpperBound = cuLowerBound + cusPerSched;
          assert(0 <= cuLowerBound && cuUpperBound <= cu_set.size());
        }
        for (unsigned k = cuLowerBound; k < cuUpperBound; k++) {
          if (cu_set[k].is_free()) {
            have_found_free_cu = true;

            collector_unit_t *cu = &cu_set[k];
            allocated = cu->allocate(inp.m_in[i], inp.m_out[i]);

            m_arbiter.add_read_requests(cu);
            break;
          }
        }
        if (allocated)
          break;
        else {
          if (j == inp.m_cu_sets.size() - 1 && !have_found_free_cu) {
            unsigned kid = (*inp.m_in[i]).get_kid(reg_id);
            unsigned wid = (*inp.m_in[i]).get_wid(reg_id);
            unsigned uid = (*inp.m_in[i]).get_uid(reg_id);
            auto _compute_instn =
                tracer->get_one_kernel_one_warp_one_instn(kid, wid, uid);
            _inst_trace_t *tmp_inst_trace = _compute_instn->inst_trace;
            auto fu = tmp_inst_trace->get_func_unit();
            if (fu == LDST_UNIT) {
              *flag_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu =
                  true;
            } else {
              *flag_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu =
                  true;
            }
          }
        }
      }
    }
  }
}

void opndcoll_rfu_t::allocate_reads(
    bool *
        flag_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
    bool *
        flag_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated,
    bool
        *flag_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated,
    bool *
        flag_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
    trace_parser *tracer) {

  std::list<op_t> allocated = m_arbiter.allocate_reads(
      flag_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
      flag_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated,
      flag_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated,
      flag_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
      tracer);
  std::map<unsigned, op_t> read_ops;
  for (std::list<op_t>::iterator r = allocated.begin(); r != allocated.end();
       r++) {
    const op_t &rr = *r;
    unsigned reg = rr.get_reg();
    unsigned wid = rr.get_wid();
    unsigned bank = register_bank_opcoll(reg, wid, m_num_banks,
                                         m_bank_warp_shift, sub_core_model,
                                         m_num_banks_per_sched, rr.get_sid());

    m_arbiter.allocate_for_read(bank, rr);
    read_ops[bank] = rr;
  }
  std::map<unsigned, op_t>::iterator r;
  for (r = read_ops.begin(); r != read_ops.end(); ++r) {
    op_t &op = r->second;
    unsigned cu = op.get_oc_id();
    unsigned operand = op.get_operand();
    m_cu[cu]->collect_operand(operand);
  }
}

bool opndcoll_rfu_t::collector_unit_t::ready() const {

  if (m_output_register != NULL)
    return (!m_free) && m_not_ready.none() &&
           (*m_output_register).has_free(m_sub_core_model, m_reg_id);
  else
    return false;
}

void opndcoll_rfu_t::collector_unit_t::init(
    unsigned n, unsigned num_banks, unsigned log2_warp_size,
    const hw_config *config, opndcoll_rfu_t *rfu, bool sub_core_model,
    unsigned reg_id, unsigned num_banks_per_sched) {
  m_rfu = rfu;
  m_cuid = n;
  m_num_banks = num_banks;
  assert(m_warp == NULL);
  m_warp = new inst_fetch_buffer_entry();
  m_bank_warp_shift = log2_warp_size;
  m_sub_core_model = sub_core_model;
  m_reg_id = reg_id;
  m_num_banks_per_sched = num_banks_per_sched;
  m_hw_cfg = config;
}

bool opndcoll_rfu_t::collector_unit_t::allocate(register_set *pipeline_reg_set,
                                                register_set *output_reg_set) {
  assert(m_free);
  assert(m_not_ready.none());
  m_free = false;
  m_output_register = output_reg_set;
  inst_fetch_buffer_entry **pipeline_reg = pipeline_reg_set->get_ready();

  if ((pipeline_reg) and ((*pipeline_reg)->m_valid)) {
    m_warp_id = (*pipeline_reg)->wid;
    compute_instn *tmp = m_tracer->get_one_kernel_one_warp_one_instn(
        (*pipeline_reg)->kid, (*pipeline_reg)->wid, (*pipeline_reg)->uid);

    _inst_trace_t *tmp_inst_trace = tmp->inst_trace;
    std::vector<int> prev_regs;

    for (unsigned op = 0; op < tmp_inst_trace->reg_srcs_num; op++) {
      int reg_num = tmp_inst_trace->reg_src[op];

      bool new_reg = true;
      for (auto r : prev_regs) {
        if (r == reg_num)
          new_reg = false;
      }
      if (reg_num >= 0 && new_reg) {
        prev_regs.push_back(reg_num);

        auto sched_id =
            (unsigned)(m_warp_id % m_hw_cfg->get_num_sched_per_sm());

        m_src_op[op] = op_t(this, op, reg_num, m_num_banks, m_bank_warp_shift,
                            m_sub_core_model, m_num_banks_per_sched, sched_id,
                            m_tracer, (*pipeline_reg)->kid,
                            (*pipeline_reg)->wid, (*pipeline_reg)->uid);
        m_not_ready.set(op);
      } else
        m_src_op[op] = op_t();
    }

    pipeline_reg_set->move_out_to(m_warp);

    return true;
  }
  return false;
}

void opndcoll_rfu_t::collector_unit_t::dispatch(
    std::map<std::tuple<unsigned, unsigned, unsigned>,
             std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned,
                        unsigned>> *clk_record,
    unsigned cycle) {
  assert(m_not_ready.none());

  if (_CALIBRATION_LOG_) {
    std::cout << "    Read Operands: (" << m_warp->kid << ", " << m_warp->wid
              << ", " << m_warp->uid << ", " << m_warp->pc << ")" << std::endl;
  }

  set_clk_record<3>(clk_record, m_warp->kid, m_warp->wid, m_warp->uid, cycle);
  m_output_register->move_in(m_sub_core_model, m_reg_id, m_warp);

  m_warp->m_valid = false;

  m_free = true;

  for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++)
    m_src_op[i].reset();
}

std::list<opndcoll_rfu_t::op_t> opndcoll_rfu_t::arbiter_t::allocate_reads(
    bool *
        flag_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
    bool *
        flag_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated,
    bool
        *flag_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated,
    bool *
        flag_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
    trace_parser *tracer) {
  std::list<op_t> result;

  int input;
  int output;
  int _inputs = m_num_banks;
  int _outputs = m_num_collectors;
  int _square = (_inputs > _outputs) ? _inputs : _outputs;
  assert(_square > 0);
  int _pri = (int)m_last_cu;

  for (int i = 0; i < _inputs; ++i)
    _inmatch[i] = -1;
  for (int j = 0; j < _outputs; ++j)
    _outmatch[j] = -1;

  for (unsigned i = 0; i < m_num_banks; i++) {
    for (unsigned j = 0; j < m_num_collectors; j++) {
      assert(i < (unsigned)_inputs);
      assert(j < (unsigned)_outputs);

      _request[i][j] = 0;
    }

    if (!m_queue[i].empty()) {
      const op_t &op = m_queue[i].front();
      int oc_id = op.get_oc_id();
      assert(i < (unsigned)_inputs);
      assert(oc_id < _outputs);
      _request[i][oc_id] = 1;
    }
    if (m_allocated_bank[i].is_write()) {

      assert(i < (unsigned)_inputs);
      _inmatch[i] = 0;
    }
  }

  for (int p = 0; p < _square; ++p) {
    output = (_pri + p) % _outputs;

    for (input = 0; input < _inputs; ++input) {
      assert(input < _inputs);
      assert(output < _outputs);
      if ((output < _outputs) && (_inmatch[input] == -1) &&

          (_request[input][output])) {

        _inmatch[input] = output;
        _outmatch[output] = input;
      }

      output = (output + 1) % _outputs;
    }
  }

  _pri = (_pri + 1) % _outputs;

  m_last_cu = _pri;
  for (unsigned i = 0; i < m_num_banks; i++) {
    if (_inmatch[i] != -1) {
      if (!m_allocated_bank[i].is_write()) {
        unsigned bank = (unsigned)i;
        op_t &op = m_queue[bank].front();
        result.push_back(op);
        m_queue[bank].pop_front();
      }
    }
  }

  for (unsigned bank = 0; bank < m_num_banks; bank++) {
    std::list<op_t>::iterator iter;
    for (iter = m_queue[bank].begin(); iter != m_queue[bank].end(); iter++) {
      if (iter->valid()) {
        unsigned kid = iter->get_instn_kid();
        unsigned wid = iter->get_instn_wid();
        unsigned uid = iter->get_instn_uid();
        if (kid == 65536 || wid == 65536 || uid == 65536)
          continue;
        else {
          auto _compute_instn =
              tracer->get_one_kernel_one_warp_one_instn(kid, wid, uid);
          _inst_trace_t *tmp_inst_trace = _compute_instn->inst_trace;
          auto fu = tmp_inst_trace->get_func_unit();
          if (fu == LDST_UNIT) {
            *flag_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated =
                true;
          } else {
            *flag_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated =
                true;
          }
        }
      }
    }
  }

  return result;
}
