#include <assert.h>
#include <bitset>
#include <list>
#include <map>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include "../trace-driven/register-set.h"
#include "../trace-driven/trace-warp-inst.h"
#include "../trace-parser/trace-parser.h"
#include "RegisterBankAllocator.h"

#ifndef OPERAND_COLLECTOR_H
#define OPERAND_COLLECTOR_H

enum collector_unit_type_t {
  SP_CUS,
  DP_CUS,
  SFU_CUS,
  TENSOR_CORE_CUS,
  INT_CUS,
  MEM_CUS,
  GEN_CUS,
};

unsigned register_bank_opcoll(unsigned regnum, unsigned wid, unsigned num_banks,
                              bool bank_warp_shift, bool sub_core_model,
                              unsigned banks_per_sched, unsigned sched_id);

class opndcoll_rfu_t {
public:
  opndcoll_rfu_t() {
    m_num_banks = 0;

    m_initialized = false;
  }

  opndcoll_rfu_t(hw_config *hw_cfg, RegisterBankAllocator *reg_bank_allocator,
                 trace_parser *tracer) {
    m_num_banks = 0;

    m_initialized = false;
    m_reg_bank_allocator = reg_bank_allocator;

    m_hw_cfg = hw_cfg;
    m_tracer = tracer;
  }

  void add_cu_set(unsigned cu_set, unsigned num_cu, unsigned num_dispatch);
  typedef std::vector<register_set *> port_vector_t;
  typedef std::vector<unsigned int> uint_vector_t;
  void add_port(port_vector_t &input, port_vector_t &ouput,
                uint_vector_t cu_sets);
  void init(hw_config *hw_cfg, RegisterBankAllocator *reg_bank_allocator,
            trace_parser *tracer);

  void step(
      bool *
          flag_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
      bool *
          flag_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated,
      bool *
          flag_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated,
      bool *
          flag_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
      trace_parser *tracer,
      std::map<std::tuple<unsigned, unsigned, unsigned>,
               std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned,
                          unsigned>> *clk_record,
      unsigned cycle) {
    dispatch_ready_cu(clk_record, cycle);
    allocate_reads(
        flag_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
        flag_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated,
        flag_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated,
        flag_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
        tracer);
    for (unsigned p = 0; p < m_in_ports.size(); p++)
      allocate_cu(
          p,
          flag_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
          flag_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated,
          flag_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated,
          flag_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
          tracer);
    process_banks();
  }

private:
  void process_banks() { m_arbiter.reset_alloction(); }

  void
  dispatch_ready_cu(std::map<std::tuple<unsigned, unsigned, unsigned>,
                             std::tuple<unsigned, unsigned, unsigned, unsigned,
                                        unsigned, unsigned>> *clk_record,
                    unsigned cycle);
  void allocate_cu(
      unsigned port,
      bool *
          flag_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
      bool *
          flag_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated,
      bool *
          flag_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated,
      bool *
          flag_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
      trace_parser *tracer);
  void allocate_reads(
      bool *
          flag_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
      bool *
          flag_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated,
      bool *
          flag_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated,
      bool *
          flag_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
      trace_parser *tracer);

  class collector_unit_t;

  class op_t {
  public:
    op_t() { m_valid = false; }
    op_t(collector_unit_t *cu, unsigned op, unsigned reg, unsigned num_banks,
         unsigned bank_warp_shift, bool sub_core_model,
         unsigned banks_per_sched, unsigned sched_id, trace_parser *tracer) {
      m_valid = true;
      m_warp = NULL;
      m_cu = cu;
      m_operand = op;
      m_register = reg;
      m_shced_id = sched_id;
      m_tracer = tracer;
      m_bank = register_bank_opcoll(reg, cu->get_warp_id(), num_banks,
                                    bank_warp_shift, sub_core_model,
                                    banks_per_sched, sched_id);
    }
    op_t(collector_unit_t *cu, unsigned op, unsigned reg, unsigned num_banks,
         unsigned bank_warp_shift, bool sub_core_model,
         unsigned banks_per_sched, unsigned sched_id, trace_parser *tracer,
         unsigned kid, unsigned wid, unsigned uid) {
      m_valid = true;
      m_warp = NULL;
      m_cu = cu;
      m_operand = op;
      m_register = reg;
      m_shced_id = sched_id;
      m_tracer = tracer;
      m_bank = register_bank_opcoll(reg, cu->get_warp_id(), num_banks,
                                    bank_warp_shift, sub_core_model,
                                    banks_per_sched, sched_id);
      m_kid = kid;
      m_wid = wid;
      m_uid = uid;
    }
    op_t(const inst_fetch_buffer_entry *warp, unsigned reg, unsigned num_banks,
         unsigned bank_warp_shift, bool sub_core_model,
         unsigned banks_per_sched, unsigned sched_id, trace_parser *tracer) {
      m_valid = true;
      m_warp = warp;
      m_register = reg;
      m_cu = NULL;
      m_operand = -1;
      m_shced_id = sched_id;
      m_tracer = tracer;
      m_bank =
          register_bank_opcoll(reg, m_warp->wid, num_banks, bank_warp_shift,
                               sub_core_model, banks_per_sched, sched_id);
    }

    bool valid() const { return m_valid; }
    unsigned get_reg() const {
      assert(m_valid);
      return m_register;
    }
    unsigned get_wid() const {
      if (m_warp)
        return m_warp->wid;
      else if (m_cu)
        return m_cu->get_warp_id();
      else
        abort();
    }
    unsigned get_kid() const {
      if (m_warp)
        return m_warp->kid;
      else
        abort();
    }
    unsigned get_uid() const {
      if (m_warp)
        return m_warp->kid;
      else
        abort();
    }

    unsigned get_instn_kid() { return m_kid; }
    unsigned get_instn_wid() { return m_wid; }
    unsigned get_instn_uid() { return m_uid; }

    unsigned get_sid() const { return m_shced_id; }
    unsigned get_active_count() const {
      if (m_warp) {
        compute_instn *tmp = m_tracer->get_one_kernel_one_warp_one_instn(
            m_warp->kid, m_warp->wid, m_warp->uid);
        trace_warp_inst_t *tmp_trace_warp_inst = &(tmp->trace_warp_inst);
        return tmp_trace_warp_inst->get_activate_count();
      } else if (m_cu)
        return m_cu->get_active_count();
      else
        abort();
    }
    const active_mask_t &get_active_mask() {
      if (m_warp) {
        compute_instn *tmp = m_tracer->get_one_kernel_one_warp_one_instn(
            m_warp->kid, m_warp->wid, m_warp->uid);
        trace_warp_inst_t *tmp_trace_warp_inst = &(tmp->trace_warp_inst);
        return tmp_trace_warp_inst->get_active_mask_ref();
      } else if (m_cu)
        return m_cu->get_active_mask();
      else
        abort();
    }
    active_mask_t get_active_mask_1() {
      if (m_warp) {
        compute_instn *tmp = m_tracer->get_one_kernel_one_warp_one_instn(
            m_warp->kid, m_warp->wid, m_warp->uid);
        trace_warp_inst_t *tmp_trace_warp_inst = &(tmp->trace_warp_inst);
        return tmp_trace_warp_inst->get_active_mask();
      } else
        abort();
    }

    unsigned get_oc_id() const { return m_cu->get_id(); }
    unsigned get_bank() const { return m_bank; }
    unsigned get_operand() const { return m_operand; }
    void dump(FILE *fp) const {
      if (m_cu)
        fprintf(fp, " <R%u, CU:%u, w:%02u> ", m_register, m_cu->get_id(),
                m_cu->get_warp_id());
      else if (!m_warp->m_valid)
        fprintf(fp, " <R%u, wid:%02u> ", m_register, m_warp->wid);
    }
    std::string get_reg_string() const {
      char buffer[64];
      snprintf(buffer, 64, "R%u", m_register);
      return std::string(buffer);
    }

    void reset() { m_valid = false; }

  private:
    bool m_valid;
    collector_unit_t *m_cu;
    const inst_fetch_buffer_entry *m_warp;
    unsigned m_operand;

    unsigned m_register;
    unsigned m_bank;
    unsigned m_shced_id;

    trace_parser *m_tracer;

    unsigned m_kid = 65536;
    unsigned m_wid = 65536;
    unsigned m_uid = 65536;
  };

  class allocation_t {
  public:
    allocation_t() { m_allocation = FREE; }
    allocation_t(RegisterBankAllocator *reg_bank_allocator, unsigned bank_id) {
      m_reg_bank_allocator = reg_bank_allocator;
      m_allocation = m_reg_bank_allocator->getBankState(bank_id);
    }
    bool is_read() const { return m_allocation == ON_READING; }
    bool is_write() const { return m_allocation == ON_WRITING; }
    bool is_free() const { return m_allocation == FREE; }
    void dump(FILE *fp) const {
      if (m_allocation == FREE) {
        fprintf(fp, "<free>");
      } else if (m_allocation == ON_READING) {
        fprintf(fp, "rd: ");
        m_op.dump(fp);
      } else if (m_allocation == ON_WRITING) {
        fprintf(fp, "wr: ");
        m_op.dump(fp);
      }
      fprintf(fp, "\n");
    }
    void alloc_read(const op_t &op) {
      assert(is_free());
      m_reg_bank_allocator->setBankState(op.get_bank(), ON_READING);

      m_op = op;
    }
    void alloc_write(const op_t &op) {
      assert(is_free());
      m_reg_bank_allocator->setBankState(op.get_bank(), ON_WRITING);

      m_op = op;
    }
    void reset() { m_reg_bank_allocator->releaseBankState(m_op.get_bank()); }

  private:
    enum Register_Bank_State m_allocation;
    op_t m_op;
    RegisterBankAllocator *m_reg_bank_allocator;
  };

  class arbiter_t {
  public:
    arbiter_t() {
      m_queue = NULL;
      m_allocated_bank = NULL;
      m_allocator_rr_head = NULL;
      _inmatch = NULL;
      _outmatch = NULL;
      _request = NULL;
      m_last_cu = 0;
    }
    arbiter_t(RegisterBankAllocator *reg_bank_allocator) {
      m_queue = NULL;
      m_allocated_bank = NULL;
      m_allocator_rr_head = NULL;
      _inmatch = NULL;
      _outmatch = NULL;
      _request = NULL;
      m_last_cu = 0;
      m_reg_bank_allocator = reg_bank_allocator;
    }

    ~arbiter_t() {
      if (m_queue) {
        delete[] m_queue;
        m_queue = nullptr;
      }
      if (m_allocated_bank) {
        delete[] m_allocated_bank;
        m_allocated_bank = nullptr;
      }
      if (m_allocator_rr_head) {
        delete[] m_allocator_rr_head;
        m_allocator_rr_head = nullptr;
      }
      if (_inmatch) {
        delete[] _inmatch;
        _inmatch = nullptr;
      }
      if (_outmatch) {
        delete[] _outmatch;
        _outmatch = nullptr;
      }
      if (_request) {
        for (unsigned i = 0; i < m_num_banks; i++)
          delete[] _request[i];
        delete[] _request;
        _request = nullptr;
      }
    }
    void init(unsigned num_cu, unsigned num_banks) {
      assert(num_cu > 0);
      assert(num_banks > 0);
      m_num_collectors = num_cu;
      m_num_banks = num_banks;
      _inmatch = new int[m_num_banks];
      _outmatch = new int[m_num_collectors];

      _request = new int *[m_num_banks];
      for (unsigned i = 0; i < m_num_banks; i++)
        _request[i] = new int[m_num_collectors];

      m_queue = new std::list<op_t>[num_banks];

      m_allocated_bank = new allocation_t[num_banks];
      for (unsigned i = 0; i < num_banks; ++i) {
        m_allocated_bank[i] = allocation_t(m_reg_bank_allocator, i);
      }
      m_allocator_rr_head = new unsigned[num_cu];
      for (unsigned n = 0; n < num_cu; n++)
        m_allocator_rr_head[n] = n % num_banks;
      reset_alloction();
    }

    void dump(FILE *fp) const {
      fprintf(fp, "\n");
      fprintf(fp, "  Arbiter State:\n");
      fprintf(fp, "  requests:\n");
      for (unsigned b = 0; b < m_num_banks; b++) {
        fprintf(fp, "    bank %u : ", b);
        std::list<op_t>::const_iterator o = m_queue[b].begin();
        for (; o != m_queue[b].end(); o++) {
          o->dump(fp);
        }
        fprintf(fp, "\n");
      }
      fprintf(fp, "  grants:\n");
      for (unsigned b = 0; b < m_num_banks; b++) {
        fprintf(fp, "    bank %u : ", b);
        m_allocated_bank[b].dump(fp);
      }
      fprintf(fp, "\n");
    }

    std::list<op_t> allocate_reads(
        bool *
            flag_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
        bool *
            flag_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated,
        bool *
            flag_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated,
        bool *
            flag_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu,
        trace_parser *tracer);

    void add_read_requests(collector_unit_t *cu) {
      const op_t *src = cu->get_operands();

      for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
        const op_t &op = src[i];
        if (op.valid()) {
          unsigned bank = op.get_bank();
          m_queue[bank].push_back(op);
        }
      }
    }
    bool bank_idle(unsigned bank) const {
      return m_allocated_bank[bank].is_free();
    }
    void allocate_bank_for_write(unsigned bank, const op_t &op) {
      assert(bank < m_num_banks);
      m_allocated_bank[bank].alloc_write(op);
    }
    void allocate_for_read(unsigned bank, const op_t &op) {
      assert(bank < m_num_banks);
      m_allocated_bank[bank].alloc_read(op);
    }
    void reset_alloction() {
      for (unsigned b = 0; b < m_num_banks; b++)
        m_allocated_bank[b].reset();
    }

  private:
    unsigned m_num_banks;
    unsigned m_num_collectors;

    allocation_t *m_allocated_bank;
    std::list<op_t> *m_queue;

    unsigned *m_allocator_rr_head;
    unsigned m_last_cu;

    int *_inmatch;
    int *_outmatch;
    int **_request;

    RegisterBankAllocator *m_reg_bank_allocator;
  };

public:
  class input_port_t {
  public:
    input_port_t(port_vector_t &input, port_vector_t &output,
                 uint_vector_t cu_sets)
        : m_in(input), m_out(output), m_cu_sets(cu_sets) {
      assert(input.size() == output.size());
      assert(not m_cu_sets.empty());
    }

    port_vector_t m_in, m_out;

    uint_vector_t m_cu_sets;
  };

private:
  class collector_unit_t {
  public:
    collector_unit_t() {
      m_free = true;
      m_warp = NULL;
      m_output_register = NULL;
      m_src_op = new op_t[MAX_REG_OPERANDS * 2];
      m_not_ready.reset();
      m_warp_id = -1;
      m_num_banks = 0;
      m_bank_warp_shift = 0;
    }
    collector_unit_t(const hw_config *hw_cfg, trace_parser *tracer) {
      m_free = true;
      m_warp = NULL;
      m_output_register = NULL;
      m_src_op = new op_t[MAX_REG_OPERANDS * 2];
      m_not_ready.reset();
      m_warp_id = -1;
      m_num_banks = 0;
      m_bank_warp_shift = 0;

      m_tracer = tracer;
      m_hw_cfg = hw_cfg;
    }
    ~collector_unit_t() {
      for (unsigned i = 0; i < MAX_REG_OPERANDS * 2; i++) {
        m_src_op[i] = op_t();
      }
      if (m_src_op) {

        delete[] m_src_op;
        m_src_op = nullptr;
      }
      if (m_warp) {
        delete m_warp;
        m_warp = nullptr;
      }
    }

    bool ready() const;
    const op_t *get_operands() const { return m_src_op; }

    unsigned get_warp_id() const { return m_warp_id; }
    unsigned get_active_count() {
      if (m_warp) {
        compute_instn *tmp = m_tracer->get_one_kernel_one_warp_one_instn(
            m_warp->kid, m_warp->wid, m_warp->uid);
        trace_warp_inst_t *tmp_trace_warp_inst = &(tmp->trace_warp_inst);
        return tmp_trace_warp_inst->get_activate_count();
      } else
        abort();
    }
    const active_mask_t &get_active_mask() {
      if (m_warp) {
        compute_instn *tmp = m_tracer->get_one_kernel_one_warp_one_instn(
            m_warp->kid, m_warp->wid, m_warp->uid);
        trace_warp_inst_t *tmp_trace_warp_inst = &(tmp->trace_warp_inst);
        return tmp_trace_warp_inst->get_active_mask_ref();
      } else
        abort();
    }
    unsigned get_sp_op() {
      if (m_warp) {
        compute_instn *tmp = m_tracer->get_one_kernel_one_warp_one_instn(
            m_warp->kid, m_warp->wid, m_warp->uid);
        trace_warp_inst_t *tmp_trace_warp_inst = &(tmp->trace_warp_inst);
        return (unsigned)(tmp_trace_warp_inst->get_sp_op());
      } else
        abort();
    }
    unsigned get_id() const { return m_cuid; }
    unsigned get_reg_id() const { return m_reg_id; }

    void init(unsigned n, unsigned num_banks, unsigned log2_warp_size,
              const hw_config *config, opndcoll_rfu_t *rfu, bool sub_core_model,
              unsigned reg_id, unsigned num_banks_per_sched);
    bool allocate(register_set *pipeline_reg, register_set *output_reg);

    std::bitset<MAX_REG_OPERANDS * 2> &get_not_ready() { return m_not_ready; }

    void collect_operand(unsigned op) { m_not_ready.reset(op); }
    unsigned get_num_operands() {
      if (m_warp) {
        compute_instn *tmp = m_tracer->get_one_kernel_one_warp_one_instn(
            m_warp->kid, m_warp->wid, m_warp->uid);
        trace_warp_inst_t *tmp_trace_warp_inst = &(tmp->trace_warp_inst);
        return tmp_trace_warp_inst->get_num_operands();
      } else
        abort();
    }
    unsigned get_num_regs() {
      if (m_warp) {
        compute_instn *tmp = m_tracer->get_one_kernel_one_warp_one_instn(
            m_warp->kid, m_warp->wid, m_warp->uid);
        trace_warp_inst_t *tmp_trace_warp_inst = &(tmp->trace_warp_inst);
        return tmp_trace_warp_inst->get_num_regs();
      } else
        abort();
    }
    void dispatch(std::map<std::tuple<unsigned, unsigned, unsigned>,
                           std::tuple<unsigned, unsigned, unsigned, unsigned,
                                      unsigned, unsigned>> *clk_record,
                  unsigned cycle);
    bool is_free() { return m_free; }

    template <unsigned pos>
    void
    set_clk_record(std::map<std::tuple<unsigned, unsigned, unsigned>,
                            std::tuple<unsigned, unsigned, unsigned, unsigned,
                                       unsigned, unsigned>> *clk_record,
                   unsigned kid, unsigned wid, unsigned uid, unsigned value) {
      const std::tuple<unsigned, unsigned, unsigned> key(kid, wid, uid);
      auto it = clk_record->find(key);

      if (it == clk_record->end()) {
        std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned, unsigned>
            new_value(0, 0, 0, 0, 0, 0);
        std::get<pos>(new_value) = value;
        (*clk_record)[key] = new_value;
      } else {
        std::get<pos>(it->second) = value;
      }
    }

  private:
    bool m_free;
    unsigned m_cuid;
    unsigned m_warp_id;
    inst_fetch_buffer_entry *m_warp;
    register_set *m_output_register;
    op_t *m_src_op;
    std::bitset<MAX_REG_OPERANDS * 2> m_not_ready;
    unsigned m_num_banks;
    unsigned m_bank_warp_shift;
    opndcoll_rfu_t *m_rfu;

    unsigned m_num_banks_per_sched;
    bool m_sub_core_model;
    unsigned m_reg_id;

    const hw_config *m_hw_cfg;
    trace_parser *m_tracer;
  };

  class dispatch_unit_t {
  public:
    dispatch_unit_t(std::vector<collector_unit_t> *cus) {
      m_last_cu = 0;
      m_collector_units = cus;
      m_num_collectors = (*cus).size();
      m_next_cu = 0;
    }
    void init(bool sub_core_model, unsigned num_warp_scheds) {
      m_sub_core_model = sub_core_model;
      m_num_warp_scheds = num_warp_scheds;
    }

    collector_unit_t *find_ready() {

      unsigned cusPerSched = m_num_collectors / m_num_warp_scheds;
      unsigned rr_increment =
          m_sub_core_model ? cusPerSched - (m_last_cu % cusPerSched) : 1;
      for (unsigned n = 0; n < m_num_collectors; n++) {
        unsigned c = (m_last_cu + n + rr_increment) % m_num_collectors;

        if ((*m_collector_units)[c].ready()) {
          m_last_cu = c;
          return &((*m_collector_units)[c]);
        }
      }
      return NULL;
    }

  private:
    unsigned m_num_collectors;
    std::vector<collector_unit_t> *m_collector_units;
    unsigned m_last_cu;
    unsigned m_next_cu;
    bool m_sub_core_model;
    unsigned m_num_warp_scheds;
  };

  bool m_initialized;

  unsigned m_num_collector_sets;

  unsigned m_num_banks;
  unsigned m_bank_warp_shift;
  unsigned m_warp_size;
  std::vector<collector_unit_t *> m_cu;
  arbiter_t m_arbiter;

  unsigned m_num_banks_per_sched;
  unsigned m_num_warp_scheds;
  bool sub_core_model;

  std::vector<input_port_t> m_in_ports;
  typedef std::map<unsigned, std::vector<collector_unit_t>> cu_sets_t;
  cu_sets_t m_cus;
  std::vector<dispatch_unit_t> m_dispatch_units;

  hw_config *m_hw_cfg;
  RegisterBankAllocator *m_reg_bank_allocator;
  trace_parser *m_tracer;

public:
  std::vector<input_port_t> *get_m_in_ports() { return &m_in_ports; }
};

#endif