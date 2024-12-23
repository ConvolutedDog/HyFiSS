#include <bitset>
#include <vector>

#include "../hw-component/RegBankAlloc.h"
#include "../hw-component/Scoreboard.h"
#include "../hw-parser/hw-parser.h"
#include "../trace-driven/register-set.h"
#include "../trace-parser/trace-parser.h"

#ifndef MAX_ALU_LATENCY
#define MAX_ALU_LATENCY 1024
#endif

#ifndef PRED_NUM_OFFSET
#define PRED_NUM_OFFSET 65536
#endif

#ifndef PIPELINE_UNIT_H
#define PIPELINE_UNIT_H

class pipelined_simd_unit {
public:
  pipelined_simd_unit(register_set *result_port,
                      const unsigned max_latency,
                      const unsigned issue_reg_id,
                      hw_config *hw_cfg,
                      trace_parser *tracer);
  virtual ~pipelined_simd_unit() = 0;

  virtual void issue(register_set &source_reg);
  virtual void issue(register_set &source_reg, unsigned reg_id);

  unsigned get_active_lanes_in_pipeline() { return 0; };

  virtual bool stallable() const { return false; }

  virtual bool can_issue(const unsigned latency) const;
  virtual bool is_issue_partitioned() = 0;

  inline unsigned get_issue_reg_id() const { return m_issue_reg_id; }
  void print() const {
    printf("%s dispatch= ", m_name.c_str());

    printf("m_dispatch_reg: (pc,wid,kid,uid) %u %u %u %u\n", 
      m_dispatch_reg->pc, m_dispatch_reg->wid, 
      m_dispatch_reg->kid, m_dispatch_reg->uid);

    for (int s = m_pipeline_depth - 1; s >= 0; s--) {
      if (m_pipeline_reg[s]->m_valid) {
        printf("      %s[%2d] ", m_name.c_str(), s);
        printf("m_dispatch_reg[%d]: (pc,wid,kid,uid) %u %u %u %u\n", s,
          m_dispatch_reg->pc, m_dispatch_reg->wid, 
          m_dispatch_reg->kid, m_dispatch_reg->uid);
      }
    }
  }

  virtual std::vector<unsigned>
  cycle(trace_parser *tracer, Scoreboard *m_scoreboard, app_config *appcfg,
        std::vector<std::pair<int, int>> *kernel_block_pair,
        std::vector<unsigned> *m_num_warps_per_sm, unsigned KERNEL_EVALUATION,
        unsigned num_scheds, regBankAlloc *m_reg_bank_allocator,
        bool *flag_Writeback_Memory_Structural_bank_of_reg_is_not_idle,
        std::map<std::tuple<unsigned, unsigned, unsigned>,
                 std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned,
                            unsigned>> *clk_record,
        unsigned cycle);

  virtual unsigned clock_multiplier() const { return 1; };

  const char *get_name() { return m_name.c_str(); }

  template <unsigned pos>
  void set_clk_record(std::map<std::tuple<unsigned, unsigned, unsigned>,
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

  inline void dump_m_pipeline_reg_front() const {
    if (CALIBRATION_LOG_ENABLED) {
      printf("    Execute: (%u, %u, %u, %u)\n", 
        m_pipeline_reg[0]->kid,
        m_pipeline_reg[0]->wid,
        m_pipeline_reg[0]->uid,
        m_pipeline_reg[0]->pc);
    }
  }

protected:
  unsigned m_pipeline_depth = MAX_ALU_LATENCY;

  std::vector<inst_fetch_buffer_entry *> m_pipeline_reg;

  register_set *m_result_port;

  unsigned m_issue_reg_id;

  unsigned active_insts_in_pipeline;

  std::string m_name;

  inst_fetch_buffer_entry *m_dispatch_reg;

  hw_config *m_hw_cfg;
  trace_parser *m_tracer;

public:
  std::bitset<MAX_ALU_LATENCY> occupied;
};

class sfu : public pipelined_simd_unit {
public:
  sfu(register_set *result_port, unsigned issue_reg_id, hw_config *hw_cfg,
      trace_parser *tracer)
      : pipelined_simd_unit(result_port,

                            MAX_ALU_LATENCY, issue_reg_id, hw_cfg, tracer) {
    m_name = "SFU";
  }

  virtual bool can_issue(const inst_fetch_buffer_entry &inst) const;
  virtual unsigned clock_multiplier() const { return 1; }
  virtual void issue(register_set &source_reg);
  virtual void issue(register_set &source_reg, unsigned reg_id);
  inline bool is_issue_partitioned() override { return true; }
  virtual bool stallable() const { return false; }
};

class dp_unit : public pipelined_simd_unit {
public:
  dp_unit(register_set *result_port, unsigned issue_reg_id, hw_config *hw_cfg,
          trace_parser *tracer)
      : pipelined_simd_unit(result_port,

                            MAX_ALU_LATENCY, issue_reg_id, hw_cfg, tracer) {
    m_name = "DP";
  }

  virtual bool can_issue(const inst_fetch_buffer_entry &inst) const;
  virtual unsigned clock_multiplier() const { return 1; }
  virtual void issue(register_set &source_reg);
  virtual void issue(register_set &source_reg, unsigned reg_id);
  inline bool is_issue_partitioned() override { return true; }
  virtual bool stallable() const { return false; }
};

class sp_unit : public pipelined_simd_unit {
public:
  sp_unit(register_set *result_port, unsigned issue_reg_id, hw_config *hw_cfg,
          trace_parser *tracer)
      : pipelined_simd_unit(result_port,

                            MAX_ALU_LATENCY, issue_reg_id, hw_cfg, tracer) {
    m_name = "SP";
  }

  virtual bool can_issue(const inst_fetch_buffer_entry &inst) const;
  virtual unsigned clock_multiplier() const { return 1; }
  virtual void issue(register_set &source_reg);
  virtual void issue(register_set &source_reg, unsigned reg_id);
  inline bool is_issue_partitioned() override { return true; }
  virtual bool stallable() const { return false; }
};

class tensor_core : public pipelined_simd_unit {
public:
  tensor_core(register_set *result_port, unsigned issue_reg_id,
              hw_config *hw_cfg, trace_parser *tracer)
      : pipelined_simd_unit(result_port,

                            MAX_ALU_LATENCY, issue_reg_id, hw_cfg, tracer) {
    m_name = "TENSOR_CORE";
  }

  virtual bool can_issue(const inst_fetch_buffer_entry &inst) const;
  virtual unsigned clock_multiplier() const { return 1; }
  virtual void issue(register_set &source_reg);
  virtual void issue(register_set &source_reg, unsigned reg_id);
  inline bool is_issue_partitioned() override { return true; }
  virtual bool stallable() const { return false; }
};

class int_unit : public pipelined_simd_unit {
public:
  int_unit(register_set *result_port, unsigned issue_reg_id, hw_config *hw_cfg,
           trace_parser *tracer)
      : pipelined_simd_unit(result_port,

                            MAX_ALU_LATENCY, issue_reg_id, hw_cfg, tracer) {
    m_name = "INT";
  }

  virtual bool can_issue(const inst_fetch_buffer_entry &inst) const;
  virtual unsigned clock_multiplier() const { return 1; }
  virtual void issue(register_set &source_reg);
  virtual void issue(register_set &source_reg, unsigned reg_id);
  inline bool is_issue_partitioned() override { return true; }
  virtual bool stallable() const { return false; }
};

class specialized_unit : public pipelined_simd_unit {
public:
  specialized_unit(register_set *result_port, unsigned issue_reg_id,
                   hw_config *hw_cfg, trace_parser *tracer, unsigned index)
      : pipelined_simd_unit(result_port,

                            MAX_ALU_LATENCY, issue_reg_id, hw_cfg, tracer) {
    m_index = index;
    m_name = std::string("SPECIALIZED_UNIT") + std::to_string(m_index);
  }

  virtual bool can_issue(const inst_fetch_buffer_entry &inst) const;
  virtual unsigned clock_multiplier() const { return 1; }
  virtual void issue(register_set &source_reg);
  virtual void issue(register_set &source_reg, unsigned reg_id);
  inline bool is_issue_partitioned() override { return true; }
  virtual bool stallable() const { return false; }

private:
  unsigned m_index;
};

class mem_unit : public pipelined_simd_unit {
public:
  mem_unit(register_set *result_port, unsigned issue_reg_id, hw_config *hw_cfg,
           trace_parser *tracer)
      : pipelined_simd_unit(result_port,

                            MAX_ALU_LATENCY, issue_reg_id, hw_cfg, tracer) {
    m_pipeline_reg_mem_unit.reserve(hw_cfg->get_num_mem_units());
    for (unsigned i = 0; i < hw_cfg->get_num_mem_units(); i++) {
      m_pipeline_reg_mem_unit.push_back(
          std::vector<inst_fetch_buffer_entry *>());
    }
    for (unsigned i = 0; i < hw_cfg->get_num_mem_units(); i++) {
      for (unsigned j = 0; j < m_pipeline_depth; j++) {
        m_pipeline_reg_mem_unit[i].push_back(new inst_fetch_buffer_entry());
      }
    }

    m_pipeline_reg.reserve(m_pipeline_depth);
    for (unsigned i = 0; i < m_pipeline_depth; i++) {
      m_pipeline_reg.push_back(new inst_fetch_buffer_entry());
    }
    m_name = "MEM";
  }

  ~mem_unit() {
    for (unsigned i = 0; i < m_hw_cfg->get_num_mem_units(); i++) {
      for (unsigned j = 0; j < m_pipeline_depth; j++) {
        delete m_pipeline_reg_mem_unit[i][j];
      }
    }
  }

  std::vector<std::vector<inst_fetch_buffer_entry *>> m_pipeline_reg_mem_unit;

  virtual bool can_issue(const inst_fetch_buffer_entry &inst) const;
  virtual unsigned clock_multiplier() const { return 1; }
  virtual void issue(register_set &source_reg);
  virtual void issue(register_set &source_reg, unsigned reg_id);
  inline bool is_issue_partitioned() override { return true; }
  virtual bool stallable() const { return true; }
  virtual std::vector<unsigned int>
  cycle(trace_parser *tracer, Scoreboard *m_scoreboard, app_config *appcfg,
        std::vector<std::pair<int, int>> *kernel_block_pair,
        std::vector<unsigned> *m_num_warps_per_sm, unsigned KERNEL_EVALUATION,
        unsigned num_scheds, regBankAlloc *m_reg_bank_allocator,
        bool *flag_Writeback_Memory_Structural_bank_of_reg_is_not_idle,
        std::map<std::tuple<unsigned, unsigned, unsigned>,
                 std::tuple<unsigned, unsigned, unsigned, unsigned, unsigned,
                            unsigned>> *clk_record,
        unsigned cycle);
};

#endif
