#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "../ISA-Def/trace_opcode.h"
#include "../ISA-Def/volta_opcode.h"
#include "../common/common_def.h"
#include "../hw-parser/hw-parser.h"
#include "inst-memadd-info.h"
#include "memory-space.h"
#include "sass-inst.h"

#ifndef INST_TRACE_H
#define INST_TRACE_H

enum FUNC_UNITS_NAME {

  NON_UNIT = 0,
  SP_UNIT,
  SFU_UNIT,
  INT_UNIT,
  DP_UNIT,
  TENSOR_CORE_UNIT,
  LDST_UNIT,
  SPEC_UNIT_1,
  SPEC_UNIT_2,
  SPEC_UNIT_3,
  NUM_FUNC_UNITS
};

struct inst_trace_t {
  inst_trace_t();
  inst_trace_t(const inst_trace_t &b);

  unsigned line_num;
  unsigned m_pc;
  unsigned mask;
  unsigned reg_dsts_num;
  unsigned reg_dest[MAX_DST];
  std::string opcode;
  unsigned reg_srcs_num;
  unsigned reg_src[MAX_SRC];
  inst_memadd_info_t *memadd_info;

  bool parse_from_string(std::string trace, unsigned tracer_version,
                         unsigned enable_lineinfo, std::string kernel_name,
                         unsigned kernel_id);

  bool check_opcode_contain(const std::vector<std::string> &opcode,
                            std::string param) const;

  unsigned
  get_datawidth_from_opcode(const std::vector<std::string> &opcode) const;

  std::vector<std::string> get_opcode_tokens() const;

  ~inst_trace_t();
};

struct _inst_trace_t {

  _inst_trace_t(unsigned _kernel_id, unsigned _pc, std::string _instn_str) {
    kernel_id = _kernel_id;
    m_pc = _pc;
    instn_str = _instn_str;

    for (unsigned it = 0; it < MAX_DST; it++) {
      reg_dest_is_pred[it] = false;
    }

    memadd_info = NULL;
    parse_from_string(_instn_str, _kernel_id);

    opcode_tokens = get_opcode_tokens();
    memadd_info->width = get_datawidth_from_opcode(opcode_tokens);
    m_valid = true;
    mask = 0x0;
  };

  _inst_trace_t(unsigned _kernel_id, unsigned _pc, std::string _instn_str,
                hw_config *hw_cfg) {
    kernel_id = _kernel_id;
    m_pc = _pc;
    instn_str = _instn_str;

    for (unsigned it = 0; it < MAX_DST; it++) {
      reg_dest_is_pred[it] = false;
    }

    memadd_info = NULL;
    parse_from_string(_instn_str, _kernel_id);

    opcode_tokens = get_opcode_tokens();
    memadd_info->width = get_datawidth_from_opcode(opcode_tokens);
    this->hw_cfg = hw_cfg;

    parse_opcode_latency_info();
    m_valid = true;
    mask = 0x0;
  };

  bool m_valid = false;

  unsigned kernel_id;
  unsigned m_pc;
  unsigned mask = 0x0;
  unsigned reg_dsts_num;
  int reg_dest[MAX_DST];
  bool reg_dest_is_pred[MAX_DST];
  std::string opcode;

  unsigned reg_srcs_num;
  int reg_src[MAX_SRC];
  inst_memadd_info_t *memadd_info;
  std::string instn_str;

  std::vector<std::string> opcode_tokens;

  std::string pred_str = "";

  unsigned initiation_interval;
  unsigned latency;
  enum FUNC_UNITS_NAME func_unit;
  hw_config *hw_cfg;

  bool parse_from_string(std::string trace, unsigned kernel_id);

  bool check_opcode_contain(const std::vector<std::string> &opcode,
                            std::string param) const;

  unsigned
  get_datawidth_from_opcode(const std::vector<std::string> &opcode) const;

  std::vector<std::string> get_opcode_tokens() const;

  inline std::vector<std::string> get_opcode_tokens_directly() const {
    return opcode_tokens;
  }

  void parse_opcode_latency_info();

  unsigned get_latency() const;
  unsigned get_initiation_interval() const;
  enum FUNC_UNITS_NAME get_func_unit() const;

  ~_inst_trace_t();
};

#endif