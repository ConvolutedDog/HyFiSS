#include <list>
#include <map>
#include <regex>
#include <string.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "../ISA-Def/accelwattch_component_mapping.h"
#include "../ISA-Def/trace_opcode.h"
#include "../common/common_def.h"
#include "../common/vector_types.h"
#include "../trace-parser/inst-trace.h"

#include "kernel-trace.h"
#include "mem-access.h"

#ifndef TRACE_WARP_INST_H
#define TRACE_WARP_INST_H

types_of_operands get_oprnd_type(op_type op, special_ops sp_op);

class trace_warp_inst_t {
public:
  trace_warp_inst_t() {
    m_opcode = 0;
    m_uid = 0;
    m_empty = true;
    m_isatomic = false;

    m_decoded = false;
    pc = (address_type)-1;
    isize = 0;

    num_operands = 0;
    num_regs = 0;

    memset(out, 0, sizeof(unsigned));
    outcount = 0;
    memset(in, 0, sizeof(unsigned));
    incount = 0;

    is_vectorin = false;
    is_vectorout = false;

    pred = -1;
    ar1 = -1;
    ar2 = -1;

    for (unsigned i = 0; i < MAX_REG_OPERANDS; i++) {
      arch_reg.src[i] = -1;
      arch_reg.dst[i] = -1;
    }

    memory_op = no_memory_op;
    data_size = 0;

    op = NO_OP;
    sp_op = OTHER_OP;
    mem_op = NOT_TEX;

    const_cache_operand = 0;

    oprnd_type = UN_OP;

    m_is_printf = false;
    should_do_atomic = false;

    m_gwarp_id = 0;
    m_warp_id = 0;
    m_dynamic_warp_id = 0;

    space = memory_space_t();
    cache_op = CACHE_UNDEFINED;
  }

  bool parse_from_trace_struct(
      const _inst_trace_t *trace,
      const std::unordered_map<std::string, OpcodeChar> *OpcodeMap,
      unsigned gwarp_id);

  inline void set_active(const active_mask_t &active);

  unsigned get_opcode() const { return m_opcode; }
  unsigned get_uid() const { return m_uid; }
  bool isempty() const { return m_empty; }
  bool isatomic() const { return m_isatomic; }
  bool isdecoded() const { return m_decoded; }
  address_type get_pc() const { return pc; }
  unsigned get_isize() const { return isize; }
  unsigned get_outcount() const { return outcount; }
  unsigned get_incount() const { return incount; }
  unsigned get_in(unsigned i) const {
    assert(i < incount);
    return in[i];
  }
  unsigned get_out(unsigned i) const {
    assert(i < outcount);
    return out[i];
  }
  bool get_is_vectorin() const { return is_vectorin; }
  bool get_is_vectorout() const { return is_vectorout; }
  int get_pred() const { return pred; }
  int get_ar1() const { return ar1; }
  int get_ar2() const { return ar2; }
  int get_arch_reg_dst(unsigned i) const {
    assert(i < outcount);
    return arch_reg.dst[i];
  }
  /// Determines whether all result registers are written back, and
  /// the value of the register is set to -1 after being written back.
  const bool allArchRegDstWriteBack() const {
    // Another implementation logic:
    //   bool all_write_back = true;
    //   for (unsigned i = 0; i < outcount; ++i) {
    //     if (trace_warp_inst.get_arch_reg_dst(i) != -1) {
    //       all_write_back = false;
    //       break;
    //     }
    //   }
    //   return all_write_back;
    return std::all_of(
      std::begin(arch_reg.dst), std::end(arch_reg.dst), 
      [&](int dstRegValue){ return dstRegValue == -1; });
  }
  int get_arch_reg_src(unsigned i) const {
    assert(i < incount);
    return arch_reg.src[i];
  }
  void set_arch_reg_dst(unsigned i, int reg) {
    assert(i < outcount);
    arch_reg.dst[i] = reg;
  }
  void set_arch_reg_src(unsigned i, int reg) {
    assert(i < incount);
    arch_reg.src[i] = reg;
  }
  _memory_op_t get_memory_op() const { return memory_op; }
  unsigned get_num_operands() const { return num_operands; }
  unsigned get_num_regs() const { return num_regs; }
  unsigned get_data_size() const { return data_size; }
  op_type get_op() const { return op; }
  special_ops get_sp_op() const { return sp_op; }
  mem_operation get_mem_op() const { return mem_op; }
  bool get_const_cache_operand() const { return const_cache_operand; }
  types_of_operands get_oprnd_type_() const { return oprnd_type; }
  bool get_should_do_atomic() const { return should_do_atomic; }
  bool get_is_printf() const { return m_is_printf; }
  unsigned get_gwarp_id() const { return m_gwarp_id; }
  unsigned get_warp_id() const { return m_warp_id; }
  unsigned get_dynamic_warp_id() const { return m_dynamic_warp_id; }
  active_mask_t get_active_mask() const { return m_warp_active_mask; }
  active_mask_t &get_active_mask_ref() { return m_warp_active_mask; }
  unsigned get_activate_count() const { return m_warp_active_mask.count(); }

private:
  unsigned m_opcode;
  unsigned m_uid;
  bool m_empty;
  bool m_isatomic;

  bool m_decoded = false;
  address_type pc = (address_type)-1;
  unsigned isize;

  unsigned out[8];

  unsigned outcount;

  unsigned in[24];

  unsigned incount;

  bool is_vectorin;
  bool is_vectorout;

  int pred;
  int ar1, ar2;

  struct {
    int dst[MAX_REG_OPERANDS];
    int src[MAX_REG_OPERANDS];
  } arch_reg;

  _memory_op_t memory_op;

  unsigned num_operands;
  unsigned num_regs;

  unsigned data_size;

  op_type op;
  special_ops sp_op;
  mem_operation mem_op;

  bool const_cache_operand;

  types_of_operands oprnd_type;

  bool should_do_atomic;
  bool m_is_printf;

  unsigned m_gwarp_id;
  unsigned m_warp_id;

  unsigned m_dynamic_warp_id;

  active_mask_t m_warp_active_mask;

  memory_space_t space;
  cache_operator_type cache_op;
};

#endif