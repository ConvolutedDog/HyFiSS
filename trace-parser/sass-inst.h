#include <bitset>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../ISA-Def/trace_opcode.h"
#include "../common/common_def.h"
#include "../common/vector_types.h"

#ifndef SASS_INST_H
#define SASS_INST_H

struct sass_inst_t {
  std::string insnStr;
  std::string kernel_name;
  unsigned kernel_id;

  unsigned line_num;
  unsigned m_pc;
  unsigned mask;
  unsigned reg_dsts_num;
  unsigned reg_dest[MAX_DST];
  std::string opcode;
  unsigned reg_srcs_num;
  unsigned reg_src[MAX_SRC];

  std::string m_source_file;
  unsigned m_source_line;

  const char *source_file() const { return m_source_file.c_str(); }
  unsigned source_line() const { return m_source_line; }

  bool m_empty = true;
};

extern std::map<unsigned, sass_inst_t> pc_to_sassStr;
extern std::vector<int> have_readed_insn_pcs;

sass_inst_t find_sass_inst_by_pc(unsigned pc);

#endif