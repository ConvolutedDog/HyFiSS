#include "sass-inst.h"

std::map<unsigned, sass_inst_t> pc_to_sassStr;
std::vector<int> have_readed_insn_pcs;

bool have_print_sass_during_this_execution = false;

sass_inst_t find_sass_inst_by_pc(unsigned pc) {
  std::map<unsigned, sass_inst_t>::iterator iter;
  iter = pc_to_sassStr.find(pc);
  if (iter != pc_to_sassStr.end()) {
    return iter->second;
  } else {
    std::cout << "Can't find sass inst by pc: " << std::hex << pc << std::endl;
    sass_inst_t null_ = sass_inst_t();
    return null_;
  }
}
