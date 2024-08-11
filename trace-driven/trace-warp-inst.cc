#include "trace-warp-inst.h"

inline types_of_operands get_oprnd_type(op_type op, special_ops sp_op) {
  switch (op) {
  case SP_OP:
  case SFU_OP:
  case SPECIALIZED_UNIT_2_OP:
  case SPECIALIZED_UNIT_3_OP:
  case DP_OP:
  case LOAD_OP:
  case STORE_OP:
    return FP_OP;
  case INTP_OP:
  case SPECIALIZED_UNIT_4_OP:
    return INT_OP;
  case ALU_OP:
    if ((sp_op == FP__OP) || (sp_op == TEX__OP) || (sp_op == OTHER_OP))
      return FP_OP;
    else if (sp_op == INT__OP)
      return INT_OP;
  default:
    return UN_OP;
  }
}

bool trace_warp_inst_t::parse_from_trace_struct(
    const _inst_trace_t *trace,
    const std::unordered_map<std::string, OpcodeChar> *OpcodeMap,
    unsigned gwarp_id) {

  active_mask_t active_mask = trace->mask;
  set_active(active_mask);

  m_decoded = true;
  pc = (address_type)trace->m_pc;
  m_gwarp_id = gwarp_id;

  isize = 16;
  for (unsigned i = 0; i < MAX_OUTPUT_VALUES; i++) {
    out[i] = 0;
  }
  for (unsigned i = 0; i < MAX_INPUT_VALUES; i++) {
    in[i] = 0;
  }

  is_vectorin = false;
  is_vectorout = false;
  ar1 = -1;
  ar2 = -1;
  memory_op = no_memory_op;
  data_size = 0;
  op = ALU_OP;
  sp_op = OTHER_OP;
  mem_op = NOT_TEX;
  const_cache_operand = 0;
  oprnd_type = UN_OP;

  const std::vector<std::string> &opcode_tokens =
      trace->get_opcode_tokens_directly();
  std::string opcode1 = opcode_tokens[0];

  std::unordered_map<std::string, OpcodeChar>::const_iterator it =
      OpcodeMap->find(opcode1);

  if (it != OpcodeMap->end()) {

    m_opcode = it->second.opcode;
    op = (op_type)(it->second.opcode_category);
    const std::unordered_map<unsigned, unsigned> *OpcPowerMap = &OpcodePowerMap;

    std::unordered_map<unsigned, unsigned>::const_iterator it2 =
        OpcPowerMap->find(m_opcode);
    if (it2 != OpcPowerMap->end())
      sp_op = (special_ops)(it2->second);
    oprnd_type = get_oprnd_type(op, sp_op);
  } else {
    std::cout << "ERROR:  undefined instruction : " << trace->opcode
              << " Opcode: " << opcode1 << std::endl;
    assert(0 && "undefined instruction");
  }

  std::string opcode = trace->opcode;
  if (opcode1 == "MUFU") {

    if ((opcode.find("MUFU.SIN") != std::string::npos) ||
        (opcode.find("MUFU.COS") != std::string::npos))
      sp_op = FP_SIN_OP;
    if ((opcode.find("MUFU.EX2") != std::string::npos) ||
        (opcode.find("MUFU.RCP") != std::string::npos))
      sp_op = FP_EXP_OP;
    if (opcode.find("MUFU.RSQ") != std::string::npos)
      sp_op = FP_SQRT_OP;
    if (opcode.find("MUFU.LG2") != std::string::npos)
      sp_op = FP_LG_OP;
  }

  if (opcode1 == "IMAD") {

    if ((opcode.find("IMAD.MOV") != std::string::npos) ||
        (opcode.find("IMAD.IADD") != std::string::npos))
      sp_op = INT__OP;
  }

  num_regs = trace->reg_srcs_num + trace->reg_dsts_num;
  num_operands = num_regs;
  outcount = trace->reg_dsts_num;
  for (unsigned m = 0; m < trace->reg_dsts_num; ++m) {
    out[m] = trace->reg_dest[m];
    arch_reg.dst[m] = trace->reg_dest[m];
  }

  incount = trace->reg_srcs_num;
  for (unsigned m = 0; m < trace->reg_srcs_num; ++m) {
    in[m] = trace->reg_src[m];
    arch_reg.src[m] = trace->reg_src[m];
  }

  if (trace->memadd_info != NULL) {
    data_size = trace->memadd_info->width;
  }

  switch (m_opcode) {
  case OP_LDC:
    data_size = 4;
    memory_op = memory_load;
    const_cache_operand = 1;

    break;
  case OP_LDG:
  case OP_LDL:
    assert(data_size > 0);
    memory_op = memory_load;

    break;
  case OP_STG:
  case OP_STL:
    assert(data_size > 0);
    memory_op = memory_store;

    break;
  case OP_ATOMG:
  case OP_RED:
  case OP_ATOM:
    assert(data_size > 0);
    memory_op = memory_load;
    op = LOAD_OP;

    m_isatomic = true;
    should_do_atomic = true;

    break;
  case OP_LDS:
    assert(data_size > 0);
    memory_op = memory_load;

    break;
  case OP_STS:
    assert(data_size > 0);
    memory_op = memory_store;

    break;
  case OP_ATOMS:
    assert(data_size > 0);
    m_isatomic = true;
    memory_op = memory_load;

    should_do_atomic = true;
    break;
  case OP_LDSM:
    assert(data_size > 0);

    break;
  case OP_ST:
  case OP_LD:
    assert(data_size > 0);
    if (m_opcode == OP_LD)
      memory_op = memory_load;
    else
      memory_op = memory_store;

    break;
  case OP_BAR:

    break;
  case OP_HADD2:
  case OP_HADD2_32I:
  case OP_HFMA2:
  case OP_HFMA2_32I:
  case OP_HMUL2_32I:
  case OP_HSET2:
  case OP_HSETP2:;
    break;
  default:
    break;
  }

  if (!trace->pred_str.empty()) {
    size_t pos_P = trace->pred_str.find('P');
    if (pos_P != std::string::npos) {
      size_t pos_space = trace->pred_str.find(' ', pos_P);
      size_t count = (pos_space != std::string::npos) ? pos_space - pos_P - 1
                                                      : std::string::npos;
      std::string num_str = trace->pred_str.substr(pos_P + 1, count);
      pred = std::stoul(num_str);
    }
  }

  m_empty = false;

  return true;
}

inline void trace_warp_inst_t::set_active(const active_mask_t &active) {
  m_warp_active_mask = active;
}