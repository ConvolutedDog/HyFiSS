#include "mem-access.h"

mem_access_t::mem_access_t(mem_access_type type, new_addr_type address,
                           unsigned size, bool wr) {

  m_type = type;

  m_addr = address;

  m_req_size = size;

  m_write = wr;
}

mem_access_t::mem_access_t(mem_access_type type, new_addr_type address,
                           unsigned size, bool wr,
                           const active_mask_t &active_mask,
                           const mem_access_byte_mask_t &byte_mask,
                           const mem_access_sector_mask_t &sector_mask)
    : m_warp_mask(active_mask), m_byte_mask(byte_mask),
      m_sector_mask(sector_mask) {
  m_type = type;
  m_addr = address;
  m_req_size = size;
  m_write = wr;
}

void mem_access_t::print(FILE *fp) const {
  fprintf(fp, "addr=0x%llx, %s, size=%u, ", m_addr, m_write ? "store" : "load ",
          m_req_size);
  switch (m_type) {
  case GLOBAL_ACC_R:
    fprintf(fp, "GLOBAL_R");
    break;
  case LOCAL_ACC_R:
    fprintf(fp, "LOCAL_R ");
    break;
  case CONST_ACC_R:
    fprintf(fp, "CONST   ");
    break;
  case TEXTURE_ACC_R:
    fprintf(fp, "TEXTURE ");
    break;
  case GLOBAL_ACC_W:
    fprintf(fp, "GLOBAL_W");
    break;
  case LOCAL_ACC_W:
    fprintf(fp, "LOCAL_W ");
    break;
  case L2_WRBK_ACC:
    fprintf(fp, "L2_WRBK ");
    break;
  case INST_ACC_R:
    fprintf(fp, "INST    ");
    break;
  case L1_WRBK_ACC:
    fprintf(fp, "L1_WRBK ");
    break;
  default:
    fprintf(fp, "unknown ");
    break;
  }
}