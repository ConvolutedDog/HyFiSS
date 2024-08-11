#include "../common/common_def.h"

#ifndef MEM_ACCESS_H
#define MEM_ACCESS_H

class mem_access_t {
public:
  mem_access_t() {}

  mem_access_t(mem_access_type type, new_addr_type address, unsigned size,
               bool wr);

  mem_access_t(mem_access_type type, new_addr_type address, unsigned size,
               bool wr, const active_mask_t &active_mask,
               const mem_access_byte_mask_t &byte_mask,
               const mem_access_sector_mask_t &sector_mask);

  new_addr_type get_addr() const { return m_addr; }

  void set_addr(new_addr_type addr) { m_addr = addr; }

  unsigned get_size() const { return m_req_size; }

  const active_mask_t &get_warp_mask() const { return m_warp_mask; }

  bool is_write() const { return m_write; }

  enum mem_access_type get_type() const { return m_type; }

  mem_access_byte_mask_t get_byte_mask() const { return m_byte_mask; }

  mem_access_sector_mask_t get_sector_mask() const { return m_sector_mask; }

  void print(FILE *fp) const;

private:
  unsigned m_uid;

  new_addr_type m_addr;

  bool m_write;

  unsigned m_req_size;

  mem_access_type m_type;

  active_mask_t m_warp_mask;

  mem_access_byte_mask_t m_byte_mask;

  mem_access_sector_mask_t m_sector_mask;
};

#endif