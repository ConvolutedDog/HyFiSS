#include "memory-space.h"

memory_space_t::memory_space_t() {
  m_type = undefined_space;
  m_bank = 0;
}

memory_space_t::memory_space_t(const enum _memory_space_t &from) {
  m_type = from;
  m_bank = 0;
}

bool memory_space_t::operator==(const memory_space_t &x) const {
  return (m_bank == x.m_bank) && (m_type == x.m_type);
}

bool memory_space_t::operator!=(const memory_space_t &x) const {
  return !(*this == x);
}

bool memory_space_t::operator<(const memory_space_t &x) const {
  if (m_type < x.m_type)
    return true;
  else if (m_type > x.m_type)
    return false;
  else if (m_bank < x.m_bank)
    return true;
  return false;
}

enum _memory_space_t memory_space_t::get_type() const { return m_type; }

void memory_space_t::set_type(enum _memory_space_t t) { m_type = t; }

unsigned memory_space_t::get_bank() const { return m_bank; }

void memory_space_t::set_bank(unsigned b) { m_bank = b; }

bool memory_space_t::is_const() const {
  return (m_type == const_space) || (m_type == param_space_kernel);
}

bool memory_space_t::is_local() const {
  return (m_type == local_space) || (m_type == param_space_local);
}

bool memory_space_t::is_global() const { return (m_type == global_space); }
