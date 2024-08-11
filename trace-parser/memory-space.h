#include "../common/common_def.h"
#include "../common/vector_types.h"

#ifndef MEMORY_SPACE_H
#define MEMORY_SPACE_H

class memory_space_t {
public:
  memory_space_t();

  memory_space_t(const enum _memory_space_t &from);

  bool operator==(const memory_space_t &x) const;
  bool operator!=(const memory_space_t &x) const;
  bool operator<(const memory_space_t &x) const;
  enum _memory_space_t get_type() const;
  void set_type(enum _memory_space_t t);
  unsigned get_bank() const;
  void set_bank(unsigned b);
  bool is_const() const;
  bool is_local() const;
  bool is_global() const;

private:
  enum _memory_space_t m_type;
  unsigned m_bank;
};

#endif