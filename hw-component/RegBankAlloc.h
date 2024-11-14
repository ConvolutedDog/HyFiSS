#include "stdio.h"
#include <assert.h>
#include <vector>

#ifndef REG_BANK_ALLOC_H
#define REG_BANK_ALLOC_H

enum RegBankState {
  FREE = 0,
  ON_READING,
  ON_WRITING,
  RegBankStateNUM,
};

class regBankAlloc {
public:
  regBankAlloc(const unsigned smid,
               const unsigned num_banks,
               const unsigned num_warp_scheds,
               const unsigned bank_warp_shift,
               const unsigned num_banks_per_sched);

  unsigned register_bank(const unsigned regnum, const unsigned wid,
                         const unsigned sched_id) const;

  RegBankState getBankState(const unsigned bank_id) const;

  RegBankState getBankState(const unsigned regnum, const unsigned wid,
                                   const unsigned sched_id) const;

  void setBankState(const unsigned bank_id, const RegBankState state);
  void setBankState(const unsigned regnum, const unsigned wid,
                    const unsigned sched_id, const RegBankState state);

  void releaseBankState(const unsigned regnum, const unsigned wid,
                        const unsigned sched_id);
  void releaseBankState(const unsigned bank_id);
  void releaseAllBankStates();

  void printBankState() const;

private:
  unsigned m_smid;
  unsigned m_num_banks;
  unsigned m_num_warp_scheds;
  unsigned m_bank_warp_shift;
  unsigned m_num_banks_per_sched;
  bool m_sub_core_model;
  std::vector<RegBankState> m_bank_state;

  // When `m_num_banks` is a power of 2, a shift operation can be
  // used instead of a modulo operation (%) to improve efficiency.
  bool isNumBankPowerOfTwo;
};

#endif
