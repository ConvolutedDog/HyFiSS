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

  inline const RegBankState& getBankState(const unsigned bank_id) const {
    return m_bank_state[bank_id];
  }

  const RegBankState& getBankState(const unsigned regnum, const unsigned wid,
                                   const unsigned sched_id) const;

  inline void setBankState(const unsigned bank_id, 
                           const RegBankState state) noexcept {
    m_bank_state[bank_id] = state;
  };

  void setBankState(const unsigned regnum, const unsigned wid,
                    const unsigned sched_id, const RegBankState state) noexcept;
  
  inline void releaseBankState(const unsigned bank_id) noexcept {
    setBankState(bank_id, FREE);
  }
  
  void releaseBankState(const unsigned regnum,
                        const unsigned wid,
                        const unsigned sched_id) noexcept;

  /// TODO: Using `std::fill_n(m_bank_state.begin(), m_num_banks, FREE);`
  /// to replace the loop can cause additional overhead due to the in-
  /// ability to take advantage of inline functions, as well as the in-
  /// ternal iterator performing bounds checks.
  inline void releaseAllBankStates() noexcept {
    for (unsigned i = 0; i < m_num_banks; ++i) {
      setBankState(i, FREE);
    }
  };

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
