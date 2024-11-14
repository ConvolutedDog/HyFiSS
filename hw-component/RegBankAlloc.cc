#include "RegBankAlloc.h"

regBankAlloc::regBankAlloc(
    const unsigned smid,
    const unsigned num_banks,
    const unsigned num_warp_scheds,
    const unsigned bank_warp_shift,
    const unsigned num_banks_per_sched) 
  : m_smid(smid), m_num_banks(num_banks),
    m_num_warp_scheds(num_warp_scheds),
    m_bank_warp_shift(bank_warp_shift),
    m_num_banks_per_sched(num_banks_per_sched),
    m_sub_core_model(num_warp_scheds > 1) {
  isNumBankPowerOfTwo = (m_num_banks & (m_num_banks - 1)) == 0;
  m_bank_state.resize(num_banks, FREE);
}

unsigned regBankAlloc::register_bank(const unsigned regnum,
                                     const unsigned wid,
                                     const unsigned sched_id) const {
  unsigned bank = regnum;
  if (m_bank_warp_shift) bank += wid;
  if (m_sub_core_model) {
    unsigned bank_num = (bank % m_num_banks_per_sched) + sched_id * m_num_banks_per_sched;
    assert(bank_num < m_num_banks);
    return bank_num;
  } else {
    // Use the `isPowerOfTwo` variable to decide whether to use bitwise
    // operations for optimization.
    return isNumBankPowerOfTwo ? bank & (m_num_banks - 1) : bank % m_num_banks;
  }
}

RegBankState
regBankAlloc::getBankState(const unsigned bank_id) const {
  return m_bank_state[bank_id];
}

RegBankState
regBankAlloc::getBankState(const unsigned regnum, const unsigned wid,
                                    const unsigned sched_id) const {
  unsigned bank_id = register_bank(regnum, wid, sched_id);
  return m_bank_state[bank_id];
}

void regBankAlloc::setBankState(const unsigned bank_id,
                                         const RegBankState state) {
  m_bank_state[bank_id] = state;
}

void regBankAlloc::setBankState(const unsigned regnum,
                                         const unsigned wid,
                                         const unsigned sched_id,
                                         const RegBankState state) {
  unsigned bank_id = register_bank(regnum, wid, sched_id);
  m_bank_state[bank_id] = state;
}

void regBankAlloc::releaseBankState(const unsigned regnum,
                                             const unsigned wid,
                                             const unsigned sched_id) {
  unsigned bank_id = register_bank(regnum, wid, sched_id);
  m_bank_state[bank_id] = FREE;
}

void regBankAlloc::releaseBankState(const unsigned bank_id) {
  m_bank_state[bank_id] = FREE;
}

void regBankAlloc::releaseAllBankStates() {
  for (unsigned i = 0; i < m_num_banks; i++) {
    m_bank_state[i] = FREE;
  }
}

void regBankAlloc::printBankState() const {
  printf("Register Bank State (smid=%u): \n", m_smid);
  for (unsigned i = 0; i < m_num_banks; i++) {
    printf("  bank %2u: %d\n", i, m_bank_state[i]);
  }
}