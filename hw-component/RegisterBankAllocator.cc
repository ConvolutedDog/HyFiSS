#include "RegisterBankAllocator.h"

RegisterBankAllocator::RegisterBankAllocator(
    const unsigned smid, const unsigned num_banks,
    const unsigned num_warp_scheds, const unsigned bank_warp_shift,
    const unsigned num_banks_per_sched) {
  m_smid = smid;
  m_num_banks = num_banks;
  m_num_warp_scheds = num_warp_scheds;
  m_bank_warp_shift = bank_warp_shift;
  m_num_banks_per_sched = num_banks_per_sched;
  m_sub_core_model = (num_warp_scheds > 1);
  m_bank_state.resize(num_banks, FREE);
}

unsigned RegisterBankAllocator::register_bank(const unsigned regnum,
                                              const unsigned wid,
                                              const unsigned sched_id) const {
  unsigned bank = regnum;
  if (m_bank_warp_shift)
    bank += wid;
  if (m_sub_core_model) {
    unsigned bank_num =
        (bank % m_num_banks_per_sched) + (sched_id * m_num_banks_per_sched);
    assert(bank_num < m_num_banks);
    return bank_num;
  } else
    return bank % m_num_banks;
}

Register_Bank_State
RegisterBankAllocator::getBankState(const unsigned bank_id) const {
  return m_bank_state[bank_id];
}

Register_Bank_State
RegisterBankAllocator::getBankState(const unsigned regnum, const unsigned wid,
                                    const unsigned sched_id) const {
  unsigned bank_id = register_bank(regnum, wid, sched_id);
  return m_bank_state[bank_id];
}

void RegisterBankAllocator::setBankState(const unsigned bank_id,
                                         const Register_Bank_State state) {
  m_bank_state[bank_id] = state;
}

void RegisterBankAllocator::setBankState(const unsigned regnum,
                                         const unsigned wid,
                                         const unsigned sched_id,
                                         const Register_Bank_State state) {
  unsigned bank_id = register_bank(regnum, wid, sched_id);
  m_bank_state[bank_id] = state;
}

void RegisterBankAllocator::releaseBankState(const unsigned regnum,
                                             const unsigned wid,
                                             const unsigned sched_id) {
  unsigned bank_id = register_bank(regnum, wid, sched_id);
  m_bank_state[bank_id] = FREE;
}

void RegisterBankAllocator::releaseBankState(const unsigned bank_id) {
  m_bank_state[bank_id] = FREE;
}

void RegisterBankAllocator::releaseAllBankStates() {
  for (unsigned i = 0; i < m_num_banks; i++) {
    m_bank_state[i] = FREE;
  }
}

void RegisterBankAllocator::printBankState() const {
  printf("Register Bank State (smid=%u): \n", m_smid);
  for (unsigned i = 0; i < m_num_banks; i++) {
    printf("  bank %2u: %d\n", i, m_bank_state[i]);
  }
}