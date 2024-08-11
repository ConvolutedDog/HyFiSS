#include "stdio.h"
#include <assert.h>
#include <vector>

#ifndef REGISTERBANKALLOCATOR_H
#define REGISTERBANKALLOCATOR_H

enum Register_Bank_State {
  FREE = 0,
  ON_READING,
  ON_WRITING,
  Register_Bank_State_NUM,
};

class RegisterBankAllocator {
public:
  RegisterBankAllocator(const unsigned smid, const unsigned num_banks,
                        const unsigned num_warp_scheds,
                        const unsigned bank_warp_shift,
                        const unsigned num_banks_per_sched);

  unsigned register_bank(const unsigned regnum, const unsigned wid,
                         const unsigned sched_id) const;

  Register_Bank_State getBankState(const unsigned bank_id) const;

  Register_Bank_State getBankState(const unsigned regnum, const unsigned wid,
                                   const unsigned sched_id) const;

  void setBankState(const unsigned bank_id, const Register_Bank_State state);
  void setBankState(const unsigned regnum, const unsigned wid,
                    const unsigned sched_id, const Register_Bank_State state);

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
  std::vector<Register_Bank_State> m_bank_state;
};

#endif