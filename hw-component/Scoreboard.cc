#include "Scoreboard.h"

Scoreboard::Scoreboard(const unsigned smid, 
                       const unsigned n_warps)
  : m_smid(smid) {
  reg_table.resize(n_warps);
  longopregs.resize(n_warps);

  /// TODO: For `std::vector<std::unordered_set<int>> reg_table;`, we need
  /// to determine in advance how many registers will be inserted, then use
  /// `reg_table[wid].reserve(size)` to pre-allocate memory. This should
  /// improve performance by reducing the number of dynamic memory allocs
  /// that occur when inserting an element.
}

void Scoreboard::reserveRegister(const unsigned wid, 
                                 const int regnum) noexcept {
  auto [iter, inserted] = reg_table[wid].insert(regnum);
  if (!inserted) {
    printf("Error: trying to reserve an already reserved register (sid=%u, "
           "wid=%u, regnum=%d).\n", m_smid, wid, regnum);
    abort();
  }
}

const bool Scoreboard::islongop(const unsigned wid, const int regnum) const {
  if (regnum == -1) return false;
  else return longopregs[wid].find(regnum) != longopregs[wid].end();
}

void Scoreboard::reserveRegisters(const unsigned wid, std::vector<int> &regnums,
                                  bool is_load) noexcept {
  std::unordered_set<int> prev_regs;
  for (auto &regnum : regnums) {
    if (regnum > 0 && prev_regs.insert(regnum).second) {
      reserveRegister(wid, regnum);
    }
  }

  if (is_load)
    for (auto &regnum : regnums)
      if (regnum > 0) longopregs[wid].insert(regnum);
}

void Scoreboard::releaseRegisters(const unsigned wid,
                                  std::vector<int> &regnums) noexcept {
  for (auto &regnum : regnums)
    releaseRegister(wid, regnum);
}

bool Scoreboard::checkCollision(const unsigned wid, std::vector<int> &regnums,
                                const int pred, const int ar1, const int ar2) const {
  if (pred > 0 && reg_table[wid].find(pred) != reg_table[wid].end()) return true;
  if (ar1 > 0 && reg_table[wid].find(ar1) != reg_table[wid].end()) return true;
  if (ar2 > 0 && reg_table[wid].find(ar2) != reg_table[wid].end()) return true;
  for (auto &reg : regnums)
    if (reg > 0 && reg_table[wid].find(reg) != reg_table[wid].end())
      return true;

  return false;
}

void Scoreboard::printContents() const {
  printf("    Scoreboard contents (sid=%u): \n", m_smid);
  for (unsigned i = 0; i < reg_table.size(); i++) {
    if (reg_table[i].size() == 0)
      continue;
    printContents(i);
  }
}

void Scoreboard::printContents(unsigned i) const {
  printf("  wid = %2u: ", i);
  std::unordered_set<int>::const_iterator it;
  for (it = reg_table[i].begin(); it != reg_table[i].end(); it++)
    printf("R%d ", *it);
  printf("\n");
}
