#include "Scoreboard.h"

Scoreboard::Scoreboard(const unsigned smid, const unsigned n_warps)
    : longopregs() {
  m_smid = smid;

  reg_table.resize(n_warps);
  longopregs.resize(n_warps);
}

void Scoreboard::printContents() const {
  printf("    Scoreboard contents (sid=%u): \n", m_smid);
  for (unsigned i = 0; i < reg_table.size(); i++) {
    if (reg_table[i].size() == 0)
      continue;
    printf("  wid = %2u: ", i);
    std::set<int>::const_iterator it;
    for (it = reg_table[i].begin(); it != reg_table[i].end(); it++)
      printf("R%d ", *it);
    printf("\n");
  }
}

void Scoreboard::printContents(unsigned i) const {

  printf("  wid = %2u: ", i);
  std::set<int>::const_iterator it;
  for (it = reg_table[i].begin(); it != reg_table[i].end(); it++)
    printf("R%d ", *it);
  printf("\n");
}

void Scoreboard::reserveRegister(const unsigned wid, const int regnum) {
  if (!(reg_table[wid].find(regnum) == reg_table[wid].end())) {
    printf("Error: trying to reserve an already reserved register (sid=%u, "
           "wid=%u, regnum=%d).",
           m_smid, wid, regnum);
    abort();
  }
  reg_table[wid].insert(regnum);
}

void Scoreboard::releaseRegister(const unsigned wid, const int regnum) {
  if (!(reg_table[wid].find(regnum) != reg_table[wid].end()))
    return;
  if (regnum != -1)
    reg_table[wid].erase(regnum);
}

const bool Scoreboard::islongop(const unsigned wid, const int regnum) {
  if (regnum == -1)
    return false;
  else
    return longopregs[wid].find(regnum) != longopregs[wid].end();
}

void Scoreboard::reserveRegisters(const unsigned wid, std::vector<int> regnums,
                                  bool is_load) {
  std::vector<int> prev_regs;

  for (unsigned r = 0; r < regnums.size(); r++) {
    if (regnums[r] > 0) {
      if (std::find(prev_regs.begin(), prev_regs.end(), regnums[r]) ==
          prev_regs.end()) {
        prev_regs.push_back(regnums[r]);

        reserveRegister(wid, regnums[r]);
      }
    }
  }

  if (is_load) {
    for (unsigned r = 0; r < regnums.size(); r++) {
      if (regnums[r] > 0) {
        longopregs[wid].insert(regnums[r]);
      }
    }
  }
}

void Scoreboard::releaseRegisters(const unsigned wid,
                                  std::vector<int> regnums) {
  for (unsigned r = 0; r < regnums.size(); r++) {
    if (regnums[r] > 0) {
      releaseRegister(wid, regnums[r]);
      longopregs[wid].erase(regnums[r]);
    }
  }
}

void Scoreboard::releaseRegisters(const unsigned wid, const int regnum) {
  if (regnum > 0) {
    releaseRegister(wid, regnum);
    longopregs[wid].erase(regnum);
  }
}

bool Scoreboard::checkCollision(const unsigned wid, std::vector<int> regnums,
                                int pred, int ar1, int ar2) const {

  std::set<int> inst_regs;

  for (unsigned iii = 0; iii < regnums.size(); iii++)
    if (regnums[iii] > 0)
      inst_regs.insert(regnums[iii]);

  if (pred > 0)
    inst_regs.insert(pred);
  if (ar1 > 0)
    inst_regs.insert(ar1);
  if (ar2 > 0)
    inst_regs.insert(ar2);

  std::set<int>::const_iterator it2;
  for (it2 = inst_regs.begin(); it2 != inst_regs.end(); it2++)
    if (reg_table[wid].find(*it2) != reg_table[wid].end()) {
      return true;
    }
  return false;
}

bool Scoreboard::pendingWrites(const unsigned wid) const {
  return !reg_table[wid].empty();
}