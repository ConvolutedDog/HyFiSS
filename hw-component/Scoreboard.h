#include <algorithm>
#include <iostream>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unordered_set> 
#include <cassert>

#ifndef SCOREBOARD_H
#define SCOREBOARD_H

class Scoreboard {
public:
  Scoreboard(const unsigned smid, const unsigned n_warps);

  void reserveRegisters(const unsigned wid, std::vector<int> &regnums,
                        bool is_load) noexcept;

  void releaseRegisters(const unsigned wid, std::vector<int> &regnums) noexcept;

  inline void releaseRegister(const unsigned wid, const int regnum) noexcept {
    if (regnum != -1) reg_table[wid].erase(regnum);
  }

  bool checkCollision(const unsigned wid, std::vector<int> &regnums, const int pred,
                      const int ar1, const int ar2) const;

  /// TODO: Maybe don't need this again.
  inline bool pendingWrites(const unsigned wid) const {
    return !reg_table[wid].empty();
  }

  /// TODO: Maybe don't need this again.
  const bool islongop(const unsigned wid, const int regnum) const;

  inline const unsigned regs_size(const unsigned wid) const {
    return reg_table[wid].size();
  }

  void printContents() const;
  void printContents(unsigned i) const;

private:
  void reserveRegister(const unsigned wid, const int regnum) noexcept;

  int get_sid() const { return m_smid; }

  unsigned m_smid;

  std::vector<std::unordered_set<int>> reg_table;
  std::vector<std::unordered_set<int>> longopregs;
};

#endif
