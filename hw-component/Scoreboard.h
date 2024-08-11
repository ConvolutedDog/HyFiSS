#include <algorithm>
#include <iostream>
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#ifndef SCOREBOARD_H
#define SCOREBOARD_H

class Scoreboard {
public:
  Scoreboard(const unsigned smid, const unsigned n_warps);

  void reserveRegisters(const unsigned wid, std::vector<int> regnums,
                        bool is_load);

  void releaseRegisters(const unsigned wid, std::vector<int> regnums);
  void releaseRegisters(const unsigned wid, const int regnum);

  void releaseRegister(const unsigned wid, const int regnum);

  bool checkCollision(const unsigned wid, std::vector<int> regnums, int pred,
                      int ar1, int ar2) const;

  bool pendingWrites(const unsigned wid) const;

  void printContents() const;
  void printContents(unsigned i) const;

  const bool islongop(const unsigned wid, const int regnum);

  const unsigned regs_size(const unsigned wid) const {
    return reg_table[wid].size();
  }

private:
  void reserveRegister(const unsigned wid, const int regnum);

  int get_sid() const { return m_smid; }

  unsigned m_smid;

  std::vector<std::set<int>> reg_table;

  std::vector<std::set<int>> longopregs;
};

#endif