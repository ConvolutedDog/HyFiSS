#include <iostream>
#include <utility>
#include <vector>

#ifndef IBUFFER_H
#define IBUFFER_H

struct ibuffer_entry {
  ibuffer_entry(unsigned _pc, unsigned _wid, unsigned _kid, unsigned _uid) {
    pc = _pc;
    wid = _wid;
    kid = _kid;
    uid = _uid;
  };
  unsigned pc;
  unsigned wid;
  unsigned kid;
  unsigned uid;
};

class IBuffer {
public:
  IBuffer(const unsigned smid, const unsigned num_warps);

  bool is_empty(unsigned global_all_kernels_warp_id) {
    return m_ibuffer[global_all_kernels_warp_id].empty();
  }

  bool has_free_slot(unsigned global_all_kernels_warp_id) {
    return m_ibuffer[global_all_kernels_warp_id].size() < 2;
  }

  bool is_not_empty(unsigned global_all_kernels_warp_id) {
    return !m_ibuffer[global_all_kernels_warp_id].empty();
  }

  void push_back(unsigned global_all_kernels_warp_id, ibuffer_entry entry) {
    m_ibuffer[global_all_kernels_warp_id].push_back(entry);
  }

  ibuffer_entry pop_front(unsigned global_all_kernels_warp_id) {
    ibuffer_entry entry = m_ibuffer[global_all_kernels_warp_id].front();
    m_ibuffer[global_all_kernels_warp_id].erase(
        m_ibuffer[global_all_kernels_warp_id].begin());
    return entry;
  }

  ibuffer_entry front(unsigned global_all_kernels_warp_id) {
    ibuffer_entry entry = m_ibuffer[global_all_kernels_warp_id].front();
    return entry;
  }

  void print_ibuffer() {
    for (unsigned i = 0; i < m_num_warps; i++) {
      std::cout << "warp - " << i << ": ";
      for (auto it = m_ibuffer[i].begin(); it != m_ibuffer[i].end(); it++) {
        std::cout << "(" << it->pc << ", " << it->wid << ", " << it->kid
                  << "), ";
      }
      std::cout << std::endl;
    }
  }

  void print_ibuffer(unsigned gwarp_start, unsigned gwarp_end) {
    std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
    for (unsigned i = gwarp_start; i < gwarp_end; i++) {
      std::cout << "    Ibuffer (pc, wid, kid) warp - " << i << ": ";
      for (auto it = m_ibuffer[i].begin(); it != m_ibuffer[i].end(); it++) {
        std::cout << "(" << it->pc << ", " << it->wid << ", " << it->kid
                  << "), ";
      }
      std::cout << std::endl;
    }
    std::cout << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << std::endl;
  }

  void print_ibuffer(unsigned gwarp_id) {
    std::cout << "    Ibuffer (pc, wid, kid) warp - " << gwarp_id << ": ";
    for (auto it = m_ibuffer[gwarp_id].begin(); it != m_ibuffer[gwarp_id].end();
         it++) {
      std::cout << "(" << it->pc << ", " << it->wid << ", " << it->kid << "), ";
    }
    std::cout << std::endl;
  }

private:
  unsigned m_smid;

  std::vector<std::vector<ibuffer_entry>> m_ibuffer;
  unsigned m_num_warps;
};

#endif