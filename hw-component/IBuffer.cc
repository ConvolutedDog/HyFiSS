#include "IBuffer.h"

#define PRINT_AT for(unsigned i=0; i<40; ++i) std::cout << "@"; std::cout << std::endl;

IBuffer::IBuffer(const unsigned smid, const unsigned num_warps)
  : m_smid(smid), m_num_warps(num_warps) {
  m_ibuffer.resize(num_warps);
}

void IBuffer::print_ibuffer() const {
  for (unsigned i = 0; i < m_num_warps; i++) {
    std::cout << "warp - " << i << ": ";
    for (auto it = m_ibuffer[i].begin(); it != m_ibuffer[i].end(); it++) {
      std::cout << "(" << it->pc << ", " << it->wid << ", " << it->kid
                << "), ";
    }
    std::cout << std::endl;
  }
}

void IBuffer::print_ibuffer(const unsigned gwarp_start, const unsigned gwarp_end) const {
PRINT_AT;
  for (unsigned i = gwarp_start; i < gwarp_end; i++) {
    std::cout << "    Ibuffer (pc, wid, kid) warp - " << i << ": ";
    for (auto it = m_ibuffer[i].begin(); it != m_ibuffer[i].end(); it++) {
      std::cout << "(" << it->pc << ", " << it->wid << ", " << it->kid
                << "), ";
    }
    std::cout << std::endl;
  }
PRINT_AT;
}

void IBuffer::print_ibuffer(const unsigned gwarp_id) const {
  std::cout << "    Ibuffer (pc, wid, kid) warp - " << gwarp_id << ": ";
  for (auto it = m_ibuffer[gwarp_id].begin(); it != m_ibuffer[gwarp_id].end();
       it++) {
    std::cout << "(" << it->pc << ", " << it->wid << ", " << it->kid << "), ";
  }
  std::cout << std::endl;
}
