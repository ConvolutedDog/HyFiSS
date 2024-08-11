#include "IBuffer.h"

IBuffer::IBuffer(const unsigned smid, const unsigned num_warps) {
  m_smid = smid;
  m_num_warps = num_warps;
  m_ibuffer.resize(num_warps);
}
