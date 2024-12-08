#include <iostream>
#include <utility>
#include <vector>

#ifndef IBUFFER_H
#define IBUFFER_H

struct ibuffer_entry {
  ibuffer_entry(const unsigned pc, const unsigned wid,
                const unsigned kid, const unsigned uid)
    : pc(pc), wid(wid), kid(kid), uid(uid) {}

  unsigned pc, wid, kid, uid;
};

class IBuffer {
public:
  IBuffer(const unsigned smid, const unsigned num_warps);

  /// The `allKernelsWarpID` uniquely identifies each warp across all
  /// kernels within the application.  For an application with multiple
  /// kernels (e.g., 100 kernels), each containing warps numbered from
  /// 0 to 10, the `allKernelsWarpID` ranges from 0 to 1000, which pro-
  /// vides a global unique identifier for every warp.
  inline bool is_empty(const unsigned allKernelsWarpID) const {
    return m_ibuffer[allKernelsWarpID].empty();
  }

  inline bool has_free_slot(const unsigned allKernelsWarpID) const {
    return m_ibuffer[allKernelsWarpID].size() < 2;
  }

  inline bool is_not_empty(const unsigned allKernelsWarpID) const {
    return !is_empty(allKernelsWarpID);
  }

  void push_back(const unsigned allKernelsWarpID, ibuffer_entry entry) {
    m_ibuffer[allKernelsWarpID].push_back(entry);
  }

  /// TODO: Using double-ended queues instead of vectors may speed up
  /// the `pop_front` function, because std::d eque supports efficient
  /// header deletion operations.
  ibuffer_entry pop_front(const unsigned allKernelsWarpID) {
    ibuffer_entry entry = std::move(m_ibuffer[allKernelsWarpID].front());
    m_ibuffer[allKernelsWarpID].erase(
        m_ibuffer[allKernelsWarpID].begin());
    return entry;
  }

  /// TODO: Merging `pop_front` and `front` to reduce the overhead of
  /// duplicate moves.
  inline const ibuffer_entry& front(const unsigned allKernelsWarpID) const {
    return m_ibuffer[allKernelsWarpID].front();
  }

  /// Return the size of `m_ibuffer`.
  inline std::size_t size() const { return m_ibuffer.size(); }

  void print_ibuffer() const;
  void print_ibuffer(const unsigned gwarp_id) const;
  void print_ibuffer(const unsigned gwarp_start, const unsigned gwarp_end) const;

private:
  unsigned m_smid;
  std::vector<std::vector<ibuffer_entry>> m_ibuffer;
  unsigned m_num_warps;
};

#endif
