#ifndef ENTRY_H
#define ENTRY_H

struct inst_fetch_buffer_entry {
  inst_fetch_buffer_entry()
    : pc(0), wid(0), kid(0), uid(0), m_valid(false), 
      latency(-1), initial_interval(-1), 
      initial_interval_dec_counter(0) {}

  inst_fetch_buffer_entry(unsigned _pc, unsigned _wid,
                          unsigned _kid, unsigned _uid)
    : pc(_pc), wid(_wid), kid(_kid), uid(_uid),
      m_valid(true), latency(-1), initial_interval(0),
      initial_interval_dec_counter(0) {}

  // inst_fetch_buffer_entry(inst_fetch_buffer_entry&& other) noexcept 
  //   : pc(other.pc), wid(other.wid), kid(other.kid), uid(other.uid),
  //     m_valid(other.m_valid), latency(other.latency),
  //     initial_interval(other.initial_interval), 
  //     initial_interval_dec_counter(other.initial_interval_dec_counter) {
  //   other.m_valid = false;
  // }

  void set_latency(unsigned _latency) { latency = _latency; }
  void set_initial_interval(unsigned _initial_interval) {
    initial_interval = _initial_interval;
    initial_interval_dec_counter = _initial_interval;
  }

  unsigned pc;
  unsigned wid;
  unsigned kid;
  unsigned uid;
  bool m_valid;
  unsigned latency;
  unsigned initial_interval;
  unsigned initial_interval_dec_counter;
};

struct curr_instn_id_per_warp_entry {
  curr_instn_id_per_warp_entry() {
    kid = 0;
    block_id = 0;
    warp_id = 0;
  };
  curr_instn_id_per_warp_entry(unsigned _kid, unsigned _block_id,
                               unsigned _warp_id) {
    kid = _kid;
    block_id = _block_id;
    warp_id = _warp_id;
  };
  unsigned kid;
  unsigned block_id;
  unsigned warp_id;
};

bool operator<(const curr_instn_id_per_warp_entry &lhs,
               const curr_instn_id_per_warp_entry &rhs);

#endif