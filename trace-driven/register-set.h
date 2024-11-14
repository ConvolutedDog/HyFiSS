#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../hw-parser/hw-parser.h"
#include "entry.h"

#ifndef REGISTER_SET_H
#define REGISTER_SET_H

class register_set {
public:
  register_set(){};

  register_set(const unsigned num, const std::string name, hw_config *hw_cfg) {

    regs.reserve(num);
    for (unsigned i = 0; i < num; i++) {
      regs.push_back(new inst_fetch_buffer_entry());
    }

    m_name = name;
    m_hw_cfg = hw_cfg;
  }

  const std::string get_name() { return m_name; }

  bool has_free() {
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->m_valid == false) {
        return true;
      }
    }
    return false;
  }
  void release_register_set() {

    for (auto ptr : regs) {

      delete ptr;
      ptr = NULL;
    }
    regs.clear();
  }

  bool has_free(bool sub_core_model, unsigned reg_id) {

    if (!sub_core_model)
      return has_free();

    assert(reg_id < regs.size());
    return (regs[reg_id]->m_valid == false);
  }

  bool has_ready() {
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->m_valid) {
        return true;
      }
    }
    return false;
  }

  bool has_ready(bool sub_core_model, unsigned reg_id) {
    if (!sub_core_model)
      return has_ready();
    assert(reg_id < regs.size());
    return (regs[reg_id]->m_valid);
  }

  unsigned get_ready_reg_id() {

    assert(has_ready());
    inst_fetch_buffer_entry **ready;
    ready = NULL;
    unsigned reg_id = 0;
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->m_valid) {
        if (ready and (*ready)->uid < regs[i]->uid) {

        } else {
          ready = &regs[i];
          reg_id = i;
        }
      }
    }
    return reg_id;
  }
  unsigned get_schd_id(unsigned reg_id) {
    assert(regs[reg_id]->m_valid);
    return (unsigned)(regs[reg_id]->wid % m_hw_cfg->get_num_sched_per_sm());
  }

  void move_warp_newalloc_src(inst_fetch_buffer_entry *&dest,
                              inst_fetch_buffer_entry *&src) {
    dest->pc = src->pc;
    dest->wid = src->wid;
    dest->kid = src->kid;
    dest->uid = src->uid;
    dest->m_valid = true;
  }

  void move_warp(inst_fetch_buffer_entry *&dest,
                 inst_fetch_buffer_entry *&src) {
    *dest = std::move(*src);
    src->m_valid = false;
    dest->m_valid = true;
  }

  void move_in(inst_fetch_buffer_entry *&src) {
    inst_fetch_buffer_entry **free = get_free();
    inst_fetch_buffer_entry *tmp = *free;
    move_warp(tmp, src);
  }

  void move_in(bool sub_core_model, unsigned reg_id,
               inst_fetch_buffer_entry *&src) {
    inst_fetch_buffer_entry *free;

    if (!sub_core_model) {
      free = *(get_free());
    } else {
      assert(reg_id < regs.size());
      free = get_free_addr(sub_core_model, reg_id);
    }

    if (free != NULL) {

      inst_fetch_buffer_entry *tmp = free;

      move_warp_newalloc_src(tmp, src);
    }
  }

  void move_out_to(inst_fetch_buffer_entry *&dest) {
    inst_fetch_buffer_entry **ready = get_ready();
    move_warp(dest, *ready);
    (*ready)->m_valid = false;
  }

  void move_out_to(bool sub_core_model, unsigned reg_id,
                   inst_fetch_buffer_entry *&dest) {
    if (!sub_core_model) {
      return move_out_to(dest);
    }
    inst_fetch_buffer_entry **ready = get_ready(sub_core_model, reg_id);

    assert(ready != NULL);
    move_warp(dest, *ready);
  }

  inst_fetch_buffer_entry **get_ready() {
    inst_fetch_buffer_entry **ready;
    ready = NULL;
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->m_valid) {
        if (ready and (*ready)->uid < regs[i]->uid) {

        } else {
          ready = &regs[i];
        }
      }
    }
    return ready;
  }
  inst_fetch_buffer_entry **
  get_ready(std::vector<inst_fetch_buffer_entry> *except_regs) {
    inst_fetch_buffer_entry **ready;
    ready = NULL;
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->m_valid) {
        if (ready and (*ready)->uid < regs[i]->uid) {

        } else {
          ready = &regs[i];
        }

        if (ready && except_regs != NULL) {
          bool is_except = false;
          for (auto except_reg : *except_regs) {
            if (except_reg.uid == regs[i]->uid &&
                except_reg.wid == regs[i]->wid &&
                except_reg.kid == regs[i]->kid) {
              is_except = true;
              break;
            }
          }
          if (is_except) {
            ready = NULL;
          }
        }
      }
    }
    return ready;
  }

  inst_fetch_buffer_entry **get_ready(bool sub_core_model, unsigned reg_id) {
    if (!sub_core_model)
      return get_ready();
    inst_fetch_buffer_entry **ready;
    ready = NULL;
    assert(reg_id < regs.size());
    if (regs[reg_id]->m_valid)
      ready = &regs[reg_id];

    return ready;
  }

  void print() const {
    std::cout << "    " << m_name << " : @ " << this << std::endl;
    for (unsigned i = 0; i < regs.size(); i++) {
      std::cout << "     ";
      if (regs[i]->m_valid) {

        std::cout << "    valid: ";
        std::cout << "pc: " << regs[i]->pc << ", wid: " << regs[i]->wid
                  << ", kid: " << regs[i]->kid << ", uid: " << regs[i]->uid;

      } else {

        std::cout << "    novalid      ";
      }
      std::cout << std::endl;
    }
  }

  inst_fetch_buffer_entry **get_free() {
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->m_valid == false) {
        return &regs[i];
      }
    }
    return NULL;
  }

  inst_fetch_buffer_entry **get_free(bool sub_core_model, unsigned reg_id) {

    if (!sub_core_model)
      return get_free();

    assert(reg_id < regs.size());
    if (regs[reg_id]->m_valid == false) {
      return &regs[reg_id];
    }
    return NULL;
  }
  inst_fetch_buffer_entry *get_free_addr(bool sub_core_model, unsigned reg_id) {

    if (!sub_core_model)
      return *(get_free());

    assert(reg_id < regs.size());
    if (regs[reg_id]->m_valid == false) {

      return regs[reg_id];
    }
    return NULL;
  }

  unsigned get_size() { return regs.size(); }
  void print_regs(unsigned reg_id) {
    std::cout << "    pipeline_reg[" << m_name << "] : @ " << this << std::endl;
    std::cout << "     ";
    if (regs[reg_id]->m_valid) {
      std::cout << "    valid: ";
      std::cout << "pc: " << regs[reg_id]->pc << ", wid: " << regs[reg_id]->wid
                << ", kid: " << regs[reg_id]->kid
                << ", uid: " << regs[reg_id]->uid;
    } else {
      std::cout << "    novalid      ";
    }
    std::cout << std::endl;
  }
  std::vector<unsigned> get_ready_reg_ids() {
    std::vector<unsigned> ready_reg_ids;
    for (unsigned i = 0; i < regs.size(); i++) {
      if (regs[i]->m_valid) {
        ready_reg_ids.push_back(i);
      }
    }
    return ready_reg_ids;
  }
  unsigned get_kid(unsigned reg_id) {
    assert(regs[reg_id]->m_valid);
    return regs[reg_id]->kid;
  }
  unsigned get_wid(unsigned reg_id) {
    assert(regs[reg_id]->m_valid);
    return regs[reg_id]->wid;
  }
  unsigned get_uid(unsigned reg_id) {
    assert(regs[reg_id]->m_valid);
    return regs[reg_id]->uid;
  }
  unsigned get_pc(unsigned reg_id) {
    assert(regs[reg_id]->m_valid);
    return regs[reg_id]->pc;
  }
  unsigned get_latency(unsigned reg_id) {
    assert(regs[reg_id]->m_valid);
    return regs[reg_id]->latency;
  }
  void set_latency(unsigned latency, unsigned reg_id) {
    assert(regs[reg_id]->m_valid);
    regs[reg_id]->latency = latency;
  }
  void set_initial_interval(unsigned initial_interval, unsigned reg_id) {
    assert(regs[reg_id]->m_valid);
    regs[reg_id]->initial_interval = initial_interval;
    regs[reg_id]->initial_interval_dec_counter = initial_interval;
  }

private:
  std::vector<inst_fetch_buffer_entry *> regs;

  std::string m_name;
  hw_config *m_hw_cfg;
};

#endif
