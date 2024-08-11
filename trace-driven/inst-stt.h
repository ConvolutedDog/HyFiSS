#include <assert.h>
#include <bitset>
#include <list>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#ifndef INST_STT_H
#define INST_STT_H

typedef bool inst_stage_t;

class inst_stt {
public:
  inst_stt();

private:
  inst_stage_t fetch_stage;

  inst_stage_t wr_bk_stage;
  inst_stage_t warp_exit_stage;
};

class mem_stat_t {
public:
  mem_stat_t();
};

struct SM_computation_instance {};

#endif