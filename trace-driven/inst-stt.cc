#include <bits/stdc++.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <string>
#include <time.h>
#include <vector>

#include "../ISA-Def/accelwattch_component_mapping.h"
#include "../ISA-Def/ampere_opcode.h"
#include "../ISA-Def/kepler_opcode.h"
#include "../ISA-Def/pascal_opcode.h"
#include "../ISA-Def/trace_opcode.h"
#include "../ISA-Def/turing_opcode.h"
#include "../ISA-Def/volta_opcode.h"
#include "inst-stt.h"

inst_stt::inst_stt() {

  fetch_stage = false;

  wr_bk_stage = false;

  warp_exit_stage = false;
}
