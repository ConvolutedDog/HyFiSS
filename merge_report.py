#!/bin/python
# -*- coding: utf-8 -*-
'''
This script is used to combine the results of parallel simulations 
of multiple processes into one, and output the performance of the 
entire program simulation.
'''

import os
import re
import glob
import argparse

# ========================================================

SMs_num = 80  # SM70
DEBUG = 0

# ========================================================
Thread_block_limit_SM = None
Thread_block_limit_registers = None
Thread_block_limit_shared_memory = None
Thread_block_limit_warps = None
Theoretical_max_active_warps_per_SM = None
Theoretical_occupancy = None

Unified_L1_cache_hit_rate = [None for _ in range(SMs_num)]
Unified_L1_cache_requests = [None for _ in range(SMs_num)]

L2_cache_hit_rate = None
L2_cache_requests = None

GMEM_read_requests = [None for _ in range(SMs_num)]
GMEM_write_requests = [None for _ in range(SMs_num)]
GMEM_total_requests = [None for _ in range(SMs_num)]
GMEM_read_transactions = [None for _ in range(SMs_num)]
GMEM_write_transactions = [None for _ in range(SMs_num)]
GMEM_total_transactions = [None for _ in range(SMs_num)]

Number_of_read_transactions_per_read_requests = [None for _ in range(SMs_num)]
Number_of_write_transactions_per_write_requests = [
    None for _ in range(SMs_num)
]

L2_read_transactions = [None for _ in range(SMs_num)]
L2_write_transactions = [None for _ in range(SMs_num)]
L2_total_transactions = [None for _ in range(SMs_num)]

DRAM_total_transactions = None

Total_number_of_global_atomic_requests = [None for _ in range(SMs_num)]
Total_number_of_global_reduction_requests = [None for _ in range(SMs_num)]
Global_memory_atomic_and_reduction_transactions = [
    None for _ in range(SMs_num)
]

Achieved_active_warps_per_SM = [None for _ in range(SMs_num)]
Achieved_occupancy = [None for _ in range(SMs_num)]

GPU_active_cycles = [None for _ in range(SMs_num)]
SM_active_cycles = [None for _ in range(SMs_num)]

Warp_instructions_executed = [None for _ in range(SMs_num)]
Instructions_executed_per_clock_cycle_IPC = [None for _ in range(SMs_num)]
Total_instructions_executed_per_seconds = [None for _ in range(SMs_num)]

Kernel_execution_time = [None for _ in range(SMs_num)]
Simulation_time_memory_model = [None for _ in range(SMs_num)]
Simulation_time_compute_model = [None for _ in range(SMs_num)]

Compute_Structural_Stall = [None for _ in range(SMs_num)]
Compute_Data_Stall = [None for _ in range(SMs_num)]
Memory_Structural_Stall = [None for _ in range(SMs_num)]
Memory_Data_Stall = [None for _ in range(SMs_num)]
Synchronization_Stall = [None for _ in range(SMs_num)]
Control_Stall = [None for _ in range(SMs_num)]
Idle_Stall = [None for _ in range(SMs_num)]
No_Stall = [None for _ in range(SMs_num)]
Other_Stall = [None for _ in range(SMs_num)]

num_Issue_Compute_Structural_out_has_no_free_slot = [
    None for _ in range(SMs_num)
]
num_Issue_Memory_Structural_out_has_no_free_slot = [
    None for _ in range(SMs_num)
]
num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute = [
    None for _ in range(SMs_num)
]
num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory = [
    None for _ in range(SMs_num)
]
num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency = [
    None for _ in range(SMs_num)
]
num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency = [
    None for _ in range(SMs_num)
]
num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty = [
    None for _ in range(SMs_num)
]
num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty = [
    None for _ in range(SMs_num)
]
num_Writeback_Compute_Structural_bank_of_reg_is_not_idle = [
    None for _ in range(SMs_num)
]
num_Writeback_Memory_Structural_bank_of_reg_is_not_idle = [
    None for _ in range(SMs_num)
]
num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated = [
    None for _ in range(SMs_num)
]
num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated = [
    None for _ in range(SMs_num)
]
num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu = [
    None for _ in range(SMs_num)
]
num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu = [
    None for _ in range(SMs_num)
]
num_Execute_Memory_Structural_icnt_injection_buffer_is_full = [
    None for _ in range(SMs_num)
]
num_Issue_Compute_Data_scoreboard = [None for _ in range(SMs_num)]
num_Issue_Memory_Data_scoreboard = [None for _ in range(SMs_num)]
num_Execute_Memory_Data_L1 = [None for _ in range(SMs_num)]
num_Execute_Memory_Data_L2 = [None for _ in range(SMs_num)]
num_Execute_Memory_Data_Main_Memory = [None for _ in range(SMs_num)]

SP_UNIT_execute_clks_sum = [None for _ in range(SMs_num)]
SFU_UNIT_execute_clks_sum = [None for _ in range(SMs_num)]
INT_UNIT_execute_clks_sum = [None for _ in range(SMs_num)]
DP_UNIT_execute_clks_sum = [None for _ in range(SMs_num)]
TENSOR_CORE_UNIT_execute_clks_sum = [None for _ in range(SMs_num)]
LDST_UNIT_execute_clks_sum = [None for _ in range(SMs_num)]
SPEC_UNIT_1_execute_clks_sum = [None for _ in range(SMs_num)]
SPEC_UNIT_2_execute_clks_sum = [None for _ in range(SMs_num)]
SPEC_UNIT_3_execute_clks_sum = [None for _ in range(SMs_num)]
Other_UNIT_execute_clks_sum = [None for _ in range(SMs_num)]

SP_UNIT_Instns_num = [None for _ in range(SMs_num)]
SFU_UNIT_Instns_num = [None for _ in range(SMs_num)]
INT_UNIT_Instns_num = [None for _ in range(SMs_num)]
DP_UNIT_Instns_num = [None for _ in range(SMs_num)]
TENSOR_CORE_UNIT_Instns_num = [None for _ in range(SMs_num)]
LDST_UNIT_Instns_num = [None for _ in range(SMs_num)]
SPEC_UNIT_1_Instns_num = [None for _ in range(SMs_num)]
SPEC_UNIT_2_Instns_num = [None for _ in range(SMs_num)]
SPEC_UNIT_3_Instns_num = [None for _ in range(SMs_num)]
Other_UNIT_Instns_num = [None for _ in range(SMs_num)]

# ========================================================
parser = argparse.ArgumentParser(description='Merge rank reports.')

parser.add_argument('--dir',
                    type=str,
                    required=True,
                    help='The directory of rank reports')
parser.add_argument('--kernel_id',
                    type=int,
                    required=True,
                    help='The kernel_id of reports')
parser.add_argument('--np',
                    type=int,
                    required=True,
                    help='The number of processes')
args = parser.parse_args()
# ========================================================

# Suppose that the kernel number and all ranks are sequential,
# and we know the kernel number.
kernel_id = args.kernel_id
reports_dir = args.dir
reports_dir = os.path.abspath(reports_dir)
np = args.np

# print(reports_dir)

file_name_template = reports_dir + "/" + r"kernel-" + str(
    kernel_id) + "-rank-*.temp.txt"

# Get all the relevant documents.
files = glob.glob(file_name_template)

# ========================================================
# Parse all rank numbers to find out the maximum value.
rank_nums = [
    int(re.search('-rank-(\d+).temp.txt', file).group(1)) for file in files
]
all_ranks_num = max(rank_nums) + 1

all_ranks_num = int(np)

print("Processes number: ", all_ranks_num)
# ========================================================

# Process each file.
for file in files:
    if int(re.search('-rank-(\d+).temp.txt', file).group(1)) >= all_ranks_num:
        continue

    # print("Processing ", reports_dir + "/" + file.split("/")[-1], "...")
    with open(file, 'r') as f:
        content = f.read()
    # print(content)
    content = content.replace("-nan", "0")

    # Extract rank_num.
    match = re.search('-rank-(\d+).temp.txt', file)
    if match:
        rank_num = int(match.group(1))
        # print("Current rank: ", rank_num)
    # ========================================================
    # Define the regular expression for which we want to extract information.
    patterns = {
        'Thread_block_limit_SM': r"Thread_block_limit_SM = (\d+)",
        'Thread_block_limit_registers':
        r"Thread_block_limit_registers = (\d+)",
        'Thread_block_limit_shared_memory':
        r"Thread_block_limit_shared_memory = (\d+)",
        'Thread_block_limit_warps': r"Thread_block_limit_warps = (\d+)",
        'Theoretical_max_active_warps_per_SM':
        r"Theoretical_max_active_warps_per_SM = (\d+)",
        'Theoretical_occupancy': r"Theoretical_occupancy = (\d+)",
        'L2_cache_requests': r"L2_cache_requests = (\d+)",
        'DRAM_total_transactions': r"DRAM_total_transactions = (\d+)",
    }

    # Loop through the pattern dictionary to extract the corresponding value for each one.
    if rank_num == 0:
        extracted_values = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                extracted_values[key] = int(match.group(1))
            else:
                extracted_values[key] = None
                print("No match found for ", key)
                exit(0)

        for key, value in extracted_values.items():
            # print(f"{key}: {value}")
            if key == "Thread_block_limit_SM":
                Thread_block_limit_SM = value
            elif key == "Thread_block_limit_registers":
                Thread_block_limit_registers = value
            elif key == "Thread_block_limit_shared_memory":
                Thread_block_limit_shared_memory = value
            elif key == "Thread_block_limit_warps":
                Thread_block_limit_warps = value
            elif key == "Theoretical_max_active_warps_per_SM":
                Theoretical_max_active_warps_per_SM = value
            elif key == "Theoretical_occupancy":
                Theoretical_occupancy = value
            elif key == "L2_cache_requests":
                L2_cache_requests = value
            elif key == "DRAM_total_transactions":
                DRAM_total_transactions = value
            else:
                print("Error: Unknown key")
                exit(0)
    if DEBUG:
        print("Thread_block_limit_SM: ", Thread_block_limit_SM)
        print("Thread_block_limit_registers: ", Thread_block_limit_registers)
        print("Thread_block_limit_shared_memory: ",
              Thread_block_limit_shared_memory)
        print("Thread_block_limit_warps: ", Thread_block_limit_warps)
        print("Theoretical_max_active_warps_per_SM: ",
              Theoretical_max_active_warps_per_SM)
        print("Theoretical_occupancy: ", Theoretical_occupancy)
        print("L2_cache_requests: ", L2_cache_requests)
        print("DRAM_total_transactions: ", DRAM_total_transactions)

    # ========================================================

    patterns = {
        'L2_cache_hit_rate': r"L2_cache_hit_rate = ([\d\s.]+)",
    }

    if rank_num == 0:
        extracted_values = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                extracted_values[key] = float(match.group(1))
            else:
                extracted_values[key] = None
                print("No match found for ", key)
                exit(0)

        for key, value in extracted_values.items():
            # print(f"{key}: {value}")
            if key == "L2_cache_hit_rate":
                L2_cache_hit_rate = value
            else:
                print("Error: Unknown key")
                exit(0)

    if DEBUG:
        print("L2_cache_hit_rate: ", L2_cache_hit_rate)

    # ========================================================

    pattern = r"Unified_L1_cache_hit_rate\[\]: ([\d\s.]+)"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        # print(numbers_str)
        numbers = [float(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for Unified_L1_cache_hit_rate[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process.
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if Unified_L1_cache_hit_rate[curr_process_idx] is not None:
                print(
                    f"Error: Unified_L1_cache_hit_rate[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                Unified_L1_cache_hit_rate[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("Unified_L1_cache_hit_rate: ", Unified_L1_cache_hit_rate)

    # ========================================================

    pattern = r"Unified_L1_cache_requests\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for Unified_L1_cache_requests[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if Unified_L1_cache_requests[curr_process_idx] is not None:
                print(
                    f"Error: Unified_L1_cache_requests[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                Unified_L1_cache_requests[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("Unified_L1_cache_requests: ", Unified_L1_cache_requests)

    # ========================================================

    pattern = r"GMEM_read_requests\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for GMEM_read_requests[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if GMEM_read_requests[curr_process_idx] is not None:
                print(
                    f"Error: GMEM_read_requests[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                GMEM_read_requests[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("GMEM_read_requests: ", GMEM_read_requests)

    # ========================================================

    pattern = r"GMEM_write_requests\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for GMEM_write_requests[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if GMEM_write_requests[curr_process_idx] is not None:
                print(
                    f"Error: GMEM_write_requests[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                GMEM_write_requests[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("GMEM_write_requests: ", GMEM_write_requests)

    # ========================================================

    pattern = r"GMEM_total_requests\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for GMEM_total_requests[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if GMEM_total_requests[curr_process_idx] is not None:
                print(
                    f"Error: GMEM_total_requests[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                GMEM_total_requests[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("GMEM_total_requests: ", GMEM_total_requests)

    # ========================================================

    pattern = r"GMEM_read_transactions\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for GMEM_read_transactions[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if GMEM_read_transactions[curr_process_idx] is not None:
                print(
                    f"Error: GMEM_read_transactions[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                GMEM_read_transactions[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("GMEM_read_transactions: ", GMEM_read_transactions)

    # ========================================================

    pattern = r"GMEM_write_transactions\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for GMEM_write_transactions[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if GMEM_write_transactions[curr_process_idx] is not None:
                print(
                    f"Error: GMEM_write_transactions[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                GMEM_write_transactions[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("GMEM_write_transactions: ", GMEM_write_transactions)

    # ========================================================

    pattern = r"GMEM_total_transactions\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for GMEM_total_transactions[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if GMEM_total_transactions[curr_process_idx] is not None:
                print(
                    f"Error: GMEM_total_transactions[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                GMEM_total_transactions[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("GMEM_total_transactions: ", GMEM_total_transactions)

    # ========================================================

    pattern = r"Number_of_read_transactions_per_read_requests\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [float(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print(
            "No match found for Number_of_read_transactions_per_read_requests[]"
        )
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if Number_of_read_transactions_per_read_requests[
                    curr_process_idx] is not None:
                print(
                    f"Error: Number_of_read_transactions_per_read_requests[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                Number_of_read_transactions_per_read_requests[
                    curr_process_idx] = numbers[curr_process_idx]
    if DEBUG:
        print("Number_of_read_transactions_per_read_requests: ",
              Number_of_read_transactions_per_read_requests)

    # ========================================================

    pattern = r"Number_of_write_transactions_per_write_requests\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [float(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print(
            "No match found for Number_of_write_transactions_per_write_requests[]"
        )
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if Number_of_write_transactions_per_write_requests[
                    curr_process_idx] is not None:
                print(
                    f"Error: Number_of_write_transactions_per_write_requests[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                Number_of_write_transactions_per_write_requests[
                    curr_process_idx] = numbers[curr_process_idx]
    if DEBUG:
        print("Number_of_write_transactions_per_write_requests: ",
              Number_of_write_transactions_per_write_requests)

    # ========================================================

    pattern = r"L2_read_transactions\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for L2_read_transactions[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if L2_read_transactions[curr_process_idx] is not None:
                print(
                    f"Error: L2_read_transactions[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                L2_read_transactions[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("L2_read_transactions: ", L2_read_transactions)

    # ========================================================

    pattern = r"L2_write_transactions\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for L2_write_transactions[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if L2_write_transactions[curr_process_idx] is not None:
                print(
                    f"Error: L2_write_transactions[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                L2_write_transactions[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("L2_write_transactions: ", L2_write_transactions)

    # ========================================================

    pattern = r"L2_total_transactions\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for L2_total_transactions[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if L2_total_transactions[curr_process_idx] is not None:
                print(
                    f"Error: L2_total_transactions[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                L2_total_transactions[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("L2_total_transactions: ", L2_total_transactions)

    # ========================================================

    pattern = r"Total_number_of_global_atomic_requests\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for Total_number_of_global_atomic_requests[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if Total_number_of_global_atomic_requests[
                    curr_process_idx] is not None:
                print(
                    f"Error: Total_number_of_global_atomic_requests[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                Total_number_of_global_atomic_requests[
                    curr_process_idx] = numbers[curr_process_idx]
    if DEBUG:
        print("Total_number_of_global_atomic_requests: ",
              Total_number_of_global_atomic_requests)

    # ========================================================

    pattern = r"Total_number_of_global_reduction_requests\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for Total_number_of_global_reduction_requests[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if Total_number_of_global_reduction_requests[
                    curr_process_idx] is not None:
                print(
                    f"Error: Total_number_of_global_reduction_requests[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                Total_number_of_global_reduction_requests[
                    curr_process_idx] = numbers[curr_process_idx]
    if DEBUG:
        print("Total_number_of_global_reduction_requests: ",
              Total_number_of_global_reduction_requests)

    # ========================================================

    pattern = r"Global_memory_atomic_and_reduction_transactions\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print(
            "No match found for Global_memory_atomic_and_reduction_transactions[]"
        )
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if Global_memory_atomic_and_reduction_transactions[
                    curr_process_idx] is not None:
                print(
                    f"Error: Global_memory_atomic_and_reduction_transactions[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                Global_memory_atomic_and_reduction_transactions[
                    curr_process_idx] = numbers[curr_process_idx]
    if DEBUG:
        print("Global_memory_atomic_and_reduction_transactions: ",
              Global_memory_atomic_and_reduction_transactions)

    # ========================================================

    pattern = r"Achieved_active_warps_per_SM\[\]: ([\d\s.]+)"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [float(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for Achieved_active_warps_per_SM[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if Achieved_active_warps_per_SM[curr_process_idx] is not None:
                print(
                    f"Error: Achieved_active_warps_per_SM[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                Achieved_active_warps_per_SM[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("Achieved_active_warps_per_SM: ", Achieved_active_warps_per_SM)

    # ========================================================

    pattern = r"Achieved_occupancy\[\]: ([\d\s.]+)"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [float(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for Achieved_occupancy[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if Achieved_occupancy[curr_process_idx] is not None:
                print(
                    f"Error: Achieved_occupancy[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                Achieved_occupancy[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("Achieved_occupancy: ", Achieved_occupancy)

    # ========================================================

    pattern = r"GPU_active_cycles\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for GPU_active_cycles[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if GPU_active_cycles[curr_process_idx] is not None:
                print(
                    f"Error: GPU_active_cycles[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                GPU_active_cycles[curr_process_idx] = numbers[curr_process_idx]
    if DEBUG:
        print("GPU_active_cycles: ", GPU_active_cycles)

    # ========================================================

    pattern = r"SM_active_cycles\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for SM_active_cycles[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if SM_active_cycles[curr_process_idx] is not None:
                print(
                    f"Error: SM_active_cycles[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                SM_active_cycles[curr_process_idx] = numbers[curr_process_idx]
    if DEBUG:
        print("SM_active_cycles: ", SM_active_cycles)

    # ========================================================

    pattern = r"Warp_instructions_executed\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for Warp_instructions_executed[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if Warp_instructions_executed[curr_process_idx] is not None:
                print(
                    f"Error: Warp_instructions_executed[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                Warp_instructions_executed[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("Warp_instructions_executed: ", Warp_instructions_executed)

    # ========================================================

    pattern = r"Instructions_executed_per_clock_cycle_IPC\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [float(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for Instructions_executed_per_clock_cycle_IPC[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if Instructions_executed_per_clock_cycle_IPC[
                    curr_process_idx] is not None:
                print(
                    f"Error: Instructions_executed_per_clock_cycle_IPC[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                Instructions_executed_per_clock_cycle_IPC[
                    curr_process_idx] = numbers[curr_process_idx]
    if DEBUG:
        print("Instructions_executed_per_clock_cycle_IPC: ",
              Instructions_executed_per_clock_cycle_IPC)

    # ========================================================

    pattern = r"Total_instructions_executed_per_seconds\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [float(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for Total_instructions_executed_per_seconds[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if Total_instructions_executed_per_seconds[
                    curr_process_idx] is not None:
                print(
                    f"Error: Total_instructions_executed_per_seconds[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                Total_instructions_executed_per_seconds[
                    curr_process_idx] = numbers[curr_process_idx]
    if DEBUG:
        print("Total_instructions_executed_per_seconds: ",
              Total_instructions_executed_per_seconds)

    # ========================================================

    pattern = r"Kernel_execution_time\[\]: ((?:\d+ )+(?:\d+))"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [int(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for Kernel_execution_time[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if Kernel_execution_time[curr_process_idx] is not None:
                print(
                    f"Error: Kernel_execution_time[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                Kernel_execution_time[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("Kernel_execution_time: ", Kernel_execution_time)

    # ========================================================

    pattern = r"Simulation_time_memory_model\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [float(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for Simulation_time_memory_model[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if Simulation_time_memory_model[curr_process_idx] is not None:
                print(
                    f"Error: Simulation_time_memory_model[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                Simulation_time_memory_model[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("Simulation_time_memory_model: ", Simulation_time_memory_model)

    # ========================================================

    pattern = r"Simulation_time_compute_model\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
    match = re.search(pattern, content)

    if match:
        numbers_str = match.group(1).strip()
        numbers = [float(n) for n in numbers_str.split()]
        # print(numbers)
    else:
        print("No match found for Simulation_time_compute_model[]")
        exit(0)

    # process_idx calculates the index corresponding to the current file/process..
    for pass_num in range(int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
        curr_process_idx = rank_num + pass_num * all_ranks_num
        if curr_process_idx < SMs_num:
            if Simulation_time_compute_model[curr_process_idx] is not None:
                print(
                    f"Error: Simulation_time_compute_model[{curr_process_idx}] is already set"
                )
                exit(0)
            else:
                # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                Simulation_time_compute_model[curr_process_idx] = numbers[
                    curr_process_idx]
    if DEBUG:
        print("Simulation_time_compute_model: ", Simulation_time_compute_model)

    try:
        # ========================================================
        pattern = r"Compute_Structural_Stall\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for Compute_Structural_Stall[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if Compute_Structural_Stall[curr_process_idx] is not None:
                    print(
                        f"Error: Compute_Structural_Stall[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    Compute_Structural_Stall[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("Compute_Structural_Stall: ", Compute_Structural_Stall)
        # ========================================================
        pattern = r"Compute_Data_Stall\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for Compute_Data_Stall[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if Compute_Data_Stall[curr_process_idx] is not None:
                    print(
                        f"Error: Compute_Data_Stall[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    Compute_Data_Stall[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("Compute_Data_Stall: ", Compute_Data_Stall)
        # ========================================================
        pattern = r"Memory_Structural_Stall\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for Memory_Structural_Stall[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if Memory_Structural_Stall[curr_process_idx] is not None:
                    print(
                        f"Error: Memory_Structural_Stall[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    Memory_Structural_Stall[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("Memory_Structural_Stall: ", Memory_Structural_Stall)
        # ========================================================
        pattern = r"Memory_Data_Stall\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for Memory_Data_Stall[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if Memory_Data_Stall[curr_process_idx] is not None:
                    print(
                        f"Error: Memory_Data_Stall[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    Memory_Data_Stall[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("Memory_Data_Stall: ", Memory_Data_Stall)
        # ========================================================
        pattern = r"Synchronization_Stall\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for Synchronization_Stall[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if Synchronization_Stall[curr_process_idx] is not None:
                    print(
                        f"Error: Synchronization_Stall[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    Synchronization_Stall[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("Synchronization_Stall: ", Synchronization_Stall)
        # ========================================================
        pattern = r"Control_Stall\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for Control_Stall[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if Control_Stall[curr_process_idx] is not None:
                    print(
                        f"Error: Control_Stall[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    Control_Stall[curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print("Control_Stall: ", Control_Stall)
        # ========================================================
        pattern = r"Idle_Stall\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for Idle_Stall[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if Idle_Stall[curr_process_idx] is not None:
                    print(
                        f"Error: Idle_Stall[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    Idle_Stall[curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print("Idle_Stall: ", Idle_Stall)
        # ========================================================
        pattern = r"No_Stall\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for No_Stall[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if No_Stall[curr_process_idx] is not None:
                    print(
                        f"Error: No_Stall[{curr_process_idx}] is already set")
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    No_Stall[curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print("No_Stall: ", No_Stall)
        # ========================================================
        pattern = r"Other_Stall\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for Other_Stall[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if Other_Stall[curr_process_idx] is not None:
                    print(
                        f"Error: Other_Stall[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    Other_Stall[curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print("Other_Stall: ", Other_Stall)
        # ========================================================
        pattern = r"num_Issue_Compute_Structural_out_has_no_free_slot\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_Issue_Compute_Structural_out_has_no_free_slot[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_Issue_Compute_Structural_out_has_no_free_slot[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_Issue_Compute_Structural_out_has_no_free_slot[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_Issue_Compute_Structural_out_has_no_free_slot[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print("num_Issue_Compute_Structural_out_has_no_free_slot: ",
                  num_Issue_Compute_Structural_out_has_no_free_slot)
        # ========================================================
        pattern = r"num_Issue_Memory_Structural_out_has_no_free_slot\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_Issue_Memory_Structural_out_has_no_free_slot[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_Issue_Memory_Structural_out_has_no_free_slot[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_Issue_Memory_Structural_out_has_no_free_slot[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_Issue_Memory_Structural_out_has_no_free_slot[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print("num_Issue_Memory_Structural_out_has_no_free_slot: ",
                  num_Issue_Memory_Structural_out_has_no_free_slot)
        # ========================================================
        pattern = r"num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print(
                "num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute: ",
                num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute
            )
        # ========================================================
        pattern = r"num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print(
                "num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory: ",
                num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory
            )
        # ========================================================
        pattern = r"num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print(
                "num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency: ",
                num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency
            )
        # ========================================================
        pattern = r"num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print(
                "num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency: ",
                num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency
            )
        # ========================================================
        pattern = r"num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print(
                "num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty: ",
                num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty
            )
        # ========================================================
        pattern = r"num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print(
                "num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty: ",
                num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty
            )
        # ========================================================
        pattern = r"num_Writeback_Compute_Structural_bank_of_reg_is_not_idle\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_Writeback_Compute_Structural_bank_of_reg_is_not_idle[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_Writeback_Compute_Structural_bank_of_reg_is_not_idle[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_Writeback_Compute_Structural_bank_of_reg_is_not_idle[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_Writeback_Compute_Structural_bank_of_reg_is_not_idle[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print("num_Writeback_Compute_Structural_bank_of_reg_is_not_idle: ",
                  num_Writeback_Compute_Structural_bank_of_reg_is_not_idle)
        # ========================================================
        pattern = r"num_Writeback_Memory_Structural_bank_of_reg_is_not_idle\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_Writeback_Memory_Structural_bank_of_reg_is_not_idle[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_Writeback_Memory_Structural_bank_of_reg_is_not_idle[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_Writeback_Memory_Structural_bank_of_reg_is_not_idle[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_Writeback_Memory_Structural_bank_of_reg_is_not_idle[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print("num_Writeback_Memory_Structural_bank_of_reg_is_not_idle: ",
                  num_Writeback_Memory_Structural_bank_of_reg_is_not_idle)
        # ========================================================
        pattern = r"num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print(
                "num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated: ",
                num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated
            )
        # ========================================================
        pattern = r"num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print(
                "num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated: ",
                num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated
            )
        # ========================================================
        pattern = r"num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print(
                "num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu: ",
                num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu
            )
        # ========================================================
        pattern = r"num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print(
                "num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu: ",
                num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu
            )
        # ========================================================
        pattern = r"num_Execute_Memory_Structural_icnt_injection_buffer_is_full\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_Execute_Memory_Structural_icnt_injection_buffer_is_full[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_Execute_Memory_Structural_icnt_injection_buffer_is_full[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_Execute_Memory_Structural_icnt_injection_buffer_is_full[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_Execute_Memory_Structural_icnt_injection_buffer_is_full[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print(
                "num_Execute_Memory_Structural_icnt_injection_buffer_is_full: ",
                num_Execute_Memory_Structural_icnt_injection_buffer_is_full)
        # ========================================================
        pattern = r"num_Issue_Compute_Data_scoreboard\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_Issue_Compute_Data_scoreboard[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_Issue_Compute_Data_scoreboard[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_Issue_Compute_Data_scoreboard[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_Issue_Compute_Data_scoreboard[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print("num_Issue_Compute_Data_scoreboard: ",
                  num_Issue_Compute_Data_scoreboard)
        # ========================================================
        pattern = r"num_Issue_Memory_Data_scoreboard\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_Issue_Memory_Data_scoreboard[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_Issue_Memory_Data_scoreboard[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_Issue_Memory_Data_scoreboard[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_Issue_Memory_Data_scoreboard[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print("num_Issue_Memory_Data_scoreboard: ",
                  num_Issue_Memory_Data_scoreboard)
        # ========================================================
        pattern = r"num_Execute_Memory_Data_L1\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_Execute_Memory_Data_L1[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_Execute_Memory_Data_L1[curr_process_idx] is not None:
                    print(
                        f"Error: num_Execute_Memory_Data_L1[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_Execute_Memory_Data_L1[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("num_Execute_Memory_Data_L1: ", num_Execute_Memory_Data_L1)
        # ========================================================
        pattern = r"num_Execute_Memory_Data_L2\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_Execute_Memory_Data_L2[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_Execute_Memory_Data_L2[curr_process_idx] is not None:
                    print(
                        f"Error: num_Execute_Memory_Data_L2[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_Execute_Memory_Data_L2[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("num_Execute_Memory_Data_L2: ", num_Execute_Memory_Data_L2)
        # ========================================================
        pattern = r"num_Execute_Memory_Data_Main_Memory\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for num_Execute_Memory_Data_Main_Memory[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if num_Execute_Memory_Data_Main_Memory[
                        curr_process_idx] is not None:
                    print(
                        f"Error: num_Execute_Memory_Data_Main_Memory[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    num_Execute_Memory_Data_Main_Memory[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print("num_Execute_Memory_Data_Main_Memory: ",
                  num_Execute_Memory_Data_Main_Memory)
        # ========================================================
        pattern = r"SP_UNIT_execute_clks_sum\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for SP_UNIT_execute_clks_sum[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if SP_UNIT_execute_clks_sum[curr_process_idx] is not None:
                    print(
                        f"Error: SP_UNIT_execute_clks_sum[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    SP_UNIT_execute_clks_sum[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("SP_UNIT_execute_clks_sum: ", SP_UNIT_execute_clks_sum)
        # ========================================================
        pattern = r"SFU_UNIT_execute_clks_sum\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for SFU_UNIT_execute_clks_sum[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if SFU_UNIT_execute_clks_sum[curr_process_idx] is not None:
                    print(
                        f"Error: SFU_UNIT_execute_clks_sum[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    SFU_UNIT_execute_clks_sum[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("SFU_UNIT_execute_clks_sum: ", SFU_UNIT_execute_clks_sum)
        # ========================================================
        pattern = r"INT_UNIT_execute_clks_sum\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for INT_UNIT_execute_clks_sum[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if INT_UNIT_execute_clks_sum[curr_process_idx] is not None:
                    print(
                        f"Error: INT_UNIT_execute_clks_sum[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    INT_UNIT_execute_clks_sum[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("INT_UNIT_execute_clks_sum: ", INT_UNIT_execute_clks_sum)
        # ========================================================
        pattern = r"DP_UNIT_execute_clks_sum\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for DP_UNIT_execute_clks_sum[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if DP_UNIT_execute_clks_sum[curr_process_idx] is not None:
                    print(
                        f"Error: DP_UNIT_execute_clks_sum[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    DP_UNIT_execute_clks_sum[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("DP_UNIT_execute_clks_sum: ", DP_UNIT_execute_clks_sum)
        # ========================================================
        pattern = r"TENSOR_CORE_UNIT_execute_clks_sum\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for TENSOR_CORE_UNIT_execute_clks_sum[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if TENSOR_CORE_UNIT_execute_clks_sum[
                        curr_process_idx] is not None:
                    print(
                        f"Error: TENSOR_CORE_UNIT_execute_clks_sum[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    TENSOR_CORE_UNIT_execute_clks_sum[
                        curr_process_idx] = numbers[curr_process_idx]
        if DEBUG:
            print("TENSOR_CORE_UNIT_execute_clks_sum: ",
                  TENSOR_CORE_UNIT_execute_clks_sum)
        # ========================================================
        pattern = r"LDST_UNIT_execute_clks_sum\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for LDST_UNIT_execute_clks_sum[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if LDST_UNIT_execute_clks_sum[curr_process_idx] is not None:
                    print(
                        f"Error: LDST_UNIT_execute_clks_sum[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    LDST_UNIT_execute_clks_sum[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("LDST_UNIT_execute_clks_sum: ", LDST_UNIT_execute_clks_sum)
        # ========================================================
        pattern = r"SPEC_UNIT_1_execute_clks_sum\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for SPEC_UNIT_1_execute_clks_sum[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if SPEC_UNIT_1_execute_clks_sum[curr_process_idx] is not None:
                    print(
                        f"Error: SPEC_UNIT_1_execute_clks_sum[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    SPEC_UNIT_1_execute_clks_sum[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("SPEC_UNIT_1_execute_clks_sum: ",
                  SPEC_UNIT_1_execute_clks_sum)
        # ========================================================
        pattern = r"SPEC_UNIT_2_execute_clks_sum\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for SPEC_UNIT_2_execute_clks_sum[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if SPEC_UNIT_2_execute_clks_sum[curr_process_idx] is not None:
                    print(
                        f"Error: SPEC_UNIT_2_execute_clks_sum[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    SPEC_UNIT_2_execute_clks_sum[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("SPEC_UNIT_2_execute_clks_sum: ",
                  SPEC_UNIT_2_execute_clks_sum)
        # ========================================================
        pattern = r"SPEC_UNIT_3_execute_clks_sum\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for SPEC_UNIT_3_execute_clks_sum[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if SPEC_UNIT_3_execute_clks_sum[curr_process_idx] is not None:
                    print(
                        f"Error: SPEC_UNIT_3_execute_clks_sum[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    SPEC_UNIT_3_execute_clks_sum[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("SPEC_UNIT_3_execute_clks_sum: ",
                  SPEC_UNIT_3_execute_clks_sum)
        # ========================================================
        pattern = r"Other_UNIT_execute_clks_sum\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for Other_UNIT_execute_clks_sum[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if Other_UNIT_execute_clks_sum[curr_process_idx] is not None:
                    print(
                        f"Error: Other_UNIT_execute_clks_sum[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    Other_UNIT_execute_clks_sum[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("Other_UNIT_execute_clks_sum: ", Other_UNIT_execute_clks_sum)
        # ========================================================
        pattern = r"SP_UNIT_Instns_num\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for SP_UNIT_Instns_num[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if SP_UNIT_Instns_num[curr_process_idx] is not None:
                    print(
                        f"Error: SP_UNIT_Instns_num[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    SP_UNIT_Instns_num[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("SP_UNIT_Instns_num: ", SP_UNIT_Instns_num)
        # ========================================================
        pattern = r"SFU_UNIT_Instns_num\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for SFU_UNIT_Instns_num[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if SFU_UNIT_Instns_num[curr_process_idx] is not None:
                    print(
                        f"Error: SFU_UNIT_Instns_num[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    SFU_UNIT_Instns_num[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("SFU_UNIT_Instns_num: ", SFU_UNIT_Instns_num)
        # ========================================================
        pattern = r"INT_UNIT_Instns_num\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for INT_UNIT_Instns_num[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if INT_UNIT_Instns_num[curr_process_idx] is not None:
                    print(
                        f"Error: INT_UNIT_Instns_num[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    INT_UNIT_Instns_num[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("INT_UNIT_Instns_num: ", INT_UNIT_Instns_num)
        # ========================================================
        pattern = r"DP_UNIT_Instns_num\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for DP_UNIT_Instns_num[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if DP_UNIT_Instns_num[curr_process_idx] is not None:
                    print(
                        f"Error: DP_UNIT_Instns_num[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    DP_UNIT_Instns_num[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("DP_UNIT_Instns_num: ", DP_UNIT_Instns_num)
        # ========================================================
        pattern = r"TENSOR_CORE_UNIT_Instns_num\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for TENSOR_CORE_UNIT_Instns_num[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if TENSOR_CORE_UNIT_Instns_num[curr_process_idx] is not None:
                    print(
                        f"Error: TENSOR_CORE_UNIT_Instns_num[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    TENSOR_CORE_UNIT_Instns_num[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("TENSOR_CORE_UNIT_Instns_num: ", TENSOR_CORE_UNIT_Instns_num)
        # ========================================================
        pattern = r"LDST_UNIT_Instns_num\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for LDST_UNIT_Instns_num[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if LDST_UNIT_Instns_num[curr_process_idx] is not None:
                    print(
                        f"Error: LDST_UNIT_Instns_num[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    LDST_UNIT_Instns_num[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("LDST_UNIT_Instns_num: ", LDST_UNIT_Instns_num)
        # ========================================================
        pattern = r"SPEC_UNIT_1_Instns_num\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for SPEC_UNIT_1_Instns_num[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if SPEC_UNIT_1_Instns_num[curr_process_idx] is not None:
                    print(
                        f"Error: SPEC_UNIT_1_Instns_num[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    SPEC_UNIT_1_Instns_num[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("SPEC_UNIT_1_Instns_num: ", SPEC_UNIT_1_Instns_num)
        # ========================================================
        pattern = r"SPEC_UNIT_2_Instns_num\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for SPEC_UNIT_2_Instns_num[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if SPEC_UNIT_2_Instns_num[curr_process_idx] is not None:
                    print(
                        f"Error: SPEC_UNIT_2_Instns_num[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    SPEC_UNIT_2_Instns_num[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("SPEC_UNIT_2_Instns_num: ", SPEC_UNIT_2_Instns_num)
        # ========================================================
        pattern = r"SPEC_UNIT_3_Instns_num\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for SPEC_UNIT_3_Instns_num[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if SPEC_UNIT_3_Instns_num[curr_process_idx] is not None:
                    print(
                        f"Error: SPEC_UNIT_3_Instns_num[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    SPEC_UNIT_3_Instns_num[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("SPEC_UNIT_3_Instns_num: ", SPEC_UNIT_3_Instns_num)
        # ========================================================
        pattern = r"Other_UNIT_Instns_num\[\]: ((?:[\d.-]+(?:e-)?\d*\s*)+)"
        match = re.search(pattern, content)

        if match:
            numbers_str = match.group(1).strip()
            numbers = [float(n) for n in numbers_str.split()]
            # print(numbers)
        else:
            # print("No match found for Other_UNIT_Instns_num[]")
            exit(0)

        # process_idx calculates the index corresponding to the current file/process..
        for pass_num in range(
                int((SMs_num + all_ranks_num - 1) / all_ranks_num)):
            curr_process_idx = rank_num + pass_num * all_ranks_num
            if curr_process_idx < SMs_num:
                if Other_UNIT_Instns_num[curr_process_idx] is not None:
                    print(
                        f"Error: Other_UNIT_Instns_num[{curr_process_idx}] is already set"
                    )
                    exit(0)
                else:
                    # print("numbers[{curr_process_idx}]: ", curr_process_idx, numbers[curr_process_idx])
                    Other_UNIT_Instns_num[curr_process_idx] = numbers[
                        curr_process_idx]
        if DEBUG:
            print("Other_UNIT_Instns_num: ", Other_UNIT_Instns_num)
        # ========================================================

    except:
        pass

# ========================================================
Unified_L1_cache_hit_rate_summary = 0.0
Unified_L1_cache_requests_summary = 0

for item in Unified_L1_cache_requests:
    if item is not None:
        Unified_L1_cache_requests_summary += item

Unified_L1_cache_hit_requests = 0

for i in range(SMs_num):
    if Unified_L1_cache_hit_rate[i] is not None and Unified_L1_cache_requests[
            i] is not None:
        Unified_L1_cache_hit_requests += Unified_L1_cache_hit_rate[
            i] * Unified_L1_cache_requests[i]

if Unified_L1_cache_requests_summary != 0:
    Unified_L1_cache_hit_rate_summary = Unified_L1_cache_hit_requests / Unified_L1_cache_requests_summary
# ========================================================
GMEM_read_requests_summary = 0
GMEM_write_requests_summary = 0
GMEM_total_requests_summary = 0
GMEM_read_transactions_summary = 0
GMEM_write_transactions_summary = 0
GMEM_total_transactions_summary = 0

for i in range(SMs_num):
    if GMEM_read_requests[i] is not None:
        GMEM_read_requests_summary += GMEM_read_requests[i]
    if GMEM_write_requests[i] is not None:
        GMEM_write_requests_summary += GMEM_write_requests[i]
    if GMEM_total_requests[i] is not None:
        GMEM_total_requests_summary += GMEM_total_requests[i]
    if GMEM_read_transactions[i] is not None:
        GMEM_read_transactions_summary += GMEM_read_transactions[i]
    if GMEM_write_transactions[i] is not None:
        GMEM_write_transactions_summary += GMEM_write_transactions[i]
    if GMEM_total_transactions[i] is not None:
        GMEM_total_transactions_summary += GMEM_total_transactions[i]
# ========================================================
if GMEM_read_requests_summary == 0:
    Number_of_read_transactions_per_read_requests_summary = 0.0
else:
    Number_of_read_transactions_per_read_requests_summary = \
    float(float(GMEM_read_transactions_summary) / float(GMEM_read_requests_summary))
if GMEM_write_requests_summary == 0:
    Number_of_write_transactions_per_write_requests_summary = 0.0
else:
    Number_of_write_transactions_per_write_requests_summary = \
    float(float(GMEM_write_transactions_summary) / float(GMEM_write_requests_summary))
# ========================================================
L2_read_transactions_summary = 0
L2_write_transactions_summary = 0
L2_total_transactions_summary = 0

for i in range(SMs_num):
    if L2_read_transactions[i] is not None:
        L2_read_transactions_summary += L2_read_transactions[i]
    if L2_write_transactions[i] is not None:
        L2_write_transactions_summary += L2_write_transactions[i]
    if L2_total_transactions[i] is not None:
        L2_total_transactions_summary += L2_total_transactions[i]
# ========================================================
Total_number_of_global_atomic_requests_summary = 0
Total_number_of_global_reduction_requests_summary = 0
Global_memory_atomic_and_reduction_transactions_summary = 0

for i in range(SMs_num):
    if Total_number_of_global_atomic_requests[i] is not None:
        Total_number_of_global_atomic_requests_summary += \
            Total_number_of_global_atomic_requests[i]
    if Total_number_of_global_reduction_requests[i] is not None:
        Total_number_of_global_reduction_requests_summary += \
            Total_number_of_global_reduction_requests[i]
    if Global_memory_atomic_and_reduction_transactions[i] is not None:
        Global_memory_atomic_and_reduction_transactions_summary += \
            Global_memory_atomic_and_reduction_transactions[i]
# ========================================================
Achieved_active_warps_per_SM_summary = 0
Achieved_occupancy_summary = 0

for i in range(SMs_num):
    if Achieved_active_warps_per_SM[i] is not None:
        Achieved_active_warps_per_SM_summary = \
            max(Achieved_active_warps_per_SM[i], \
                Achieved_active_warps_per_SM_summary)
    if Achieved_occupancy[i] is not None:
        Achieved_occupancy_summary = \
            max(Achieved_occupancy[i], \
                Achieved_occupancy_summary)

if Achieved_active_warps_per_SM_summary > Theoretical_max_active_warps_per_SM:
    Achieved_active_warps_per_SM = Theoretical_max_active_warps_per_SM

# print(Achieved_occupancy_summary, Theoretical_occupancy)

if Achieved_occupancy_summary > float(Theoretical_occupancy) / 100.0:
    Achieved_occupancy_summary = Theoretical_occupancy / 100.0
# ========================================================
GPU_active_cycles_summary = 0
SM_active_cycles_summary = 0

for i in range(SMs_num):
    if GPU_active_cycles[i] is not None:
        GPU_active_cycles_summary = max(GPU_active_cycles[i],
                                        GPU_active_cycles_summary)
    if SM_active_cycles[i] is not None:
        SM_active_cycles_summary = max(SM_active_cycles[i],
                                       SM_active_cycles_summary)
# ========================================================
Kernel_execution_time_summary = 0
Simulation_time_memory_model_summary = 0
Simulation_time_compute_model_summary = 0

for i in range(SMs_num):
    if Kernel_execution_time[i] is not None:
        Kernel_execution_time_summary = \
            max(Kernel_execution_time[i], Kernel_execution_time_summary)
    if Simulation_time_memory_model[i] is not None:
        Simulation_time_memory_model_summary = \
            max(Simulation_time_memory_model[i], Simulation_time_memory_model_summary)
    if Simulation_time_compute_model[i] is not None:
        Simulation_time_compute_model_summary = \
            max(Simulation_time_compute_model[i], Simulation_time_compute_model_summary)
# ========================================================
Compute_Structural_Stall_summary = 0
Compute_Data_Stall_summary = 0
Memory_Structural_Stall_summary = 0
Memory_Data_Stall_summary = 0
Synchronization_Stall_summary = 0
Control_Stall_summary = 0
Idle_Stall_summary = 0
No_Stall_summary = 0
Other_Stall_summary = 0

Compute_Structural_Stall_ratio = 0.0
Compute_Data_Stall_ratio = 0.0
Memory_Structural_Stall_ratio = 0.0
Memory_Data_Stall_ratio = 0.0
Synchronization_Stall_ratio = 0.0
Control_Stall_ratio = 0.0
Idle_Stall_ratio = 0.0
No_Stall_ratio = 0.0
Other_Stall_ratio = 0.0

num_Issue_Compute_Structural_out_has_no_free_slot_summary = 0
num_Issue_Memory_Structural_out_has_no_free_slot_summary = 0
num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute_summary = 0
num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory_summary = 0
num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency_summary = 0
num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency_summary = 0
num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty_summary = 0
num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty_summary = 0
num_Writeback_Compute_Structural_bank_of_reg_is_not_idle_summary = 0
num_Writeback_Memory_Structural_bank_of_reg_is_not_idle_summary = 0
num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated_summary = 0
num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated_summary = 0
num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu_summary = 0
num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu_summary = 0
num_Execute_Memory_Structural_icnt_injection_buffer_is_full_summary = 0
num_Issue_Compute_Data_scoreboard_summary = 0
num_Issue_Memory_Data_scoreboard_summary = 0
num_Execute_Memory_Data_L1_summary = 0
num_Execute_Memory_Data_L2_summary = 0
num_Execute_Memory_Data_Main_Memory_summary = 0

SP_UNIT_execute_clks_sum_summary = 0
SFU_UNIT_execute_clks_sum_summary = 0
INT_UNIT_execute_clks_sum_summary = 0
DP_UNIT_execute_clks_sum_summary = 0
TENSOR_CORE_UNIT_execute_clks_sum_summary = 0
LDST_UNIT_execute_clks_sum_summary = 0
SPEC_UNIT_1_execute_clks_sum_summary = 0
SPEC_UNIT_2_execute_clks_sum_summary = 0
SPEC_UNIT_3_execute_clks_sum_summary = 0
Other_UNIT_execute_clks_sum_summary = 0

SP_UNIT_Instns_num_summary = 0
SFU_UNIT_Instns_num_summary = 0
INT_UNIT_Instns_num_summary = 0
DP_UNIT_Instns_num_summary = 0
TENSOR_CORE_UNIT_Instns_num_summary = 0
LDST_UNIT_Instns_num_summary = 0
SPEC_UNIT_1_Instns_num_summary = 0
SPEC_UNIT_2_Instns_num_summary = 0
SPEC_UNIT_3_Instns_num_summary = 0
Other_UNIT_Instns_num_summary = 0

for i in range(SMs_num):
    if Compute_Structural_Stall[i] is not None:
        Compute_Structural_Stall_summary += Compute_Structural_Stall[i]
    if Compute_Data_Stall[i] is not None:
        Compute_Data_Stall_summary += Compute_Data_Stall[i]
    if Memory_Structural_Stall[i] is not None:
        Memory_Structural_Stall_summary += Memory_Structural_Stall[i]
    if Memory_Data_Stall[i] is not None:
        Memory_Data_Stall_summary += Memory_Data_Stall[i]
    if Synchronization_Stall[i] is not None:
        Synchronization_Stall_summary += Synchronization_Stall[i]
    if Control_Stall[i] is not None:
        Control_Stall_summary += Control_Stall[i]
    if Idle_Stall[i] is not None:
        Idle_Stall_summary += Idle_Stall[i]
    if No_Stall[i] is not None:
        No_Stall_summary += No_Stall[i]
    if Other_Stall[i] is not None:
        Other_Stall_summary += Other_Stall[i]

    if num_Issue_Compute_Structural_out_has_no_free_slot[i] is not None:
        num_Issue_Compute_Structural_out_has_no_free_slot_summary += num_Issue_Compute_Structural_out_has_no_free_slot[
            i]
    if num_Issue_Memory_Structural_out_has_no_free_slot[i] is not None:
        num_Issue_Memory_Structural_out_has_no_free_slot_summary += num_Issue_Memory_Structural_out_has_no_free_slot[
            i]
    if num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute[
            i] is not None:
        num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute_summary += \
            num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute[i]
    if num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory[
            i] is not None:
        num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory_summary += \
            num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory[i]
    if num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency[
            i] is not None:
        num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency_summary += \
            num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency[i]
    if num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency[
            i] is not None:
        num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency_summary += \
            num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency[i]
    if num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty[
            i] is not None:
        num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty_summary += \
            num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty[i]
    if num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty[
            i] is not None:
        num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty_summary += \
            num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty[i]
    if num_Writeback_Compute_Structural_bank_of_reg_is_not_idle[i] is not None:
        num_Writeback_Compute_Structural_bank_of_reg_is_not_idle_summary += \
            num_Writeback_Compute_Structural_bank_of_reg_is_not_idle[i]
    if num_Writeback_Memory_Structural_bank_of_reg_is_not_idle[i] is not None:
        num_Writeback_Memory_Structural_bank_of_reg_is_not_idle_summary += \
            num_Writeback_Memory_Structural_bank_of_reg_is_not_idle[i]
    if num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated[
            i] is not None:
        num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated_summary += \
            num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated[i]
    if num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated[
            i] is not None:
        num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated_summary += \
            num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated[i]
    if num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu[
            i] is not None:
        num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu_summary += \
            num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu[i]
    if num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu[
            i] is not None:
        num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu_summary += \
            num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu[i]
    if num_Execute_Memory_Structural_icnt_injection_buffer_is_full[
            i] is not None:
        num_Execute_Memory_Structural_icnt_injection_buffer_is_full_summary += \
            num_Execute_Memory_Structural_icnt_injection_buffer_is_full[i]
    if num_Issue_Compute_Data_scoreboard[i] is not None:
        num_Issue_Compute_Data_scoreboard_summary += num_Issue_Compute_Data_scoreboard[
            i]
    if num_Issue_Memory_Data_scoreboard[i] is not None:
        num_Issue_Memory_Data_scoreboard_summary += num_Issue_Memory_Data_scoreboard[
            i]
    if num_Execute_Memory_Data_L1[i] is not None:
        num_Execute_Memory_Data_L1_summary += num_Execute_Memory_Data_L1[i]
    if num_Execute_Memory_Data_L2[i] is not None:
        num_Execute_Memory_Data_L2_summary += num_Execute_Memory_Data_L2[i]
    if num_Execute_Memory_Data_Main_Memory[i] is not None:
        num_Execute_Memory_Data_Main_Memory_summary += num_Execute_Memory_Data_Main_Memory[
            i]

    if SP_UNIT_execute_clks_sum[i] is not None:
        SP_UNIT_execute_clks_sum_summary += SP_UNIT_execute_clks_sum[i]
    if SFU_UNIT_execute_clks_sum[i] is not None:
        SFU_UNIT_execute_clks_sum_summary += SFU_UNIT_execute_clks_sum[i]
    if INT_UNIT_execute_clks_sum[i] is not None:
        INT_UNIT_execute_clks_sum_summary += INT_UNIT_execute_clks_sum[i]
    if DP_UNIT_execute_clks_sum[i] is not None:
        DP_UNIT_execute_clks_sum_summary += DP_UNIT_execute_clks_sum[i]
    if TENSOR_CORE_UNIT_execute_clks_sum[i] is not None:
        TENSOR_CORE_UNIT_execute_clks_sum_summary += TENSOR_CORE_UNIT_execute_clks_sum[
            i]
    if LDST_UNIT_execute_clks_sum[i] is not None:
        LDST_UNIT_execute_clks_sum_summary += LDST_UNIT_execute_clks_sum[i]
    if SPEC_UNIT_1_execute_clks_sum[i] is not None:
        SPEC_UNIT_1_execute_clks_sum_summary += SPEC_UNIT_1_execute_clks_sum[i]
    if SPEC_UNIT_2_execute_clks_sum[i] is not None:
        SPEC_UNIT_2_execute_clks_sum_summary += SPEC_UNIT_2_execute_clks_sum[i]
    if SPEC_UNIT_3_execute_clks_sum[i] is not None:
        SPEC_UNIT_3_execute_clks_sum_summary += SPEC_UNIT_3_execute_clks_sum[i]
    if Other_UNIT_execute_clks_sum[i] is not None:
        Other_UNIT_execute_clks_sum_summary += Other_UNIT_execute_clks_sum[i]

    if SP_UNIT_Instns_num[i] is not None:
        SP_UNIT_Instns_num_summary += SP_UNIT_Instns_num[i]
    if SFU_UNIT_Instns_num[i] is not None:
        SFU_UNIT_Instns_num_summary += SFU_UNIT_Instns_num[i]
    if INT_UNIT_Instns_num[i] is not None:
        INT_UNIT_Instns_num_summary += INT_UNIT_Instns_num[i]
    if DP_UNIT_Instns_num[i] is not None:
        DP_UNIT_Instns_num_summary += DP_UNIT_Instns_num[i]
    if TENSOR_CORE_UNIT_Instns_num[i] is not None:
        TENSOR_CORE_UNIT_Instns_num_summary += TENSOR_CORE_UNIT_Instns_num[i]
    if LDST_UNIT_Instns_num[i] is not None:
        LDST_UNIT_Instns_num_summary += LDST_UNIT_Instns_num[i]
    if SPEC_UNIT_1_Instns_num[i] is not None:
        SPEC_UNIT_1_Instns_num_summary += SPEC_UNIT_1_Instns_num[i]
    if SPEC_UNIT_2_Instns_num[i] is not None:
        SPEC_UNIT_2_Instns_num_summary += SPEC_UNIT_2_Instns_num[i]
    if SPEC_UNIT_3_Instns_num[i] is not None:
        SPEC_UNIT_3_Instns_num_summary += SPEC_UNIT_3_Instns_num[i]
    if Other_UNIT_Instns_num[i] is not None:
        Other_UNIT_Instns_num_summary += Other_UNIT_Instns_num[i]


total_num_stalls = Compute_Structural_Stall_summary + Compute_Data_Stall_summary + \
                   Memory_Structural_Stall_summary + Memory_Data_Stall_summary + \
                   Synchronization_Stall_summary + Control_Stall_summary + \
                   Idle_Stall_summary + No_Stall_summary + Other_Stall_summary

if total_num_stalls != 0:
    Compute_Structural_Stall_ratio = float(
        Compute_Structural_Stall_summary) / float(total_num_stalls)
    Compute_Data_Stall_ratio = float(Compute_Data_Stall_summary) / float(
        total_num_stalls)
    Memory_Structural_Stall_ratio = float(
        Memory_Structural_Stall_summary) / float(total_num_stalls)
    Memory_Data_Stall_ratio = float(Memory_Data_Stall_summary) / float(
        total_num_stalls)
    Synchronization_Stall_ratio = float(Synchronization_Stall_summary) / float(
        total_num_stalls)
    Control_Stall_ratio = float(Control_Stall_summary) / float(
        total_num_stalls)
    Idle_Stall_ratio = float(Idle_Stall_summary) / float(total_num_stalls)
    No_Stall_ratio = float(No_Stall_summary) / float(total_num_stalls)
    Other_Stall_ratio = float(Other_Stall_summary) / float(total_num_stalls)
else:
    total_num_stalls = 1
    Compute_Structural_Stall_ratio = float(
        Compute_Structural_Stall_summary) / float(total_num_stalls)
    Compute_Data_Stall_ratio = float(Compute_Data_Stall_summary) / float(
        total_num_stalls)
    Memory_Structural_Stall_ratio = float(
        Memory_Structural_Stall_summary) / float(total_num_stalls)
    Memory_Data_Stall_ratio = float(Memory_Data_Stall_summary) / float(
        total_num_stalls)
    Synchronization_Stall_ratio = float(Synchronization_Stall_summary) / float(
        total_num_stalls)
    Control_Stall_ratio = float(Control_Stall_summary) / float(
        total_num_stalls)
    Idle_Stall_ratio = float(Idle_Stall_summary) / float(total_num_stalls)
    No_Stall_ratio = float(No_Stall_summary) / float(total_num_stalls)
    Other_Stall_ratio = float(Other_Stall_summary) / float(total_num_stalls)
# ========================================================
### MAY ERROR
Warp_instructions_executed_summary = 0

# for i in range(SMs_num):
#     if Warp_instructions_executed[i] is not None:
#         Warp_instructions_executed_summary += Warp_instructions_executed[i]

# Warp_instructions_executed_summary *= SMs_num
import subprocess

sass_files_dir = reports_dir + "/../sass_traces/kernel_" + str(kernel_id +
                                                               1) + ".sass"
command = f"awk '{{count += gsub(/ /,\"&\")}} END{{print count}}' {sass_files_dir}"

result = subprocess.run(command, shell=True, text=True, capture_output=True)
if result.returncode == 0:
    Warp_instructions_executed_summary = int(int(result.stdout.strip()) / 3)
else:
    for i in range(SMs_num):
        if Warp_instructions_executed[i] is not None:
            Warp_instructions_executed_summary += Warp_instructions_executed[i]

    Warp_instructions_executed_summary *= SMs_num

Instructions_executed_per_clock_cycle_IPC_summary = \
    float(Warp_instructions_executed_summary) / float(GPU_active_cycles_summary * SMs_num)

Total_instructions_executed_per_seconds_summary = \
    float(Warp_instructions_executed_summary) / float(Kernel_execution_time_summary * SMs_num) * 1024
# ========================================================

# ========================================================
# Finally, write the summarized content into a new file.
with open(reports_dir + '/' + f'kernel-{kernel_id}-summary.txt', 'w') as f:
    f.write("Summary:\n")
    f.write("\n")
    print("Reports have been written to " + \
          reports_dir + '/' + f'kernel-{kernel_id}-summary.txt')
    f.write(" - Config: " + reports_dir + "\n")
    f.write("\n")
    f.write(" - Theoretical Performance: " + "\n")
    f.write("       * Thread_block_limit_SM: " + str(Thread_block_limit_SM) +
            "\n")
    f.write("       * Thread_block_limit_registers: " +
            str(Thread_block_limit_registers) + "\n")
    f.write("       * Thread_block_limit_shared_memory: " +
            str(Thread_block_limit_shared_memory) + "\n")
    f.write("       * Thread_block_limit_warps: " +
            str(Thread_block_limit_warps) + "\n")
    f.write("       * Theoretical_max_active_warps_per_SM: " +
            str(Theoretical_max_active_warps_per_SM) + "\n")
    f.write("       * Theoretical_occupancy: " +
            str(format(float(Theoretical_occupancy) / 100.0, '.2f')) + "\n")
    f.write("\n")
    f.write(" - L1 Cache Performance: " + "\n")
    f.write("       * Unified_L1_cache_hit_rate: " +
            str(format(Unified_L1_cache_hit_rate_summary, '.4f')) + "\n")
    f.write("       * Unified_L1_cache_requests: " +
            str(Unified_L1_cache_requests_summary) + "\n")
    f.write("\n")
    f.write(" - L2 Cache Performance: " + "\n")
    f.write("       * L2_cache_hit_rate: " +
            str(format(L2_cache_hit_rate, '.4f')) + "\n")
    f.write("       * L2_cache_requests: " + str(L2_cache_requests) + "\n")
    f.write("       * L2_read_transactions: " +
            str(L2_read_transactions_summary) + "\n")
    f.write("       * L2_write_transactions: " +
            str(L2_write_transactions_summary) + "\n")
    f.write("       * L2_total_transactions: " +
            str(L2_total_transactions_summary) + "\n")
    f.write("\n")
    f.write(" - DRAM Performance: " + "\n")
    f.write("       * DRAM_total_transactions: " +
            str(DRAM_total_transactions) + "\n")
    f.write("\n")
    f.write(" - Global Memory Performance: " + "\n")
    f.write("       * GMEM_read_requests: " + str(GMEM_read_requests_summary) +
            "\n")
    f.write("       * GMEM_write_requests: " +
            str(GMEM_write_requests_summary) + "\n")
    f.write("       * GMEM_total_requests: " +
            str(GMEM_total_requests_summary) + "\n")
    f.write("       * GMEM_read_transactions: " +
            str(GMEM_read_transactions_summary) + "\n")
    f.write("       * GMEM_write_transactions: " +
            str(GMEM_write_transactions_summary) + "\n")
    f.write("       * GMEM_total_transactions: " +
            str(GMEM_total_transactions_summary) + "\n")
    f.write("       * Number_of_read_transactions_per_read_requests: "+\
        str(format(Number_of_read_transactions_per_read_requests_summary, '.2f'))+"\n")
    f.write("       * Number_of_write_transactions_per_write_requests: "+\
        str(format(Number_of_write_transactions_per_write_requests_summary, '.2f'))+"\n")
    f.write("       * Total_number_of_global_atomic_requests: "+\
        str(Total_number_of_global_atomic_requests_summary)+"\n")
    f.write("       * Total_number_of_global_reduction_requests: "+\
        str(Total_number_of_global_reduction_requests_summary)+"\n")
    f.write("       * Global_memory_atomic_and_reduction_transactions: "+\
        str(Global_memory_atomic_and_reduction_transactions_summary)+"\n")
    f.write("\n")
    f.write(" - SMs Performance: " + "\n")
    f.write("       * Achieved_active_warps_per_SM: " +
            str(format(Achieved_active_warps_per_SM_summary, '.2f')) + "\n")
    f.write("       * Achieved_occupancy: " +
            str(format(Achieved_occupancy_summary, '.2f')) + "\n")
    f.write("\n")
    f.write("       * GPU_active_cycles: " + str(GPU_active_cycles_summary) +
            "\n")
    f.write("       * SM_active_cycles: " + str(SM_active_cycles_summary) +
            "\n")
    f.write("\n")
    f.write("       * Warp_instructions_executed: " +
            str(Warp_instructions_executed_summary) + "\n")
    f.write("       * Instructions_executed_per_clock_cycle_IPC: "+\
        str(format(Instructions_executed_per_clock_cycle_IPC_summary, '.2f'))+"\n")
    f.write("       * Total_instructions_executed_per_seconds (MIPS): "+\
        str(format(Total_instructions_executed_per_seconds_summary, '.2f'))+"\n")
    f.write("\n")
    f.write("       * Kernel_execution_time (ns): " +
            str(Kernel_execution_time_summary) + "\n")
    f.write("\n")
    f.write(" - Simulator Performance: " + "\n")
    f.write("       * Simulation_time_memory_model (s): " +
            str(format(Simulation_time_memory_model_summary, '.2f')) + "\n")
    f.write("       * Simulation_time_compute_model (s): " +
            str(format(Simulation_time_compute_model_summary, '.2f')) + "\n")
    f.write("\n")
    f.write(" - Stall Cycles: " + "\n")
    f.write("       * Compute_Structural_Stall: " +
            str(int(Compute_Structural_Stall_summary)) + "\n")
    f.write("       * Compute_Data_Stall: " +
            str(int(Compute_Data_Stall_summary)) + "\n")
    f.write("       * Memory_Structural_Stall: " +
            str(int(Memory_Structural_Stall_summary)) + "\n")
    f.write("       * Memory_Data_Stall: " +
            str(int(Memory_Data_Stall_summary)) + "\n")
    f.write("       * Synchronization_Stall: " +
            str(int(Synchronization_Stall_summary)) + "\n")
    f.write("       * Control_Stall: " + str(int(Control_Stall_summary)) +
            "\n")
    f.write("       * Idle_Stall: " + str(int(Idle_Stall_summary)) + "\n")
    f.write("       * Other_Stall: " + str(int(Other_Stall_summary)) + "\n")
    f.write("       * No_Stall: " + str(int(No_Stall_summary)) + "\n")
    f.write("\n")
    f.write(" - Stall Cycles Distribution: " + "\n")
    f.write("       * Compute_Structural_Stall_ratio: " +
            str(format(Compute_Structural_Stall_ratio, '.6f')) + "\n")
    f.write("       * Compute_Data_Stall_ratio: " +
            str(format(Compute_Data_Stall_ratio, '.6f')) + "\n")
    f.write("       * Memory_Structural_Stall_ratio: " +
            str(format(Memory_Structural_Stall_ratio, '.6f')) + "\n")
    f.write("       * Memory_Data_Stall_ratio: " +
            str(format(Memory_Data_Stall_ratio, '.6f')) + "\n")
    f.write("       * Synchronization_Stall_ratio: " +
            str(format(Synchronization_Stall_ratio, '.6f')) + "\n")
    f.write("       * Control_Stall_ratio: " +
            str(format(Control_Stall_ratio, '.6f')) + "\n")
    f.write("       * Idle_Stall_ratio: " +
            str(format(Idle_Stall_ratio, '.6f')) + "\n")
    f.write("       * Other_Stall_ratio: " +
            str(format(Other_Stall_ratio, '.6f')) + "\n")
    f.write("       * No_Stall_ratio: " + str(format(No_Stall_ratio, '.6f')) +
            "\n")
    f.write("\n")

    f.write(" - Memory Structural Stall Cycles Breakdown: " + "\n")
    f.write("       * Issue_out_has_no_free_slot: "+\
        str(num_Issue_Memory_Structural_out_has_no_free_slot_summary)+"\n")
    f.write("       * Issue_previous_issued_inst_exec_type_is_memory: "+\
        str(num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory_summary)+"\n")
    f.write("       * Execute_result_bus_has_no_slot_for_latency: "+\
        str(num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency_summary)+"\n")
    f.write("       * Execute_m_dispatch_reg_of_fu_is_not_empty: "+\
        str(num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty_summary)+"\n")
    f.write("       * Writeback_bank_of_reg_is_not_idle: "+\
        str(num_Writeback_Memory_Structural_bank_of_reg_is_not_idle_summary)+"\n")
    f.write("       * ReadOperands_bank_reg_belonged_to_was_allocated: "+\
        str(num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated_summary)+"\n")
    f.write("       * ReadOperands_port_num_m_in_ports_m_in_fails_as_not_found_free_cu: "+\
        str(num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu_summary)+"\n")
    f.write("       * Execute_icnt_injection_buffer_is_full: "+\
        str(num_Execute_Memory_Structural_icnt_injection_buffer_is_full_summary)+"\n")
    f.write("\n")
    all_Memory_Structural_cycles = num_Issue_Memory_Structural_out_has_no_free_slot_summary + \
                                   num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory_summary + \
                                   num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency_summary + \
                                   num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty_summary + \
                                   num_Writeback_Memory_Structural_bank_of_reg_is_not_idle_summary + \
                                   num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated_summary + \
                                   num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu_summary + \
                                   num_Execute_Memory_Structural_icnt_injection_buffer_is_full_summary

    if all_Memory_Structural_cycles == 0:
        all_Memory_Structural_cycles = 1

    f.write(" - Memory Structural Stall Cycles Breakdown Distribution: " +
            "\n")
    f.write("       * Issue_out_has_no_free_slot: "+\
        str(format(num_Issue_Memory_Structural_out_has_no_free_slot_summary / all_Memory_Structural_cycles, '.6f'))+"\n")
    f.write("       * Issue_previous_issued_inst_exec_type_is_memory: "+\
        str(format(num_Issue_Memory_Structural_previous_issued_inst_exec_type_is_memory_summary / all_Memory_Structural_cycles, '.6f'))+"\n")
    f.write("       * Execute_result_bus_has_no_slot_for_latency: "+\
        str(format(num_Execute_Memory_Structural_result_bus_has_no_slot_for_latency_summary / all_Memory_Structural_cycles, '.6f'))+"\n")
    f.write("       * Execute_m_dispatch_reg_of_fu_is_not_empty: "+\
        str(format(num_Execute_Memory_Structural_m_dispatch_reg_of_fu_is_not_empty_summary / all_Memory_Structural_cycles, '.6f'))+"\n")
    f.write("       * Writeback_bank_of_reg_is_not_idle: "+\
        str(format(num_Writeback_Memory_Structural_bank_of_reg_is_not_idle_summary / all_Memory_Structural_cycles, '.6f'))+"\n")
    f.write("       * ReadOperands_bank_reg_belonged_to_was_allocated: "+\
        str(format(num_ReadOperands_Memory_Structural_bank_reg_belonged_to_was_allocated_summary / all_Memory_Structural_cycles, '.6f'))+"\n")
    f.write("       * ReadOperands_port_num_m_in_ports_m_in_fails_as_not_found_free_cu: "+\
        str(format(num_ReadOperands_Memory_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu_summary / all_Memory_Structural_cycles, '.6f'))+"\n")
    f.write("       * Execute_icnt_injection_buffer_is_full: "+\
        str(format(num_Execute_Memory_Structural_icnt_injection_buffer_is_full_summary / all_Memory_Structural_cycles, '.6f'))+"\n")
    f.write("\n")
    f.write(" - Compute Structural Stall Cycles Breakdown: " + "\n")
    f.write("       * Issue_out_has_no_free_slot: "+\
        str(num_Issue_Compute_Structural_out_has_no_free_slot_summary)+"\n")
    f.write("       * Issue_previous_issued_inst_exec_type_is_compute: "+\
        str(num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute_summary)+"\n")
    f.write("       * Execute_result_bus_has_no_slot_for_latency: "+\
        str(num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency_summary)+"\n")
    f.write("       * Execute_m_dispatch_reg_of_fu_is_not_empty: "+\
        str(num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty_summary)+"\n")
    f.write("       * Writeback_bank_of_reg_is_not_idle: "+\
        str(num_Writeback_Compute_Structural_bank_of_reg_is_not_idle_summary)+"\n")
    f.write("       * ReadOperands_bank_reg_belonged_to_was_allocated: "+\
        str(num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated_summary)+"\n")
    f.write("       * ReadOperands_port_num_m_in_ports_m_in_fails_as_not_found_free_cu: "+\
        str(num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu_summary)+"\n")
    f.write("\n")
    all_Compute_Structural_cycles = num_Issue_Compute_Structural_out_has_no_free_slot_summary + \
                                    num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute_summary + \
                                    num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency_summary + \
                                    num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty_summary + \
                                    num_Writeback_Compute_Structural_bank_of_reg_is_not_idle_summary + \
                                    num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated_summary + \
                                    num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu_summary

    if all_Compute_Structural_cycles == 0:
        all_Compute_Structural_cycles = 1

    f.write(" - Compute Structural Stall Cycles Breakdown Distribution: " +
            "\n")
    f.write("       * Issue_out_has_no_free_slot: "+\
        str(format(num_Issue_Compute_Structural_out_has_no_free_slot_summary / all_Compute_Structural_cycles, '.6f'))+"\n")
    f.write("       * Issue_previous_issued_inst_exec_type_is_compute: "+\
        str(format(num_Issue_Compute_Structural_previous_issued_inst_exec_type_is_compute_summary / all_Compute_Structural_cycles, '.6f'))+"\n")
    f.write("       * Execute_result_bus_has_no_slot_for_latency: "+\
        str(format(num_Execute_Compute_Structural_result_bus_has_no_slot_for_latency_summary / all_Compute_Structural_cycles, '.6f'))+"\n")
    f.write("       * Execute_m_dispatch_reg_of_fu_is_not_empty: "+\
        str(format(num_Execute_Compute_Structural_m_dispatch_reg_of_fu_is_not_empty_summary / all_Compute_Structural_cycles, '.6f'))+"\n")
    f.write("       * Writeback_bank_of_reg_is_not_idle: "+\
        str(format(num_Writeback_Compute_Structural_bank_of_reg_is_not_idle_summary / all_Compute_Structural_cycles, '.6f'))+"\n")
    f.write("       * ReadOperands_bank_reg_belonged_to_was_allocated: "+\
        str(format(num_ReadOperands_Compute_Structural_bank_reg_belonged_to_was_allocated_summary / all_Compute_Structural_cycles, '.6f'))+"\n")
    f.write("       * ReadOperands_port_num_m_in_ports_m_in_fails_as_not_found_free_cu: "+\
        str(format(num_ReadOperands_Compute_Structural_port_num_m_in_ports_m_in_fails_as_not_found_free_cu_summary / all_Compute_Structural_cycles, '.6f'))+"\n")
    f.write("\n")
    f.write(" - Memory Data Stall Cycles Breakdown: " + "\n")
    f.write("       * Issue_scoreboard: " +
            str(num_Issue_Memory_Data_scoreboard_summary) + "\n")
    f.write("       * Execute_L1: " + str(num_Execute_Memory_Data_L1_summary) +
            "\n")
    f.write("       * Execute_L2: " + str(num_Execute_Memory_Data_L2_summary) +
            "\n")
    f.write("       * Execute_Main_Memory: " +
            str(num_Execute_Memory_Data_Main_Memory_summary) + "\n")
    f.write("\n")
    all_Memory_Data_cycles = num_Issue_Memory_Data_scoreboard_summary + \
                             num_Execute_Memory_Data_L1_summary + \
                             num_Execute_Memory_Data_L2_summary + \
                             num_Execute_Memory_Data_Main_Memory_summary

    if all_Memory_Data_cycles == 0:
        all_Memory_Data_cycles = 1

    f.write(" - Memory Data Stall Cycles Breakdown Distribution: " + "\n")
    f.write("       * Issue_scoreboard: " + str(
        format(
            num_Issue_Memory_Data_scoreboard_summary /
            all_Memory_Data_cycles, '.6f')) + "\n")
    f.write("       * Execute_L1: " + str(
        format(num_Execute_Memory_Data_L1_summary /
               all_Memory_Data_cycles, '.6f')) + "\n")
    f.write("       * Execute_L2: " + str(
        format(num_Execute_Memory_Data_L2_summary /
               all_Memory_Data_cycles, '.6f')) + "\n")
    f.write("       * Execute_Main_Memory: " + str(
        format(
            num_Execute_Memory_Data_Main_Memory_summary /
            all_Memory_Data_cycles, '.6f')) + "\n")
    f.write("\n")
    f.write(" - Compute Data Stall Cycles Breakdown: " + "\n")
    f.write("       * Issue_scoreboard: " +
            str(num_Issue_Compute_Data_scoreboard_summary) + "\n")
    f.write("\n")
    all_Compute_Data_cycles = num_Issue_Compute_Data_scoreboard_summary
    
    if all_Compute_Data_cycles == 0:
        all_Compute_Data_cycles = 1
    
    f.write(" - Compute Data Stall Cycles Breakdown Distribution: " + "\n")
    f.write("       * Issue_scoreboard: " + str(
        format(
            num_Issue_Compute_Data_scoreboard_summary /
            all_Compute_Data_cycles, '.6f')) + "\n")
    f.write("\n")

    f.write(" - Function Unit Execution Cycles Breakdown: " + "\n")
    f.write("       * SP_UNIT_execute_clks: " +
            str(SP_UNIT_execute_clks_sum_summary) + "\n")
    f.write("       * SFU_UNIT_execute_clks: " +
            str(SFU_UNIT_execute_clks_sum_summary) + "\n")
    f.write("       * INT_UNIT_execute_clks: " +
            str(INT_UNIT_execute_clks_sum_summary) + "\n")
    f.write("       * DP_UNIT_execute_clks: " +
            str(DP_UNIT_execute_clks_sum_summary) + "\n")
    f.write("       * TENSOR_CORE_UNIT_execute_clks: " +
            str(TENSOR_CORE_UNIT_execute_clks_sum_summary) + "\n")
    f.write("       * LDST_UNIT_execute_clks: " +
            str(LDST_UNIT_execute_clks_sum_summary) + "\n")
    f.write("       * SPEC_UNIT_1_execute_clks: " +
            str(SPEC_UNIT_1_execute_clks_sum_summary) + "\n")
    f.write("       * SPEC_UNIT_2_execute_clks: " +
            str(SPEC_UNIT_2_execute_clks_sum_summary) + "\n")
    f.write("       * SPEC_UNIT_3_execute_clks: " +
            str(SPEC_UNIT_3_execute_clks_sum_summary) + "\n")
    f.write("       * Other_UNIT_execute_clks: " +
            str(Other_UNIT_execute_clks_sum_summary) + "\n")
    f.write("\n")
    all_Execution_Cycles = SP_UNIT_execute_clks_sum_summary + \
                           SFU_UNIT_execute_clks_sum_summary + \
                           INT_UNIT_execute_clks_sum_summary + \
                           DP_UNIT_execute_clks_sum_summary + \
                           TENSOR_CORE_UNIT_execute_clks_sum_summary + \
                           LDST_UNIT_execute_clks_sum_summary + \
                           SPEC_UNIT_1_execute_clks_sum_summary + \
                           SPEC_UNIT_2_execute_clks_sum_summary + \
                           SPEC_UNIT_3_execute_clks_sum_summary + \
                           Other_UNIT_execute_clks_sum_summary

    if all_Execution_Cycles == 0:
        all_Execution_Cycles = 1

    f.write(" - Function Unit Execution Cycles Breakdown Distribution: " +
            "\n")
    f.write("       * SP_UNIT_execute_clks: " + str(
        format(SP_UNIT_execute_clks_sum_summary /
               all_Execution_Cycles, '.6f')) + "\n")
    f.write("       * SFU_UNIT_execute_clks: " + str(
        format(SFU_UNIT_execute_clks_sum_summary /
               all_Execution_Cycles, '.6f')) + "\n")
    f.write("       * INT_UNIT_execute_clks: " + str(
        format(INT_UNIT_execute_clks_sum_summary /
               all_Execution_Cycles, '.6f')) + "\n")
    f.write("       * DP_UNIT_execute_clks: " + str(
        format(DP_UNIT_execute_clks_sum_summary /
               all_Execution_Cycles, '.6f')) + "\n")
    f.write("       * TENSOR_CORE_UNIT_execute_clks: " + str(
        format(
            TENSOR_CORE_UNIT_execute_clks_sum_summary /
            all_Execution_Cycles, '.6f')) + "\n")
    f.write("       * LDST_UNIT_execute_clks: " + str(
        format(LDST_UNIT_execute_clks_sum_summary /
               all_Execution_Cycles, '.6f')) + "\n")
    f.write("       * SPEC_UNIT_1_execute_clks: " + str(
        format(SPEC_UNIT_1_execute_clks_sum_summary /
               all_Execution_Cycles, '.6f')) + "\n")
    f.write("       * SPEC_UNIT_2_execute_clks: " + str(
        format(SPEC_UNIT_2_execute_clks_sum_summary /
               all_Execution_Cycles, '.6f')) + "\n")
    f.write("       * SPEC_UNIT_3_execute_clks: " + str(
        format(SPEC_UNIT_3_execute_clks_sum_summary /
               all_Execution_Cycles, '.6f')) + "\n")
    f.write("       * Other_UNIT_execute_clks: " + str(
        format(Other_UNIT_execute_clks_sum_summary /
               all_Execution_Cycles, '.6f')) + "\n")
    f.write("\n")

    f.write(" - Function Unit Execution Instns Number: " + "\n")
    f.write("       * SP_UNIT_Instns_num: " + str(SP_UNIT_Instns_num_summary) +
            "\n")
    f.write("       * SFU_UNIT_Instns_num: " +
            str(SFU_UNIT_Instns_num_summary) + "\n")
    f.write("       * INT_UNIT_Instns_num: " +
            str(INT_UNIT_Instns_num_summary) + "\n")
    f.write("       * DP_UNIT_Instns_num: " + str(DP_UNIT_Instns_num_summary) +
            "\n")
    f.write("       * TENSOR_CORE_UNIT_Instns_num: " +
            str(TENSOR_CORE_UNIT_Instns_num_summary) + "\n")
    f.write("       * LDST_UNIT_Instns_num: " +
            str(LDST_UNIT_Instns_num_summary) + "\n")
    f.write("       * SPEC_UNIT_1_Instns_num: " +
            str(SPEC_UNIT_1_Instns_num_summary) + "\n")
    f.write("       * SPEC_UNIT_2_Instns_num: " +
            str(SPEC_UNIT_2_Instns_num_summary) + "\n")
    f.write("       * SPEC_UNIT_3_Instns_num: " +
            str(SPEC_UNIT_3_Instns_num_summary) + "\n")
    f.write("       * Other_UNIT_Instns_num: " +
            str(Other_UNIT_Instns_num_summary) + "\n")
    f.write("\n")

    all_Instns_num = SP_UNIT_Instns_num_summary + \
                     SFU_UNIT_Instns_num_summary + \
                     INT_UNIT_Instns_num_summary + \
                     DP_UNIT_Instns_num_summary + \
                     TENSOR_CORE_UNIT_Instns_num_summary + \
                     LDST_UNIT_Instns_num_summary + \
                     SPEC_UNIT_1_Instns_num_summary + \
                     SPEC_UNIT_2_Instns_num_summary + \
                     SPEC_UNIT_3_Instns_num_summary + \
                     Other_UNIT_Instns_num_summary

    if all_Instns_num == 0:
        all_Instns_num = 1

    f.write(" - Function Unit Execution Instns Number Distribution: " + "\n")
    f.write("       * SP_UNIT_Instns_num: " +
            str(format(SP_UNIT_Instns_num_summary / all_Instns_num, '.6f')) +
            "\n")
    f.write("       * SFU_UNIT_Instns_num: " +
            str(format(SFU_UNIT_Instns_num_summary / all_Instns_num, '.6f')) +
            "\n")
    f.write("       * INT_UNIT_Instns_num: " +
            str(format(INT_UNIT_Instns_num_summary / all_Instns_num, '.6f')) +
            "\n")
    f.write("       * DP_UNIT_Instns_num: " +
            str(format(DP_UNIT_Instns_num_summary / all_Instns_num, '.6f')) +
            "\n")
    f.write("       * TENSOR_CORE_UNIT_Instns_num: " + str(
        format(TENSOR_CORE_UNIT_Instns_num_summary / all_Instns_num, '.6f')) +
            "\n")
    f.write("       * LDST_UNIT_Instns_num: " +
            str(format(LDST_UNIT_Instns_num_summary / all_Instns_num, '.6f')) +
            "\n")
    f.write("       * SPEC_UNIT_1_Instns_num: " +
            str(format(SPEC_UNIT_1_Instns_num_summary /
                       all_Instns_num, '.6f')) + "\n")
    f.write("       * SPEC_UNIT_2_Instns_num: " +
            str(format(SPEC_UNIT_2_Instns_num_summary /
                       all_Instns_num, '.6f')) + "\n")
    f.write("       * SPEC_UNIT_3_Instns_num: " +
            str(format(SPEC_UNIT_3_Instns_num_summary /
                       all_Instns_num, '.6f')) + "\n")
    f.write("       * Other_UNIT_Instns_num: " +
            str(format(Other_UNIT_Instns_num_summary /
                       all_Instns_num, '.6f')) + "\n")
    f.write("\n")

    f.write(" - Function Unit Execution Average Cycles Per Instn : " + "\n")
    if SP_UNIT_Instns_num_summary > 0:
        f.write("       * SP_UNIT_average_cycles_per_instn: "+\
            str(format(SP_UNIT_execute_clks_sum_summary / SP_UNIT_Instns_num_summary, '.6f'))+"\n")
    else:
        f.write("       * SP_UNIT_average_cycles_per_instn: " +
                str(format(0, '.6f')) + "\n")
    if SFU_UNIT_Instns_num_summary > 0:
        f.write("       * SFU_UNIT_average_cycles_per_instn: "+\
            str(format(SFU_UNIT_execute_clks_sum_summary / SFU_UNIT_Instns_num_summary, '.6f'))+"\n")
    else:
        f.write("       * SFU_UNIT_average_cycles_per_instn: " +
                str(format(0, '.6f')) + "\n")
    if INT_UNIT_Instns_num_summary > 0:
        f.write("       * INT_UNIT_average_cycles_per_instn: "+\
            str(format(INT_UNIT_execute_clks_sum_summary / INT_UNIT_Instns_num_summary, '.6f'))+"\n")
    else:
        f.write("       * INT_UNIT_average_cycles_per_instn: " +
                str(format(0, '.6f')) + "\n")
    if DP_UNIT_Instns_num_summary > 0:
        f.write("       * DP_UNIT_average_cycles_per_instn: "+\
            str(format(DP_UNIT_execute_clks_sum_summary / DP_UNIT_Instns_num_summary, '.6f'))+"\n")
    else:
        f.write("       * DP_UNIT_average_cycles_per_instn: " +
                str(format(0, '.6f')) + "\n")
    if TENSOR_CORE_UNIT_Instns_num_summary > 0:
        f.write("       * TENSOR_CORE_UNIT_average_cycles_per_instn: "+\
            str(format(TENSOR_CORE_UNIT_execute_clks_sum_summary / TENSOR_CORE_UNIT_Instns_num_summary, '.6f'))+"\n")
    else:
        f.write("       * TENSOR_CORE_UNIT_average_cycles_per_instn: " +
                str(format(0, '.6f')) + "\n")
    if LDST_UNIT_Instns_num_summary > 0:
        f.write("       * LDST_UNIT_average_cycles_per_instn: "+\
            str(format(LDST_UNIT_execute_clks_sum_summary / LDST_UNIT_Instns_num_summary, '.6f'))+"\n")
    else:
        f.write("       * LDST_UNIT_average_cycles_per_instn: " +
                str(format(0, '.6f')) + "\n")
    if SPEC_UNIT_1_Instns_num_summary > 0:
        f.write("       * SPEC_UNIT_1_average_cycles_per_instn: "+\
            str(format(SPEC_UNIT_1_execute_clks_sum_summary / SPEC_UNIT_1_Instns_num_summary, '.6f'))+"\n")
    else:
        f.write("       * SPEC_UNIT_1_average_cycles_per_instn: " +
                str(format(0, '.6f')) + "\n")
    if SPEC_UNIT_2_Instns_num_summary > 0:
        f.write("       * SPEC_UNIT_2_average_cycles_per_instn: "+\
            str(format(SPEC_UNIT_2_execute_clks_sum_summary / SPEC_UNIT_2_Instns_num_summary, '.6f'))+"\n")
    else:
        f.write("       * SPEC_UNIT_2_average_cycles_per_instn: " +
                str(format(0, '.6f')) + "\n")
    if SPEC_UNIT_3_Instns_num_summary > 0:
        f.write("       * SPEC_UNIT_3_average_cycles_per_instn: "+\
            str(format(SPEC_UNIT_3_execute_clks_sum_summary / SPEC_UNIT_3_Instns_num_summary, '.6f'))+"\n")
    else:
        f.write("       * SPEC_UNIT_3_average_cycles_per_instn: " +
                str(format(0, '.6f')) + "\n")
    if Other_UNIT_Instns_num_summary > 0:
        f.write("       * Other_UNIT_average_cycles_per_instn: "+\
            str(format(Other_UNIT_execute_clks_sum_summary / Other_UNIT_Instns_num_summary, '.6f'))+"\n")
    else:
        f.write("       * Other_UNIT_average_cycles_per_instn: " +
                str(format(0, '.6f')) + "\n")
    f.write("\n")

    import datetime

    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    f.write(" - Report Time: " + formatted_time + "\n")

    # ========================================================
    """
    /**********************************************************************************************
    The following events that may cause stalls:
    1      Issue: ibuffer_empty
    2      Issue: control hazard
    3      Issue: m_[memory]_out has no free slot
    4      Issue: previous_issued_inst_exec_type is [memory]
    5      Issue: m_[compute]_out has no free slot
    6      Issue: previous_issued_inst_exec_type is [compute]
    7      Issue: scoreboard
    8      Execute: m_dispatch_reg of fu\[\d+\]-\w+\s is not empty
    9      Execute: result_bus has no slot for latency-\d+
    10     Execute: icnt_injection_buffer is full
    11     Execute: bank-\d+ of reg-\d+ is not idle
    12     ReadOperands: bank\[\d+\] reg-\d+ \(order:\d+\) belonged to was allocated for write
    13     ReadOperands: bank\[\d+\] reg-\d+ \(order:\d+\) belonged to was allocated for other regs
    14     ReadOperands: port_num-\d+/m_in_ports\[\d+\].m_in\[\d+\] fails as not found free cu
    15     Writeback: bank-\d+ of reg-\d+ is not idle
    **********************************************************************************************/
    /**********************************************************************************************
    The following events that may cause stalls:
    
    Compute_Structural_Stall:
        1      Issue: m_[compute]_out has no free slot
        2      Issue: previous_issued_inst_exec_type is [compute]
        3      Execute: [compute] result_bus has no slot for latency-\d+
        4      Execute: [compute] m_dispatch_reg of fu\[\d+\]-\w+\s is not empty
        5      Writeback: bank-\d+ of reg-\d+ is not idle
        6      ReadOperands: bank\[\d+\] reg-\d+ \(order:\d+\) belonged to was allocated
        7      ReadOperands: port_num-\d+/m_in_ports\[\d+\].m_in\[\d+\] fails as not found free cu
    Compute_Data_Stall:
        9      Issue: [compute] scoreboard
    Memory_Structural_Stall:
        1      Issue: m_[memory]_out has no free slot
        2      Issue: previous_issued_inst_exec_type is [memory]
        3      Execute: [memory] result_bus has no slot for latency-\d+
        4      Execute: [memory] m_dispatch_reg of fu\[\d+\]-\w+\s is not empty
        5      Writeback: bank-\d+ of reg-\d+ is not idle
        6      ReadOperands: bank\[\d+\] reg-\d+ \(order:\d+\) belonged to was allocated
        7      ReadOperands: port_num-\d+/m_in_ports\[\d+\].m_in\[\d+\] fails as not found free cu
        8      Execute: icnt_injection_buffer is full
    Memory_Data_Stall:
        9      Issue: [memory] scoreboard
        10     L1 // first calculate 9, and then allocate the remaining stalls to 10,11,12
        11     L2
        12     Main Memory
    **********************************************************************************************/
    """