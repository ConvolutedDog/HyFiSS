#include "common_def.h"

int kernel_info_m_next_uid = 0;

unsigned long long GLOBAL_HEAP_START = 0xC0000000;

unsigned long long SHARED_MEM_SIZE_MAX = 96 * (1 << 10);

unsigned long long LOCAL_MEM_SIZE_MAX = 1 << 14;

unsigned MAX_STREAMING_MULTIPROCESSORS = 80;

unsigned MAX_THREAD_PER_SM = 1 << 11;

unsigned MAX_WARP_PER_SM = 1 << 6;
unsigned long long TOTAL_LOCAL_MEM_PER_SM =
    MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;
unsigned long long TOTAL_SHARED_MEM =
    MAX_STREAMING_MULTIPROCESSORS * SHARED_MEM_SIZE_MAX;
unsigned long long TOTAL_LOCAL_MEM =
    MAX_STREAMING_MULTIPROCESSORS * MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;
unsigned long long SHARED_GENERIC_START = GLOBAL_HEAP_START - TOTAL_SHARED_MEM;
unsigned long long LOCAL_GENERIC_START = SHARED_GENERIC_START - TOTAL_LOCAL_MEM;
unsigned long long STATIC_ALLOC_LIMIT =
    GLOBAL_HEAP_START - (TOTAL_LOCAL_MEM + TOTAL_SHARED_MEM);
