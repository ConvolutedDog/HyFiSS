USE_BOOST ?= 1
DEBUG ?= 0
USE_GPROF ?= 0

BOOST_PATH := $(shell echo $$LD_LIBRARY_PATH | tr ':' '\n' | grep boost/lib | head -n 1)
ifeq ($(BOOST_PATH),)
	BOOST_HOME ?=
else
	BOOST_HOME := $(shell dirname $(BOOST_PATH))
#	$(info Using BOOST_HOME: $(BOOST_HOME))
endif

MPICC_PATH := $(shell which mpicc)
# $(info Using MPICC_PATH: $(MPICC_PATH))
MPI_PATH := $(shell dirname $(MPICC_PATH))
# $(info Using MPI_PATH: $(MPI_PATH))
MPI_HOME ?= $(shell dirname $(MPI_PATH))
# $(info Using MPI_HOME: $(MPI_HOME))

MPICXX = $(shell which mpic++)
MPIRUN = $(shell which mpirun)

ifeq ($(USE_BOOST),1)
	CXX = $(MPICXX)
	CC = $(MPICXX)
else
	CXX = g++
	CC = gcc
endif

CXXFLAGS = -Wall -finline-functions -funswitch-loops -MMD -MP

ifeq ($(USE_GPROF),1)
	CXXFLAGS += -pg
endif

CFLAGS = $(CXXFLAGS)

# Detect Support for C++11 (C++0x) from GCC Version 
GNUC_CPP0X := $(shell mpic++ --version | perl -ne 'if (/g++\s+\(.*\)\s+([0-9.]+)/){ if($$1 >= 4.3) {$$n=1} else {$$n=0;} } END { print $$n; }')

ifeq ($(GNUC_CPP0X), 1)
	CXXFLAGS += -std=c++11
endif

INC_DIRS = -I./hw-parser -I./hw-component -I./ISA-Def -I./DEV-Def -I./trace-parser -I./trace-driven -I./common -I./common/CLI -I./common/CLI/impl -I$(MPI_HOME)/include -I$(BOOST_HOME)/include -I./parda
CXXFLAGS += $(INC_DIRS) $(shell pkg-config --cflags glib-2.0)
CFLAGS += $(INC_DIRS)

LIBRARIES = -L$(BOOST_HOME)/lib -lboost_mpi -lboost_serialization
LIBRARIES += $(shell pkg-config --libs glib-2.0)

ifeq ($(DEBUG),1)
	OPTFLAGS = -O0 -g3 -fPIC
else
	OPTFLAGS = -O3 -fPIC
endif

OBJ_PATH = obj

TARGET = gpu-simulator.x

exist_OBJ_PATH = $(shell if [ -d $(OBJ_PATH) ]; then echo "exist"; else echo "noexist"; fi)

ifeq ("$(exist_OBJ_PATH)", "noexist")
$(shell mkdir $(OBJ_PATH))
endif

CC_SRCS := $(wildcard *.c) $(wildcard parda/*.c)
CC_SRCS := $(filter-out parda/parda_mpi.c parda/parda_omp.c parda/main.c parda/seperate.c, $(CC_SRCS))

CXX_SRCS := $(wildcard *.cc) $(wildcard trace-parser/*.cc) $(wildcard trace-driven/*.cc) 
CXX_SRCS += $(wildcard hw-component/*.cc) $(wildcard hw-parser/*.cc) $(wildcard common/*.cc)

SRCS := $(CC_SRCS) $(CXX_SRCS)

CC_OBJS := $(CC_SRCS:%.c=$(OBJ_PATH)/%.o)
CXX_OBJS := $(CXX_SRCS:%.cc=$(OBJ_PATH)/%.o)

OBJS := $(CXX_OBJS) $(CC_OBJS) 

DEPS := $(OBJS:.o=.d)

-include $(DEPS)

default: all

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ $^ $(LIBRARIES) 

$(OBJ_PATH)/%.o: %.cc
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -c $< -o $@

$(OBJ_PATH)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) $(OPTFLAGS) -c $< -o $@

.PHONY: clean

clean:
	rm -f $(OBJS) $(DEPS)
	rm -f $(TARGET)
	rm -rf $(OBJ_PATH)
