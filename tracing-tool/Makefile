NVCC=/usr/local/cuda/bin/nvcc -ccbin=$(CXX) -D_FORCE_INLINES --compiler-options "-pipe"
ARCH=70 75 80

NVCC_VER_REQ=11
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool! Instrumented applications can still use lower versions of nvcc.)
endif

NVBIT_PATH=nvbit
INCLUDES=-I$(NVBIT_PATH)

LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

SOURCES=$(wildcard *.cu)

OBJECTS=$(SOURCES:.cu=.o)

$(foreach sm,$(ARCH),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

NVBIT_TOOL=tracer.so

all: $(NVBIT_TOOL)

$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) $(GENCODE_FLAGS) -O3 $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

%.o: %.cu
	$(NVCC) -dc -c -std=c++11 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall $(GENCODE_FLAGS) -O3 -Xcompiler -fPIC $< -o $@

inject_funcs.o: inject_funcs.cu
	$(NVCC) $(INCLUDES) -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions $(GENCODE_FLAGS) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

clean:
	rm -f *.so *.o