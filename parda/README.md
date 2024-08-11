Reuse distance is a well established approach to characterizing data cache locality based on the stack histogram model. 
This analysis so far has been restricted to ofﬂine use due to the high cost, often several orders of magnitude larger than the execution time of the analyzed code. Parda is the ﬁrst parallel algorithm to compute accurate reuse distances by analysis of memory address traces. The algorithm uses a tunable parameter that enables faster analysis when the maximum needed reuse distance is limited by a cache size upper bound. 

This program is a Parda implementation on file input. parda omp implementation is mainly in `parda_omp.c` and `parda_omp.h`. 

## Instructions to run file input Parda. 

### A. Setup and compile

`Step 0:` parda use glib standard linux library. If on ubuntu system just execute following sudo command.

```shell
sudo apt-get install glib
```

`Step 1:` Download sample trace files from project git web page. 

`normal_137979.trace` is text file and 
`binary_137979.trace` is the binary file.
This two files record trace data of `ls` command. 

`Step 2:`

```shell
cd /path/to/parda
```

Current program only tests with `gcc` and `icc`. 
Edit the first three lines of makefile. 
If machine has `mpicc`, give `MPI=1` option to enable mpi parallelism. 
Otherwise, give `OMP=1`. If use only sequential algorithm, comments both `OMP` and `MPI`.

```makefile
DEBUG = 1
OMP = 1
MPI = 1
```

```shell
make
```

### B. Execution instructions

```shell
./parda.x --help to see how to run with different flags and run with sequential algorithm. 
```

#### Execution arguments:

```makefile
--input: the input trace file name.
--lines: the total number of lines in the input trace file. 
--enable-omp: enable program to parallelly run with OpenMP threads.
--enable-mpi: enable program to parallelly run with MPI.
--enable-seperate: Seperate the input file to prepare for running with paralellization.
```

#### 1) Sequential execution:

```shell
./parda.x --input=normal_137979.trace --lines=137979 > seq.hist 
```

#### 2) Run parda with OpenMP `--enable-omp` flag. 

Before running with omp we need to seperate the trace files to threads number. For example if we want to run with 4 threads. 

```shell
./parda.x --enable-seperate --input=normal_137979.trace --lines=137979 --threads=4
```

We will find 4 seperated trace files:

`4_normal_137979.trace_p0.txt`  `4_normal_137979.trace_p1.txt`
`4_normal_137979.trace_p2.txt`  `4_normal_137979.trace_p3.txt`

```shell
./parda.x --enable-omp --input=normal_137979.trace --lines=137979 --threads=4 > omp.re 
```

#### 3) Run parda with MPI

```shell
mpirun -np 4 ./parda.x --input=normal_137979.trace --lines=137979 --enable-mpi
```

### Parda

Parda is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Parda is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

### Author: 
Qingpeng Niu

### Contact: 
niuqingpeng at gmail.com

### Documententation

#### Related publications:

```
PARDA: A Fast Parallel Reuse Distance Analysis Algorithm.
Qingpeng Niu, James Dinan, Qingda Lu and P. Sadayappan.
IEEE IPDPS (IPDPS'12), May 2012, Shanghai, China.
```