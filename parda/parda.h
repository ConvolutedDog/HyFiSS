#ifndef _PARDA_H
#define _PARDA_H

#include "narray.h"
#include "process_args.h"
#include "splay.h"

#include <assert.h>
#include <glib.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#ifdef enable_mpi
#ifdef enable_omp
#define enable_hybrid
#endif
#endif

#define enable_timing
#ifdef enable_timing
#define PTIME(cmd) cmd
#else
#define PTIME(cmd)
#endif

#ifdef enable_debugging
#define DEBUG(cmd) cmd
#else
#define DEBUG(cmd)
#endif

#ifdef enable_profiling
#define PROF(cmd) cmd
#else
#define PROF(cmd)
#endif

#define DEFAULT_NBUCKETS 1000000
#define B_OVFL nbuckets
#define B_INF nbuckets + 1
#define SLEN 20

extern int nbuckets;
#ifdef ENABLE_PROFILING
extern char pfile[30];
extern FILE *pid_fp;
#endif

typedef char HKEY[SLEN];

typedef struct end_keytime_s {
  narray_t *gkeys;
  narray_t *gtimes;
} end_keytime_t;

typedef struct processor_info_s {
  int pid, psize;
  long tstart, tlen, tend, sum;
} processor_info_t;

typedef struct program_data_s {
  GHashTable *gh;
  narray_t *ga;
  end_keytime_t ekt;
  Tree *root;
  unsigned int *histogram;
} program_data_t;

void classical_tree_based_stackdist(char *inputFileName, long lines);

gboolean compare_strings(gconstpointer a, gconstpointer b);
void iterator(gpointer key, gpointer value, gpointer ekt);

program_data_t parda_init(void);
void parda_input_with_filename(char *inFileName, program_data_t *pdt,
                               long begin, long end);
void parda_input_with_textfilepointer(FILE *fp, program_data_t *pdt, long begin,
                                      long end);
void parda_input_with_binaryfilepointer(FILE *fp, program_data_t *pdt,
                                        long begin, long end);
void parda_free(program_data_t *pdt);
end_keytime_t parda_generate_end(const program_data_t *pdt);
processor_info_t parda_get_processor_info(int pid, int psize, long sum);
void parda_get_abfront(program_data_t *pdt_a, const narray_t *gb,
                       const processor_info_t *pit_a);
int parda_get_abend(program_data_t *pdt_b, const end_keytime_t *ekt_a);
program_data_t parda_merge(program_data_t *pdt_a, program_data_t *pdt_b,
                           const processor_info_t *pit_b);

void parda_print_front(const program_data_t *pdt);
void parda_print_end(const end_keytime_t *ekt);
void parda_print_tree(const program_data_t *pdt);
void parda_print_hash(const program_data_t *pdt);
void parda_print(const program_data_t *pdt);
void print_iterator(gpointer key, gpointer value, gpointer ekt);
void parda_print_histogram(const unsigned *histogram);
void parda_fprintf_histogram(const unsigned *histogram, FILE *file);
float parda_fprintf_histogram_r(const unsigned *histogram, FILE *file,
                                bool print);

int parda_findopt(char *option, char **value, int *argc, char ***argv);
void parda_process(char *input, T tim, program_data_t *pdt);

void show_hkey(void *data, int i, FILE *fp);
void show_T(void *data, int i, FILE *fp);

double rtclock(void);

static inline T parda_low(int pid, int psize, T sum) {
  return (((long long)(pid)) * (sum) / (psize));
}

static inline T parda_high(int pid, int psize, T sum) {
  return parda_low(pid + 1, psize, sum) - 1;
}

static inline T parda_size(int pid, int psize, T sum) {
  return (parda_low(pid + 1, psize, sum)) - (parda_low(pid, psize, sum));
}

static inline T parda_owner(T index, int psize, T sum) {
  return (((long long)psize) * (index + 1) - 1) / sum;
}

static inline char *parda_generate_pfilename(char filename[], int pid,
                                             int psize) {
  char pfilename[30];
  sprintf(pfilename, "%d_%s_p%d.txt", psize, filename, pid);
  return strdup(pfilename);
}

static inline void process_one_access(char *input, program_data_t *pdt,
                                      const long tim) {
  int distance;
  int *lookup;
  lookup = (T *)g_hash_table_lookup(pdt->gh, input);

  if (lookup == NULL) {
    char *data = strdup(input);
    pdt->root = insert(tim, pdt->root);
    long *p_data;
    narray_append_val(pdt->ga, input);
    if (!(p_data = (long *)malloc(sizeof(long)))) {
      printf("no memory for p_data\n");
      assert(0);
      exit(-1);
    }
    *p_data = tim;
    g_hash_table_insert(pdt->gh, data, p_data);
  }

  else {
    char *data = strdup(input);
    pdt->root = insert((*lookup), pdt->root);
    distance = node_size(pdt->root->right);
    pdt->root = delete_(*lookup, pdt->root);
    pdt->root = insert(tim, pdt->root);
    int *p_data;
    if (!(p_data = (int *)malloc(sizeof(int)))) {
      printf("no memory for p_data\n");
      assert(0);
      exit(-1);
    }
    *p_data = tim;
    g_hash_table_replace(pdt->gh, data, p_data);

    if (distance > nbuckets)
      pdt->histogram[B_OVFL] += 1;
    else
      pdt->histogram[distance] += 1;
  }
}

static inline int process_one_access_and_get_distance(char *input,
                                                      program_data_t *pdt,
                                                      const long tim) {
  int distance;
  int *lookup;
  lookup = (T *)g_hash_table_lookup(pdt->gh, input);

  if (lookup == NULL) {
    char *data = strdup(input);
    pdt->root = insert(tim, pdt->root);
    long *p_data;
    narray_append_val(pdt->ga, input);
    if (!(p_data = (long *)malloc(sizeof(long)))) {
      printf("no memory for p_data\n");
      assert(0);
      exit(-1);
    }
    *p_data = tim;
    g_hash_table_insert(pdt->gh, data, p_data);

    return B_INF;
  }

  else {
    char *data = strdup(input);
    pdt->root = insert((*lookup), pdt->root);
    distance = node_size(pdt->root->right);
    pdt->root = delete_(*lookup, pdt->root);
    pdt->root = insert(tim, pdt->root);
    int *p_data;
    if (!(p_data = (int *)malloc(sizeof(int)))) {
      printf("no memory for p_data\n");
      assert(0);
      exit(-1);
    }
    *p_data = tim;
    g_hash_table_replace(pdt->gh, data, p_data);

    if (distance > nbuckets)
      pdt->histogram[B_OVFL] += 1;
    else
      pdt->histogram[distance] += 1;

    if (distance > nbuckets)
      return B_OVFL;
    else
      return distance;
  }
}
#endif
