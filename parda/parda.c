#include "parda.h"
#include "narray.h"

#include <omp.h>

int nbuckets = DEFAULT_NBUCKETS;
void iterator(gpointer key, gpointer value, gpointer ekt) {
  HKEY temp;

  strcpy((char *)temp, (char *)key);
  narray_append_val(((end_keytime_t *)ekt)->gkeys, temp);
  narray_append_val(((end_keytime_t *)ekt)->gtimes, value);
}

end_keytime_t parda_generate_end(const program_data_t *pdt) {
  GHashTable *gh = pdt->gh;
  end_keytime_t ekt;
  ekt.gkeys = narray_new(sizeof(HKEY), 1000);
  ekt.gtimes = narray_new(sizeof(T), 1000);
  g_hash_table_foreach(gh, (GHFunc)iterator, &ekt);
  return ekt;
}

program_data_t parda_init() {
  program_data_t pdt;
  GHashTable *gh;
  narray_t *ga = narray_new(sizeof(HKEY), 1000);
  Tree *root;
  unsigned int *histogram;
  histogram = (unsigned int *)malloc(sizeof(unsigned int) * (nbuckets + 2));
  gh = g_hash_table_new_full(g_str_hash, compare_strings, free, free);
  root = NULL;
  memset(histogram, 0, (nbuckets + 2) * sizeof(unsigned int));
  pdt.ga = ga;
  pdt.gh = gh;
  pdt.root = root;
  pdt.histogram = histogram;
  return pdt;
}

gboolean compare_strings(gconstpointer a, gconstpointer b) {

  if (strcmp((const char *)(a), (const char *)(b)) == 0)
    return TRUE;
  else
    return FALSE;
}

void parda_process(char *input, T tim, program_data_t *pdt) {
  GHashTable *gh = pdt->gh;
  Tree *root = pdt->root;
  narray_t *ga = pdt->ga;
  unsigned int *histogram = pdt->histogram;
  int distance;
  T *lookup;
  lookup = (T *)g_hash_table_lookup(gh, input);
  if (lookup == NULL) {
    char *data = strdup(input);
    root = insert(tim, root);
    T *p_data;

    narray_append_val(ga, input);
    if (!(p_data = (T *)malloc(sizeof(T))))
      exit(1);
    *p_data = tim;
    g_hash_table_insert(gh, data, p_data);
  }

  else {
    root = insert((*lookup), root);
    distance = node_size(root->right);
    root = delete_(*lookup, root);
    root = insert(tim, root);
    int *p_data;
    if (!(p_data = (int *)malloc(sizeof(int))))
      exit(1);
    *p_data = tim;
    g_hash_table_replace(gh, strdup(input), p_data);

    if (distance > nbuckets)
      histogram[B_OVFL]++;
    else
      histogram[distance]++;
  }
  pdt->root = root;
}

processor_info_t parda_get_processor_info(int pid, int psize, long sum) {
  processor_info_t pit;
  pit.tstart = parda_low(pid, psize, sum);
  pit.tend = parda_high(pid, psize, sum);
  pit.tlen = parda_size(pid, psize, sum);
  pit.sum = sum;
  pit.pid = pid;
  pit.psize = psize;
  return pit;
}

void parda_get_abfront(program_data_t *pdt_a, const narray_t *gb,
                       const processor_info_t *pit_a) {

  T tim = pit_a->tend + 1;
  GHashTable *gh = pdt_a->gh;
  narray_t *ga = pdt_a->ga;
  Tree *root = pdt_a->root;
  unsigned *histogram = pdt_a->histogram;
  unsigned i;
  T *lookup;
  T distance;
  unsigned len = narray_get_len(gb);

  for (i = 0; i < len; i++, tim++) {
    HKEY entry;
    char *temp = ((HKEY *)gb->data)[i];
    strcpy(entry, temp);
    lookup = (T *)g_hash_table_lookup(gh, entry);

    if (lookup == NULL) {
      narray_append_val(ga, entry);
      root = insert(tim, root);
    } else {
      root = insert((*lookup), root);
      distance = node_size(root->right);
      root = delete_(*lookup, root);
      root = insert(tim, root);
      if (distance > nbuckets)
        histogram[B_OVFL]++;
      else
        histogram[distance]++;
    }
  }
  pdt_a->root = root;
  pdt_a->gh = gh;
}

int parda_get_abend(program_data_t *pdt_b, const end_keytime_t *ekt_a) {
  Tree *root = pdt_b->root;
  GHashTable *gh = pdt_b->gh;
  narray_t *gkeys = ekt_a->gkeys;
  narray_t *gtimes = ekt_a->gtimes;
  unsigned len = narray_get_len(gkeys);
  unsigned i;
  HKEY key;
  T tim;
  T *lookup;
  for (i = 0; i < len; i++) {
    char *temp = ((HKEY *)gkeys->data)[i];
    strcpy(key, temp);
    tim = ((T *)(gtimes->data))[i];
    lookup = (T *)g_hash_table_lookup(gh, key);
    if (lookup == NULL) {
      char *data = strdup(key);
      root = insert(tim, root);
      T *p_data;
      if (!(p_data = (T *)malloc(sizeof(T))))
        return -1;
      *p_data = tim;
      g_hash_table_insert(gh, data, p_data);
    }
  }
  pdt_b->root = root;
  pdt_b->gh = gh;
  return 0;
}

program_data_t parda_merge(program_data_t *pdt_a, program_data_t *pdt_b,
                           const processor_info_t *pit_b) {
  program_data_t pdt;
  parda_get_abfront(pdt_a, pdt_b->ga, pit_b);
  DEBUG(printf("after get abfront %d\n", pit_b->pid);)
  narray_free(pdt_b->ga);
  pdt_a->ekt = parda_generate_end(pdt_a);
  parda_get_abend(pdt_b, &pdt_a->ekt);
  narray_free(pdt_a->ekt.gkeys);
  narray_free(pdt_a->ekt.gtimes);
  pdt.ga = pdt_a->ga;
  pdt.root = pdt_b->root;
  pdt.gh = pdt_b->gh;
  pdt.histogram = pdt_a->histogram;
  int i;
  for (i = 0; i < nbuckets + 2; i++)
    pdt.histogram[i] += pdt_b->histogram[i];
  free(pdt_b->histogram);
  return pdt;
}

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0)
    printf("Error return from gettimeofday: %d", stat);
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void classical_tree_based_stackdist(char *inputFileName, long lines) {
#ifdef enable_timing
  double ts, te, t_init, t_input, t_print, t_free;
  ts = rtclock();
#endif
  program_data_t pdt_c = parda_init();
  PTIME(te = rtclock();)
  PTIME(t_init = te - ts;)
  parda_input_with_filename(inputFileName, &pdt_c, 0, lines - 1);
  program_data_t *pdt = &pdt_c;
  pdt->histogram[B_INF] += narray_get_len(pdt->ga);
  PTIME(ts = rtclock();)
  PTIME(t_input = ts - te;)
  parda_print_histogram(pdt->histogram);
  PTIME(te = rtclock();)
  PTIME(t_print = te - ts;)
  parda_free(pdt);
  PTIME(ts = rtclock();)
  PTIME(t_free = ts - te;)
#ifdef enable_timing
  printf("seq\n");
  printf("init time is %lf\n", t_init);
  printf("input time is %lf\n", t_input);
  printf("print time is %lf\n", t_print);
  printf("free time is %lf\n", t_free);
#endif
}

void parda_input_with_filename(char *inFileName, program_data_t *pdt,
                               long begin, long end) {
  DEBUG(printf("enter parda_input < %s from %ld to %ld\n", inFileName, begin,
               end);)
  FILE *fp;
  if (!is_binary) {
    fp = fopen(inFileName, "r");
    printf("text open haha\n");
    parda_input_with_textfilepointer(fp, pdt, begin, end);
  } else {
    fp = fopen(inFileName, "rb");
    printf("binary open haha\n");
    parda_input_with_binaryfilepointer(fp, pdt, begin, end);
  }
  fclose(fp);
}

void parda_input_with_binaryfilepointer(FILE *fp, program_data_t *pdt,
                                        long begin, long end) {
  HKEY input;
  long t, i;
  long count;
  void **buffer = (void **)malloc(buffersize * sizeof(void *));
  for (t = begin; t <= end; t += count) {
    count = fread(buffer, sizeof(void *), buffersize, fp);
    for (i = 0; i < count; i++) {
      sprintf(input, "%p", buffer[i]);
      DEBUG(printf("%s %d\n", input, i + t);)
      process_one_access(input, pdt, i + t);
    }
  }
}

void parda_input_with_textfilepointer(FILE *fp, program_data_t *pdt, long begin,
                                      long end) {
  HKEY input;
  long i;
  for (i = begin; i <= end; i++) {
    assert(fscanf(fp, "%s", input) != EOF);
    DEBUG(printf("%s %d\n", input, i);)
    process_one_access(input, pdt, i);
  }
}

void parda_free(program_data_t *pdt) {
  narray_free(pdt->ga);

  g_hash_table_destroy(pdt->gh);
  free(pdt->histogram);
  freetree(pdt->root);
}
