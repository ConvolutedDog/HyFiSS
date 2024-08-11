#include "parda.h"

void parda_print_front(const program_data_t *pdt) {
  narray_t *ga = pdt->ga;
  unsigned i;
  unsigned len = narray_get_len(ga);
  printf("< ");
  for (i = 0; i < len; i++) {
    printf("%s ", ((HKEY *)ga->data)[i]);
  }
  printf(">\n");
}

void parda_print_end(const end_keytime_t *ekt) {
  narray_t *gkeys = ekt->gkeys;
  narray_t *gtimes = ekt->gtimes;

  unsigned len = narray_get_len(gkeys);

  unsigned i;
  printf("[ ");
  for (i = 0; i < len; i++) {
    printf("(%s:%d) ", ((HKEY *)(gkeys->data))[i], ((T *)(gtimes->data))[i]);
  }
  printf("]\n");
}

void parda_print_tree(const program_data_t *pdt) {
  Tree *root = pdt->root;
  printtree(root, 0);
}

void print_iterator(gpointer key, gpointer value, gpointer ekt) {
  printf("(%s:%d) ", (char *)key, *(T *)value);
}

void parda_print_hash(const program_data_t *pdt) {
  printf("[ ");
  g_hash_table_foreach(pdt->gh, (GHFunc)print_iterator, NULL);
  printf("]\n");
}

void parda_print(const program_data_t *pdt) {
  parda_print_front(pdt);
  parda_print_tree(pdt);
  parda_print_hash(pdt);
}

void parda_print_histogram(const unsigned *histogram) {
  int last_bucket;
  int i;
  unsigned long long sum = 0;
  unsigned long long cum = 0;

  last_bucket = nbuckets - 1;
  while (histogram[last_bucket] == 0)
    last_bucket--;

  for (i = 0; i <= last_bucket; i++)
    sum += histogram[i];
  sum += histogram[B_OVFL];
  sum += histogram[B_INF];

  printf("# Dist\t     Refs\t   Refs(%%)\t  Cum_Ref\tCum_Ref(%%)\n");

  for (i = 0; i <= last_bucket; i++) {
    cum += histogram[i];
    if (histogram[i])
      printf("%6d\t%9u\t%0.8lf\t%9llu\t%0.8lf\n", i, histogram[i],
             histogram[i] / (double)sum, cum, cum / (double)sum);
  }

  cum += histogram[B_OVFL];
  printf("#OVFL \t%9u\t%0.8f\t%9llu\t%0.8lf\n", histogram[B_OVFL],
         histogram[B_OVFL] / (double)sum, cum, cum / (double)sum);
  cum += histogram[B_INF];
  printf("#INF  \t%9u\t%0.8f\t%9llu\t%0.8lf\n", histogram[B_INF],
         histogram[B_INF] / (double)sum, cum, cum / (double)sum);
}

void parda_fprintf_histogram(const unsigned *histogram, FILE *file) {
  int last_bucket;
  int i;
  unsigned long long sum = 0;
  unsigned long long cum = 0;

  last_bucket = nbuckets - 1;
  while (histogram[last_bucket] == 0)
    last_bucket--;

  for (i = 0; i <= last_bucket; i++)
    sum += histogram[i];
  sum += histogram[B_OVFL];
  sum += histogram[B_INF];

  fprintf(file, "# Dist\t     Refs\t   Refs(%%)\t  Cum_Ref\tCum_Ref(%%)\n");

  for (i = 0; i <= last_bucket; i++) {
    cum += histogram[i];
    if (histogram[i])
      fprintf(file, "%6d\t%9u\t%0.8lf\t%9llu\t%0.8lf\n", i, histogram[i],
              histogram[i] / (double)sum, cum, cum / (double)sum);
  }

  cum += histogram[B_OVFL];
  fprintf(file, "#OVFL \t%9u\t%0.8f\t%9llu\t%0.8lf\n", histogram[B_OVFL],
          histogram[B_OVFL] / (double)sum, cum, cum / (double)sum);
  cum += histogram[B_INF];
  fprintf(file, "#INF  \t%9u\t%0.8f\t%9llu\t%0.8lf\n", histogram[B_INF],
          histogram[B_INF] / (double)sum, cum, cum / (double)sum);
}

float parda_fprintf_histogram_r(const unsigned *histogram, FILE *file,
                                bool print = true) {
  int last_bucket;
  int i;
  unsigned long long sum = 0;
  unsigned long long cum = 0;

  last_bucket = nbuckets - 1;
  while (histogram[last_bucket] == 0)
    last_bucket--;

  for (i = 0; i <= last_bucket; i++)
    sum += histogram[i];
  sum += histogram[B_OVFL];
  sum += histogram[B_INF];

  if (print)
    fprintf(file, "# Dist\t     Refs\t   Refs(%%)\t  Cum_Ref\tCum_Ref(%%)\n");

  for (i = 0; i <= last_bucket; i++) {
    cum += histogram[i];
    if (histogram[i] && print)
      fprintf(file, "%6d\t%9u\t%0.8lf\t%9llu\t%0.8lf\n", i, histogram[i],
              histogram[i] / (double)sum, cum, cum / (double)sum);
  }

  cum += histogram[B_OVFL];
  if (print)
    fprintf(file, "#OVFL \t%9u\t%0.8f\t%9llu\t%0.8lf\n", histogram[B_OVFL],
            histogram[B_OVFL] / (double)sum, cum, cum / (double)sum);

  float hit_rate = cum / (double)sum;

  cum += histogram[B_INF];
  if (print)
    fprintf(file, "#INF  \t%9u\t%0.8f\t%9llu\t%0.8lf\n", histogram[B_INF],
            histogram[B_INF] / (double)sum, cum, cum / (double)sum);

  return hit_rate;
}