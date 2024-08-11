#include "parda.h"
#ifdef enable_omp
#include "parda_omp.h"
#endif
#include "parda_mpi.h"

narray_t *parda_recv_array(int source, int *tag, unsigned element_size) {
  narray_t *ga;
  MPI_Status status;
  unsigned blen;
  void *bdata;
  MPI_Recv(&blen, 1, MPI_UNSIGNED, source, (*tag)++, MPI_COMM_WORLD, &status);
  bdata = (char *)calloc(blen, 1);
  MPI_Recv(bdata, blen, MPI_CHAR, source, (*tag)++, MPI_COMM_WORLD, &status);
  ga = narray_heaparray_new(bdata, blen, element_size);
  return ga;
}

void parda_send_array(narray_t *ga, int dest, int *tag) {
  MPI_Send(&ga->len, 1, MPI_UNSIGNED, dest, (*tag)++, MPI_COMM_WORLD);
  MPI_Send(ga->data, ga->len, MPI_CHAR, dest, (*tag)++, MPI_COMM_WORLD);
}

unsigned *parda_mpi_merge(program_data_t *pdt, processor_info_t *pit) {
  int i, len;
  int psize = pit->psize;
  int pid = pit->pid;
  int var, tag = 1;
  for (var = pid, len = 1; var % 2 == 1; var = (var >> 1), len = (len << 1)) {
    end_keytime_t ekt_a;
    int dest = pid - len;
    parda_send_array(pdt->ga, dest, &tag);
    ekt_a.gkeys = parda_recv_array(dest, &tag, sizeof(HKEY));
    ekt_a.gtimes = parda_recv_array(dest, &tag, sizeof(T));
    parda_get_abend(pdt, &ekt_a);
    narray_t *mga = parda_recv_array(dest, &tag, sizeof(HKEY));
    narray_free(pdt->ga);
    pdt->ga = mga;
  }
  if (pid + len < psize) {
    int source = pid + len;
    pdt->ekt = parda_generate_end(pdt);
    narray_t *gb = parda_recv_array(source, &tag, sizeof(HKEY));
    parda_send_array(pdt->ekt.gkeys, source, &tag);
    parda_send_array(pdt->ekt.gtimes, source, &tag);
    parda_get_abfront(pdt, gb, pit);
    parda_send_array(pdt->ga, source, &tag);
    narray_free(pdt->ekt.gkeys);
    narray_free(pdt->ekt.gtimes);
  } else if (pid == psize - 1) {
    pdt->histogram[B_INF] += narray_get_len(pdt->ga);
  }
  unsigned *global_his = (unsigned *)malloc(sizeof(unsigned) * (nbuckets + 2));

  for (i = 0; i < nbuckets + 2; i++) {
    global_his[i] = 0;
  }
  MPI_Reduce(pdt->histogram, global_his, nbuckets + 2, MPI_UNSIGNED, MPI_SUM, 0,
             MPI_COMM_WORLD);
  return global_his;
}
int parda_MPI_IO_binary_input(program_data_t *pdt, char filename[],
                              const processor_info_t *pit) {
  MPI_File thefile;
  MPI_Status status;
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL,
                &thefile);
  MPI_File_set_view(thefile, pit->tstart * sizeof(void *), MPI_LONG, MPI_LONG,
                    "native", MPI_INFO_NULL);
#ifdef ENABLE_PROFILING
  double t3, t4;
  MPI_Barrier(MPI_COMM_WORLD);
  t3 = MPI_Wtime();
#endif
  GHashTable *gh = pdt->gh;
  Tree *root = pdt->root;
  narray_t *ga = pdt->ga;
  int bufsize = 10000;
  void **buf = (void **)malloc(bufsize * sizeof(void *));
  unsigned int *histogram = pdt->histogram;
  HKEY input;
  long tim, begin;
  int count, i;
  for (tim = begin = pit->tstart; begin <= pit->tend; begin += count) {
    MPI_File_read(thefile, buf, bufsize, MPI_LONG, &status);
    MPI_Get_count(&status, MPI_LONG, &count);
    if (begin + count > pit->tend + 1) {
      count = pit->tend + 1 - begin;
    }
    for (i = 0; i < count; i++) {
      sprintf(input, "%p", buf[i]);
      int distance;
      T *lookup;
      lookup = g_hash_table_lookup(gh, input);

      if (lookup == NULL) {
        char *data = strdup(input);
        root = insert(tim, root);
        T *p_data;

        narray_append_val(ga, input);
        if (!(p_data = (T *)malloc(sizeof(T))))
          return -1;
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
          return -1;
        *p_data = tim;
        g_hash_table_replace(gh, strdup(input), p_data);

        if (distance > nbuckets)
          histogram[B_OVFL]++;
        else
          histogram[distance]++;
      }
      tim++;
    }
  }
  printf("start from %ld to %ld\n", pit->tstart, tim);
#ifdef ENABLE_PROFILING
  t4 = MPI_Wtime();
  int pid = pit->pid;
  fprintf(pid_fp,
          "parda input time with barrier = %.3lf sec for processor %d; \n",
          t4 - t3, pid);
#endif
  pdt->root = root;
  return 1;
}

void parda_mpi_stackdist(char *inputFileName, long lines, int processors,
                         int argc, char **argv) {
  int pid, psize;
  program_data_t pdt;
  long psum;
  processor_info_t pit;
  MPI_Init(&argc, &argv);
  process_args(argc, argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);
  psum = lines;
  MPI_Bcast(&psum, 1, MPI_INT, 0, MPI_COMM_WORLD);
#ifdef enable_timing
  double ts, te, t_init, t_input, t_print, t_free;
  ts = MPI_Wtime();
#endif
  pit = parda_get_processor_info(pid, psize, psum);
  pdt = parda_init();
  PTIME(MPI_Barrier(MPI_COMM_WORLD);)
  PTIME(te = MPI_Wtime();)
  PTIME(t_init = te - ts;)
  parda_input_with_filename(parda_generate_pfilename(inputFileName, pid, psize),
                            &pdt, pit.tstart, pit.tend);
  unsigned *global_his = parda_mpi_merge(&pdt, &pit);
  PTIME(MPI_Barrier(MPI_COMM_WORLD);)
  PTIME(ts = MPI_Wtime();)
  PTIME(t_input = ts - te;)
  if (pid == 0) {
    parda_print_histogram(global_his);
  }
  PTIME(te = MPI_Wtime();)
  PTIME(t_print = te - ts;)
  parda_free(&pdt);
  free(global_his);
  PTIME(ts = MPI_Wtime();)
  PTIME(t_free = ts - te;)
#ifdef enable_timing
  if (pid == 0) {
    printf("mpi\n");
    printf("init time is %lf\n", t_init);
    printf("input time is %lf\n", t_input);
    printf("print time is %lf\n", t_print);
    printf("free time is %lf\n", t_free);
  }
#endif
  MPI_Finalize();
}

#if defined(enable_omp) && defined(enable_mpi)
void parda_hybrid_stackdist(char *inputFileName, long lines, int processors,
                            int argc, char **argv) {
  int pid, psize;
  program_data_t pdt;
  long psum;
  processor_info_t pit;
  MPI_Init(&argc, &argv);
  process_args(argc, argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &psize);
  DEBUG(if (pid == 0))
  DEBUG(printf("enter hybrid\n");)

  psum = lines;
  MPI_Bcast(&psum, 1, MPI_INT, 0, MPI_COMM_WORLD);
  pit = parda_get_processor_info(pid, psize, psum);
  program_data_t *pdt_a = parda_omp_init(threads);

  pdt = parda_omp_input(inputFileName, pdt_a, pit.tstart, pit.tend, pid, psize);
  parda_omp_free(pdt_a, threads);

  unsigned *global_his = parda_mpi_merge(&pdt, &pit);

  if (pid == 0) {
    parda_print_histogram(global_his);
  }
  parda_free(&pdt);
  free(global_his);
  MPI_Finalize();
}
#endif
