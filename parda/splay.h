#ifndef _splay_h
#define _splay_h
#include <stdio.h>
#include <stdlib.h>

typedef struct tree_node Tree;
typedef int T;
struct tree_node {
  Tree *left, *right;
  T key;
  T size;
};

#define compare(i, j) ((i) - (j))

#define node_size(x) (((x) == NULL) ? 0 : ((x)->size))

Tree *splay(T i, Tree *t);
Tree *insert(T i, Tree *t);
Tree *delete_(T i, Tree *t);
Tree *find_rank(T r, Tree *t);
void printtree(Tree *t, int d);
void freetree(Tree *t);
#endif
