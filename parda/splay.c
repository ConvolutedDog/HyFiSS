#include "splay.h"

Tree *splay(T i, Tree *t)

{
  Tree N, *l, *r, *y;
  T comp, l_size, r_size;
  if (t == NULL)
    return t;
  N.left = N.right = NULL;
  l = r = &N;
  l_size = r_size = 0;

  for (;;) {
    comp = compare(i, t->key);
    if (comp < 0) {
      if (t->left == NULL)
        break;
      if (compare(i, t->left->key) < 0) {
        y = t->left;
        t->left = y->right;
        y->right = t;
        t->size = node_size(t->left) + node_size(t->right) + 1;
        t = y;
        if (t->left == NULL)
          break;
      }
      r->left = t;
      r = t;
      t = t->left;
      r_size += 1 + node_size(r->right);
    } else if (comp > 0) {
      if (t->right == NULL)
        break;
      if (compare(i, t->right->key) > 0) {
        y = t->right;
        t->right = y->left;
        y->left = t;
        t->size = node_size(t->left) + node_size(t->right) + 1;
        t = y;
        if (t->right == NULL)
          break;
      }
      l->right = t;
      l = t;
      t = t->right;
      l_size += 1 + node_size(l->left);
    } else {
      break;
    }
  }
  l_size += node_size(t->left);
  r_size += node_size(t->right);
  t->size = l_size + r_size + 1;

  l->right = r->left = NULL;

  for (y = N.right; y != NULL; y = y->right) {
    y->size = l_size;
    l_size -= 1 + node_size(y->left);
  }
  for (y = N.left; y != NULL; y = y->left) {
    y->size = r_size;
    r_size -= 1 + node_size(y->right);
  }

  l->right = t->left;
  r->left = t->right;
  t->left = N.right;
  t->right = N.left;

  return t;
}

Tree *insert(T i, Tree *t) {

  Tree *new_;

  if (t != NULL) {
    t = splay(i, t);
    if (compare(i, t->key) == 0) {
      return t;
    }
  }
  new_ = (Tree *)malloc(sizeof(Tree));
  if (new_ == NULL) {
    printf("Ran out of space\n");
    exit(1);
  }
  if (t == NULL) {
    new_->left = new_->right = NULL;
  } else if (compare(i, t->key) < 0) {
    new_->left = t->left;
    new_->right = t;
    t->left = NULL;
    t->size = 1 + node_size(t->right);
  } else {
    new_->right = t->right;
    new_->left = t;
    t->right = NULL;
    t->size = 1 + node_size(t->left);
  }
  new_->key = i;
  new_->size = 1 + node_size(new_->left) + node_size(new_->right);
  return new_;
}

Tree *delete_(T i, Tree *t) {

  Tree *x;
  T tsize;

  if (t == NULL)
    return NULL;
  tsize = t->size;
  t = splay(i, t);
  if (compare(i, t->key) == 0) {
    if (t->left == NULL) {
      x = t->right;
    } else {
      x = splay(i, t->left);
      x->right = t->right;
    }
    free(t);
    if (x != NULL) {
      x->size = tsize - 1;
    }
    return x;
  } else {
    return t;
  }
}

Tree *find_rank(T r, Tree *t) {

  T lsize;
  if ((r < 0) || (r >= node_size(t)))
    return NULL;
  for (;;) {
    lsize = node_size(t->left);
    if (r < lsize) {
      t = t->left;
    } else if (r > lsize) {
      r = r - lsize - 1;
      t = t->right;
    } else {
      return t;
    }
  }
}
void freetree(Tree *t) {
  if (t == NULL)
    return;
  freetree(t->right);
  freetree(t->left);
  free(t);
}
void printtree(Tree *t, int d) {

  int i;
  if (t == NULL)
    return;
  printtree(t->right, d + 1);
  for (i = 0; i < d; i++)
    printf("  ");
  printf("%d(%d)\n", t->key, t->size);
  printtree(t->left, d + 1);
}
