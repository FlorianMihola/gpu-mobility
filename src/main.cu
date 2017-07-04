#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

struct struct_linked_list_node;

typedef struct struct_linked_list_node {
  int data;
  struct struct_linked_list_node *next;
} linked_list_node;

typedef struct {
  linked_list_node *head, *tail;
} linked_list;

__device__ bool _ll_is_marked(linked_list_node *n) {
  return ((unsigned long long int) n & 1);
}

__device__ linked_list_node *_ll_marked(linked_list_node *n) {
  return (linked_list_node *) ((unsigned long long int) n | 1);
}

__device__ linked_list_node *_ll_unmarked(linked_list_node *n) {
  return (linked_list_node *) ((unsigned long long int) n & (~1));
}

__device__ linked_list_node *ll_create_node(int data) {
  linked_list_node *lln = (linked_list_node *) malloc(sizeof(linked_list_node));
  lln->data = data;
  lln->next = NULL;
  return lln;
}

__device__ linked_list_node *ll_first(linked_list *ll) {
  return ll->head->next;
}

__device__ void ll_reset(linked_list *ll) {
  ll->head = ll_create_node(INT_MIN);
  ll->tail = ll_create_node(INT_MAX);
  ll->head->next = ll->tail;
}

__device__ linked_list *ll_create() {
  linked_list *ll = (linked_list *) malloc(sizeof(linked_list));
  ll_reset(ll);
  return ll;
}

__device__ void ll_print(linked_list *ll) {
  linked_list_node *n = ll->head;
  while (n) {
    printf("%d @ ", n->data);
    printf("%x\n", n);
    n = _ll_unmarked(n->next);
  }
}

__device__ void _ll_free_nodes(linked_list_node *from, linked_list_node *to) {
  //*
  from = _ll_unmarked(from);
  while (from && from != to) {
    linked_list_node *next = from->next;
    free(from);
    from = _ll_unmarked(next);
  }
  //*/
}

__device__ linked_list_node *ll_search(linked_list *ll,
                                       int data,
                                       linked_list_node **left_node
                                       ) {
  linked_list_node *left_node_next, *right_node;

  while (true) {
    linked_list_node *t = ll->head;
    linked_list_node *t_next = ll->head->next;

    do {
      if (!_ll_is_marked(t_next)) {
        *left_node = t;
        left_node_next = t_next;
      }
      t = _ll_unmarked(t_next);
      if (t == ll->tail)
        break;
      t_next = t->next;
    } while (_ll_is_marked(t_next) || (t->data < data));
    right_node = t;

    if (left_node_next == right_node)
      if ((right_node != ll->tail) && _ll_is_marked(right_node->next))
        continue;
      else
        return right_node;

    unsigned long long int old = (unsigned long long int) left_node_next;
    if (old == atomicCAS((unsigned long long int *) &((*left_node)->next),
                         old,
                         (unsigned long long int  ) right_node
                         )
        ) {
      _ll_free_nodes((linked_list_node *) old, right_node);
      if ((right_node != ll->tail) && _ll_is_marked(right_node->next))
        continue;
      else
        return right_node;
    }
  }
}

__device__ bool ll_insert(linked_list *ll, int data) {
  linked_list_node *new_node = ll_create_node(data);
  linked_list_node *right_node, *left_node;

  while (true) {
    right_node = ll_search(ll, data, &left_node);
    if ((right_node != ll->tail) && (right_node->data == data))
      return false;
    new_node->next = right_node;
    unsigned long long int old = (unsigned long long int) right_node;
    if (old == atomicCAS((unsigned long long int *) &(left_node->next),
                         old,
                         (unsigned long long int  ) new_node
                         )
        )
      return true;
  }
}

__device__ bool ll_remove(linked_list *ll, int data) {
  linked_list_node * right_node, *right_node_next, *left_node;

  //printf("ll_remove(%d)\n", data);
  while (true) {
    right_node = ll_search(ll, data, &left_node);
    //printf("ll_remove: found %d\n", right_node->data);
    if ((right_node == ll->tail) || (right_node->data != data))
      return false;
    right_node_next = right_node->next;
    if (!_ll_is_marked(right_node_next)) {
      unsigned long long int old = (unsigned long long int) right_node_next;
      if (old == atomicCAS((unsigned long long int *) &(right_node->next),
                           old,
                           (unsigned long long int  ) _ll_marked(right_node_next)
                           )
          )
        break;
    }
  }
  unsigned long long int old = (unsigned long long int) right_node;
  if (old == atomicCAS((unsigned long long int *) &(left_node->next),
                       old,
                       (unsigned long long int  ) right_node_next
                       )
      )
    _ll_free_nodes((linked_list_node *) old, right_node_next);
  else
    right_node = ll_search(ll, right_node->data, &left_node);
  return true;
}

// call ONCE!
__device__ void ll_free(linked_list *ll) {
  // todo
  if (ll) {
    linked_list_node *lln = ll->head;

    while (lln) {
      linked_list_node *next = lln->next;
      free(lln);
      lln = next;
    }

    free(ll);
    ll = NULL;
  }
}

__device__ void ll_free_safe(linked_list *ll) {
  if (ll) {
    linked_list_node *lln = _ll_unmarked(ll->head);

    while (lln) {
      linked_list_node *next = _ll_unmarked(lln->next);
      free(lln);
      lln = next;
    }

    free(ll);
    ll = NULL;
  }
}

__device__ void ll_to_array(linked_list *ll, int *arr, unsigned int l) {
  linked_list_node *node = ll->head;

  if (node) // skip head
    node = node->next;

  unsigned int i = 0;
  while (node && i < l) {
    node = node->next;
    arr[i++] = node->data;
  }

  // fill remaining cells with 0, if any
  while (i < l)
    arr[i++] = 13373; // todo
}

__device__ void ll_to_array_safe(linked_list *ll, int *arr, unsigned int l) {
  linked_list_node *node = ll->head;
  linked_list_node *unmarked = _ll_unmarked(node);

  if (node) { // skip head
    node = node->next;
    unmarked = _ll_unmarked(node);
  }

  unsigned int i = 0;
  while (unmarked && i < l) {
    if (!_ll_is_marked(unmarked->next))
      arr[i++] = unmarked->data;

    node = unmarked->next;
    unmarked = _ll_unmarked(node);
  }

  while (i < l)
    arr[i++] = 13373; // todo
}

// grid

typedef struct {
  float lower_x, lower_y, cell_width, cell_height;
  unsigned int cols, rows;
  linked_list **cells;
} grid;

/*
__device__ grid *grid_create() {
  grid *g = (grid *) malloc(sizeof(grid));
  return g;
}
*/

grid *grid_host_create(float lower_x,
                       float lower_y,
                       float cell_width,
                       float cell_height,
                       unsigned int cols,
                       unsigned int rows
                       ) {
  grid *d_grid, h_grid;
  cudaMalloc((void **) &d_grid, sizeof(grid));

  h_grid.lower_x = lower_x;
  h_grid.lower_y = lower_y;
  h_grid.cell_width = cell_width;
  h_grid.cell_height = cell_height;
  h_grid.cols = cols;
  h_grid.rows = rows;
  linked_list **cells;
  cudaMalloc((void **) &cells, rows * cols * sizeof(linked_list *));
  h_grid.cells = cells;

  printf("h_grid.cells %x\n", cells);
  printf("h_grid.cell_height %f\n", h_grid.cell_height);
  printf("h_grid.cell_width %f\n", h_grid.cell_width);

  cudaMemcpy(d_grid,
             &h_grid,
             sizeof(grid),
             cudaMemcpyHostToDevice
             );

  return d_grid;
}

__device__ void grid_alloc_cells(grid *g) {
  unsigned int num_cells = g->rows * g->cols;
  printf("num_cells %d\n", num_cells);
  printf("grid.cells %x\n", g->cells);
  for (unsigned int i = 0; i < num_cells; i++) {
    g->cells[i] = ll_create();
    //printf("created cell ll at %x\n", g->cells[i]);
  }
}

__device__ linked_list *_grid_cell(grid *g, float2 pos) {
  //printf("x %f\n", pos.x);
  //printf("y %f\n", pos.y);

  //printf("cell_width %f\n", g->cell_width);
  //printf("cell_height %f\n", g->cell_height);

  unsigned int col = floor((pos.x - g->lower_x) / g->cell_width);
  unsigned int row = floor((pos.y - g->lower_y) / g->cell_height);

  //printf("col %d\n", col);
  //printf("row %d\n", row);

  unsigned int i = row * (g->cols) + col;

  if (col < g->cols && row < g->rows) {
    //printf("_grid_cell: %x\n", g->cells[i]);
    return g->cells[i];
  } else {
    printf("_grid_cell: NULL\n");
    return NULL;
  }
}

__device__ linked_list *grid_add_node(grid *g, float2 pos, int n) {
  linked_list *cell = _grid_cell(g, pos);

  /*
  printf("not calling ll_insert\n");
  return cell;
  //*/

  /*
  printf("cell at %x\n", cell);
  return cell;
  */

  if (cell) {
    //printf("calling ll_insert NOT\n");
    ll_insert(cell, n);
  }

  return cell;
}

__device__ bool grid_remove_node(grid *g, float2 pos, int n) {
  linked_list *cell = _grid_cell(g, pos);
  if (cell)
    return ll_remove(cell, n);
  else
    return false;
}

__device__ linked_list **grid_neighbours(grid *g, float2 pos) {
  unsigned int col = floor((pos.x - g->lower_x) / g->cell_width);
  unsigned int row = floor((pos.y - g->lower_y) / g->cell_height);

  if (col >= g->cols || row >= g->rows) {
    printf("this cell does not exist: col %d, row %d\n", col, row);
    printf("no such cell, pos: %f, %f\n", pos.x, pos.y);
    return NULL;
  }

  linked_list **neighbours = (linked_list **) malloc(9 * sizeof(linked_list *));
  unsigned int i = 0;

  neighbours[i++] = g->cells[row * g->cols + col];

  unsigned int left_most = (col > 0) ? col - 1 : 0;
  unsigned int right_most = (col + 1 < g->cols) ? col + 1 : col;

  if (row > 0) {
    for (unsigned int c = left_most; c <= right_most; c++) {
      neighbours[i++] = g->cells[(row - 1) * g->cols +  c];
    }
  }
  if (left_most < col)
    neighbours[i++] = g->cells[row * g->cols + left_most];

  if (right_most > col)
    neighbours[i++] = g->cells[row * g->cols + right_most];
  if (row + 1 < g->rows) {
    for (unsigned int c = left_most; c <= right_most; c++) {
      neighbours[i++] = g->cells[(row + 1) * g->cols +  c];
    }
  }

  while (i < 9)
    neighbours[i++] = NULL;

  return neighbours;
}

typedef struct {
  float2 pos;
  float2 waypoint;
  float vel;
  int pause;
  float so;
} random_waypoint_node;

typedef struct {
  float2 pos;
  float direction;
  float vel;
  int pause;
  float so;
} random_direction_node;

typedef struct {
  float x_lower;
  float x_upper;
  float y_lower;
  float y_upper;
} world_aabb;

typedef struct {
  world_aabb world;
  float max_velocity;
  float min_so, max_so;
  unsigned int pause;
} random_waypoint_config;

__device__ float random_float_in_range(curandState *rand_state,
                                       int i,
                                       float lower,
                                       float upper
                                       ) {
  return curand_uniform(&rand_state[i]) * (upper - lower) + lower;
}

__device__ unsigned int random_int_in_range(curandState *rand_state,
                                            int i,
                                            unsigned int lower,
                                            unsigned int upper
                                            ) {
  return curand(&rand_state[i]) % (upper - lower) + lower;
}

__global__ void grid_init(grid *grid,
                          unsigned int num_nodes
                          ) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i == 0) {
    printf("grid_alloc_cells\n");
    grid_alloc_cells(grid);
  }
}

__global__ void random_waypoint_init(curandState *rand_state,
                                     random_waypoint_config* config,
                                     random_waypoint_node* nodes,
                                     grid *grid,
                                     unsigned int num_nodes
                                     ) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < num_nodes) {
    curand_init(clock64(), i, 0, &rand_state[i]);

    nodes[i].pos = make_float2(random_float_in_range(rand_state,
                                                     i,
                                                     config->world.x_lower,
                                                     config->world.x_upper
                                                     ),
                               random_float_in_range(rand_state,
                                                     i,
                                                     config->world.y_lower,
                                                     config->world.y_upper
                                                     )
                               );
    nodes[i].waypoint = nodes[i].pos;
    nodes[i].vel = 0.0;
    nodes[i].pause = 1;
    nodes[i].so = random_float_in_range(rand_state,
                                        i,
                                        config->min_so,
                                        config->max_so
                                        );

    grid_add_node(grid, nodes[i].pos, i);
    printf("%d pos %f, %f\n", i, nodes[i].pos.x, nodes[i].pos.y);
  }
}

__device__ bool random_waypoint_is_safe(random_waypoint_node *nodes,
                                        grid *grid,
                                        int i,
                                        float2 pos
                                        ) {
  linked_list **neighbours = grid_neighbours(grid, pos);

  unsigned int count = 0;
  for (unsigned int j = 0; j < 9; j++)
    if (neighbours[j]) {
      count++;

      linked_list_node *n = ll_first(neighbours[j]);
      while (n != neighbours[j]->tail) {
        if (n->data != i) {
          float dx = nodes[n->data].pos.x - pos.x;
          float dy = nodes[n->data].pos.y - pos.y;
          float dsq = dx * dx + dy * dy;
          float safe_d = nodes[i].so + nodes[n->data].vel + nodes[n->data].so;
          //printf("dsq %f, safe_d**2 %f\n", dsq, safe_d * safe_d);
          if (dsq < (safe_d * safe_d)) {
            /*
              printf("neighbour at %f, %f is too close to %f, %f\n",
              nodes[n->data].pos.x,
              nodes[n->data].pos.y,
              nodes[i].pos.x,
              nodes[i].pos.y
              );
            */
            free(neighbours);
            return false;
          }
        }
        n = n->next;
      }
    }

  //printf("%d neighbour regions, all safe\n", count);

  free(neighbours);

  return true;
}


__device__ bool random_direction_is_safe(random_direction_node *nodes,
                                         grid *grid,
                                         int i,
                                         float2 pos
                                         ) {
  linked_list **neighbours = grid_neighbours(grid, pos);

  unsigned int count = 0;
  for (unsigned int j = 0; j < 9; j++) {
    //printf("neighbours %x\n", neighbours[j]);
    if (neighbours[j]) {
      count++;

      linked_list_node *n = ll_first(neighbours[j]);
      while (n && n != neighbours[j]->tail) {
        //printf("neighbour %x\n", n);
        while (_ll_is_marked(n->next)) {
          //printf("skipping %x\n", n);
          n = _ll_unmarked(n->next);
        }
        if (n) {
          if (n->data != i) {
            float dx = nodes[n->data].pos.x - pos.x;
            float dy = nodes[n->data].pos.y - pos.y;
            float dsq = dx * dx + dy * dy;
            float safe_d = nodes[i].so + nodes[n->data].vel + nodes[n->data].so;
            //printf("dsq %f, safe_d**2 %f\n", dsq, safe_d * safe_d);
            if (dsq < (safe_d * safe_d)) {
              /*
              printf("neighbour at %f, %f is too close to %f, %f\n",
                     nodes[n->data].pos.x,
                     nodes[n->data].pos.y,
                     nodes[i].pos.x,
                     nodes[i].pos.y
                     );
              */
              free(neighbours);
              return false;
            }
          }
          n = _ll_unmarked(n->next);
        }
      }
    }
  }

  //printf("%d neighbour regions, all safe\n", count);

  free(neighbours);
  return true;
}

__global__ void random_waypoint_step(curandState *rand_state,
                                     random_waypoint_config* config,
                                     random_waypoint_node* nodes,
                                     grid *grid,
                                     unsigned int num_nodes
                                     ) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < num_nodes) {
    //*
    if (nodes[i].pause > 0) {
      if (--nodes[i].pause <= 0) {
        nodes[i].waypoint =
          make_float2(random_float_in_range(rand_state,
                                            i,
                                            config->world.x_lower,
                                            config->world.x_upper
                                            ),
                      random_float_in_range(rand_state,
                                            i,
                                            config->world.y_lower,
                                            config->world.y_upper
                                            )
                      );
        nodes[i].vel = curand_uniform(&rand_state[i]) * config->max_velocity;
      }
    } else {
      float2 to_waypoint = make_float2(nodes[i].waypoint.x - nodes[i].pos.x,
                                       nodes[i].waypoint.y - nodes[i].pos.y
                                       );
      float d = sqrt(to_waypoint.x * to_waypoint.x
                     + to_waypoint.y * to_waypoint.y
                     );
      if (d <= nodes[i].vel) {
        nodes[i].pos = nodes[i].waypoint;
        nodes[i].pause = config->pause;
      } else {
        to_waypoint.x *= nodes[i].vel / d;
        to_waypoint.y *= nodes[i].vel / d;

        float2 candidate = make_float2(nodes[i].pos.x + to_waypoint.x,
                                       nodes[i].pos.y + to_waypoint.y
                                       );

        if (random_waypoint_is_safe(nodes, grid, i, candidate)) {
          //printf("moving %d\n", i);
          if (grid_remove_node(grid, nodes[i].pos, i))
            grid_add_node(grid, candidate, i);
          nodes[i].pos = candidate;
        } else {
          printf("can't move %d\n", i);
        }
      }
    }
    //*/
  }
}

// random direction

__global__ void random_direction_init(curandState *rand_state,
                                      random_waypoint_config* config,
                                      random_direction_node* nodes,
                                      grid *grid,
                                      unsigned int num_nodes
                                      ) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  //*
  if (i < num_nodes) {
    curand_init(clock64(), i, 0, &rand_state[i]);

    nodes[i].pos = make_float2(random_float_in_range(rand_state,
                                                     i,
                                                     config->world.x_lower,
                                                     config->world.x_upper
                                                     ),
                               random_float_in_range(rand_state,
                                                     i,
                                                     config->world.y_lower,
                                                     config->world.y_upper
                                                     )
                               );
    nodes[i].direction = 0.0;
    nodes[i].vel = 0.0;
    nodes[i].pause = 1;
    nodes[i].so = random_float_in_range(rand_state,
                                        i,
                                        config->min_so,
                                        config->max_so
                                        );

    linked_list *cell = grid_add_node(grid, nodes[i].pos, i);
  }
  //*/
}

__device__ float2 cut(float2 p1, float2 p2, float2 p3, float2 p4) {
  return make_float2((
                      (p4.x - p3.x) * (p2.x * p1.y - p1.x * p2.y)
                      - (p2.x - p1.x) * (p4.x * p3.y - p3.x * p4.y)
                      )
                     /
                     (
                      (p4.y - p3.y) * (p2.x - p1.x)
                      - (p2.y - p1.y) * (p4.x - p3.x)
                      ),
                     (
                      (p1.y - p2.y) * (p4.x * p3.y - p3.x * p4.y)
                      - (p3.y - p4.y) * (p2.x * p1.y - p1.x * p2.y)
                      )
                     /
                     (
                      (p4.y - p3.y) * (p2.x - p1.x)
                      - (p2.y - p1.y) * (p4.x - p3.x)
                      )
                     );
}

__device__ float distance_squared(float2 p1, float2 p2) {
  float dx = p1.x - p2.x;
  float dy = p1.y - p2.y;
  return dx * dx + dy * dy;
}

__global__ void random_direction_step(curandState *rand_state,
                                      random_waypoint_config* config,
                                      random_direction_node* nodes,
                                      grid *grid,
                                      unsigned int num_nodes
                                      ) {
  printf("%d * %d + %d\n", blockDim.x, blockIdx.x, threadIdx.x);
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  printf("%d\n", i);

  float2 top_left     = make_float2(config->world.x_lower, config->world.y_lower);
  float2 bottom_left  = make_float2(config->world.x_lower, config->world.y_upper);
  float2 top_right    = make_float2(config->world.x_upper, config->world.y_lower);
  float2 bottom_right = make_float2(config->world.x_upper, config->world.y_upper);


  //printf("tl %f, %f\n", top_left.x, top_left.y);
  //printf("bl %f, %f\n", bottom_left.x, bottom_left.y);
  //printf("tr %f, %f\n", top_right.x, top_right.y);
  //printf("br %f, %f\n", bottom_right.x, bottom_right.y);


  if (i < num_nodes) {
    printf("node %d\n", i);
    if (nodes[i].pause > 0) {
      if (--nodes[i].pause <= 0) {
        printf("%d: choose new direction\n", i);
        nodes[i].direction =
          random_float_in_range(rand_state,
                                i,
                                0,
                                2 * M_PI
                                );
        nodes[i].vel = curand_uniform(&rand_state[i]) * config->max_velocity;
      }
    } else {
      float2 forward = make_float2(cos(nodes[i].direction),
                                   sin(nodes[i].direction)
                                   );
      float2 candidate = make_float2(nodes[i].pos.x + forward.x,
                                     nodes[i].pos.y + forward.y
                                     );

      printf("candidate for %d: %f, %f\n", i, candidate.x, candidate.y);

      // out of bounds?
      bool oob_left  = candidate.x < config->world.x_lower;
      bool oob_right = candidate.x > config->world.x_upper;
      bool oob_up    = candidate.y < config->world.y_lower;
      bool oob_down  = candidate.y > config->world.y_upper;

      //printf("%d: %f, %f | oob? l%d r%d u%d d%d\n",
      //       i,
      //       candidate.x,
      //       candidate.y,
      //       oob_left,
      //       oob_right,
      //       oob_up,
      //       oob_down
      //       );


      if (oob_left) {
        if (oob_up) {
          float2 c1 = cut(nodes[i].pos,
                          candidate,
                          top_left,
                          bottom_left
                          );
          float2 c2 = cut(nodes[i].pos,
                          candidate,
                          top_left,
                          top_right
                          );
          float d1 = distance_squared(candidate, c1);
          float d2 = distance_squared(candidate, c2);
          if (d1 < d2)
            candidate = c1;
          else
            candidate = c2;
        }
        else if (oob_down) {
          float2 c1 = cut(nodes[i].pos,
                          candidate,
                          top_left,
                          bottom_left
                          );
          float2 c2 = cut(nodes[i].pos,
                          candidate,
                          bottom_left,
                          bottom_right
                          );
          float d1 = distance_squared(candidate, c1);
          float d2 = distance_squared(candidate, c2);
          if (d1 < d2)
            candidate = c1;
          else
            candidate = c2;
        }
        else {
          candidate = cut(nodes[i].pos,
                          candidate,
                          top_left,
                          bottom_left
                          );
        }
      } else if (oob_right) {
        if (oob_up) {
          float2 c1 = cut(nodes[i].pos,
                          candidate,
                          top_right,
                          bottom_right
                          );
          float2 c2 = cut(nodes[i].pos,
                          candidate,
                          top_left,
                          top_right
                          );
          float d1 = distance_squared(candidate, c1);
          float d2 = distance_squared(candidate, c2);
          if (d1 < d2)
            candidate = c1;
          else
            candidate = c2;
        }
        else if (oob_down) {
          float2 c1 = cut(nodes[i].pos,
                          candidate,
                          top_right,
                          bottom_right
                          );
          float2 c2 = cut(nodes[i].pos,
                          candidate,
                          bottom_left,
                          bottom_right
                          );
          float d1 = distance_squared(candidate, c1);
          float d2 = distance_squared(candidate, c2);
          if (d1 < d2)
            candidate = c1;
          else
            candidate = c2;
        }
        else {
          candidate = cut(nodes[i].pos,
                          candidate,
                          top_right,
                          bottom_right
                          );
        }
      } else if (oob_up) {
        candidate = cut(nodes[i].pos,
                        candidate,
                        top_left,
                        top_right
                        );
      } else if (oob_down) {
        candidate = cut(nodes[i].pos,
                        candidate,
                        bottom_left,
                        bottom_right
                        );
      }

      __syncthreads();

      if (random_direction_is_safe(nodes, grid, i, candidate)) {
        //printf("moving %d\n", i);
        if (grid_remove_node(grid, nodes[i].pos, i))
          grid_add_node(grid, candidate, i);
        nodes[i].pos = candidate;

        if (oob_left || oob_right || oob_up || oob_down) {
          nodes[i].pause = config->pause;
          printf("clamped to %f, %f\n", candidate.x, candidate.y);
        }
      } else {
        printf("can't move %d\n", i);
      }

    }
  }
}

int main(int argc, char **argv) {
  cudaError_t err;

  unsigned int num_threads = 32; // 1024;
  unsigned int num_blocks = 1;
  unsigned int num_nodes = num_threads * num_blocks;
  unsigned int num_frames = 10;

  random_waypoint_node *h_nodes;
  //random_direction_node *h_nodes;
  random_waypoint_config h_config;
  h_config.world.x_lower = -10.0;
  h_config.world.x_upper =  10.0;
  h_config.world.y_lower = -10.0;
  h_config.world.y_upper =  10.0;
  h_config.max_velocity  =   0.5;
  h_config.pause = 2;
  h_config.min_so = 0.01;
  h_config.max_so = 0.1;

  float cell_size = h_config.max_so * 2 + h_config.max_velocity;
  unsigned int grid_cols =
    ceil((h_config.world.x_upper - h_config.world.x_lower) / cell_size);
  unsigned int grid_rows =
    ceil((h_config.world.y_upper - h_config.world.y_lower) / cell_size);

  printf("%d nodes\n", num_nodes);
  //printf("grid: %d x %d\n", grid_cols, grid_rows);

  random_waypoint_node *d_nodes;
  //random_direction_node *d_nodes;
  random_waypoint_config *d_config;
  grid *d_grid;
  curandState *d_rand_state;

  h_nodes = (random_waypoint_node *)
    malloc(num_nodes * sizeof(random_waypoint_node));

  /*
  h_nodes = (random_direction_node *)
    malloc(num_nodes * sizeof(random_direction_node));
  */

  cudaMalloc((void **) &d_nodes, num_nodes * sizeof(random_waypoint_node));
  //cudaMalloc((void **) &d_nodes, num_nodes * sizeof(random_direction_node));
  cudaMalloc((void **) &d_config, sizeof(random_waypoint_config));
  cudaMalloc((void **) &d_rand_state, num_threads * sizeof(curandState));

  cudaMemcpy(d_config,
             &h_config,
             sizeof(random_waypoint_config),
             cudaMemcpyHostToDevice
             );

  d_grid = grid_host_create(h_config.world.x_lower,
                            h_config.world.y_lower,
                            cell_size,
                            cell_size,
                            grid_cols,
                            grid_rows
                            );

  printf("d_grid %x\n", d_grid);
  printf("init\n");
  printf("grid_init\n");
  grid_init<<<num_blocks, num_threads>>>(d_grid,
                                         num_nodes
                                         );
  err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("ERR: %s\n", cudaGetErrorString(err));

  printf("random_waypoint_init\n");
  random_waypoint_init<<<num_blocks, num_threads>>>(d_rand_state,
  //random_direction_init<<<num_blocks, num_threads>>>(d_rand_state,
                                                     d_config,
                                                     d_nodes,
                                                     d_grid,
                                                     num_nodes
                                                     );
  err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("ERR: %s\n", cudaGetErrorString(err));

  printf("/init\n");


  for (unsigned int i = 0; i < num_frames; i++) {
    printf("\nframe %u\n", i);
    //*
    random_waypoint_step<<<num_blocks, num_threads>>>(d_rand_state,
    //random_direction_step<<<num_blocks, num_threads>>>(d_rand_state,
                                                       d_config,
                                                       d_nodes,
                                                       d_grid,
                                                       num_nodes
                                                       );

    err = cudaGetLastError();
    if (err != cudaSuccess)
      printf("ERR: %s\n", cudaGetErrorString(err));
    //*/

    printf("memcpy %x (device) to %x (host)\n", d_nodes, h_nodes);
    cudaMemcpy(h_nodes,
               d_nodes,
               num_nodes * sizeof(random_waypoint_node),
               //num_nodes * sizeof(random_direction_node),
               cudaMemcpyDeviceToHost
               );
    err = cudaGetLastError();
    if (err != cudaSuccess)
      printf("ERR: %s\n", cudaGetErrorString(err));

    for (unsigned int i = 0; i < num_nodes; i++) {
      printf("%u: %f,%f (v = %f) -> %f, %f | p = %d\n",
             i,
             h_nodes[i].pos.x,
             h_nodes[i].pos.y,
             h_nodes[i].vel,
             //h_nodes[i].direction,
             h_nodes[i].waypoint.x,
             h_nodes[i].waypoint.y,
             h_nodes[i].pause
             );
    }
  }

  cudaFree(d_rand_state);
  cudaFree(d_config);
  cudaFree(d_nodes);

  free(h_nodes);

  exit(EXIT_SUCCESS);
}
