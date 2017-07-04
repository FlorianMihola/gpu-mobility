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


__device__ void ll_reset(linked_list *ll) {
  ll->head = ll_create_node(INT_MIN);
  ll->tail = ll_create_node(INT_MAX);
  ll->head->next = ll->tail;
}

__device__ linked_list *ll_create() {
  linked_list *ll = (linked_list *) malloc(sizeof(linked_list));
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
/*
typedef struct {
  float cell_height;
  float cell_width;
  unsigned int rows;
  unsigned int cols;
  linked_list_node *cells[];
} grid;

*/
//

typedef struct {
  float2 pos;
  float2 waypoint;
  float vel;
  int pause;
} random_waypoint_node;

typedef struct {
  float x_lower;
  float x_upper;
  float y_lower;
  float y_upper;
} world_aabb;

typedef struct {
  world_aabb world;
  float max_velocity;
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

__global__ void random_waypoint_init(curandState *rand_state,
                                     random_waypoint_config* config,
                                     random_waypoint_node* nodes,
                                     unsigned int num_nodes,
                                     linked_list *ll
                                     ) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i == 0) {
    ll_reset(ll);
    for (unsigned int j = 0; j < num_nodes; j++) {
      ll_insert(ll, j);
    }

    ll_print(ll);
  }

  if (i < num_nodes) {
    //ll_prepend(ll, i);

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

    nodes[i].pos = make_float2(0, 0);
    nodes[i].waypoint = nodes[i].pos;
  }
}

__global__ void random_waypoint_step(curandState *rand_state,
                                     random_waypoint_config* config,
                                     random_waypoint_node* nodes,
                                     unsigned int num_nodes,
                                     linked_list *ll
                                     ) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < num_nodes) {
    if (true
        //0 <= (i % 32) && (i & 32) <= 32
        //0 <= (i % 32) && (i & 32) <= 3
        ) {
      unsigned int r = random_int_in_range(rand_state,
                                           i,
                                           0,
                                           num_nodes
                                           );
      //r = 4 + (r % 2);
      nodes[i].vel = r;
      if (ll_remove(ll, r)) {
        nodes[i].pos.y = 1337.1337;

        ll_insert(ll, r);
      }
      else {
        nodes[i].pos.y = -1337.1337;
      }
    }
  }
}

__global__ void test_cleanup(linked_list *ll,
                             int *arr,
                             unsigned int num_nodes
                             ) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i == 0) {
    for (unsigned int j = 0; j < num_nodes; j++)
      arr[j] = -1337;

    ll_to_array_safe(ll, arr, num_nodes);
    ll_free_safe(ll);
  }
}

int main(int argc, char **argv) {
  cudaError_t err;
  int exit_status = EXIT_SUCCESS;

  unsigned int num_threads = 1024;
  unsigned int num_blocks = 5;
  unsigned int num_nodes = num_threads * num_blocks;
  unsigned int num_frames = 5000;

  random_waypoint_node *h_nodes;
  random_waypoint_config h_config;
  h_config.world.x_lower = -10.0;
  h_config.world.x_upper =  10.0;
  h_config.world.y_lower = -10.0;
  h_config.world.y_upper =  10.0;
  h_config.max_velocity  =   3.0;
  h_config.pause = 2;

  random_waypoint_node *d_nodes;
  random_waypoint_config *d_config;
  curandState *d_rand_state;
  linked_list *test_list;
  int *test_array, *h_test_array;

  h_nodes = (random_waypoint_node *)
    malloc(num_nodes * sizeof(random_waypoint_node));


  cudaMalloc((void **) &d_nodes, num_nodes * sizeof(random_waypoint_node));
  cudaMalloc((void **) &d_config, sizeof(random_waypoint_config));
  cudaMalloc((void **) &d_rand_state, num_threads * sizeof(curandState));
  cudaMalloc((void **) &test_list, sizeof(linked_list));
  cudaMalloc((void **) &test_array, num_nodes * sizeof(int));


  h_test_array = (int *) malloc(num_nodes * sizeof(int));

  cudaMemcpy(d_config,
             &h_config,
             sizeof(random_waypoint_config),
             cudaMemcpyHostToDevice
             );

  random_waypoint_init<<<num_blocks, num_threads>>>(d_rand_state,
                                                    d_config,
                                                    d_nodes,
                                                    num_nodes,
                                                    test_list
                                                    );

  for (unsigned int i = 0; i < num_frames; i++) {
    random_waypoint_step<<<num_blocks, num_threads>>>(d_rand_state,
                                                      d_config,
                                                      d_nodes,
                                                      num_nodes,
                                                      test_list
                                                      );
    err = cudaGetLastError();
    if (err != cudaSuccess)
      printf("ERR: %s\n", cudaGetErrorString(err));
    //printf("frame %u\n", i);

    cudaMemcpy(h_nodes,
               d_nodes,
               num_nodes * sizeof(random_waypoint_node),
               cudaMemcpyDeviceToHost
               );
    err = cudaGetLastError();
    if (err != cudaSuccess)
      printf("ERR (memcpy): %s\n", cudaGetErrorString(err));
    /*
    for (unsigned int i = 0; i < num_nodes; i++) {
      float2 to_waypoint = make_float2(h_nodes[i].waypoint.x - h_nodes[i].pos.x,
                                       h_nodes[i].waypoint.y - h_nodes[i].pos.y
                                       );
      float d = sqrt(to_waypoint.x * to_waypoint.x
                     + to_waypoint.y * to_waypoint.y
                     );
      printf("%u: @ %f,%f (v = %f) -> %f,%f (d = %f) (p = %d)\n",
             i,
             h_nodes[i].pos.x,
             h_nodes[i].pos.y,
             h_nodes[i].vel,
             h_nodes[i].waypoint.x,
             h_nodes[i].waypoint.y,
             d,
             h_nodes[i].pause
             );
    }
    */
  }


  test_cleanup<<<num_blocks, num_threads>>>(test_list,
                                            test_array,
                                            num_nodes
                                            );

  //cudaFree(test_list);
  cudaFree(d_rand_state);
  cudaFree(d_config);
  cudaFree(d_nodes);

  free(h_nodes);

  cudaMemcpy(h_test_array,
             test_array,
             num_nodes * sizeof(int),
             cudaMemcpyDeviceToHost
             );

  cudaFree(test_array);

  //*
  unsigned int seen = 0;
  for (unsigned int i = 0; i < num_nodes; i++) {
    printf("> %d\n", h_test_array[i]);
    //if (seen & (1<<h_test_array[i]))
    //  printf("Duplicate %d\n", h_test_array[i]);
    seen |= (1<<h_test_array[i]);
  }
  //*/

  printf("missing\n");
  for (unsigned int i = 0; i < num_nodes; i++) {
    bool found = false;
    for (unsigned int j = 0; j < num_nodes; j++)
      if (h_test_array[j] == i) {
        found = true;
        break;
      }
    if (!found) {
      printf("%d\n", i);
      exit_status = EXIT_FAILURE;
    }
  }

  printf("/missing\n");

  free(h_test_array);

  exit(exit_status);
}
