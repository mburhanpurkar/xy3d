////#include "modxy3d.h"
#include "parallelize.h"

// void test_3d() {
//      const int len = 3;
//      int sizes[len] = {19, 20, 21};
//      for (int i=0; i < len; i++) {
//          Metropolis metropolis(sizes[i], "test");
//          metropolis.simulate(1.0, 3e3, 0.1, 13.38, 0.01);
//      }
// }

void test_parallel() {
  mpi_run(0.0, 0.4, 0.7, 64 * 8, 16, 3e3, "test");
}

int main(int argc, char** argv) {
    // Do unit testing
    // test_3d();
    test_parallel();
    return 1;
}

