#include "xy2d.h"

void test_2d() {
    const int len = 3;
    int sizes[len] = {25, 30, 35};
    for (int i=0; i < len; i++) {
	Metropolis metropolis(sizes[i], "test");
	metropolis.simulate(0.1, 4.5, 0.2, 1e4);
    }
}

int main(int argc, char** argv) {
    // Do unit testing
    test_2d();
    return 1;
}
