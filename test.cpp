#include <iostream>
#include <cmath>
#include <math.h>
#include "xy2d.h"
#define _USE_MATH_DEFINES
using namespace std;


void test_2d() {
    const int len = 3;
    int sizes[len] = {25, 30, 35};
    for (int i=0; i < len; i++) {
	Metropolis metropolis(sizes[i], "test");
	metropolis.simulate(0.1, 4.5, 0.2, 1e4);
    }
}


int main(int argc, char** argv) {
    if (argc == 2) {
	// A system size has been specified--simulate on that
	Metropolis metropolis(atoi(argv[1]), "");
	metropolis.simulate(0.1, 4.5, 0.2, 1e4);
	return 0;
    }
    else {
	// Do unit testing
	test_2d();
	return 1;
    }
}

