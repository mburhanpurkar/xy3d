// Note, this produces questionable results with a system size < 20!
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <cassert>
#define _USE_MATH_DEFINES
using namespace std;


// Global variables for random number generation
random_device rd;
mt19937 gen(100);//gen(rd());     // Mersenne Twister RNG    
uniform_real_distribution<double> ran_u(0.0, 1.0);

class Metropolis {
    int L;
    int SIZE;
    double SIZEd;
    double Jstar;
    double Kstar;
    double* state;
    int** neighs;
    int*** plaqs;
    double ENERGY = 0.0;
    int* nu; // this will have length L (one winding number for each row)
	
    inline int index_to_n(int, int);
    inline double bond_energy(double, double);
    inline double p_energy(int, double);
    void step();
    void flip(int);
    void neighbours();
    double magnetization();
    void get_energy();
    int pbc(int);
    double constrain(double);
    int winding(int);
    int angle_to_nu(double, double, int);

public:
    Metropolis(int, double, double);
    ~Metropolis();
    void simulate();
};

Metropolis::Metropolis(int L, double J, double K) {
    this->L = L;
    SIZE = L * L;
    SIZEd = (double) SIZE;
    Jstar = J;
    Kstar = K;
	
    // Instead of randomly initializing the state, include a nu = 1 spin wave!
    nu = new int[L];
    state = new double[SIZE];
    uniform_real_distribution<double> ran_u(0.0, 1.0);
    double start = 0.0;
    double delta = 2.0 * M_PI / L;
    for (int i=0; i < L; ++i) {
	for (int j=0; j < L; ++j) {
	    int idx = index_to_n(i, j);
	    state[idx] = constrain(start + delta * j);
	}
    }

    // As a check to make sure this worked, initialize nu with the winding number
    // and check that each element is equal to 1
    for (int i=0; i < L; ++i) {
	nu[i] = winding(i);
	assert (nu[i] == 1);
    }
    
    // Set up neighbour and plaquette tables
    neighs = new int*[SIZE];
    plaqs = new int**[SIZE];
    for(int i = 0; i < SIZE; ++i) {
  	neighs[i] = new int[6];
  	plaqs[i] = new int*[4];
  	for(int j = 0; j < 4; ++j)
	      plaqs[i][j] = new int[3];
    }
    neighbours();
}

Metropolis::~Metropolis() {
    delete[] state;
    for(int i = 0; i < SIZE; ++i)
	delete[] neighs[i];
    delete[] neighs;
}

inline int Metropolis::index_to_n(int i, int j) {
    return i * L + j;
}

void Metropolis::neighbours() {
    int u,d,r,l,n;
    
    for (int i=0; i < L; ++i) {
	for (int j=0; j < L; ++j) {
	    // Periodic boundary
	    u = j + 1 == L  ? 0     : j + 1;
	    d = j - 1 == -1 ? L - 1 : j - 1;
	    r = i + 1 == L  ? 0     : i + 1;
	    l = i - 1 == -1 ? L - 1 : i - 1;
	    
	    // Fill in neighbours table
	    n = index_to_n(i, j);
	    neighs[n][0] = index_to_n(i, u);
	    neighs[n][1] = index_to_n(r, j);
	    neighs[n][2] = index_to_n(l, j);
	    neighs[n][3] = index_to_n(i, d);

	    // Add in plaquettes!
	    plaqs[n][0][0] = index_to_n(r, j); plaqs[n][0][1] = index_to_n(r, u); plaqs[n][0][2] = index_to_n(i, u);
	    plaqs[n][1][0] = index_to_n(i, u); plaqs[n][1][1] = index_to_n(l, u); plaqs[n][1][2] = index_to_n(l, j);
	    plaqs[n][2][0] = index_to_n(l, j); plaqs[n][2][1] = index_to_n(l, d); plaqs[n][2][2] = index_to_n(i, d);
	    plaqs[n][3][0] = index_to_n(i, d); plaqs[n][3][1] = index_to_n(r, d); plaqs[n][3][2] = index_to_n(r, j);
	}
    }
}

void Metropolis::simulate() {
    get_energy();
    // Here we pick the number of MC timesteps to run
    // Note that we only need to thermalize RIGHT NOW once at the start since we're not changing parameters
    // After each flip, measure the winding number of the line on which the flip occured and store it
    // Do this for some large number of timesteps (or until they all have winding number 0)
    int nmc = 1000;
    for (int t=0; t < nmc; t++) {
    	// Let's pack the measurement action into flip
    	flip(t);
    }
}

inline double Metropolis::bond_energy(double angle1, double angle2) {
    return -1.0 * cos(angle1 - angle2);
}

inline double Metropolis::p_energy(int n, double central_angle) {
    double prod;
    double energy = 0.0;

    for (int p=0; p < 4; p++) {
	// Select each of the four elements
	prod = 1.0;
	prod *= cos((central_angle - state[plaqs[n][p][0]]) / 2.0);
	prod *= cos((state[plaqs[n][p][2]] - central_angle) / 2.0);
	prod *= cos((state[plaqs[n][p][0]] - state[plaqs[n][p][1]]) / 2.0);
	prod *= cos((state[plaqs[n][p][1]] - state[plaqs[n][p][2]]) / 2.0);
	energy += prod;
    }
    return -1.0 * energy;
}

void Metropolis::flip(int t) {
    // It's not great to have this here, but we can't make it global because we need SIZE
    uniform_int_distribution<int> ran_pos(0, SIZE-1);
    int index = ran_pos(gen);
    double flip_axis = ran_u(gen) * M_PI;
    double old_angle = state[index];
    double new_angle = constrain(2.0 * flip_axis - old_angle);
    double E1 = 0.0;
    double E2 = 0.0;

    for (int i=0; i < 4; ++i) {
	E1 += bond_energy(state[neighs[index][i]], old_angle) * Jstar;
	E2 += bond_energy(state[neighs[index][i]], new_angle) * Jstar;
    }

    E1 += p_energy(index, old_angle) * Kstar;
    E2 += p_energy(index, new_angle) * Kstar;

    // If E2 < E1, then we definitely flip
    double p = E2 < E1 ? 1.0 : exp(-(E2 - E1));

    if (ran_u(gen) < p) {
	state[index] = new_angle;
	ENERGY += (E2 - E1) / SIZEd / 2.0;

	// I'm p sure the row is just index / L, right???
	// Check if something changed
	int new_nu = winding(index / L);
	if (new_nu != nu[index / L]) {
	    cout << t << " " << index / L << " " << new_nu - nu[index / L] << endl;
	    nu[index / L] = new_nu;
	}
	return;
    }
    else
	return flip(t);
}

double Metropolis::magnetization() {
    double sum_x = 0.0;
    double sum_y = 0.0;

    for (int i=0; i < SIZE; i++) {
	sum_x += cos(state[i]);
	sum_y += sin(state[i]);
    }
    sum_x /= SIZEd;
    sum_y /= SIZEd;

    return sqrt(sum_x * sum_x + sum_y * sum_y);
}

void Metropolis::get_energy() {
    for (int i=0; i < SIZE; i++) {
	for (int j=0; j < 4; j++) 
	    ENERGY += bond_energy(state[neighs[i][j]], state[i]) * Jstar;
	ENERGY += p_energy(i, state[i]) * Kstar;
    }
    ENERGY = ENERGY / SIZEd / 2.0;
}

inline int Metropolis::pbc(int n) {
    return n >= 0 ? n % L : L + (n % L);
}

inline double another_constrain(double x) {
    if (x < -M_PI)
	return x + 2 * M_PI;
    if (x > M_PI)
	return x - 2 * M_PI;
    return x;
}

inline double Metropolis::constrain(double alpha) {
    double x = fmod(alpha, 2 * M_PI);
    return x > 0 ? x : x += 2 * M_PI;
}

int Metropolis::angle_to_nu(double x, double tol=0.1, int depth=0) {
    // Given a change in angle x, this computes the winding number. Note that arguments must
    // be positive and depth should always be 0 when called.
    if (depth == 0) 
	assert (x >= 0);
    if (abs(x - 2 * M_PI) < tol)
	return 1 + depth;
    if (x < 0)
	return 0;
    return angle_to_nu(x - 2 * M_PI, tol, depth + 1);
}

int Metropolis::winding(int row) {
    // Compute the winding number for a given row
    // The functionality will be similar to vorticity computations
    double delta = 0.0;
    for (int col=0; col < L; col++) {
	int id1 = index_to_n(row, col % L);
	int id2 = index_to_n(row, (col + 1) % L);
	delta += another_constrain(state[id2] - state[id1]);
    }
    return delta > 0 ? angle_to_nu(abs(delta), 0.1, 0) : -angle_to_nu(abs(delta), 0.1, 0);
}
