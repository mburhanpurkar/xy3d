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
mt19937 gen(rd());     // Mersenne Twister RNG    
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
	
    inline int index_to_n(int, int);
    inline double p_energy(int, double);
    void step();
    void flip();
    void neighbours();
    void get_energy();
    int pbc(int);
    double constrain_to_2pi(double);
    double constrain_to_pi(double);

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
    state = new double[SIZE];
    uniform_real_distribution<double> ran_u(0.0, 1.0);
    double start = 0.0;
    double delta = 2.0 * M_PI / L;
    /* for (int i=0; i < L; ++i) { */
    /* 	for (int j=0; j < L; ++j) { */
    /* 	    int idx = index_to_n(i, j); */
    /* 	    state[idx] = 0.0; */
    /* 	} */
    /* } */
    for (int i=0; i < SIZE; ++i) state[i] = ran_u(gen) * 2 * M_PI;
    
    // Set up neighbour and plaquette tables
    neighs = new int*[SIZE];
    plaqs = new int**[SIZE];
    for(int i = 0; i < SIZE; ++i) {
  	neighs[i] = new int[4];
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
	    n = index_to_n(j,  i);
	    neighs[n][0] = index_to_n(u, i);
	    neighs[n][1] = index_to_n(j, r);
	    neighs[n][2] = index_to_n(j, l);
	    neighs[n][3] = index_to_n(d, i);

	    // Add in plaquettes!
	    plaqs[n][0][0] = index_to_n(j, r); plaqs[n][0][1] = index_to_n(u, r); plaqs[n][0][2] = index_to_n(u, i);
	    plaqs[n][1][0] = index_to_n(u, i); plaqs[n][1][1] = index_to_n(u, l); plaqs[n][1][2] = index_to_n(j, l);
	    plaqs[n][2][0] = index_to_n(j, l); plaqs[n][2][1] = index_to_n(d, l); plaqs[n][2][2] = index_to_n(d, i);
	    plaqs[n][3][0] = index_to_n(d, i); plaqs[n][3][1] = index_to_n(d, r); plaqs[n][3][2] = index_to_n(j, r);
	}
    }
}

void Metropolis::simulate() {
    get_energy();

    // Thermalize first with 1000 MC steps (overkill, surely)
    int nmc = 500;
    nmc *= L * L;

    for (int t=0; t < nmc; t++) 
    	flip();

    // Now output the spins
    for (int i=0; i < L; i++) {
	for (int j=0; j < L; j++)
	    cout << state[index_to_n(i, j)] << " ";
	cout << "\n";
    }
    for (int t=0; t < 1000; t++) 
    	flip();

    cout << "\n\n" << endl;
    // Now output the spins
    for (int i=0; i < L; i++) {
	for (int j=0; j < L; j++)
	    cout << state[index_to_n(i, j)] << " ";
	cout << "\n";
    }
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

void Metropolis::flip() {
    // It's not great to have this here, but we can't make it global because we need SIZE
    uniform_int_distribution<int> ran_pos(0, SIZE-1);
    int index = ran_pos(gen);
    double flip_axis = ran_u(gen) * M_PI;
    double old_angle = state[index];
    double new_angle = constrain_to_2pi(2.0 * flip_axis - old_angle);
    double E1 = 0.0;
    double E2 = 0.0;

    E1 += p_energy(index, old_angle) * Kstar;
    E2 += p_energy(index, new_angle) * Kstar;

    // If E2 < E1, then we definitely flip
    double p = E2 < E1 ? 1.0 : exp(-(E2 - E1));

    if (ran_u(gen) < p) {
	state[index] = new_angle;
	ENERGY += (E2 - E1) / SIZEd / 2.0;
    }
    return;
}

void Metropolis::get_energy() {
    for (int i=0; i < SIZE; i++)
	ENERGY += p_energy(i, state[i]) * Kstar;
    ENERGY = ENERGY / SIZEd / 2.0;
}

inline int Metropolis::pbc(int n) {
    return n >= 0 ? n % L : L + (n % L);
}

inline double Metropolis::constrain_to_2pi(double alpha) {
    double x = fmod(alpha, 2 * M_PI);
    return x > 0 ? x : x += 2 * M_PI;
}

inline double Metropolis::constrain_to_pi(double alpha) {
    if (alpha < -M_PI)
	return alpha + 2 * M_PI;
    if (alpha > M_PI)
	return alpha - 2 * M_PI;
    return alpha;
}
