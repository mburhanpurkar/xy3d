#include <cstdlib>
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <cstdlib>
#define _USE_MATH_DEFINES


// Define indexing structure
#define UP 0
#define RIGHT 1
#define LEFT 2
#define DOWN 3
#define AZ 4
#define AZD 5


// Output array
#define DATALEN 5
#define MAG 0
#define MAG2 1
#define MAG4 2
#define ENE 3
#define ENE2 4


// Coupling constants
double J = 1.0;
double K = 0.0;

using namespace std;

random_device rd;
mt19937 gen(rd());     // Mersenne Twister RNG    


class Metropolis {
    int L;
    int SIZE;
    double SIZEd;
    double* state;
    int** neighs;
    int*** plaqs;
    
    // Forward declarations
    inline int index_to_n(int i, int j, int k);
    inline double penergy_delta(int n, double old_angle, double new_angle);
    inline double bond_energy(double angle1, double angle2);
    void metro_step(double t, int N, double& energy, double m[DATALEN]);
    void flip(double& energy, double t);
    void neighbours();
    double magnetization();
    double get_energy();

public:
    Metropolis(int L);
    ~Metropolis();
    void simulate(double tmin, double tmax, double deltat, int N);
};

Metropolis::Metropolis(int L) {
    this->L = L;
    SIZE = L * L * L;
    SIZEd = (double) SIZE;
    
    // Randomly initialize the state
    state = new double[SIZE];
    uniform_real_distribution<double> ran_u(0.0, 1.0);
    for (int i=0; i < SIZE; ++i) state[i] = ran_u(gen) * 2 * M_PI;
    
    // Set up neighbour and plaquette tables
    neighs = new int*[SIZE];
    plaqs = new int**[SIZE];
    for(int i = 0; i < SIZE; ++i) {
	neighs[i] = new int[6];
	plaqs[i] = new int*[12];
	for(int j = 0; j < 12; ++j)
	    plaqs[i][j] = new int[3];
    }
    neighbours();
}


Metropolis::~Metropolis() {
    delete[] state;
    for(int i = 0; i < SIZE; ++i)
	delete[] neighs[i];
    delete[] neighs;
    for(int i = 0; i < SIZE; ++i) {
	for(int j = 0; j < 12; ++j)
	    delete[] plaqs[i][j];
	delete[] plaqs[i];
    }
    delete[] plaqs;
}

inline int Metropolis::index_to_n(int i, int j, int k) {
    return i  * L * L + j * L + k;
}

void Metropolis::neighbours() {
    int u,d,r,l,a,b,n;
    
    for (int i=0; i < L; i++) {
	for (int j=0; j < L; j++) {
	    for (int k=0; k < L; k++) {
		// Periodic boundary
		u = j + 1 == L  ? 0     : j + 1;
		d = j - 1 == -1 ? L - 1 : j - 1;
		r = k + 1 == L  ? 0     : k + 1;
		l = k - 1 == -1 ? L - 1 : k - 1;
		a = i + 1 ==  L ? 0     : i + 1;
		b = i - 1 == -1 ? L - 1 : i - 1;
		
		// Fill in neighbours table
		n = i * L * L + j * L + k;
		cout << n << endl;
		neighs[n][UP]    = index_to_n(i, u, k);
		neighs[n][DOWN]  = index_to_n(i, d, k);
		neighs[n][RIGHT] = index_to_n(i, j, r);
		neighs[n][LEFT]  = index_to_n(i, j, l);
		neighs[n][AZ]    = index_to_n(a, j, k);
		neighs[n][AZD]   = index_to_n(b, j, k);
		
		// xy plane
		plaqs[n][0][0] = index_to_n(d, j, k); plaqs[n][0][1] = index_to_n(d, r, k); plaqs[n][0][2] = index_to_n(i, r, k);
		plaqs[n][1][0] = index_to_n(u, j, k); plaqs[n][1][1] = index_to_n(u, r, k); plaqs[n][1][2] = index_to_n(i, r, k);
		plaqs[n][2][0] = index_to_n(u, j, k); plaqs[n][2][1] = index_to_n(u, l, k); plaqs[n][2][2] = index_to_n(i, l, k);
		plaqs[n][3][0] = index_to_n(d, j, k); plaqs[n][3][1] = index_to_n(d, l, k); plaqs[n][3][2] = index_to_n(i, l, k);
		    
		// yz plane
		plaqs[n][4][0] = index_to_n(i, r, k); plaqs[n][4][1] = index_to_n(i, r, a); plaqs[n][4][2] = index_to_n(i, j, a);
		plaqs[n][5][0] = index_to_n(i, l, k); plaqs[n][5][1] = index_to_n(i, l, a); plaqs[n][5][2] = index_to_n(i, k, a);
		plaqs[n][6][0] = index_to_n(i, l, k); plaqs[n][6][1] = index_to_n(i, l, b); plaqs[n][6][2] = index_to_n(i, j, b);
		plaqs[n][7][0] = index_to_n(i, r, k); plaqs[n][7][1] = index_to_n(i, r, b); plaqs[n][7][2] = index_to_n(i, j, b);
		
		// xz plane
		plaqs[n][8][0] = index_to_n(d, j, a); plaqs[n][8][1] = index_to_n(d, j, k); plaqs[n][8][2] = index_to_n(i, j, a);
		plaqs[n][9][0] = index_to_n(i, j, a); plaqs[n][9][1] = index_to_n(u, j, k); plaqs[n][9][2] = index_to_n(u, j, a); 
		plaqs[n][10][0] = index_to_n(u, j, k); plaqs[n][10][1] = index_to_n(u, j, b); plaqs[n][10][2] = index_to_n(i, j, b);
		plaqs[n][11][0] = index_to_n(d, j, k); plaqs[n][11][1] = index_to_n(d, j, b); plaqs[n][11][2] = index_to_n(i, j, b);
	    }
	}
    }
}

void Metropolis::simulate(double tmin, double tmax, double deltat, int N) {
    ofstream output;
    
    output.open(("mod_n" + to_string(L) + ".txt")); // Outfile name
    // Get the initial energy
    double energy = get_energy();
    double m[DATALEN];     // For writing-out observables
    for (double t = tmax; t > tmin; t -= deltat) {
    	metro_step(t, N, energy, m);
	output << t << " " << m[MAG] << " " << m[ENE] << " " << m[MAG2] - m[MAG] * m[MAG] << " "
	       <<  m[ENE2] - m[ENE] * m[ENE] << " " << 1.0 - m[MAG4]/(3.0 * m[MAG2] * m[MAG2]) << endl;
    	cout << t << endl;
    }
    output.close();
}

void Metropolis::metro_step(double t, int N, double& energy, double m[DATALEN]) {
    double sum, chi, heat;
 
    for (int i=0; i < DATALEN; i++)
    	m[i] = 0.0;

    // Thermalize--TODO: optimize number of thermalization steps later!
    for (int i=0; i < SIZE * 1000; i++)
    	flip(energy, t);

    for (int i=0; i < N; i++) {
    	for (int j=0; j < SIZE; j++)
    	    flip(energy, t);
	
        // Once the state is updated, re-compute quantities
    	sum = magnetization();
    	chi = sum * sum;
    	heat = energy * energy;
	m[MAG] += sum;        // Magnetization
    	m[MAG2] += chi;       // Susceptibility
    	m[MAG4] += chi * chi; // Binder
    	m[ENE] += energy;     // Energy
    	m[ENE2] += heat;      // Specific heat
    }
    
    // Take an average
    for (int i=0; i < DATALEN; i++)
	m[i] /= (1.0 * N);

    return;
}

inline double Metropolis::bond_energy(double angle1, double angle2) {
    return -J * cos(angle1 - angle2);
}

inline double Metropolis::penergy_delta(int n, double old_angle, double new_angle) {
    // Maybe the best way to define this function is such that it returns the plaqsuette energy
    // difference between the new spin and old spin? I can't think of a better way at the moment
    // that wouldn't have massive array-copying cost...
    double prod_cur, prod_new, res, ref_angle1, ref_angle2;
    double energy_cur = 0.0;
    double energy_new = 0.0;
    
    // Compute the angle product over the plaquettes for the selected state
    for (int p=0; p < 12; p++) {
	// Select each of the four elements
	prod_cur = 1.0;
	prod_new = 1.0;
	for (int x=1; x < 4; x++) {
	    ref_angle1 = state[plaqs[n][p][x]];
	    for (int y=0; y < x; y++) {
		ref_angle2 = state[plaqs[n][p][y]];
		// This never hits y = 3, so hopefully no indexing problems here!
		if (x == 3) {
		    // We've picked the plaqsuette "seed"--different factors
		    // for flipped and unflipped spins
		    prod_cur *= cos((old_angle - ref_angle2) / 2.0);
		    prod_new *= cos((new_angle - ref_angle2) / 2.0);
		}
		else {
		    // The factor should be the same
		    res = cos((ref_angle1 - ref_angle2) / 2.0);
		    prod_cur *= res;
		    prod_new *= res;
		}
	    }
	}
	energy_cur += prod_cur;
	energy_new += prod_new;
    }    
    return - K * (energy_new - energy_cur);
}

void Metropolis::flip(double& energy, double t) {
    uniform_int_distribution<int> ran_pos(0, SIZE-1);
    uniform_real_distribution<double> ran_u(0.0, 1.0);
    int index = ran_pos(gen);
    double flip_axis = ran_u(gen) * M_PI; // Random angle between 0 and pi
    double old_angle = state[index];
    double new_angle = 2.0 * flip_axis - old_angle;
    double E1 = 0.0;
    double E2 = 0.0;

    for (int i=0; i < 6; i++) {
	E1 += bond_energy(state[neighs[index][i]], old_angle);
	E2 += bond_energy(state[neighs[index][i]], new_angle);
    }

    // Now we need to account for the plaquette energy. For efficiency reasons, I wrote the energy
    // function to compute the energy difference between the flipped and unflipped spins, such that
    // penergy_delta returns energy_new - energy_cur--that is, if it is energetically favourable to
    // flip in terms of plaquette energy, the energy difference will be positive. Thus, we should
    // ADD the result of penergy_delta to E2 and then do the comparison...
    E2 += penergy_delta(index, old_angle, new_angle);

    // If E2 < E1, then we definitely flip
    double p = E2 < E1 ? 1.0 : exp(-(E2 - E1) / t);

    if (ran_u(gen) < p) {
	state[index] = new_angle;
	energy += (E2 - E1) / SIZEd;
    } 
    return;
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

double Metropolis::get_energy() {
    double energy = 0.0;
    double energy2 = 0.0;
    double prod, ref_angle, seed_angle;

    // This is almost identical to the xy energy computation, except it
    // has a different normalization and uses J^2 instead of J
    for (int i=0; i < SIZE; ++i) {
	for (int j=0; j < 6; ++j) {
	    int x =  neighs[i][j];
	    double y = state[x];
	    double z = state[i];
	    energy += bond_energy(y, z);
	}
    }

    // Then we tack on an extra K term, involving a plaquette sum
    for (int i=0; i < SIZE; i++) {
	seed_angle = state[i];
	// Compute the angle product over the plaquettes for the selected state
	for (int p=0; p < 12; p++) {
	    // Select each of the four elements
	    prod = 1.0;
	    for (int x=1; x < 4; x++) {
		for (int y=0; y < x; y++) {
		    ref_angle = state[plaqs[i][p][y]];
		    if (x == 3)
			prod *= cos((seed_angle - ref_angle) / 2.0);
		    else
			prod *= cos((state[plaqs[i][p][x]] - ref_angle) / 2.0);
		}
	    }
	    energy2 += prod;
	}    
    }
	
    return (energy - K * energy2) / SIZEd;
}


int main(int argc, char** argv) {
    if (argc == 2) {
	Metropolis metropolis(atoi(argv[1]));
	metropolis.simulate(0.1, 5.0, 0.1, 1000);
	return 0;
    }
    else
	return 1;
}
