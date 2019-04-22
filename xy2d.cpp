#include <cstdlib>
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <cstdlib>
#define _USE_MATH_DEFINES

// Output array
#define DATALEN 5
#define MAG 0
#define MAG2 1
#define MAG4 2
#define ENE 3
#define ENE2 4

// Coupling constants
double J = 1.0;

using namespace std;

random_device rd;
mt19937 gen(rd());     // Mersenne Twister RNG    
uniform_real_distribution<double> ran_u(0.0, 1.0);

class Metropolis {
    int L;
    int SIZE;
    double SIZEd;
    double* state;
    int** neighs;
    int pbc(int n);
    void print_for_py();
    
    inline int index_to_n(int, int);
    inline double bond_energy(double angle1, double angle2);
    void metro_step(double t, int N, double& energy, double m[DATALEN]);
    void flip(double& energy, double t);
    void neighbours();
    void total_vorticity();
    double magnetization();
    double get_energy();

public:
    Metropolis(int L);
    ~Metropolis();
    void simulate(double tmin, double tmax, double deltat, int N);
};

Metropolis::Metropolis(int L) {
    this->L = L;
    SIZE = L * L;
    SIZEd = (double) SIZE;
    
    // Randomly initialize the state
    state = new double[SIZE];
    uniform_real_distribution<double> ran_u(0.0, 1.0);
    for (int i=0; i < SIZE; ++i) state[i] = ran_u(gen) * 2 * M_PI;
    
    // Set up neighbour and plaquette tables
    neighs = new int*[SIZE];
    for(int i = 0; i < SIZE; ++i)
	neighs[i] = new int[4];
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
	    r = j + 1 == L  ? 0     : j + 1;
	    l = j - 1 == -1 ? L - 1 : j - 1;
	    u = i + 1 == L  ? 0     : i + 1;
	    d = i - 1 == -1 ? L - 1 : i - 1;
	    
	    // Fill in neighbours table
	    n = index_to_n(i, j);
	    neighs[n][0] = index_to_n(i, u);
	    neighs[n][1] = index_to_n(i, d);
	    neighs[n][2] = index_to_n(r, j);
	    neighs[n][3] = index_to_n(l, j);
	}
    }
}

void Metropolis::simulate(double tmin, double tmax, double deltat, int N) {
    ofstream output;
    cout << "Writing data to " << "xy2d_n" + to_string(L) + ".txt" << endl;
    output.open(("xy2d_n" + to_string(L) + ".txt")); // Outfile name
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
	total_vorticity();
    }
    
    // Take an average
    for (int i=0; i < DATALEN; i++)
	m[i] /= (1.0 * N);

    return;
}

inline double Metropolis::bond_energy(double angle1, double angle2) {
    return -J * cos(angle1 - angle2);
}

inline double constrain(double alpha) {
    // Assumes you haven't screwed up too badly (i.e. you're 2 * PI away from 0)
    double x = fmod(alpha, 2 * M_PI);
    return x > 0 ? x : x += 2 * M_PI;
}

void Metropolis::flip(double& energy, double t) {
    // It's not great to have this here, but we can't make it global because we need SIZE
    uniform_int_distribution<int> ran_pos(0, SIZE-1);
    int index = ran_pos(gen);
    double flip_axis = ran_u(gen) * M_PI; // Random angle between 0 and pi
    double old_angle = state[index];
    double new_angle = constrain(2.0 * flip_axis - old_angle);
    double E1 = 0.0;
    double E2 = 0.0;

    for (int i=0; i < 4; ++i) {
	E1 += bond_energy(state[neighs[index][i]], old_angle);
	E2 += bond_energy(state[neighs[index][i]], new_angle);
    }

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
    double prod;

    // This is almost identical to the xy energy computation, except it
    // has a different normalization and uses J^2 instead of J
    for (int i=0; i < SIZE; ++i) 
	for (int j=0; j < 4; ++j) 
	    energy += bond_energy(state[i], state[neighs[i][j]]);

    return energy / SIZEd;
}


inline int Metropolis::pbc(int n){
    return n >= 0 ? n % L : L + (n % L);
}

void Metropolis::total_vorticity() {
    // Compute the vorticity in one dimension I guess?
    // Want to do a line integral around 3x3 plquette


    for (int x=0; x < L; ++x) {
	for (int y=0; y < L; ++y) {
	    // For each point in the xy plane, get the neigbours
	    // To do pbc in C++, just do % L for every index
	    
	    // The indices we want for a given (x, y, z) are....
	    // (x + 1, y + 1, z) -> (x, y + 1, z) -> (x - 1, y + 1, z) ->
	    // (x - 1 , y, z) -> (x - 1, y - y, z) -> (x, y - 1, z) ->
	    // (x + 1, y - 1, z) -> (x + 1, y, z) -> start
	    // Check that we have roughly 2pi precession?
	    
	    // Actually, it might be interesting just to output
	    // all of this data to a file an analyze in python
	    const int len = 12;
	    //int arr[len][2] = {{x, y}, {x + 1, y}, {x + 1, y + 1}, {x, y + 1}};
	    
	    // int arr[8][2] = {{x + 1, y + 1}, {x, y + 1}, {x - 1, y + 1},
	    // 		      {x - 1 , y}, {x - 1, y - 1}, {x, y - 1},
	    // 		      {x + 1, y - 1}, {x + 1, y}};
	    
	    // MORE POWER
	    int arr[len][2] = {{x, y}, {x + 1, y}, {x + 2, y}, {x + 3, y},
	    		      {x + 3, y - 1}, {x + 3, y - 2}, {x + 3, y - 3},
	    		      {x + 2, y - 3}, {x + 1, y - 3}, {x, y - 3},
	    		      {x, y - 2}, {x, y - 1}};
	    
	    double prev = state[index_to_n(arr[0][0] % L, arr[0][1] % L)];
	    double delta = 0.0;
	    double newangle;

	    for (int i=1; i < len; ++i) {
		newangle = state[index_to_n(pbc(arr[i][0]), pbc(arr[i][1]))];
		delta += newangle - prev;
		//cout << newangle - prev << endl;
		prev = newangle;
	    }		
	    // Output delta
	    if (delta > 2 * M_PI - 0.1) {
		cout << "**" << delta;
		print_for_py();
	    }
	}
    }
	    
}

void Metropolis::print_for_py() {
    cout << "\n\n[";
    for (int i=0; i < L; ++i) {
	cout << "[";
	for (int j=0; j < L; ++j)
	    (j == L - 1) ? cout << state[index_to_n(i, j)] : cout << state[index_to_n(i, j)] << ", ";
	(i == L - 1) ? cout << "]" : cout << "],\n";
    }
    cout << "]\n\n";
}

int main(int argc, char** argv) {
    if (argc == 2) {
	Metropolis metropolis(atoi(argv[1]));
	metropolis.simulate(1.0, 1.1, 0.1, 1e4);
	return 0;
    }
    else
	return 1;
}
