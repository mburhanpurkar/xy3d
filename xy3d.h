#include <cstdlib>
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <cstdlib>
#define _USE_MATH_DEFINES

using namespace std;

// Output array
const int DATALEN = 6;
const int MAG = 0;
const int MAG2 = 1;
const int MAG4 = 2;
const int ENE = 3;
const int ENE2 = 4;
const int VORT = 5;

// Coupling constant
double J = 1.0; 

random_device rd;
mt19937 gen(rd());     // Mersenne Twister RNG    
uniform_real_distribution<double> ran_u(0.0, 1.0);

class Metropolis {
    int L;
    int SIZE;
    double SIZEd;
    double* state;
    int** neighs;
    double ENERGY = 0.0;
    double m[DATALEN];
    string fname;
	
    inline int index_to_n(int, int, int);
    inline double bond_energy(double, double);
    void metro_step(double, int);
    void flip(double);
    void neighbours();
    double total_vorticity();
    double magnetization();
    void get_energy();
    int pbc(int);
    void test();

public:
    Metropolis(int, string);
    ~Metropolis();
    void simulate(double, double, double, int);
};

Metropolis::Metropolis(int L, string pref) {
    this->L = L;
    SIZE = L * L * L;
    SIZEd = (double) SIZE;
    fname = pref + "xy3d_n" + to_string(L) + ".txt";
	
    // Randomly initialize the state
    state = new double[SIZE];
    uniform_real_distribution<double> ran_u(0.0, 1.0);
    for (int i=0; i < SIZE; ++i) state[i] = ran_u(gen) * 2 * M_PI;
    
    // Set up neighbour and plaquette tables
    neighs = new int*[SIZE];
    for(int i = 0; i < SIZE; ++i)
	neighs[i] = new int[6];
    neighbours();
}

Metropolis::~Metropolis() {
    delete[] state;
    for(int i = 0; i < SIZE; ++i)
	delete[] neighs[i];
    delete[] neighs;
}

inline int Metropolis::index_to_n(int i, int j, int k) {
    return i * L * L + j * L + k;
}

void Metropolis::neighbours() {
    int u,d,r,l,a,b,n;
    
    for (int i=0; i < L; ++i) {
	for (int j=0; j < L; ++j) {
	    for (int k=0; k < L; ++k) {
		// Periodic boundary
		u = j + 1 == L  ? 0     : j + 1;
		d = j - 1 == -1 ? L - 1 : j - 1;
		r = k + 1 == L  ? 0     : k + 1;
		l = k - 1 == -1 ? L - 1 : k - 1;
		a = i + 1 ==  L ? 0     : i + 1;
		b = i - 1 == -1 ? L - 1 : i - 1;

		// Fill in neighbours table
		n = index_to_n(i, j, k);
		neighs[n][0] = index_to_n(i, u, k);
		neighs[n][1] = index_to_n(i, d, k);
		neighs[n][2] = index_to_n(i, j, r);
		neighs[n][3] = index_to_n(i, j, l);
		neighs[n][4] = index_to_n(a, j, k);
		neighs[n][5] = index_to_n(b, j, k);
	    }
	}
    }
}

void Metropolis::simulate(double tmin, double tmax, double deltat, int N) {
    ofstream output;
    cout << "Writing data to " << fname << endl;
    output.open(fname); // Outfile name
    get_energy();
    for (double t = tmax; t > tmin; t -= deltat) {
    	metro_step(t, N);
    	output << t << " " << m[MAG] << " " << m[ENE] << " " << m[MAG2] - m[MAG] * m[MAG] << " "
    	       <<  m[ENE2] - m[ENE] * m[ENE] << " " << 1.0 - m[MAG4]/(3.0 * m[MAG2] * m[MAG2])
    	       <<  " " << m[VORT] << endl;
    	cout << t << endl;
    }
    output.close();
}

void Metropolis::metro_step(double t, int N) {
    double sum, chi, heat;
 
    for (int i=0; i < DATALEN; i++)
    	m[i] = 0.0;

    for (int i=0; i < SIZE * 1000; i++)
    	flip(t);

    for (int i=0; i < N; i++) {
    	for (int j=0; j < SIZE; j++)
    	    flip(t);
	
	sum = magnetization();
    	chi = sum * sum;
    	heat = ENERGY * ENERGY;
	m[MAG] += sum;        // Magnetization
    	m[MAG2] += chi;       // Susceptibility
    	m[MAG4] += chi * chi; // Binder
    	m[ENE] += ENERGY;     // Energy
    	m[ENE2] += heat;      // Specific heat
	m[VORT] += total_vorticity();
    }

    for (int i=0; i < DATALEN; i++)
	m[i] /= (1.0 * N);
    return;
}

inline double Metropolis::bond_energy(double angle1, double angle2) {
    return -J * cos(angle1 - angle2);
}

inline double constrain(double alpha) {
    double x = fmod(alpha, 2 * M_PI);
    return x > 0 ? x : x += 2 * M_PI;
}

void Metropolis::flip(double t) {
    // It's not great to have this here, but we can't make it global because we need SIZE
    uniform_int_distribution<int> ran_pos(0, SIZE-1);
    int index = ran_pos(gen);
    double flip_axis = ran_u(gen) * M_PI;
    double old_angle = state[index];
    double new_angle = constrain(2.0 * flip_axis - old_angle);
    double E1 = 0.0;
    double E2 = 0.0;

    for (int i=0; i < 6; ++i) {
	E1 += bond_energy(state[neighs[index][i]], old_angle);
	E2 += bond_energy(state[neighs[index][i]], new_angle);
    }

    // If E2 < E1, then we definitely flip
    double p = E2 < E1 ? 1.0 : exp(-(E2 - E1) / t);

    if (ran_u(gen) < p) {
	state[index] = new_angle;
	ENERGY += (E2 - E1) / SIZEd / 3.0;
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

void Metropolis::get_energy() {
    for (int i=0; i < SIZE; i++) 
	for (int j=0; j < 6; j++) 
	    ENERGY += bond_energy(state[neighs[i][j]], state[i]);
    ENERGY = ENERGY / SIZEd;
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



double Metropolis::total_vorticity()
{
    int count = 0;
    for (int x=0; x < L; ++x) {
	for (int y=0; y < L; ++y) {
	    for (int z=0; z < L; ++z) {
		const int len = 5;
		int arr[len][3] = {{x, y, z}, {x + 1, y, z}, {x + 1, y + 1, z}, {x, y + 1, z}, {x, y}};
		// int arr[len][3] = {{x, y, z}, {x + 1, y, z}, {x + 2, y, z}, {x + 3, y, z},
		// 		       {x + 3, y - 1, z}, {x + 3, y - 2, z}, {x + 3, y - 3, z},
		// 		       {x + 2, y - 3, z}, {x + 1, y - 3, z}, {x, y - 3},
		// 		       {x, y - 2, z}, {x, y - 1, z}, {x, y, z}};

		// int arr[len][3] = {{x, y, z}, {x + 1, y, z}, {x + 2, y, z}, {x + 3, y},
		// 		   {x + 3, y - 1, z}, {x + 3, y - 2, z}, {x + 3, y - 3},
		// 		   {x + 2, y - 3, z}, {x + 1, y - 3, z}, {x, y - 3},
		// 		   {x, y - 2, z}, {x, y - 1, z}, {x, y, z}};

		double prev = state[index_to_n(arr[0][0] % L, arr[0][1] % L, arr[0][2] % L)];
		double delta = 0.0;
		double newangle;

		for (int i=1; i < len; ++i) {
		    newangle = state[index_to_n(pbc(arr[i][0]), pbc(arr[i][1]), pbc(arr[i][2]))];
		    delta += another_constrain(newangle - prev);
		    prev = newangle;
		}

		// Now we've gone around our plaquette and know what the value of the line
		// integral is. If it's any multiple of 2pi, we should add it to the count
		if (abs(delta - 2 * M_PI) < 0.01 || abs(delta + 2 * M_PI) < 0.01) ++count;
	    }
	}
    }
    return count / SIZEd;
}
