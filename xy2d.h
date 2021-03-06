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

// Output array
#define DATALEN 6
#define MAG 0
#define MAG2 1
#define MAG4 2
#define ENE 3
#define ENE2 4
#define VORT 5

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
    double ENERGY = 0.0;
    double m[DATALEN];
    string fname;
    //ofstream spinfile;
	
    inline int index_to_n(int, int);
    inline double bond_energy(double, double);
    void metro_step(double, int);
    void flip(double);
    void neighbours();
    double total_vorticity();
    double magnetization();
    void get_energy(double);
    int pbc(int);
    void print_for_py();

public:
    Metropolis(int, string);
    ~Metropolis();
    void simulate(double, double, double, int);
};

Metropolis::Metropolis(int L, string pref) {
    this->L = L;
    SIZE = L * L;
    SIZEd = (double) SIZE;
    fname = pref + "xy2d_n" + to_string(L) + ".txt";
	
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
	    u = j + 1 == L  ? 0     : j + 1;
	    d = j - 1 == -1 ? L - 1 : j - 1;
	    r = i + 1 == L  ? 0     : i + 1;
	    l = i - 1 == -1 ? L - 1 : i - 1;
	    
	    // Fill in neighbours table
	    n = index_to_n(j, i);
	    neighs[n][0] = index_to_n(u, i);
	    neighs[n][1] = index_to_n(j, r);
	    neighs[n][2] = index_to_n(j, l);
	    neighs[n][3] = index_to_n(d, i);
	}
    }
}

void Metropolis::simulate(double Jmin, double Jmax, double delta, int N) {
    ofstream output;
    cout << "Writing data to " << fname << endl;
    output.open(fname); // Outfile name
    get_energy(Jmax);

    //spinfile.open("spins.txt");
    
    for (double Jstar = Jmax; Jstar > Jmin; Jstar -= delta) {
    	metro_step(Jstar, N);
    	output << Jstar << " " << m[MAG] << " " << m[ENE] << " " << m[MAG2] - m[MAG] * m[MAG] << " "
    	       <<  m[ENE2] - m[ENE] * m[ENE] << " " << 1.0 - m[MAG4]/(3.0 * m[MAG2] * m[MAG2])
    	       <<  " " << m[VORT] << endl;
    	cout << Jstar << endl;
    }
    output.close();

    //spinfile.close();
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

	// Output the spins
	/* for (int i=0; i < L; i++) { */
	/*     for (int j=0; j < L; j++) { */
	/* 	spinfile << state[index_to_n(i, j)] << " "; */
	/*     } */
	/*     spinfile << "\n"; */
	/* } */
	/* spinfile << "\n"; */
    }

    for (int i=0; i < DATALEN; i++)
	m[i] /= (1.0 * N);
    return;
}

inline double Metropolis::bond_energy(double angle1, double angle2) {
    return -1.0 * cos(angle1 - angle2);
}

inline double constrain(double alpha) {
    double x = fmod(alpha, 2 * M_PI);
    return x > 0 ? x : x += 2 * M_PI;
}

void Metropolis::flip(double Jstar) {
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

    // If E2 < E1, then we definitely flip
    double p = E2 < E1 ? 1.0 : exp(-(E2 - E1));

    if (ran_u(gen) < p) {
	state[index] = new_angle;
	ENERGY += (E2 - E1) / SIZEd / 2.0;
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

void Metropolis::get_energy(double Jstar) {
    for (int i=0; i < SIZE; i++) 
	for (int j=0; j < 4; j++) 
	    ENERGY += bond_energy(state[neighs[i][j]], state[i]) * Jstar;
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

/* double Metropolis::total_vorticity() { */
/*     int count = 0; */
/*     for (int x=0; x < L; ++x) { */
/* 	for (int y=0; y < L; ++y) { */
/* 	    const int len = 5; */
/* 	    int arr[len][2] = {{x, y}, {x + 1, y}, {x + 1, y + 1}, {x, y + 1}, {x, y}}; */
/* 	    // int arr[len][2] = {{x, y}, {x + 1, y}, {x + 2, y}, {x + 3, y}, */
/* 	    // 		       {x + 3, y - 1}, {x + 3, y - 2}, {x + 3, y - 3}, */
/* 	    // 		       {x + 2, y - 3}, {x + 1, y - 3}, {x, y - 3}, */
/* 	    // 		       {x, y - 2}, {x, y - 1}, {x, y}}; */
	    
/* 	    double prev = state[index_to_n(arr[0][0] % L, arr[0][1] % L)]; */
/* 	    double delta = 0.0; */
/* 	    double newangle; */

/* 	    for (int i=1; i < len; ++i) { */
/* 		newangle = state[index_to_n(pbc(arr[i][0]), pbc(arr[i][1]))]; */
/* 		delta += another_constrain(newangle - prev); */
/* 		prev = newangle; */
/* 	    } */

/* 	    if (abs(delta - 2 * M_PI) < 0.01 || abs(delta + 2 * M_PI) < 0.01) */
/* 		++count; */
/* 	} */
/*     } */
/*     return count / SIZEd; */
/* } */

double angle_mod(double alpha) {
    if (alpha < -M_PI)
	return alpha + 2 * M_PI;
    if (alpha > M_PI)
	return alpha - 2 * M_PI;
    return alpha;
}

double Metropolis::total_vorticity()
{
    int count = 0;
    int antivort = 0;
    for (int x=0; x < L; ++x) {
	for (int y=0; y < L; ++y) {
	    const int len = 5;
	    int arr[len][2] = {{x, y}, {x + 1, y}, {x + 1, y + 1}, {x, y + 1}, {x, y}};
	    /* const int len = 13; */
	    /* int arr[len][2] = {{x, y}, {x + 1, y}, {x + 2, y}, {x + 3, y}, */
	    /* 		       {x + 3, y - 1}, {x + 3, y - 2}, {x + 3, y - 3}, */
	    /* 		       {x + 2, y - 3}, {x + 1, y - 3}, {x, y - 3}, */
	    /* 		       {x, y - 2}, {x, y - 1}, {x, y}}; */

	    double prev = state[index_to_n(pbc(arr[0][0]), pbc(arr[0][1]))];
	    double delta = 0.0;
	    double newangle;

	    for (int i=1; i < len; ++i) {
		newangle = state[index_to_n(pbc(arr[i][0]), pbc(arr[i][1]))];
		delta += angle_mod(newangle - prev);
		prev = newangle;
	    }

	    // Now we've gone around our plaquette and know what the value of the line
	    // integral is. If it's any multiple of 2pi, we should add it to the count
	    if (abs(delta - 2 * M_PI) < 0.01 || abs(delta - 4 * M_PI) < 0.01 || abs(delta - 6 * M_PI) < 0.01) {
		count++;
	    }
	    else if (abs(delta + 2 * M_PI) < 0.01 || abs(delta + 4 * M_PI) < 0.01 || abs(delta + 6 * M_PI) < 0.01) {
		antivort++;
	    }
	}
    }
    /* if (count != 0) */
    /* 	cout << count << " " << antivort << endl; */
    /* if (count != antivort) */
    /* 	cout << "very bad!" << endl; */
    return count / SIZEd;
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
