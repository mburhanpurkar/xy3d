#include <cstdlib>
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
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
double K = 1.0;

// 3D XY model
int L = 8;
int SIZE = L * L * L;
double SIZEd = (double) L * (double) L * (double) L;


using namespace std;

void init(double state[SIZE], mt19937& gen, uniform_real_distribution<double>& aran);
void neighbours(int neighs[SIZE][6], int plaq[SIZE][12][3]);
void metro_step(double state[SIZE], int neighs[SIZE][6], int plaq[SIZE][12][3], double t, int N,
		double& energy, mt19937& gen, uniform_real_distribution<double>& ran_u,
		uniform_int_distribution<int>& ran_pos, double m[DATALEN]);
double magnetization(double state[SIZE]);
void flip(double state[SIZE], int neighs[SIZE][6], double& energy, mt19937& gen,
	  uniform_real_distribution<double>& ran_u, uniform_int_distribution<int>& ran_pos, double t);
double get_energy(double state[SIZE], int neighs[SIZE][6], int plaq[SIZE][12][3]);
void write(ofstream& file, double t, int N, double m[DATALEN]);


int main(void)
{
    ofstream output;
    double state[SIZE];    // Stores the spins
    int N = 1e4;           // Number of system averages taken for each call to metro_step
    double energy;
    double m[DATALEN];     // For writing-out observables
    int neighs[SIZE][6];   // Store neighbours
    int plaq[SIZE][12][3]; // Store plaquettes
    
    random_device rd;
    mt19937 gen(rd());     // Mersenne Twister RNG
    uniform_int_distribution<int> ran_pos(0, SIZE-1);  // Get any random integer
    uniform_real_distribution<double> ran_u(0.0, 1.0); // Unif(0, 1)--will need Unif(0, pi) and Unif(0, 2pi) in this code
    
    double deltat, deltat_crit;              // Temperature spacing near and away critical temperature
    double tmax, tmin, tcrit_up, tcrit_down; // Define critical temperature range (corresponds to spacing above)
    tmax = 5.0;
    tmin = 0.0;
    deltat = 0.1;
    
    output.open(("mod_n" + to_string(L) + ".txt")); // Outfile name
    init(state, gen, ran_u);
    neighbours(neighs, plaq);                 // Create a table of neighbours to prevent extra calculation later
    energy = get_energy(state, neighs, plaq); // Compute initial energy--only time we do a complete direct energy calculation
    cout << "Got energy "<< energy << endl;
    
    for (double t = tmax; t > tmin; t -= deltat)
    {
    	metro_step(state, neighs, plaq, t, N, energy, gen, ran_u, ran_pos, m);
    	write(output, t, N, m);
    	cout << t << endl;
    }
    output.close();

    return 0;
}

void init(double state[SIZE], mt19937& gen, uniform_real_distribution<double>& aran)
{
    for (int i=0; i < SIZE; i++) state[i] = aran(gen) * 2 * M_PI;
    return;
}

inline int index_to_n(int i, int j, int k)
{
    return i  * L * L + j * L + k;
}

void neighbours(int neighs[SIZE][6], int plaq[SIZE][12][3])
{
    int u,d,r,l,a,b,n;

    for (int i=0; i < L; i++)
    {
	for (int j=0; j < L; j++)
	{
	    for (int k=0; k < L; k++)
	    {
		// Periodic boundary
		u = j + 1 == L  ? 0     : j + 1;
		d = j - 1 == -1 ? L - 1 : j - 1;
		r = k + 1 == L  ? 0     : k + 1;
		l = k - 1 == -1 ? L - 1 : k - 1;
		a = i + 1 ==  L ? 0     : i + 1;
		b = i - 1 == -1 ? L - 1 : i - 1;

		// Fill in neighbours table
		n = i * L * L + j * L + k;
		neighs[n][UP]    = index_to_n(i, u, k);
		neighs[n][DOWN]  = index_to_n(i, d, k);
		neighs[n][RIGHT] = index_to_n(i, j, r);
		neighs[n][LEFT]  = index_to_n(i, j, l);
		neighs[n][AZ]    = index_to_n(a, j, k);
		neighs[n][AZD]   = index_to_n(b, j, k);
		
		// xy plane
		plaq[n][0][0] = index_to_n(d, j, k); plaq[n][0][1] = index_to_n(d, r, k); plaq[n][0][2] = index_to_n(i, r, k);
		plaq[n][1][0] = index_to_n(u, j, k); plaq[n][1][1] = index_to_n(u, r, k); plaq[n][1][2] = index_to_n(i, r, k);
		plaq[n][2][0] = index_to_n(u, j, k); plaq[n][2][1] = index_to_n(u, l, k); plaq[n][2][2] = index_to_n(i, l, k);
		plaq[n][3][0] = index_to_n(d, j, k); plaq[n][3][1] = index_to_n(d, l, k); plaq[n][3][2] = index_to_n(i, l, k);

		// yz plane
		plaq[n][4][0] = index_to_n(i, r, k); plaq[n][4][1] = index_to_n(i, r, a); plaq[n][4][2] = index_to_n(i, j, a);
		plaq[n][5][0] = index_to_n(i, l, k); plaq[n][5][1] = index_to_n(i, l, a); plaq[n][5][2] = index_to_n(i, k, a);
		plaq[n][6][0] = index_to_n(i, l, k); plaq[n][6][1] = index_to_n(i, l, b); plaq[n][6][2] = index_to_n(i, j, b);
		plaq[n][7][0] = index_to_n(i, r, k); plaq[n][7][1] = index_to_n(i, r, b); plaq[n][7][2] = index_to_n(i, j, b);
		
		// xz plane
		plaq[n][8][0] = index_to_n(d, j, a); plaq[n][8][1] = index_to_n(d, j, k); plaq[n][8][2] = index_to_n(i, j, a);
		plaq[n][9][0] = index_to_n(i, j, a); plaq[n][9][1] = index_to_n(u, j, k); plaq[n][9][2] = index_to_n(u, j, a); 
		plaq[n][10][0] = index_to_n(u, j, k); plaq[n][10][1] = index_to_n(u, j, b); plaq[n][10][2] = index_to_n(i, j, b);
		plaq[n][11][0] = index_to_n(d, j, k); plaq[n][11][1] = index_to_n(d, j, b); plaq[n][11][2] = index_to_n(i, j, b);

	    }
	}
    }
    return;
}

void metro_step(double state[SIZE], int neighs[SIZE][6], int plaq[SIZE][12][3], double t, int N, double& energy, mt19937& gen,
		uniform_real_distribution<double>& ran_u, uniform_int_distribution<int>& ran_pos, double m[DATALEN])
{
    double sum, chi, heat;
 
    for (int i=0; i < DATALEN; i++)
    	m[i] = 0.0;

    // Thermalize--TODO: optimize number of thermalization steps later!
    for (int i=0; i < SIZE * 1000; i++)
    	flip(state, neighs, energy, gen, ran_u, ran_pos, t);

    for (int i=0; i < N; i++)
    {
    	for (int j=0; j < SIZE; j++)
    	    flip(state, neighs, energy, gen, ran_u, ran_pos, t);
	
        // Once the state is updated, re-compute quantities
    	sum = magnetization(state);
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

inline double bond_energy(double angle1, double angle2)
{
    return -1.0 * J * J * cos(angle1 - angle2);
}

void flip(double state[SIZE], int neighs[SIZE][6], double& energy, mt19937& gen,
	  uniform_real_distribution<double>& ran_u, uniform_int_distribution<int>& ran_pos, double t)
{
    int index = ran_pos(gen);
    double flip_axis = ran_u(gen) * M_PI; // Random angle between 0 and pi
    double old_angle = state[index];
    double new_angle = 2.0 * flip_axis - old_angle;
    double E1 = 0.0;
    double E2 = 0.0;

    for (int i=0; i < 6; i++)
    {
	E1 += bond_energy(state[neighs[index][i]], old_angle);
	E2 += bond_energy(state[neighs[index][i]], new_angle);
    }
     
    // If E2 < E1, then we definitely flip
    double p = E2 < E1 ? 1.0 : exp(-(E2 - E1) / t);

    if (ran_u(gen) < p)
    {
	state[index] = new_angle;
	energy += (E2 - E1) / SIZEd / 3.0;
    } 
    return;
}

double magnetization(double state[SIZE])
{
    double sum_x = 0.0;
    double sum_y = 0.0;

    for (int i=0; i < SIZE; i++)
    {
	sum_x += cos(state[i]);
	sum_y += sin(state[i]);
    }
    sum_x /= SIZEd;
    sum_y /= SIZEd;
    
    return sqrt(sum_x * sum_x + sum_y * sum_y);
}

double get_energy(double state[SIZE], int neighs[SIZE][6], int plaq[SIZE][12][3])
{
    double energy = 0.0;
    double energy2 = 0.0;
    double prod;

    // This is almost identical to the xy energy computation, except it
    // has a different normalization and uses J^2 instead of J
    for (int i=0; i < SIZE; i++)
    {
	for (int j=0; j < 6; j++)
	    energy += bond_energy(state[neighs[i][j]], state[i]);
    }

    // Then we tack on an extra K term, involving a plaquette sum
    for (int i=0; i < SIZE; i++)
    {
	// Iterate over all 12 plaquettes for each point
	for (int p=0; p < 12; p++)
	{
	    // Select each of the four elements
	    prod = 1.0;
	    for (int elt1=0; elt1 < 4; elt1++)
	    {
		for (int elt2=3; elt2 > elt1; elt2--) prod *= cos((state[plaq[i][p][elt1]] - state[plaq[i][p][elt2]]) / 2.0);
	    }
	    energy2 += prod;
	}
    }
	
    return energy / SIZEd / 4.0 - K * J * J * J * J * energy2;
}

void write(ofstream& file, double t, int N, double m[DATALEN])
{
    cout << "Writing to file..." << endl;

    // Index Key for Lazy Plotting
    // 0 -- temperature
    // 1 -- magnetization
    // 2 -- energy
    // 3 -- susceptibility
    // 4 -- specific heat
    // 5 -- binder cumulant
    file << t << " " << m[MAG] << " " << m[ENE] << " " << m[MAG2] - m[MAG] * m[MAG] << " "
     	 <<  m[ENE2] - m[ENE] * m[ENE] << " " << 1.0 - m[MAG4]/(3.0 * m[MAG2] * m[MAG2]) << endl;
    return;
}
