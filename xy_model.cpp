#include <cstdlib>
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>

#define _USE_MATH_DEFINES

#define UP 0
#define RIGHT 1
#define LEFT 2
#define DOWN 3
#define AZ 4
#define AZD 5

// This is a 3D XY model
int L = 8;
int SIZE = L * L * L;
double SIZEd = (double) L * (double) L * (double) L;

#define DATALEN 9
#define MAG 0
#define MAG2 1
#define MAG4 2
#define MAGE 3
#define SUSE 4
#define ENE 5
#define ENE2 6
#define ENE4 7
#define ENEE 8
#define CHE 9


using namespace std;

void init(double state[SIZE], mt19937& gen, uniform_real_distribution<double>& aran);
void neighbours(int neighs[SIZE][6]);
void metro_step(double state[SIZE], int neighs[SIZE][6], double tstar, int N,
		double& energy, mt19937& gen, uniform_real_distribution<double>& ran_u,
		uniform_int_distribution<int>& ran_pos, double m[DATALEN]);
double magnetization(double state[SIZE]);
void flip(double state[SIZE], int neighs[SIZE][6], double& energy, mt19937& gen,
	  uniform_real_distribution<double>& ran_u, uniform_int_distribution<int>& ran_pos, double tstar);
double get_energy(double state[SIZE], int neighs[SIZE][6]);
void write(ofstream& file, double tstar, int N, double m[DATALEN]);


int main(void)
{
    double state[SIZE];  // Stores the spins
    int N;               // Number of averages done into the system
    double energy;
    double m[DATALEN];   // Magnetization- and energy-related output data: see defined indices above
    int neighs[SIZE][6]; // Store neighbours
    
    random_device rd;
    mt19937 gen(rd()); //Mersenne Twister RNG
    //mt19937 gen(1834760); //Mersenne Twister RNG
    uniform_int_distribution<int> ran_pos(0, SIZE-1); //Get any random integer
    uniform_real_distribution<double> ran_u(0.0, 1.0); //Unif(0, 1)--will need Unif(0, pi) and Unif(0, 2pi) in this code
    
    double deltat, deltat_crit; //Temperature deltas for Metropolis and Wolff updates
    double tmax, tmin, tcrit_up, tcrit_down; //Max and min temperature, and interval where we apply Wolff
    tmax = 5.0;
    tmin = 0.0;
    tcrit_up = 2.3;
    tcrit_down = 2.1;
    deltat = 0.1;
    deltat_crit = 0.01;

    ofstream output; //Output of the stream

    init(state, gen, ran_u); //Init randomly
    neighbours(neighs); //Get neighbour table
    energy = get_energy(state, neighs); //Compute initial energy

    N = 1e4;

    output.open(("n" + to_string(L) + "_nice.txt"));
    for (double tstar = tmax; tstar > tcrit_up; tstar -= deltat)
    {
    	metro_step(state, neighs, tstar, N, energy, gen, ran_u, ran_pos, m);
    	write(output, tstar, N, m);
    	cout << tstar << endl;
    }
    for (double tstar = tcrit_up - deltat_crit; tstar > tcrit_down; tstar -= deltat_crit)
    {
    	metro_step(state, neighs, tstar, N, energy, gen, ran_u, ran_pos, m);
    	write(output, tstar, N, m);
    	cout << tstar << endl;
    }
    for (double tstar = tcrit_down - deltat; tstar >= tmin; tstar -= deltat)
    {
	metro_step(state, neighs, tstar, N, energy, gen, ran_u, ran_pos, m);
	write(output, tstar, N, m);
	cout << tstar << endl;
    }
    output.close();

    return 0;
}


//Initialices the grid in which we are going to do the Ising, using random values
void init(double state[SIZE], mt19937& gen, uniform_real_distribution<double>& aran)
{
    // Note that aran is Unif(0, 1)--we need to multiply by 2pi to scale
    for (int i=0; i < SIZE; i++) state[i] = aran(gen) * 2 * M_PI; //Generate numbers
    return;
}

inline int index_to_n(int i, int j, int k)
{
    return i  * L * L + j * L + k;
}

//Fills the neigbour table
void neighbours(int neighs[SIZE][6])
{
    int u,d,r,l,a,b,n;

    for (int i=0; i < L; i++)
    {
	for (int j=0; j < L; j++)
	{
	    for (int k=0; k < L; k++)
	    {
		//Get the (x,y) with periodic boundaries
		u = j + 1 == L  ? 0     : j + 1;
		d = j - 1 == -1 ? L - 1 : j - 1;
		r = k + 1 == L  ? 0     : k + 1;
		l = k - 1 == -1 ? L - 1 : k - 1;
		a = i + 1 ==  L ? 0     : i + 1;
		b = i - 1 == -1 ? L - 1 : i - 1;
	    
		//(x,y) to index notation and store in table
		n = i * L * L + j * L + k;
		neighs[n][UP]    = index_to_n(i, u, k);
		neighs[n][DOWN]  = index_to_n(i, d, k);
		neighs[n][RIGHT] = index_to_n(i, j, r);
		neighs[n][LEFT]  = index_to_n(i, j, l);
		neighs[n][AZ]    = index_to_n(a, j, k);
		neighs[n][AZD]   = index_to_n(b, j, k);
	    }
	}
    }
    return;
}

//Do all the things needed for a certain temperature
void metro_step(double state[SIZE], int neighs[SIZE][6], double tstar, int N, double& energy, mt19937& gen,
		uniform_real_distribution<double>& ran_u, uniform_int_distribution<int>& ran_pos, double m[DATALEN])
{
    double sum = 0.0;
    double energysum;
    double chi, heat;
    double old_sum, old_chi, old_heat, old_energy;
 
    for (int i=0; i < DATALEN; i++)
	m[i] = 0.0; //Init the values

    //Thermalize the state
    for (int i=0; i < SIZE * 1000; i++)
	flip(state, neighs, energy, gen, ran_u, ran_pos, tstar);

    old_sum = 0.0;
    old_chi = 0.0;
    old_heat = 0.0;
    old_energy = 0.0;
 
    for (int i=0; i < N; i++)
    {
	//make changes and then average
	for (int j=0; j < SIZE; j++)
	    flip(state, neighs, energy, gen, ran_u, ran_pos, tstar);
	
        //Compute quantities at time j
	sum = magnetization(state);
	chi = sum * sum;
	heat = energy * energy;
		
        //Add all the quantities
	m[MAG] += sum; //Magnetization
	m[MAG2] += chi; //For the susceptibility
	m[MAG4] += chi * chi; //For the Binder cumulant and also variance of susceptibility
	m[ENE] += energy; //Energy
	m[ENE2] += heat; //For specific heat
	m[ENE4] += heat * heat; //For the variance of specific heat
	//This are used for errors,
	m[MAGE] += old_sum * sum; //in magnetization
	m[SUSE] += old_chi * chi; //in susceptibility
	m[ENEE] += old_energy * energy; //in energy
	m[CHE] += old_heat * heat; //in specific heat

	//Get the value for the next iteration
	old_sum = sum;
	old_energy = energy;
	old_chi = chi;
	old_heat = heat;
    }

    //Finish the average
    for (int i=0; i < DATALEN; i++)
	m[i] /= (1.0 * N);

    return;
}


inline double bond_energy(double angle1, double angle2)
{
    return -1.0 * cos(angle1 - angle2);
}


//Flip a spin via Metropolis
void flip(double state[SIZE], int neighs[SIZE][6], double& energy, mt19937& gen,
	  uniform_real_distribution<double>& ran_u, uniform_int_distribution<int>& ran_pos, double tstar)
{
    int index = ran_pos(gen); //Get a random position to flip
    double flip_axis = ran_u(gen) * M_PI; //Random angle between 0 and pi
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
    double p = E2 < E1 ? 1.0 : exp(-(E2 - E1) / tstar);

    if (ran_u(gen) < p)
    {
	state[index] = new_angle;
	energy += (E2 - E1) / SIZEd / 3.0;
    } 
    return;
}

//Compute the magnetization
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

//Computes the energy of the system
double get_energy(double state[SIZE], int neighs[SIZE][6])
{
    double energy = 0.0; //Sum
    //For every spin,
    for (int i=0; i < SIZE; i++)
    {
	// for every neightbour
	for (int j=0; j < 6; j++)
	    energy += bond_energy(state[neighs[i][j]], state[i]);
    }
    return energy / SIZEd / 3.0; //Return the energy
}



void write(ofstream& file, double tstar, int N, double m[DATALEN])
{
    file << tstar << " " << 1.0/tstar << " "; //Write T and B
    cout << "writing to file" << endl;
    //We here take in account residual errors, which, for low T, makes the quantities chi, ch, etc.
    //to diverge. This must be substracted. That's why we use an abs for correlation time and also
    //a check to avoid zero value of variances.

    //Then write the quantities and the corresponding errors to a file. The four things are equal,
    //but each one referred to a different quantity.

    double chi = m[MAG2] - m[MAG] * m[MAG]; //Magnetic susceptibility (up to T factor)
    double rhom = chi != 0 ? (m[MAGE] - m[MAG] * m[MAG]) / chi : 0.0; //Rho magnetization, computed if chi != 0
    double taugm = rhom != 1.0 ? rhom / (1.0 - rhom) : 0.0; //Taug magnetization, computed if rhom != 0
    file << m[MAG] << " " << sqrt(chi * abs(2.0 * taugm + 1) / (1.0*N)) << " "; //Write everything

    double fourth = m[MAG4] - m[MAG2] * m[MAG2]; //Susceptibility variance
    double rhos = fourth != 0.0 ? (m[SUSE] - m[MAG2] * m[MAG2]) / fourth : 0.0; //Rho susceptibility
    double taugs = rhos != 1.0 ? rhos /(1.0 - rhos) : 0.0; //Taug susceptibility
    double error_sq = sqrt(fourth * abs(2.0 * taugs + 1) / (1.0*N));
    file << " " << chi << " " << error_sq << " ";

    double heat = m[ENE2] - m[ENE] * m[ENE]; //Specific heat (up to T^2 factor)
    double rhoe = heat != 0.0 ? (m[ENEE] - m[ENE]*m[ENE]) / heat : 0.0;
    double tauge = rhoe != 1.0 ? rhoe / (1.0 - rhoe) : 0.0;
    file << " " << m[ENE] << " " << sqrt(heat * abs(2.0 * tauge + 1) / (1.0*N)) << " ";

    double fourth_ene = m[ENE4] - m[ENE2] * m[ENE2];
    double rhoc = fourth_ene != 0.0 ? (m[CHE] - m[ENE2] * m[ENE2]) / fourth_ene : 0.0;
    double taugc = rhoc != 1.0 ? rhoc / (1.0 - rhoc) : 0.0;
    file << " " << heat << " " << sqrt(fourth_ene * abs(2.0 * taugc + 1) / (1.0*N)) << " ";

    //Binder cumulant
    double binder = 1.0 - m[MAG4]/(3.0 * m[MAG2] * m[MAG2]); //Computes 4th cumulant minus one, b-1.
    file << binder << " " << 2.0 * (1.0 - binder) * (error_sq / m[MAG2]) << endl;
    return;
}
