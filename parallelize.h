#include <cstdlib>
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <mpi.h>
#include <cassert>
#define _USE_MATH_DEFINES

using namespace std;

// Output array
#define DATALEN 11
#define MAG 0
#define MAG2 1
#define MAG4 2
#define ENE 3
#define ENE2 4
#define VORT0 5
#define VORT1 6
#define VORT2 7
#define VORT3 8
#define JSTAR 9
#define KSTAR 10

random_device rd;
mt19937 gen(rd());     // Mersenne Twister RNG
uniform_real_distribution<double> ran_u(0.0, 1.0);

class Metropolis {
    int L;
    int SIZE;
    double SIZEd;
    double* state;
    int** neighs;
    int*** plaqs;
    double ENERGY = 0.0;
    double m[DATALEN];
    string fname;
    ofstream spinfile;

    inline int index_to_n(int, int, int);
    inline double bond_energy(double, double);
    inline double penergy(int, double);
    void metro_step(double, int, double, double);
    void flip(double, double, double);
    void neighbours();
    void total_vorticity();
    double magnetization();
    void get_energy(double, double);
    int pbc(int);
    void test();

public:
    Metropolis(int, string);
    ~Metropolis();
    void simulate_parallel(double*, double, double, int, double*, double);
};

Metropolis::Metropolis(int L, string pref) {
    this->L = L;
    SIZE = L * L * L;
    SIZEd = (double) SIZE;
    fname = pref + "_commented_mod_xy3d_n" + to_string(L) + ".txt";

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
		
		// The only ordering relevant here is that the middle index should be the diagonal term
		plaqs[n][0][0] = index_to_n(b, j, k); plaqs[n][0][1] = index_to_n(b, u, k); plaqs[n][0][2] = index_to_n(i, u, k);
		plaqs[n][1][0] = index_to_n(a, j, k); plaqs[n][1][1] = index_to_n(a, u, k); plaqs[n][1][2] = index_to_n(i, u, k);
		plaqs[n][2][0] = index_to_n(a, j, k); plaqs[n][2][1] = index_to_n(a, d, k); plaqs[n][2][2] = index_to_n(i, d, k);
		plaqs[n][3][0] = index_to_n(i, d, k); plaqs[n][3][1] = index_to_n(b, d, k); plaqs[n][3][2] = index_to_n(b, j, k);

		plaqs[n][4][0] = index_to_n(i, u, k); plaqs[n][4][1] = index_to_n(i, u, r); plaqs[n][4][2] = index_to_n(i, j, r);
		plaqs[n][5][0] = index_to_n(i, j, r); plaqs[n][5][1] = index_to_n(i, d, r); plaqs[n][5][2] = index_to_n(i, d, k);
		plaqs[n][6][0] = index_to_n(i, d, k); plaqs[n][6][1] = index_to_n(i, d, l); plaqs[n][6][2] = index_to_n(i, j, l);
		plaqs[n][7][0] = index_to_n(i, j, l); plaqs[n][7][1] = index_to_n(i, u, l); plaqs[n][7][2] = index_to_n(i, u, k);

		plaqs[n][8][0] = index_to_n(a, j, k); plaqs[n][8][1] = index_to_n(a, j, r); plaqs[n][8][2] = index_to_n(i, j, r);
		plaqs[n][9][0] = index_to_n(i, j, r); plaqs[n][9][1] = index_to_n(b, j, r); plaqs[n][9][2] = index_to_n(b, j, k);
		plaqs[n][10][0] = index_to_n(b, j, k); plaqs[n][10][1] = index_to_n(b, j, l); plaqs[n][10][2] = index_to_n(i, j, l);
		plaqs[n][11][0] = index_to_n(i, j, l); plaqs[n][11][1] = index_to_n(a, j, l); plaqs[n][11][2] = index_to_n(a, j, k);
	    }
	}
    }
}

void Metropolis::simulate_parallel(double* Jstars, double Kstar, double t, int N, double* sub_outvec, double n_per_proc) {
  get_energy(Jstars[0], Kstar);
  for (int i=0; i < n_per_proc; ++i) {
      metro_step(t, N, Jstars[i], Kstar);
      for (int j=0; j < DATALEN; ++j)
        sub_outvec[i * DATALEN + j] = m[j];
  }
}

void Metropolis::metro_step(double t, int N, double Jstar, double Kstar) {
    double sum, chi, heat;

    for (int i=0; i < DATALEN; i++)
    	m[i] = 0.0;

    for (int i=0; i < SIZE * 1000; i++)
    	flip(t, Jstar, Kstar);

    for (int i=0; i < N; i++) {
    	for (int j=0; j < SIZE; j++)
    	    flip(t, Jstar, Kstar);

	sum = magnetization();
	chi = sum * sum;
	heat = ENERGY * ENERGY;
	m[MAG] += sum;        // Magnetization
	m[MAG2] += chi;       // Susceptibility
	m[MAG4] += chi * chi; // Binder
	m[ENE] += ENERGY;     // Energy
	m[ENE2] += heat;      // Specific heat
	total_vorticity();

	/* for (int i=0; i < L; i++) { */
	/*     for (int j=0; j < L; j++) { */
	/* 	for (int k=0; k < L; k++) { */
	/* 	    spinfile << state[index_to_n(i, j, k)] << " "; */
	/* 	} */
	/* 	spinfile << "\n"; */
	/*     } */
	/*     spinfile << "\n"; */
	/* } */
	/* spinfile << "\n"; */
    }

    for (int i=0; i < DATALEN; i++)
	m[i] /= (1.0 * N);

    // Don't average over these guys!
    m[JSTAR] = Jstar;
    m[KSTAR] = Kstar;
    return;
}

inline double Metropolis::bond_energy(double angle1, double angle2) {
    return -1.0 * cos(angle1 - angle2);
}

inline double Metropolis::penergy(int n, double central_angle) {
    double prod;
    double energy = 0.0;

    for (int p=0; p < 12; p++) {
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

inline double constrain(double alpha) {
    // this guy makes sure our angles are in [0, 2pi]
    double x = fmod(alpha, 2 * M_PI);
    return x > 0 ? x : x += 2 * M_PI;
}

void Metropolis::flip(double t, double Jstar, double Kstar) {
    // It's not great to have this here, but we can't make it global because we need SIZE
    uniform_int_distribution<int> ran_pos(0, SIZE-1);
    int index = ran_pos(gen);
    double flip_axis = ran_u(gen) * M_PI;
    double old_angle = state[index];
    double new_angle = constrain(2.0 * flip_axis - old_angle);
    double E1 = 0.0;
    double E2 = 0.0;

    for (int i=0; i < 6; ++i) {
	E1 += bond_energy(state[neighs[index][i]], old_angle) * Jstar;
	E2 += bond_energy(state[neighs[index][i]], new_angle) * Jstar;
    }

    E1 += penergy(index, old_angle) * Kstar;
    E2 += penergy(index, new_angle) * Kstar;

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

void Metropolis::get_energy(double Jstar, double Kstar) {
    for (int i=0; i < SIZE; i++) {
	for (int j=0; j < 6; j++)
	    ENERGY += bond_energy(state[neighs[i][j]], state[i]) * Jstar;
	ENERGY += penergy(i, state[i]) * Kstar;
    }
    ENERGY = ENERGY / SIZEd / 3.0;
}

inline int Metropolis::pbc(int n) {
    return n >= 0 ? n % L : L + (n % L);
}

inline double another_constrain(double x) {
    // for the 100th time!! this guy is different than the other guy!!
    // provided angles (not the input angles, which can be in [-2pi, 2pi], but the two angles you're subtracting)
    // begin in [0, 2pi], this guy puts them between [-pi, pi]
    if (x < -M_PI)
        return another_constrain(x + 2 * M_PI);
    if (x > M_PI)
        return another_constrain(x - 2 * M_PI);
    return x;
}

void Metropolis::total_vorticity() {
    double t1, t2, tmp, delta;
    const int len = 13;

    for (int x=0; x < L; ++x) {
	for (int y=0; y < L; ++y) {
	    for (int z=0; z < L; ++z) {
		// Note that here we only consider vortices in the xy plane. This should be fine.
	        int arr[len][3] = {{x, y, z}, {x + 1, y, z}, {x + 2, y, z}, {x + 3, y},
				   {x + 3, y - 1, z}, {x + 3, y - 2, z}, {x + 3, y - 3},
				   {x + 2, y - 3, z}, {x + 1, y - 3, z}, {x, y - 3},
				   {x, y - 2, z}, {x, y - 1, z}, {x, y, z}};
		
		delta = 0.0;
		for (int i=1; i < len; ++i) {
		    t1 = state[index_to_n(pbc(arr[i - 1][0]), pbc(arr[i - 1][1]), pbc(arr[i - 1][2]))];
		    t2 = state[index_to_n(pbc(arr[i][0]), pbc(arr[i][1]), pbc(arr[i][2]))];
		    tmp = another_constrain(t2 - t1);
		    
		    /* // Check if spins are too far apart to constitute a vortex */
		    /* if (abs(tmp) > M_PI / 2.0) { */
		    /*     i = len; */
		    /* 	delta = 0.0; */
		    /* } */
		    /* else  */
		    delta += tmp;
		}
		
		// Allow for +/-1 or +/-2 vortices
		if (abs(delta + 4 * M_PI) < 0.1)
		    m[VORT0] += 1;
		else if (abs(delta + 2 * M_PI) < 0.1)
		    m[VORT1] += 1;
		else if (abs(delta - 2 * M_PI) < 0.1)
		    m[VORT2] += 1;
		else if (abs(delta - 4 * M_PI) < 0.1)
		    m[VORT3] += 1;
	    }
	}
    }
}

void mpi_run(double Kstar, double Jmin, double Jmax, int npts, int size, double N, string pref) {
    MPI_Init(NULL, NULL);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    cout << "MPI world size: " << world_size << endl;
    
    // 'shared' has 32 cores per node and 'general' has 64--ensure number of test
    // couplings is a multiple of 32 for rc use!
    double* couplings = NULL;
    if (world_rank == 0) {
        assert (npts % 32 == 0);
	double spacing = (Jmax - Jmin) / npts;
	couplings  = (double*) malloc(sizeof(double) * npts);
	for (int i=0; i < npts; ++i)
	    couplings[i] = Jmin + spacing * i;
	
	// This should now also be true
	assert (npts % world_size == 0);
    }

    // First, make an array containing the sets of parameters we want
    double* sub_couplings = (double *) malloc(sizeof(double) * npts / world_size);
    assert(sub_couplings != NULL);
    // Now we can push the couplings out to the different nodes!
    MPI_Scatter(couplings, npts / world_size, MPI_DOUBLE, sub_couplings,
		npts / world_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Make a vector for storing the (flattened) 2d array
    double* out_vec = NULL;
    if (world_rank == 0) {
        out_vec = (double*) malloc(sizeof(double) * npts * DATALEN);
	assert(out_vec != NULL);
    }

    // Now we tell the subprocesses what to do
    double* sub_outvec = (double*) malloc(sizeof(double) * npts / world_size * DATALEN);
    Metropolis metropolis(size, "test");
    metropolis.simulate_parallel(sub_couplings, Kstar, 1.0, N, sub_outvec, npts / world_size);
    
    // Finally, gather the flattened arrays
    int displs[world_size];
    int counts[world_size];
    for (int i=0; i < world_size; ++i) {
        counts[i] = DATALEN * npts / world_size;
	displs[i] = counts[i] * i;
    }
    MPI_Gatherv(sub_outvec, DATALEN * npts / world_size, MPI_DOUBLE, out_vec,
		counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD);

    // Let's check what gets put into out_vec
    
    MPI_Finalize();
    if (world_rank == 0) {  
        cout << "Checking outputted array..." << endl;
	for (int i=0; i < npts; i++) {
	    for (int j=0; j < DATALEN; j++)
	      cout << out_vec[i * DATALEN + j] << " ";
	    cout << "\n" << endl;
	}
    }

    // Now, write everything from root
    if (world_rank == 0) {
        string fname = pref + "parallel_mod_xy3d_n" + to_string(size) + ".txt";
	ofstream output;
	output.open(fname);
	cout << "Writing to file..." << endl;
	for (int i=0; i < npts; i++) {
	    for (int j=0; j < DATALEN; j++) {
	      output << out_vec[i * DATALEN + j] << " ";
	    }
	    output << "\n" << endl;
	    cout << "Finished writing!" << endl;
	}
	output.close();
    }
    
    // Cleanup
    free(couplings);
    free(sub_couplings);
    free(sub_outvec);
    if (world_rank == 0)
        free(out_vec);
}
