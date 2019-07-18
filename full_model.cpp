#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <tuple>
#define _USE_MATH_DEFINES

using namespace std;

// Output array
#define DATALEN 11
#define MAG 0
#define MAG2 1
#define MAG4 2
#define ENE 3
#define ENE2 4
#define FLUX 5
#define FLUX2 6

#define NEIGH1E 7
#define NEIGH1E2 8
#define NEIGH2E 9
#define NEIGH2E2 10

random_device rd;
mt19937 gen(rd());     // Mersenne Twister RNG
uniform_real_distribution<double> ran_u(0.0, 1.0);
uniform_int_distribution<int> ran_bond(0, 1);

class Metropolis {
    int L;
    int SIZE;
    double SIZEd;
    double* state;
    int** dual;
    int*** spin_plaqs;
    tuple<int,int>*** plaqs;
    double ENERGY = 0.0;
    double m[DATALEN];
    string fname;
    double flux_energy_change;
    double flux_energy;
    double neigh1e;
    double neigh2e;

    inline int cvt(int);
    inline int index_to_n(int, int, int);
    void lattice_info();
    void metro_step(int, double, double);
    inline double constrain(double);
    inline double senergy(int, int, int, double);
    inline double penergy(int, int);
    // inline double neigh1efunc(int, int, int, double);
    // inline double neigh2efunc(int, int, int, double);
    inline double neigh1efunc(int, double);
    inline double neigh2efunc(int, double);
    double flip_energy_change(double, double, int, int, double, double);
    void flip(double, double, int);
    double magnetization();
    void get_energy(double, double);
    inline int pbc(int);

public:
    Metropolis(int, string);
    ~Metropolis();
    void simulate(int, double, double, double, double);
};

Metropolis::Metropolis(int L, string pref) {
    this->L = L;
    SIZE = L * L * L;
    SIZEd = (double) SIZE;
    fname = pref + "full_xy3d_n" + to_string(L) + ".txt";

    // Construct arrays
    state = new double[SIZE];
    dual = new int*[SIZE];
    plaqs = new tuple<int,int>**[SIZE];
    spin_plaqs = new int**[SIZE];

    for(int i=0; i < SIZE; ++i) {
        plaqs[i] = new tuple<int,int>*[3];
        dual[i] = new int[3];
        spin_plaqs[i] = new int*[12];
        for (int j=0; j < 3; ++j)
            plaqs[i][j] = new tuple<int,int>[4];
        for (int j=0; j < 12; ++j)
            spin_plaqs[i][j] = new int[4];
    }

    // Fill
    for (int i=0; i < SIZE; ++i) {
        state[i] = ran_u(gen) * 2 * M_PI;
        for (int j=0; j < 3; ++j)
            dual[i][j] = cvt(ran_bond(gen));
    }
    lattice_info();
}

Metropolis::~Metropolis() {
    delete[] state;
    for(int i=0; i < SIZE; ++i) {
        delete[] dual[i];
        for (int j=0; j < 3; ++j) {
            delete[] plaqs[i][j];
        }
        delete[] plaqs[i];
    }
    delete[] plaqs;
}

inline int Metropolis::cvt(int i) {
    // So that we can use random integers in [0, 1] for initialization
    return i == 1 ? i : -1;
}

inline int Metropolis::index_to_n(int i, int j, int k) {
    return i * L * L + j * L + k;
}

void Metropolis::lattice_info() {
    for (int i=0; i < L; ++i) {
        for (int j=0; j < L; ++j) {
            for (int k=0; k < L; ++k) {
                // plaqs tells us how to index the dual lattice
                // 0 -- which spin we're considering
                // 1 -- which plaquette we're considering (side, back, top)
                // 2 -- which of the four sides we're on (no particular order)
                // At 2, we have a tuple, containing (spin, bond_idx)
                // bond_idx can be sigma_x, sigma_y, or sigma_z
                int n = index_to_n(i, j, k);
                plaqs[n][0][0] = make_tuple(n, 0);
                plaqs[n][0][1] = make_tuple(n, 2);
                plaqs[n][0][2] = make_tuple(index_to_n(i, j, pbc(k + 1)), 0);
                plaqs[n][0][3] = make_tuple(index_to_n(pbc(i + 1), j, k), 2);
                plaqs[n][1][0] = make_tuple(n, 1);
                plaqs[n][1][1] = make_tuple(n, 2);
                plaqs[n][1][2] = make_tuple(index_to_n(i, j, pbc(k + 1)), 1);
                plaqs[n][1][3] = make_tuple(index_to_n(i, pbc(j + 1), k), 2);
                plaqs[n][2][0] = make_tuple(n, 0);
                plaqs[n][2][1] = make_tuple(n, 1);
                plaqs[n][2][2] = make_tuple(index_to_n(i, pbc(j + 1), k), 0);
                plaqs[n][2][3] = make_tuple(index_to_n(pbc(i + 1), j, k), 1);

                // Let's code in the spin plaquettes too--XY plane first
                spin_plaqs[n][0][0] = n;
                spin_plaqs[n][0][1] = index_to_n(pbc(i + 1), j, k);
                spin_plaqs[n][0][2] = index_to_n(pbc(i + 1), pbc(j + 1), k);
                spin_plaqs[n][0][3] = index_to_n(i, pbc(j + 1), k);
                spin_plaqs[n][1][0] = n;
                spin_plaqs[n][1][1] = index_to_n(i, pbc(j + 1), k);
                spin_plaqs[n][1][2] = index_to_n(pbc(i - 1), pbc(j + 1), k);
                spin_plaqs[n][1][3] = index_to_n(pbc(i - 1), j, k);
                spin_plaqs[n][2][0] = n;
                spin_plaqs[n][2][1] = index_to_n(pbc(i - 1), j, k);
                spin_plaqs[n][2][2] = index_to_n(pbc(i - 1), pbc(j - 1), k);
                spin_plaqs[n][2][3] = index_to_n(i, pbc(j - 1), k);
                spin_plaqs[n][3][0] = n;
                spin_plaqs[n][3][1] = index_to_n(i, pbc(j - 1), k);
                spin_plaqs[n][3][2] = index_to_n(pbc(i + 1), pbc(j - 1), k);
                spin_plaqs[n][3][3] = index_to_n(pbc(i + 1), j, k);

                // YZ plane next
                spin_plaqs[n][4][0] = n;
                spin_plaqs[n][4][1] = index_to_n(i, j, pbc(k + 1));
                spin_plaqs[n][4][2] = index_to_n(i, pbc(j + 1), pbc(k + 1));
                spin_plaqs[n][4][3] = index_to_n(i, pbc(j + 1), k);
                spin_plaqs[n][5][0] = n;
                spin_plaqs[n][5][1] = index_to_n(i, pbc(j + 1), k);
                spin_plaqs[n][5][2] = index_to_n(i, pbc(j + 1), pbc(k - 1));
                spin_plaqs[n][5][3] = index_to_n(i, j, pbc(k - 1));
                spin_plaqs[n][6][0] = n;
                spin_plaqs[n][6][1] = index_to_n(i, j, pbc(k - 1));
                spin_plaqs[n][6][2] = index_to_n(i, pbc(j - 1), pbc(k - 1));
                spin_plaqs[n][6][3] = index_to_n(i, pbc(j - 1), k);
                spin_plaqs[n][7][0] = n;
                spin_plaqs[n][7][1] = index_to_n(i, pbc(j - 1), k);
                spin_plaqs[n][7][2] = index_to_n(i, pbc(j - 1), pbc(k + 1));
                spin_plaqs[n][7][3] = index_to_n(i, j, pbc(k + 1));

                // Lastly, XZ
                spin_plaqs[n][8][0] = n;
                spin_plaqs[n][8][1] = index_to_n(pbc(i + 1), j, k);
                spin_plaqs[n][8][2] = index_to_n(pbc(i + 1), j, pbc(k + 1));
                spin_plaqs[n][8][3] = index_to_n(i, j, pbc(k + 1));
                spin_plaqs[n][9][0] = n;
                spin_plaqs[n][9][1] = index_to_n(i, j, pbc(k + 1));
                spin_plaqs[n][9][2] = index_to_n(pbc(i - 1), j, pbc(k + 1));
                spin_plaqs[n][9][3] = index_to_n(pbc(i - 1), j, k);
                spin_plaqs[n][10][0] = n;
                spin_plaqs[n][10][1] = index_to_n(pbc(i - 1), j, k);
                spin_plaqs[n][10][2] = index_to_n(pbc(i - 1), j, pbc(k - 1));
                spin_plaqs[n][10][3] = index_to_n(i, j, pbc(k - 1));
                spin_plaqs[n][11][0] = n;
                spin_plaqs[n][11][1] = index_to_n(i, j, pbc(k - 1));
                spin_plaqs[n][11][2] = index_to_n(pbc(i + 1), j, pbc(k - 1));
                spin_plaqs[n][11][3] = index_to_n(pbc(i + 1), j, k);
            }
        }
    }
}

void Metropolis::simulate(int N, double Jstar, double Kmin, double Kmax, double delta) {
    ofstream output;
    cout << "Writing data to " << fname << endl;
    output.open(fname); // Outfile name
    get_energy(Jstar, Kmax);

    for (double Kstar=Kmax; Kstar > Kmin; Kstar -= delta) {
        metro_step(N, Jstar, Kstar);
        output << Jstar << " " << Kstar << " " << m[MAG] << " " << m[ENE] << " "
               << m[MAG2] - m[MAG] * m[MAG] << " " <<  m[ENE2] - m[ENE] * m[ENE] << " "
               << m[FLUX2] - m[FLUX] * m[FLUX] << " " << m[NEIGH1E2] - m[NEIGH1E] * m[NEIGH1E] << " "
	           << m[NEIGH2E2] - m[NEIGH2E] * m[NEIGH2E] << endl;
        cout << Jstar << " " << Kstar << endl;
    }
    output.close();
}

void Metropolis::metro_step(int N, double Jstar, double Kstar) {
    double sum, chi, heat;

    for (int i=0; i < DATALEN; i++) m[i] = 0.0;

    for (int i=0; i < SIZE * 200; i++) flip(Jstar, Kstar, i);

    for (int i=0; i < N; i++) {
        for (int j=0; j < SIZE * 30; j++) flip(Jstar, Kstar, j);

        sum = magnetization();
        chi = sum * sum;
        heat = ENERGY * ENERGY;
        m[MAG] += sum;        // Magnetization
        m[MAG2] += chi;       // Susceptibility
        m[MAG4] += chi * chi; // Binder
        m[ENE] += ENERGY;     // Energy
        m[ENE2] += heat;      // Specific heat
        m[FLUX] += flux_energy;
        m[FLUX2] += flux_energy * flux_energy;
    	m[NEIGH1E] += neigh1e;
    	m[NEIGH1E2] += neigh1e * neigh1e;
    	m[NEIGH2E] += neigh2e;
    	m[NEIGH2E2] += neigh2e * neigh2e;
    }

    for (int i=0; i < DATALEN; i++) m[i] /= (1.0 * N);
}

inline double Metropolis::constrain(double alpha) {
    double x = fmod(alpha, 2 * M_PI);
    return x >= 0 ? x : constrain(x + 2 * M_PI);
}

inline double Metropolis::senergy(int x, int y, int z, double central) {
    return (cos((state[index_to_n(pbc(x - 1), y, z)] - central) / 2.0) * dual[index_to_n(pbc(x - 1), y, z)][0] +
            cos((state[index_to_n(pbc(x + 1), y, z)] - central) / 2.0) * dual[index_to_n(x, y, z)][0] +
            cos((state[index_to_n(x, pbc(y - 1), z)] - central) / 2.0) * dual[index_to_n(x, pbc(y - 1), z)][1] +
            cos((state[index_to_n(x, pbc(y + 1), z)] - central) / 2.0) * dual[index_to_n(x, y, z)][1] +
            cos((state[index_to_n(x, y, pbc(z - 1))] - central) / 2.0) * dual[index_to_n(x, y, pbc(z - 1))][2] +
            cos((state[index_to_n(x, y, pbc(z + 1))] - central) / 2.0) * dual[index_to_n(x, y, z)][2]);
}

// inline double Metropolis::neigh1efunc(int x, int y, int z, double central) {
//     return (cos(state[index_to_n(pbc(x - 1), y, z)] - central) +
//             cos(state[index_to_n(pbc(x + 1), y, z)] - central) +
//             cos(state[index_to_n(x, pbc(y - 1), z)] - central) +
//             cos(state[index_to_n(x, pbc(y + 1), z)] - central) +
//             cos(state[index_to_n(x, y, pbc(z - 1))] - central) +
// 	        cos(state[index_to_n(x, y, pbc(z + 1))] - central));
// }
//
// inline double Metropolis::neigh2efunc(int x, int y, int z, double central) {
//     return (cos((state[index_to_n(pbc(x - 1), y, z)] - central) / 2.0) +
//             cos((state[index_to_n(pbc(x + 1), y, z)] - central) / 2.0) +
//             cos((state[index_to_n(x, pbc(y - 1), z)] - central) / 2.0) +
//             cos((state[index_to_n(x, pbc(y + 1), z)] - central) / 2.0) +
//             cos((state[index_to_n(x, y, pbc(z - 1))] - central) / 2.0) +
//             cos((state[index_to_n(x, y, pbc(z + 1))] - central) / 2.0));
// }

inline double Metropolis::neigh1efunc(int n, double central) {
    double tot = 0.0;
    for (int p=0; p < 12; ++p) {
        double prod = cos(central - spin_plaqs[n][p][1]);
        prod *= cos(spin_plaqs[n][p][1] - spin_plaqs[n][p][2]);
        prod *= cos(spin_plaqs[n][p][2] - spin_plaqs[n][p][3]);
        prod *= cos(spin_plaqs[n][p][3] - central);
        tot += prod;
    }
    return tot;
}

inline double Metropolis::neigh2efunc(int n, double central) {
    int x = n / (L * L);
    int y = (n - x * L * L) / L;
    int z = n - (x * L * L + y * L);
    double tot = 0.0;
    for (int p=0; p < 12; ++p) {
        double prod = cos((central - spin_plaqs[n][p][1]) / 2.0);
        prod *= cos((spin_plaqs[n][p][1] - spin_plaqs[n][p][2]) / 2.0);
        prod *= cos((spin_plaqs[n][p][2] - spin_plaqs[n][p][3]) / 2.0);
        prod *= cos((spin_plaqs[n][p][3] - central) / 2.0);
        tot += prod;
    }
    return tot + (cos((state[index_to_n(pbc(x - 1), y, z)] - central) / 2.0) +
                cos((state[index_to_n(pbc(x + 1), y, z)] - central) / 2.0) +
                cos((state[index_to_n(x, pbc(y - 1), z)] - central) / 2.0) +
                cos((state[index_to_n(x, pbc(y + 1), z)] - central) / 2.0) +
                cos((state[index_to_n(x, y, pbc(z - 1))] - central) / 2.0) +
                cos((state[index_to_n(x, y, pbc(z + 1))] - central) / 2.0));;
}

inline double Metropolis::penergy(int n, int i) {
    // Specify a spin number and a plaquette index, and this boi gets you the energy
    // (up to a factor of -K)
    double energy = 1.0;
    for (int idx=0; idx < 4; ++idx) {
        tuple<int,int> tup = plaqs[n][i][idx];
        energy *= dual[get<0>(tup)][get<1>(tup)];
    }
    return energy;
}

double Metropolis::flip_energy_change(double Jstar, double Kstar, int n, int bond, double oldangle, double newangle) {
    int x = n / (L * L);
    int y = (n - x * L * L) / L;
    int z = n - (x * L * L + y * L);

    // First take care of the plaquette energy--this is nice because a spin flip results in an energy change of 2 * p_energy
    double p_energy;
    switch (bond) {
        case 0:
            p_energy = penergy(n, 2) + penergy(index_to_n(x, pbc(y - 1), z), 2) + penergy(n, 0) + penergy(index_to_n(x, y, pbc(z - 1)), 0);
            break;
        case 1:
            p_energy = penergy(n, 2) + penergy(index_to_n(pbc(x - 1), y, z), 2) + penergy(n, 1) + penergy(index_to_n(x, y, pbc(z - 1)), 1);
            break;
        default:
            p_energy = penergy(n, 1) + penergy(index_to_n(x, pbc(y - 1), z), 1) + penergy(n, 0) + penergy(index_to_n(pbc(x - 1), y, z), 0);
    }

    // Now take care of the neighbour energy--the easiest way to do this is to just flip the relevant spin + bond, compute the
    // energy, flip them back, compute the energy, and add the difference to p_energy
    dual[n][bond] = -dual[n][bond];
    double newe = senergy(x, y, z, newangle);
    dual[n][bond] = -dual[n][bond];
    double olde = senergy(x, y, z, oldangle);

    flux_energy_change = 2 * p_energy * Kstar;
    // It is not a mistake that the second term has a + sign here
    return - Jstar * (newe - olde) + 2 * Kstar * p_energy;
}

void Metropolis::flip(double Jstar, double Kstar, int i) {
    int index = (int) (ran_u(gen) * SIZEd);
    int bond = (int) (ran_u(gen) * 3.0);
    double flip_axis = ran_u(gen) * M_PI;
    double old_angle = state[index];
    double new_angle = constrain(2.0 * flip_axis - old_angle);
    double deltaE = flip_energy_change(Jstar, Kstar, index, bond, old_angle, new_angle);

    if (ran_u(gen) < (deltaE < 0 ? 1.0 : exp(-(deltaE)))) {
        state[index] = new_angle;
        dual[index][bond] = -dual[index][bond];
        ENERGY += deltaE;
        flux_energy += flux_energy_change;

	int x = index / (L * L);
	int y = (index - x * L * L) / L;
	int z = index - (x * L * L + y * L);
	// neigh1e += neigh1efunc(x, y, z, new_angle) - neigh1efunc(x, y, z, old_angle);
	// neigh2e += neigh2efunc(x, y, z, new_angle) - neigh2efunc(x, y, z, old_angle);
    neigh1e += neigh1efunc(index, new_angle) - neigh1efunc(index, old_angle);
	neigh2e += neigh2efunc(index, new_angle) - neigh2efunc(index, old_angle);
    }
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
    double bond_energy = 0.0;
    double p_energy = 0.0;
    // These bois are member variables
    neigh1e = 0.0;
    neigh2e = 0.0;

    for (int i=0; i < SIZE; ++i) {
        // Get plaquette energy
        for (int j=0; j < 3; ++j) {
            double prod = 1.0;
            for (int k=0; k < 4; k++) {
                tuple<int,int> tup = plaqs[i][j][k];
                prod *= dual[get<0>(tup)][get<1>(tup)];
            }
            p_energy += prod;
        }
        // Now get energy from neighbours--each spin has three
        int x = i / (L * L);
        int y = (i - x * L * L) / L;
        int z = i - (x * L * L + y * L);
        bond_energy += cos((state[i] - state[index_to_n(pbc(x + 1), y, z)]) / 2.0) * dual[i][0];
        bond_energy += cos((state[i] - state[index_to_n(x, pbc(y + 1), z)]) / 2.0) * dual[i][1];
        bond_energy += cos((state[i] - state[index_to_n(x, y, pbc(z + 1))]) / 2.0) * dual[i][2];

	    // We can also add neighbour contributions for the two modified energy terms--rip these didn't work
        // neigh1e += cos((state[i] - state[index_to_n(pbc(x + 1), y, z)]));
        // neigh1e += cos((state[i] - state[index_to_n(x, pbc(y + 1), z)]));
        // neigh1e += cos((state[i] - state[index_to_n(x, y, pbc(z + 1))]));
        neigh2e += cos((state[i] - state[index_to_n(pbc(x + 1), y, z)]));
        neigh2e += cos((state[i] - state[index_to_n(x, pbc(y + 1), z)]));
        neigh2e += cos((state[i] - state[index_to_n(x, y, pbc(z + 1))]));

        // We're going to do two of these guys again--one with the one half factor and one without
        for (int i=0; i < SIZE; ++i) {
            for (int p=0; p < 12; ++p) {
                double prod1 = 1.0;
                double prod2 = 1.0;
                for (int j=0; j < 4; ++j) {
                    prod1 *= cos(state[spin_plaqs[i][p][j]] - state[spin_plaqs[i][p][(j + 1) % 4]]);
                    prod2 *= cos((state[spin_plaqs[i][p][j]] - state[spin_plaqs[i][p][(j + 1) % 4]]) / 2.0);
                }
                neigh2e += prod1;
                neigh2e += prod2;
            }
        }

    }
    flux_energy = - Kstar * p_energy;
    ENERGY = (-Jstar * bond_energy - Kstar * p_energy);
}

inline int Metropolis::pbc(int n) {
    return n >= 0 ? n % L : L + (n % L);
}

int main() {
    // simulate(int N, double J, double Kmin, double Kmax, double delta)
    // for (double K=30; K < 51; K += 10) {
	double Jstar = 0.0;
	std::stringstream ss;
	ss << "J=" << std::setprecision(2) << Jstar << "_";
	std::string mystring = ss.str();
    Metropolis metropolis(10, mystring);
	metropolis.simulate(3e3, Jstar, 0.49, 1.0, 0.025);
    //}
}
