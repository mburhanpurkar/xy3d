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
#define DATALEN 7
#define MAG 0
#define MAG2 1
#define MAG4 2
#define ENE 3
#define ENE2 4
#define FLUX 5
#define FLUX2 6

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
    tuple<int,int>*** plaqs;
    double ENERGY = 0.0;
    string fname;
    double flux_energy_change;
    double flux_energy;

    inline int index_to_n(int, int, int);
    void lattice_info();
    void metro_step(int, double, double);
    inline double constrain(double);
    inline double senergy(int, int, int, double);
    inline double penergy(int, int);
    double flip_energy_change(double, double, int, int, double, double);
    void flip(double, double, int);
    void get_energy(double, double);
    inline int pbc(int);
    void winding(string);

public:
    Metropolis(int, string);
    ~Metropolis();
    void simulate(int, double, double);
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

    for(int i=0; i < SIZE; ++i) {
        plaqs[i] = new tuple<int,int>*[3];
        dual[i] = new int[3];
        for (int j=0; j < 3; ++j)
            plaqs[i][j] = new tuple<int,int>[4];
    }

    // Fill
    for (int i=0; i < SIZE; ++i) {
        state[i] = ran_u(gen) * 2 * M_PI;
        for (int j=0; j < 3; ++j)
            dual[i][j] = 1;
    }

    // Instead, let's initialize to have a winding number of 1:
    for (int k=0; k < L; ++k) {
        for (int j=0; j < L; ++j) {
            for (int i=0; i < L; ++i) state[index_to_n(i, j, k)] = M_PI * 2.0 / L * i;
        }
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
            }
        }
    }
}

void Metropolis::simulate(int N, double Jstar, double Kstar) {
    ofstream output;
    cout << "Writing data to " << fname << endl;
    get_energy(Jstar, Kstar);
    metro_step(N, Jstar, Kstar);
}

void Metropolis::metro_step(int N, double Jstar, double Kstar) {
    for (int i=0; i < SIZE * 200; i++) flip(Jstar, Kstar, i);

    for (int i=0; i < N; i++) {
        cout << i << endl;
        for (int j=0; j < SIZE * 30; j++) flip(Jstar, Kstar, j);
    }
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
    // If i is 1/10 of an MC step, we should output the winding number
    if (i % (SIZE / 10) == 0) winding("winding_" + fname);

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
    }
}

void Metropolis::get_energy(double Jstar, double Kstar) {
    double bond_energy = 0.0;
    double p_energy = 0.0;
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

    }
    flux_energy = - Kstar * p_energy;
    ENERGY = (-Jstar * bond_energy - Kstar * p_energy);
}

inline int Metropolis::pbc(int n) {
    return n >= 0 ? n % L : L + (n % L);
}

inline double another_constrain(double x) {
    // Everyone's favourite is back
    if (x <= -M_PI) return another_constrain(x + 2 * M_PI);
    if (x > M_PI) return another_constrain(x - 2 * M_PI);
    return x;
}

void Metropolis::winding(string fname) {
    ofstream output;
    output.open(fname);
    // Given the current state, compute the winding number across each row:
    for (int k=0; k < L; ++k) {
        for (int j=0; j < L; ++j) {
            double dtheta = 0.0;
            for (int i=0; i < L; ++i) dtheta += another_constrain(state[index_to_n(i, j, k)] - state[index_to_n(pbc(i + 1), j, k)]);
            // Output the value for the line:
            output << dtheta << " ";
        }
    }
    output << "\n";
    output.close();
}

int main() {
	double Jstar = 2.0;
    double Kstar = 0.0;
	std::stringstream ss;
	ss << "J=" << std::setprecision(2) << Jstar << "_" << "K=" << std::setprecision(2) << Kstar << "_";
	std::string mystring = ss.str();
    Metropolis metropolis(10, mystring);
	metropolis.simulate(5e3, Jstar, Kstar);
}
