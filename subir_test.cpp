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

    inline int index_to_n(int, int, int);
    void lattice_info();
    inline double constrain(double);
    inline double senergy(int, int, int, double);
    inline double penergy(int, int);
    double flip_energy_change(double, double, int, int, double, double);
    void flip(double, double, int);
    void get_energy(double, double);
    double get_partial_energy(double, double, int);
    inline int pbc(int);
    inline int cvt(int);

public:
    Metropolis(int, bool);
    ~Metropolis();
    double simulate(int, double, double);
};

Metropolis::Metropolis(int L, bool vison) {
    this->L = L;
    SIZE = L * L * L;
    SIZEd = (double) SIZE;

    // Construct arrays
    state = new double[SIZE];
    dual = new int*[SIZE];
    plaqs = new tuple<int,int>**[SIZE];

    for (int i=0; i < SIZE; ++i) {
        plaqs[i] = new tuple<int,int>*[3];
        dual[i] = new int[3];
        for (int j=0; j < 3; ++j)
            plaqs[i][j] = new tuple<int,int>[4];
    }

    // Initialize spins and sigmas randomly
    for (int i=0; i < SIZE; i++) {
        state[i] = 2 * M_PI;
        for (int j=0; j < 3; j++) dual[i][j] = cvt(ran_bond(gen));
    }

    if (vison) {
        // This isn't great, but good enough for now--edge will contain an ordered
        // list of coordinates in the xy plane that will contain the twist
        tuple<int, int> edge[4 * L];

        // Start in the bottom left corner:
        int x = 0;
        int y = 0;
        edge[0] = make_tuple(x, y);

        for (int i=0; i < L - 1; i++) {
            x += 1;
            edge[i + 1] = make_tuple(x, y);
        }
        for (int i=0; i < L - 1; i++) {
            y += 1;
            edge[L + i + 1] = make_tuple(x, y);
        }
        for (int i=0; i < L - 1; i++) {
            x -= 1;
            edge[2 * L + i + 1] = make_tuple(x, y);
        }
        for (int i=0; i < L - 2; i++) {
            y -= 1;
            edge[3 * L + i + 1] = make_tuple(x, y);
        }

        // Initialize the spins
        for (int z=0; z < L; z++) {
            for (int i=0; i < 4 * L; i++) {
                state[index_to_n(get<0>(edge[i]), get<1>(edge[i]), z)] = 2 * M_PI / (4.0 * L) * i;
                // // If we're at the last spin, put a -1:
                // if (i == 4 * L - 1) dual[index_to_n(get<0>(edge[i]), get<1>(edge[i]), z)][1] = -1;
            }
        }
    }

    // Initialize plaquettes
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

inline int Metropolis::cvt(int i) {
    // So that we can use random integers in [0, 1] for initialization
    return i == 1 ? i : -1;
}

void Metropolis::lattice_info() {
    for (int i=0; i < L; ++i) {
        for (int j=0; j < L; ++j) {
            for (int k=0; k < L; ++k) {
                // Bookkeeping convention: each site has three associated plaquettes
                // 0 -- which spin we're considering
                // 1 -- which plaquette we're considering (side, back, top)
                // 2 -- which of the four sites we're on (no particular order)
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

double Metropolis::simulate(int N, double Jstar, double Kstar) {
    double cumulative_energy = 0.0;

    // Thermalize
    // get_energy(Jstar, Kstar); DONT CALL THIS--TEST
    for (int i=0; i < SIZE * 7000; i++) flip(Jstar, Kstar, i);

    // Then, compute the total energy after each step
    for (int i=0; i < SIZE * N; i++) {
        flip(Jstar, Kstar, i);
        if (i % SIZE == 0) {
            // If we use ENERGY, it will include sigma flips on the border
            cumulative_energy += get_partial_energy(Jstar, Kstar, 2);
        }
    }
    return cumulative_energy / N;
}

double Metropolis::get_partial_energy(double Jstar, double Kstar, int boundary) {
    // This is the same as get_energy() but we only consider the interior spins,
    // specified by boundary
    double bond_energy = 0.0;
    double p_energy = 0.0;
    for (int z=0; z < L; z++) {
        for (int x=boundary; x < L - boundary; ++x) {
            for (int y=boundary; y < L - boundary; ++y) {
                int idx = index_to_n(x, y, z);
                // Get plaquette energy
                for (int j=0; j < 3; ++j) {
                    double prod = 1.0;
                    for (int k=0; k < 4; k++) {
                        tuple<int,int> tup = plaqs[idx][j][k];
                        prod *= dual[get<0>(tup)][get<1>(tup)];
                    }
                    p_energy += prod;
                }
                // Now get energy from neighbours--each spin has three
                bond_energy += cos((state[idx] - state[index_to_n(pbc(x + 1), y, z)]) / 2.0) * dual[idx][0];
                bond_energy += cos((state[idx] - state[index_to_n(x, pbc(y + 1), z)]) / 2.0) * dual[idx][1];
                bond_energy += cos((state[idx] - state[index_to_n(x, y, pbc(z + 1))]) / 2.0) * dual[idx][2];
            }
        }
    }
    return -Jstar * bond_energy - Kstar * p_energy;
}

inline double Metropolis::constrain(double alpha) {
    // alpha -> alpha in [0, 2pi)
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
    // Specify a spin number and a plaquette index, return the energy
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

    // The second term has a + sign here
    return - Jstar * (newe - olde) + 2 * Kstar * p_energy;
}

void Metropolis::flip(double Jstar, double Kstar, int i) {
    // Here we can pick any index to flip BUT if we choose one on the boundary,
    // we don't let the spin change, but we do allow the sigma to change
    uniform_int_distribution<int> ran_idx(0, L-1); // this shouldn't really be here, but it's okay
    int x = ran_idx(gen);
    int y = ran_idx(gen);
    int z = ran_idx(gen);
    int index = index_to_n(x, y, z);
    int bond = (int) (ran_u(gen) * 3.0);
    double old_angle = state[index];
    double new_angle;
    if (x != 0 && y != 0 && x != L - 1 && y != L - 1) {
        double flip_axis = ran_u(gen) * M_PI;
        new_angle = constrain(2.0 * flip_axis - old_angle);
    }
    else new_angle = old_angle;
    double deltaE = flip_energy_change(Jstar, Kstar, index, bond, old_angle, new_angle);

    if (ran_u(gen) < (deltaE < 0 ? 1.0 : exp(-(deltaE)))) {
        state[index] = new_angle;
        dual[index][bond] = -dual[index][bond];
        ENERGY += deltaE;
    }
}

void Metropolis::get_energy(double Jstar, double Kstar) {
    // Get the energy of the whole system (no need to modify since the initial offset is irrelevant)
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
    ENERGY = (-Jstar * bond_energy - Kstar * p_energy);
}

inline int Metropolis::pbc(int n) {
    // % in C++ returns negative values--this function gives us periodic boundaries
    // without negative values!
    return n >= 0 ? n % L : L + (n % L);
}

inline double another_constrain(double x) {
    // x -> x in [-pi, pi]
    if (x <= -M_PI) return another_constrain(x + 2 * M_PI);
    if (x > M_PI) return another_constrain(x - 2 * M_PI);
    return x;
}

int main() {
    double res[10];
    for (int i=0; i < 10; i++) {
        Metropolis metropolis(20, true);
        Metropolis metropolis2(20, false);
        res[i] = metropolis.simulate(20000, 0.1, (double) i / 2.0) / metropolis2.simulate(20000, 0.1, (double) i / 2.0);
        cout << "K=" << (double) i / 2.0 << "\t\t" << res[i] << endl;
    }
}
