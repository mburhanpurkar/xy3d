#include <iostream>
#include <cmath>
#define _USE_MATH_DEFINES

using namespace std;

int L = 2;
double state[4] = {0.1, M_PI - 0.1, M_PI + 0.1, -0.1};
int plaqs[4][4][3];

inline double constrain(double alpha) {
    double x = fmod(alpha, 2 * M_PI);
    return x > 0 ? x : x += 2 * M_PI;
}

inline int index_to_n(int i, int j) {
    return i * L + j;
}

inline double p_energy(int n, double central_angle) {
    double prod;
    double energy = 0.0;

    for (int p=0; p < 4; p++) {
	// Select each of the four elements
	prod = 1.0;
	prod *= cos((central_angle - state[plaqs[n][p][0]]) / 2.0);
	prod *= cos((state[plaqs[n][p][0]] - state[plaqs[n][p][1]]) / 2.0);
	prod *= cos((state[plaqs[n][p][1]] - state[plaqs[n][p][2]]) / 2.0);
	prod *= cos((state[plaqs[n][p][2]] - central_angle) / 2.0);
	energy += prod;
    }
    return -1.0 * energy;
}

void neighbours() {
    int u,d,r,l,n;
    
    for (int i=0; i < L; ++i) {
	for (int j=0; j < L; ++j) {
	    // Periodic boundary
	    u = j + 1 == L  ? 0     : j + 1;
	    d = j - 1 == -1 ? L - 1 : j - 1;
	    r = i + 1 == L  ? 0     : i + 1;
	    l = i - 1 == -1 ? L - 1 : i - 1;
	    
	    n = index_to_n(j,  i);
	    // Add in plaquettes!
	    plaqs[n][0][0] = index_to_n(j, r); plaqs[n][0][1] = index_to_n(u, r); plaqs[n][0][2] = index_to_n(u, i);
	    plaqs[n][1][0] = index_to_n(u, i); plaqs[n][1][1] = index_to_n(u, l); plaqs[n][1][2] = index_to_n(j, l);
	    plaqs[n][2][0] = index_to_n(j, l); plaqs[n][2][1] = index_to_n(d, l); plaqs[n][2][2] = index_to_n(d, i);
	    plaqs[n][3][0] = index_to_n(d, i); plaqs[n][3][1] = index_to_n(d, r); plaqs[n][3][2] = index_to_n(j, r);
	}
    }
}

int main(void) {
    double pi_2[4] = {0.001, M_PI / 2.0 + 0.001, M_PI + 0.001, 3 * M_PI / 2.0 + 0.001};
    double pi_4[4] = {0.1, M_PI + 0.1, 0.1, M_PI + 0.1};

    double prod = 1.0;
    for (int i=0; i < 4; i++) {
	prod *= cos((pi_2[(i + 1) % 4] - pi_2[i % 4]) / 2);
	cout << cos((pi_2[(i + 1) % 4] - pi_2[i % 4]) / 2) << " ";
    }
    cout << "\n";
    cout << "2 pi Prod " << -1.0 * prod << endl;
    
    prod = 1.0;
    for (int i=0; i < 4; i++) {
	prod *= cos((pi_4[(i + 1) % 4] - pi_4[i % 4]) / 2);
	cout << cos((pi_4[(i + 1) % 4] - pi_4[i % 4]) / 2) << " ";
    }
    cout << "\n";
    cout << "4 pi Prod " << -1.0 * prod << endl;
    cout << "*****" << endl;
    
    // That looked pretty clearly fine... Now test two actual ground states...
    neighbours();
    double energy = 0.0;
    for (int i=0; i < 4; i++) {
    	energy += p_energy(i, state[i]);
    }
    cout << "4 pi Energy " << energy << endl;

    // Do it again for a 2pi one
    state[0] = 0.0;
    state[1] = M_PI / 2.0 - 0.0;
    state[3] = M_PI + 0.0;
    state[2] = 3 * M_PI / 2.0 - 0.0;

    energy = 0.0;
    for (int i=0; i < 4; i++) {
    	energy += p_energy(i, state[i]);
    }
    cout << "2 pi Energy " << energy << endl;

    // // Do it again on a perfectly aligned one
    for (int i=0; i < 4; i++)
    	state[i] = i % 2 == 1 ? 0.1 : -0.1;
    energy = 0.0;
    for (int i=0; i < 4; i++) {
    	energy += p_energy(i, state[i]);
    }
    cout << "Aligned Energy " << energy << endl;

}
