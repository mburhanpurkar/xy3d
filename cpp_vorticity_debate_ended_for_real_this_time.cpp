#include <iostream>
#include <cmath>
#define _USE_MATH_DEFINES
using namespace std;

// Okay, now I just need to reproduce the python result in C++ and then we're goldennnn
// Because C++ sucks, I'm just going to copy and paste the relevant arrays here
double charge_1[4][4] = { { 3.9269908169872414 ,  3.4033920413889436 ,  2.879793265790645 ,  2.356194490192346 },
			  { 4.4505895925855405 ,  3.9269908169872414 ,  2.356194490192345 ,  1.8325957145940475 },
			  { 4.97418836818384 ,  5.497787143782138 ,  0.7853981633974483 ,  1.3089969389957485 },
			  { 5.497787143782139 ,  6.021385919380438 ,  0.2617993877991509 ,  0.7853981633974497 } };
    
double charge_n1[4][4] = { { 3.9269908169872414 ,  4.4505895925855405 ,  4.97418836818384 ,  5.497787143782139 },
			   { 3.4033920413889427 ,  3.9269908169872414 ,  5.497787143782138 ,  6.021385919380438 },
			   { 2.879793265790644 ,  2.356194490192345 ,  0.7853981633974483 ,  0.26179938779915024 },
			   { 2.3561944901923453 ,  1.8325957145940466 ,  1.3089969389957479 ,  0.7853981633974491 } };

double charge_2[4][4] = { { 3.9269908169872414 ,  2.8797932657906418 ,  1.8325957145940441 ,  0.7853981633974465 },
			  { 4.974188368183839 ,  3.9269908169872414 ,  0.7853981633974483 ,  6.021385919380435 },
			  { 6.021385919380436 ,  0.7853981633974483 ,  3.9269908169872414 ,  4.974188368183838 },
			  { 0.7853981633974474 ,  1.832595714594045 ,  2.8797932657906427 ,  3.9269908169872405 } };

double charge_n2[4][4] = { { 3.9269908169872414 ,  4.974188368183841 ,  6.021385919380438 ,  0.7853981633974496 },
			   { 2.879793265790644 ,  3.9269908169872414 ,  0.7853981633974483 ,  1.8325957145940472 },
			   { 1.8325957145940464 ,  0.7853981633974483 ,  3.9269908169872414 ,  2.879793265790645 },
			   { 0.7853981633974487 ,  6.021385919380437 ,  4.97418836818384 ,  3.9269908169872423 } };

int outer_loop[13][2] = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {3, 1}, {3, 2}, {3, 3}, {2, 3}, {1, 3}, {0, 3}, {0, 2}, {0, 1}, {0, 0}};


inline double another_constrain(double x) {
    if (x < -M_PI)
	return x + 2 * M_PI;
    if (x > M_PI)
	return x - 2 * M_PI;
    return x;
}

double vorticity(double arr[4][4]) {
    double delta = 0.0;

    for (int i=0; i < 12; i++)
	delta += another_constrain(arr[outer_loop[i + 1][0]][outer_loop[i + 1][1]] - arr[outer_loop[i][0]][outer_loop[i][1]]);

    return delta / 2.0 / M_PI;
}


int main(void) {
    cout << vorticity(charge_1) << endl;
    cout << vorticity(charge_n1) << endl;
    cout << vorticity(charge_2) << endl;
    cout << vorticity(charge_n2) << endl;
}