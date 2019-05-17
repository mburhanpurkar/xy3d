.PHONY: all test clean

all:
	g++ -O3 -o xy_model xy_model.cpp
	g++ -O3 -o mod_xy_model -fsanitize=address -ggdb3 mod_xy_model.cpp
	g++ -O3 -o xy2d -std=c++11 -fsanitize=address -ggdb3 xy2d.cpp
test:
	g++ -O3 -o test -std=c++11 -fsanitize=address -ggdb3 test.cpp
