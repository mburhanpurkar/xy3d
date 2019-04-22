.PHONY: all test clean

all:
	g++ -O3 -o xy_model xy_model.cpp
	g++ -O3 -o mod_xy_model -fsanitize=address -ggdb3 mod_xy_model.cpp
	g++ -O3 -o xy2d -fsanitize=address -ggdb3 xy2d.cpp
test:
	g++ -o test test.cpp
