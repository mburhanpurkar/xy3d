.PHONY: all test clean

test:
	g++ -O3 -o test2d -std=c++11 -fsanitize=address -ggdb3 test2d.cpp
	g++ -O3 -o test3d -std=c++11 -fsanitize=address -ggdb3 test3d.cpp
	g++ -O3 -o testmod3d -std=c++11 -fsanitize=address -ggdb3 testmod3d.cpp
