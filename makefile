all:
	g++ -O3 -o xy_model xy_model.cpp
	g++ -O3 -o mod_xy_model -fsanitize=address -ggdb3 mod_xy_model.cpp
