#pragma once
#include "datatype.h"
#include <string>

extern const int DIM;
extern const double dataset_rho;
extern const double dataset_dmin;
extern const std::string dataset_name;

std::vector<double> read_point();
