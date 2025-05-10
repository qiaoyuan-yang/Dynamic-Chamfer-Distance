#pragma once
#include "datatype.h"
#include <string>

// extern const int DIM;
extern const double dataset_rho;
extern const double dataset_dmin;
extern const double dataset_dmax;
extern const int dataset_window_size;
inline constexpr std::size_t DIM = 300;  // C++17 起可用 inline；老标准写成：constexpr std::size_t DIM = 300;
