#pragma once
#include "datatype.h"
#include <vector>

class BruteForceANN {
public:
    BruteForceANN();

    void insert(const point& p);
    point query(const point& p) const;

private:
    std::vector<point> points;
};
