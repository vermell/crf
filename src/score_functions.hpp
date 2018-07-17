#ifndef SCOREFUNCTIONS_H_INCLUDED /* any name uniquely mapped to file name */
#define SCOREFUNCTIONS_H_INCLUDED

#include <vector>
#include <iostream>
#include "factor.hpp"
#include <numeric>
#include <boost/geometry/arithmetic/dot_product.hpp>

namespace pgm {
	inline double weight_product(FeatureVector fv, int variablevalue, std::vector<double> params, int stepsize){
		std::vector<double> r;
		double score = 0.0;
		for(auto& f: fv){
			int index = variablevalue * stepsize + f.first;
			score += params[index];
		}

		return score;
	};
}
#endif
