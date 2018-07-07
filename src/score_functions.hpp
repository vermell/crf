#ifndef SCOREFUNCTIONS_H_INCLUDED /* any name uniquely mapped to file name */
#define SCOREFUNCTIONS_H_INCLUDED

#include <vector>
#include <iostream>
#include "factor.hpp"
#include <numeric>

namespace pgm {
	inline double weight_product(FeatureVector fv, int variablevalue, std::vector<double> params, int stepsize){
		std::vector<double> r;
		for(auto& f: fv){
			int index = variablevalue * stepsize + f.first;
			//std::cout << "index = " << index << " - v: " << variablevalue << std::endl; 
			double score = params[index];
			r.push_back(score);
		}

		// sum up features
		double sum = 0.0;
		for(auto& n : r)
			sum += n;
		return sum;
	};
}
#endif
