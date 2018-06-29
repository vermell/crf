#ifndef GRAPHICALMODEL_H_INCLUDED /* any name uniquely mapped to file name */
#define GRAPHICALMODEL_H_INCLUDED
#include <vector>
#include "factor.hpp"
#include <Eigen/Core>
namespace pgm {
	class GraphicalModel{
		std::vector<NodePotential> unaries;
		int featureSize;
		int labelSize;
		
	public:
		Parameter unaryParameter;
		Parameter pairwiseParameter;
		GraphicalModel(int featureSize, int labelSize, Parameter& unaryParameter, Parameter& pairwiseParameter);
		void addNodePotential(NodePotential unary);
		void printInfo();
		
		double logLikelihood();

		std::vector<double> gradientLogLikelihood();
		
		void learnModel();
	};
}
#endif
