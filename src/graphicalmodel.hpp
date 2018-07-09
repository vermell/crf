#ifndef GRAPHICALMODEL_H_INCLUDED /* any name uniquely mapped to file name */
#define GRAPHICALMODEL_H_INCLUDED
#include <vector>
#include "factor.cpp"
#include <Eigen/Core>


namespace pgm {
	class GraphicalModel{
		
		DiscreteDimension yDim = 0;
		FeatureDimension featureDim = 0;
	public:

		std::vector<NodePotential> unaries;
		
		int featureSize;
		int labelSize;

		std::vector<FeatureVariable> xVariables;
		std::vector<DiscreteVariable> yVariables;
		
		Parameter unaryParameter;
		Parameter pairwiseParameter;

		GraphicalModel(int featureSize, int labelSize, Parameter& unaryParameter, Parameter& pairwiseParameter);

		GraphicalModel(int featureSize, int labelSize);
		
		void addNodePotential(NodePotential unary);

		void printInfo();
		
		double logLikelihood();

		std::vector<double> gradientLogLikelihood();
		
		void learnModel();

		void addX(int idx, std::vector<int> values);
		void addY(int idy, int value);
		void addUnary(int idx, int idy);
		void addPairwise();
	};
}
#endif
