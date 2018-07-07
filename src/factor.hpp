#ifndef FACTOR_H_INCLUDED /* any name uniquely mapped to file name */
#define FACTOR_H_INCLUDED

#include <vector>
#include "featurefactory.hpp"
#include "score_functions.hpp"
#include <math.h>
#include <algorithm>

namespace pgm {

	template<class T>
	class Dimension {
		int maxrange;
	public:
		Dimension(int maxrange): maxrange(maxrange){};
		virtual bool isValidValue(T value);

		int size() { return maxrange;};
	};

	
	template<>
	inline bool Dimension<int>::isValidValue(int value) {
		if(0 <= value && value < this->maxrange) return true;

		return false;
	};
	
	template<>
	inline bool Dimension<FeatureVector>::isValidValue(FeatureVector value) {
		//for(auto & v: value){
		//	if(0 > v || value >= maxrange) return false;
		//}
		if(value.size() <= maxrange)
			return true;

		return false;
	}

	typedef Dimension<int> DiscreteDimension;
	typedef Dimension<FeatureVector> FeatureDimension;
    	
	template <class T, class D>
	class Variable {
		T value;
		int id;
		D dimension;
	public:
		Variable(int id, T value, D dimension): id(id), value(value), dimension(dimension){
			if(!dimension.isValidValue(value))
				throw std::invalid_argument("Variable out of range.");
		};
		
		int getId() {return id;};
		T getValue(){return value;};
		D getDimension(){return dimension;};
	};


	typedef Variable<FeatureVector, FeatureDimension> FeatureVariable;
	typedef Variable<int, DiscreteDimension> DiscreteVariable;

	typedef std::map<int,int> Evidence;
	
	struct Factor{
		virtual double score() = 0;
		virtual double scoreWithEvidence(Evidence evidence) = 0;
		//virtual vector<int> getScope() = 0;
		
	};

	typedef std::vector<double> Parameter;

	class UnaryParameter {
    private:
        /* Here will be the instance stored. */
        static UnaryParameter* instance;
		
        /* Private constructor to prevent instancing. */
        UnaryParameter() {
			
		}

    public:
        /* Static access method. */
        static UnaryParameter* getInstance() {
			if (instance == 0)
				{
					instance = new UnaryParameter();
				}

			return instance;

		}

		std::vector<double> theta;
	};

	/* Null, because instance will be initialized on demand. */
    inline pgm::UnaryParameter* UnaryParameter::instance = 0;

	
	class NodePotential: Factor {
		FeatureVariable x;
		DiscreteVariable y;
		UnaryParameter* params = UnaryParameter::getInstance();
		
	public:
		NodePotential(FeatureVariable x, DiscreteVariable y): x(x), y(y) {
			std::cout << "Size: " <<x.getDimension().size() << std::endl;
			std::cout << "Size: " <<y.getDimension().size() << std::endl;
			
			if(params->theta.size() != x.getDimension().size() * y.getDimension().size()){
				std::cout << "Size: " <<params->theta.size() << std::endl;
				throw std::invalid_argument("Parameter-Vector has wrong size.");
			}
		};

		double score() override {
			//std::cout << "Score. " << y.getId() << std::endl;
			double inner_feature_product = weight_product(x.getValue(), y.getValue(), params->theta, x.getDimension().size());
			//std::cout << "Params0 = " << params[0] << " - > "<< inner_feature_product << std::endl;

			
			//double inner_feature_product2 = weight_product(x.getValue(), 0, params, x.getDimension().size());
			//std::cout << "Score = " << exp(inner_feature_product) << " - > " << std::endl;
			return exp(inner_feature_product);
		};

		double scoreWithEvidence(Evidence evidence) override {
			//std::cout << "Score with Evidence:" << std::endl;
			if(evidence.find(y.getId()) == evidence.end()){
				return score();
			}
			else {
				
				int yvalue = evidence[y.getId()];
				// check if valid -> maybe we can skip this
				if(!y.getDimension().isValidValue(yvalue)){
					throw std::invalid_argument("Wrong range of Variable in Score-Function.");
				}
				double inner_feature_product = weight_product(x.getValue(), yvalue, params->theta, x.getDimension().size());
				return exp(inner_feature_product);
			}
		}

		double getFeatureFunction(int id) {
			int id_mod = id % y.getDimension().size();
			if(x.getValue().count(id_mod) > 0){
				return 1.0;
			}
			else return 0.0;
		}

		double conditionalProbabilityWithEvidence(int yvalue) {
			// p(y|x) see paper ~/studium/pgm/loglinear.pdf
			double partitionFunction = 0.0;

			for(int i=0; i<y.getDimension().size(); i++){
				partitionFunction += scoreWithEvidence(pgm::Evidence({{y.getId(), i}}));
			}
			double prob = scoreWithEvidence(pgm::Evidence({{y.getId(), yvalue}})) / partitionFunction;
			return prob;
		}

		double conditionalProbability(){
			// p(y|x) see paper ~/studium/pgm/loglinear.pdf
			double partitionFunction = 0.0;

			for(int i=0; i<y.getDimension().size(); i++){
				partitionFunction += scoreWithEvidence(pgm::Evidence({{y.getId(), i}}));
			}
			double prob = score() / partitionFunction;
			
			return prob;
		}

		double logConditionalProbability(){
			return log(conditionalProbability());
		}

	};


	/* Helper Function */
	inline void printFeatureVector(FeatureVector f) {
		std::cout << "Vector[ ";
		for(auto& v: f){
			std::cout << v.first << " ";
		}
		std::cout << "]" << std::endl;
		
	};
}

#endif
