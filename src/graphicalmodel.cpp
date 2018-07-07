#include <iostream>

#include "graphicalmodel.hpp"
#include "factor.hpp"
#include "featurefactory.hpp"
//#include "score_functions.hpp"
#include <math.h>
#include <algorithm>
#include <Eigen/Core>
#include <LBFGS.h>
#include <vector>

pgm::Parameter unariesP {-1.0,-1.0,-1.0,0.0,0.0,0.0};
pgm::Parameter pairwiseP;
pgm::DiscreteDimension boolDim(2);
pgm::FeatureDimension featureDim(3);
int featureSize = 3;
int labelSize = 2;



// real params
pgm::UnaryParameter* paramU = pgm::UnaryParameter::getInstance();



void printParameter(){
	std::cout << "Unaries: [ ";
	for(auto& p: paramU->theta){
		std::cout << p << " ";
	}
	std::cout << "]" << std::endl;
					   
}

class LogLikelihood
{
private:
    int n;
	pgm::GraphicalModel model;
public:
    LogLikelihood(int n_, pgm::GraphicalModel model_) : n(n_), model(model_) {}
    double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad)
    {

		std::vector<double> grad_k = model.gradientLogLikelihood();
		for(int i = 0; i < n; i++){
			//std::cout << x[i] << std::endl;
			//unariesP[i] = x[i];
			//model.unaryParameter[i] = x[i];
			paramU->theta[i] = -x[i];
			grad[i] = grad_k[i];
			std::cout << "Gradient: " << grad[i] << std::endl;
		}

		
		//paramU->theta[1] = -12;
		//printParameter();
	    double fx = model.logLikelihood();
			//model.logLikelihood();
	  
        return fx;
    }

	
};

std::vector<double> pgm::GraphicalModel::gradientLogLikelihood(){
	std::vector<double> v;

	for(int k=0; k < paramU->theta.size(); k++){
		long double count = 0.0L;
		for(auto& factor: unaries){
			count += factor.getFeatureFunction(k);
		}

		long double sum = 0.0L;
		for(auto& factor: unaries){
			double s = 0.0;
			for(int y = 0; y< labelSize; y++){
				s += factor.conditionalProbabilityWithEvidence(y)*factor.getFeatureFunction(k);
				std::cout << factor.conditionalProbabilityWithEvidence(y) << std::endl;
				sum += factor.conditionalProbabilityWithEvidence(y)*factor.getFeatureFunction(k);
			}
			//std::cout << "s = " << s << std::endl;
		}

		std::cout << k << " = " <<count <<" :: " << sum << std::endl;
		double regularization = 1.0 * paramU->theta[k];
		v.push_back(count - sum);
	
    }
	
	return v;
}


pgm::GraphicalModel::GraphicalModel(int featureSize, int labelSize, Parameter& unaryParameter, Parameter& pairwiseParameter):
	featureSize(featureSize), labelSize(labelSize), unaryParameter(unaryParameter), pairwiseParameter(pairwiseParameter)  {
	std::cout << "Creating Graphical Model." << std::endl;
}

pgm::GraphicalModel::GraphicalModel(int featureSize, int labelSize):
	featureSize(featureSize), labelSize(labelSize) {
	std::cout << "Creating Graphical Model." << std::endl;
	
 
}

//void addX(int idx, double value[]);
//		void addY(int idy, int value);
//		void addUnary(int idx, int idy);

void pgm::GraphicalModel::addX(int idx, std::vector<int> values){
	
    pgm::FeatureFactory featureFactory(featureSize);

	pgm::FeatureVector fv = featureFactory.generateFeatureVector(values);

	pgm::FeatureVariable x(idx, fv, featureDim);
	xVariables.push_back(x);

	std::cout << "Added Variable " << idx << std::endl;
}

void pgm::GraphicalModel::addY(int idy, int value){
	pgm::DiscreteVariable y(idy, value, boolDim);
	yVariables.push_back(y);
}

void pgm::GraphicalModel::addUnary(int idx, int idy){
	// check if id's exists

	FeatureVariable* x = nullptr;
	for(auto& xi: xVariables){
		if(xi.getId() == idx)
			x = &xi;
	}

	DiscreteVariable* y = nullptr;
	for(auto& yi: yVariables){
		if(yi.getId() == idy)
			y = &yi;
	}
	

	//auto y = std::find(yVariables.begin(), yVariables.end(), [&](DiscreteVariable & o) {
	//																o.getId() == idy;
	//});
	

	
	if(x && y) {
		std::cout<<"Add Potential (" << idx << "," << idy << ")" << std::endl;
		addNodePotential(pgm::NodePotential(*x,*y));
	}
	
}

void pgm::GraphicalModel::addNodePotential(pgm::NodePotential unary) {
	//	if(std::find(unaries.begin(), unaries.end(), [&unary](const Type& obj) {return obj.getName() == myString.get;}) != unaries.end()){
		unaries.push_back(unary);
		//	}
		//else {
		//	throw std::invalid_argument("Factor already exists in Graph.");
		//	}
}

double pgm::GraphicalModel::logLikelihood() {
    double lambda = 1.0d;
	double sum = 0.0;
	for(auto& factor: this->unaries)
		sum -= factor.logConditionalProbability();


	double rSum = 0.0d;
	for(auto& p: paramU->theta){
		rSum += pow(p,2);
	}
	double regularization = lambda*0.5 * rSum;
	std::cout << "Regularization-Term: " << regularization << std::endl;
	
	return sum + regularization;
}
	

void pgm::GraphicalModel::learnModel(){
	const int n = 6;
    // Set up parameters
	LBFGSpp::LBFGSParam<double> param;

	
	param.epsilon = 1e-50;
    param.max_iterations = 100;

    // Create solver and function object
	LBFGSpp::LBFGSSolver<double> solver(param);
    

    // Initial guess
	Eigen::VectorXd x = Eigen::VectorXd::Ones(n);
    // x will be overwritten to be the best point found
    double fx;

	LogLikelihood fun(n, *this);
	
    int niter = solver.minimize(fun, x, fx);

	printParameter();
    std::cout << niter << " iterations" << std::endl;
    std::cout << "x = \n" << x.transpose() << std::endl;
    std::cout << "f(x) = " << fx << std::endl;
   
}

void pgm::GraphicalModel::printInfo(){
	std::cout << "Model contains " << unaries.size() << " factors." << std::endl;
	
	std::cout << "Model contains " << xVariables.size() << " X-Variables." << std::endl;

	std::cout << "Model contains " << yVariables.size() << " Y-Variables." << std::endl;
}

// be carefull -> global vars

int currId = -1;
pgm::NodePotential getFactorFrom(std::vector<int> featurevector, int yValue) {
	currId += 1;
	pgm::FeatureFactory featureFactory(featureSize);

	pgm::FeatureVector fv = featureFactory.generateFeatureVector(featurevector);

	pgm::DiscreteVariable y(currId,yValue,boolDim);
	std::cout << y.getId() << std::endl;

	currId +=1;
	pgm::FeatureVariable x(currId, fv, featureDim);

	pgm::printFeatureVector(x.getValue());

	return pgm::NodePotential(x,y);
}



int main(){
	
	paramU->theta = {-1.0,-1.0,-1.0,0.0,0.0,0.0};
	
	
	pgm::GraphicalModel model (3,2, unariesP, pairwiseP);
	//std::cout << model.logLiklihood() << std::endl;

    for(int i=0; i<320; i++){
		
		//model.addNodePotential(getFactorFrom({0,1,1}, 0));
		
		model.addNodePotential(getFactorFrom({1,1,1}, 1));
		
		model.addNodePotential(getFactorFrom({0,1,1}, 0));
		model.addNodePotential(getFactorFrom({0,1,1}, 0));
		//model.addNodePotential(getFactorFrom({0,1,1}, 0));
		
		//model.addNodePotential(getFactorFrom({0,1,1}, 0));
		
		
		//model.addNodePotential(getFactorFrom({0,0,1}, 0));
		
		//model.addNodePotential(getFactorFrom({1,0,1}, 1));
		//model.addNodePotential(getFactorFrom({1,1,0}, 1));
	}

	model.printInfo();

	//double logLikelihood = model.logLikelihood();

	//std::cout << "L(v) = " << exp(logLikelihood) << std::endl;

	//model.learnModel();
	//std::cout << "SCORE: " << model.logLikelihood() << std::endl;

	//model.unaryParameter[1] = -12;
	double t = model.logLikelihood();
	std::cout << "SCORE1" << t << std::endl;

	
    double t2 = model.logLikelihood();
	std::cout << "SCORE2" << t2 << std::endl;

	model.learnModel();
	//model.gradientLogLikelihood();

	//paramU->theta[0]= -0.3;
	pgm::NodePotential pot = getFactorFrom({1,1,1}, 1);
	std::cout << pot.conditionalProbability() << std::endl;
}
