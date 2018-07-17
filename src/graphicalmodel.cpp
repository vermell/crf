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
#include "infer.hpp"
#include <ctime>
#include <omp.h>


pgm::Parameter unariesP {-1.0,-1.0,-1.0,0.0,0.0,0.0};
pgm::Parameter pairwiseP;
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
	int iter = 0;
	pgm::GraphicalModel model;
public:
    LogLikelihood(int n_, pgm::GraphicalModel model_) : n(n_), model(model_) {}
    double operator()(const Eigen::VectorXd& x, Eigen::VectorXd& grad)
    {

		iter++;
		// Start the timer
		std::clock_t begin = std::clock();
		
		
		std::vector<double> grad_k = model.gradientLogLikelihood();
		for(int i = 0; i < n; i++){
			if(!(grad_k[i] != grad_k[i]) && !(x[i] != x[i])){
				paramU->theta[i] = x[i];
				grad[i] = grad_k[i];
			}
			std::cout << "Gradient: " << grad[i] << std::endl;
		}

		
	    double fx = model.logLikelihood();
		
		std::clock_t end = std::clock();
		double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

		std::cout << "Round " << iter << " -- Elapsed secs " << elapsed_secs << std::endl;
        return fx;
    }

	
};

std::vector<double> pgm::GraphicalModel::gradientLogLikelihood(){
	std::vector<double> v;

	for(int h=0; h < featureSize; h++)
	for(int k=0; k < labelSize; k++){
		
		long double sum = 0.0L;
		for(auto& factor: unaries){
			double s = 0.0;
			for(int y = 0; y< labelSize; y++){
				
				long double d  = factor.conditionalProbabilityWithEvidence(y)*factor.getFeatureFunction(h*labelSize + y);
				if(!(d !=d))
					sum += d;
			}
		}

		double regularization = 1.0 * paramU->theta[h*labelSize + k];
		v.push_back(featureStatistics[k + labelSize*h] - sum - regularization);
	
    }
	
	return v;
}

void pgm::GraphicalModel::generateStatistics(){

	for(int h=0; h < featureSize; h++)
		for(int k=0; k < labelSize; k++){

			long double count = 0.0L;
			for(auto& factor: unaries) {
				count += factor.getFeatureFunction(k+labelSize*h);
			}
			
			featureStatistics.push_back(count);
		}
	
}


pgm::GraphicalModel::GraphicalModel(int featureSize, int labelSize, Parameter& unaryParameter, Parameter& pairwiseParameter):
	featureSize(featureSize), labelSize(labelSize), unaryParameter(unaryParameter), pairwiseParameter(pairwiseParameter)  {
	featureDim = pgm::FeatureDimension(featureSize);
	
	yDim = pgm::DiscreteDimension(labelSize);
	std::cout << "Creating Graphical Model." << std::endl;
}

pgm::GraphicalModel::GraphicalModel(int featureSize, int labelSize):
	featureSize(featureSize), labelSize(labelSize) {

	featureDim = pgm::FeatureDimension(featureSize);
	
	yDim = pgm::DiscreteDimension(labelSize);

	std::cout << "Creating Graphical Model." << std::endl;
	
 
}

void pgm::GraphicalModel::addX(int idx, std::vector<int> values){
	
    pgm::FeatureFactory featureFactory(featureSize);

	pgm::FeatureVector fv = featureFactory.generateFeatureVector(values);

	pgm::FeatureVariable x(idx, fv, featureDim);
	xVariables.push_back(x);

	//std::cout << "Added Variable " << idx << std::endl;
}

void pgm::GraphicalModel::addY(int idy, int value){
	//std::cout << value << std::endl;
	pgm::DiscreteVariable y(idy, value, yDim);
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
	
	
	if(x && y) {
		addNodePotential(pgm::NodePotential(*x,*y));
	}
	
}

void pgm::GraphicalModel::addNodePotential(pgm::NodePotential unary) {
		unaries.push_back(unary);
}

double pgm::GraphicalModel::logLikelihood() {
    double lambda = 1.0d;
	double sum = 0.0d;


#pragma omp parallel for reduction(-: sum)
	for (auto it = 0; it < this->unaries.size(); it++){
    
     	double d = this->unaries[it].logConditionalProbability();

			if(!(d !=d))
				sum -= d;
    
	}
	

	double rSum = 0.0d;
	for(auto& p: paramU->theta){
		double add = pow(p,2);

		//std::cout << "b " << add << std::endl;
		if (!(add != add))
			rSum += add;
	}
	double regularization = lambda*0.5 * rSum;
	//std::cout << "Regularization-Term: " << regularization << "rsum: " << rSum  << std::endl;
	//std::cout << sum << std::endl;
	return sum + regularization;
}
	

void pgm::GraphicalModel::learnModel(){
	generateStatistics();

	const int n = labelSize * featureSize;
    // Set up parameters
	LBFGSpp::LBFGSParam<double> param;

	
	param.epsilon = 1e-2;
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
	pgm::DiscreteDimension boolDim(2);
	pgm::FeatureDimension featureDim(3);


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
	
	paramU->theta = {-1.0,-1.0,-1.0,-1.0,-1.0,-1.0};
	
	
	pgm::GraphicalModel model (3,2);//, unariesP, pairwiseP);
	//std::cout << model.logLiklihood() << std::endl;

    for(int i=0; i<1000; i++){
		
		//model.addNodePotential(getFactorFrom({0,1,1}, 0));
		
		model.addNodePotential(getFactorFrom({1,0,1}, 0));
		
		model.addNodePotential(getFactorFrom({0,1,0}, 1));
		model.addNodePotential(getFactorFrom({0,1,0}, 1));
	}

	model.printInfo();

	double t = model.logLikelihood();
	std::cout << "SCORE1: " << t << std::endl;

	
    double t2 = model.logLikelihood();
	std::cout << "SCORE2: " << t2 << std::endl;

	model.learnModel();
	//model.gradientLogLikelihood();

	//paramU->theta[0]= -0.3;
	pgm::NodePotential pot = getFactorFrom({1,0,1}, 0);
	std::cout << pot.conditionalProbability() << std::endl;

	pgm::GraphicalModel model2 (3,2, unariesP, pairwiseP);

	
	inferMQPBO(model);
	
}
