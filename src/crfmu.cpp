// foo.cpp
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "graphicalmodel.hpp"
#include<boost/python/module.hpp>
#include<boost/python/def.hpp>
#include<boost/python/extract.hpp>
#include "infer.hpp"

using namespace boost::python;
namespace np = boost::python::numpy;
// real params
//pgm::UnaryParameter* paramU = pgm::UnaryParameter::getInstance();

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

class GraphicalModelWrapper {

 public:

	GraphicalModelWrapper(int featureSize, int labelSize)
		: featureSize(featureSize), labelSize(labelSize) {
		pgm::UnaryParameter::getInstance()->theta = std::vector<double>(featureSize*labelSize, -1.0);

		
		//pgm::UnaryParameter::getInstance()->theta = {-1.0,-1.0,-1.0,0.0,0.0,0.0};

		for(int i = 0; i < (featureSize * labelSize); i++){
			pgm::UnaryParameter::getInstance()-> theta[i] = fRand(-1.0, 0.0);
		}
		
		gm = new pgm::GraphicalModel(featureSize, labelSize);
	}

	void addX(int id, np::ndarray values){
		int n = values.shape(0);
		std::vector<int> f;

		for(int i = 0; i < n; i++){
			f.push_back(extract<int>(values[i]));
		}

		gm->addX(id, f);
	}

	void addY(int id, int value){
		gm->addY(id, value);
	}

	void addUnary(int idx, int idy){
		gm->addUnary(idx, idy);
	}

	void learnModel(){
		gm->learnModel();
	}

	void printInfo(){
		gm->printInfo();
	}

	void infer() {
		inferMQPBO(*gm);
	}

 private:
	int featureSize;
	int labelSize;
	pgm::GraphicalModel* gm;
	
};

int testFunction(np::ndarray values){
	int n = values.shape(0);
	for(int i = 0; i < n; i++){
		double t = extract<double>(values[i]);
		std::cout << t << std::endl;
	}
	return -13;
}
//BOOST_PYTHON_MODULE(foo) {
//  using namespace boost::python;

//  class_<A>("A", init<int>())
//      .def("get_i", &A::get_i, "This is the docstring for A::get_i")
//      ;
//}

BOOST_PYTHON_MODULE(crf) {
	np::initialize();
  class_<GraphicalModelWrapper>("GraphicalModel", init<int,int>())
      .def("addX", &GraphicalModelWrapper::addX, "Add an X-Variable to the model. Note: Id should be uniqe.")
	  .def("print_info", &GraphicalModelWrapper::printInfo, "Print some statistics from Model.")
      .def("addY", &GraphicalModelWrapper::addY, "Add Y-Variable to the model.")
	  .def("addUnary", &GraphicalModelWrapper::addUnary, "Add potential between var x_idx and y_idy")
	  .def("learnModel", &GraphicalModelWrapper::learnModel, "Learn Parameters for model.")
	  .def("infer", &GraphicalModelWrapper::infer, "Infer Model with Parameters.");
  def("test", testFunction, "This is a test.");
}
