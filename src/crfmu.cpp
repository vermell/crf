// foo.cpp
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "graphicalmodel.hpp"
#include<boost/python/module.hpp>
#include<boost/python/def.hpp>
#include<boost/python/extract.hpp>

using namespace boost::python;
namespace np = boost::python::numpy;
// real params
//pgm::UnaryParameter* paramU = pgm::UnaryParameter::getInstance();


class GraphicalModelWrapper {

 public:

	GraphicalModelWrapper(int featureSize, int labelSize)
		: featureSize(featureSize), labelSize(labelSize) {
		pgm::UnaryParameter::getInstance()->theta = std::vector<double>(featureSize*labelSize, -0.5);
	
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
	  .def("learnModel", &GraphicalModelWrapper::learnModel, "Learn Parameters for model.");
  def("test", testFunction, "This is a test.");
}
