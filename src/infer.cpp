#include <iostream>
#include <opengm/graphicalmodel/graphicalmodel.hxx>

#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/gibbs.hxx>
#include <opengm/inference/mqpbo.hxx>
#include <opengm/inference/graphcut.hxx> 
#include <opengm/inference/alphaexpansion.hxx> 
#include <opengm/inference/auxiliary/minstcutkolmogorov.hxx> 
#include "graphicalmodel.hpp"

std::vector<size_t> inferMQPBO(pgm::GraphicalModel graphicalModel){

	

	//*******************
   //** Typedefs
   //*******************
   typedef double                                                                 ValueType;          // type used for values
   typedef size_t                                                                 IndexType;          // type used for indexing nodes and factors (default : size_t)
   typedef size_t                                                                 LabelType;          // type used for labels (default : size_t)
   typedef opengm::Adder                                                          OpType;             // operation used to combine terms
   typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType>                ExplicitFunction;   // shortcut for explicite function 
   typedef opengm::PottsFunction<ValueType,IndexType,LabelType>                   PottsFunction;      // shortcut for Potts function
   typedef opengm::meta::TypeListGenerator<ExplicitFunction,PottsFunction>::type  FunctionTypeList;   // list of all function the model cal use (this trick avoids virtual methods) - here only one
   typedef opengm::DiscreteSpace<IndexType, LabelType>                            SpaceType;          // type used to define the feasible statespace
   typedef opengm::GraphicalModel<ValueType,OpType,FunctionTypeList,SpaceType>    Model;              // type of the model
   typedef Model::FunctionIdentifier                                              FunctionIdentifier; // type of the function identifier

   typedef opengm::external::QPBO<Model> QPBO;

   
   //******************
   //** DATA
   //******************
   IndexType N = 6;
   IndexType M = 4;  
   int data[] = { 0, 0, 0, 0, 0, 0,
                  0, 7, 2, 0, 4, 0,
                  6, 9, 8, 8, 9, 9,
                  9, 9, 9, 9, 9, 9 };



   //*******************
   //** Code
   //*******************

   std::cout << "Start building the model ... "<<std::endl;
   // Build empty Model
   LabelType numLabel = graphicalModel.labelSize;
   std::vector<LabelType> numbersOfLabels(graphicalModel.yVariables.size(),numLabel);


   std::cout << graphicalModel.yVariables.size()  << " : " << graphicalModel.labelSize << std::endl;
   Model gm(SpaceType(numbersOfLabels.begin(), numbersOfLabels.end()));

   // Add 1st order functions and factors to the model
   for(IndexType variable = 0; variable < gm.numberOfVariables(); ++variable) {
      // construct 1st order function
      const LabelType shape[] = {gm.numberOfLabels(variable)};
      ExplicitFunction f(shape, shape + 1);
      //f(0) = std::fabs(data[variable] - 2.0);
      //f(1) = std::fabs(data[variable] - 8.0);
	  //f(0) = 1.95;
	  //f(1) = 1.0;
	  //f(2) = 3.0;

	  for(int l = 0; l < numLabel; l++){
		  //		  std::cout << "Id "<< variable << " Score " << l << ": "<< graphicalModel.unaries[variable].scoreWithEvidence(pgm::Evidence({{variable, l}})) << std::endl;
		  int id = graphicalModel.unaries[variable].y.getId();
		  double value  = std::fabs(graphicalModel.unaries[variable].scoreWithEvidence(pgm::Evidence({{id, l}})));

		  std::cout << "Id " << variable << " : " << l << " = " << value << std::endl;
		 
		  f(l) = value;
	  }
	  //f(0) = 0.2888209;
	  //f(1) = 0.2888208;
	  //f(2) = 0.36787924;
      // add function
      FunctionIdentifier id = gm.addFunction(f);
      // add factor
      IndexType variableIndex[] = {variable};
      gm.addFactor(id, variableIndex, variableIndex + 1);
   }
   // add 2nd order functions for all variables neighbored on the grid
   {
      // add a potts function to the model
      PottsFunction potts(numLabel, numLabel, 0.0, 2.0);
      FunctionIdentifier pottsid = gm.addFunction(potts);

      IndexType vars[]  = {0,1}; 
      for(IndexType n=0; n<N;++n){
         for(IndexType m=0; m<M;++m){
            vars[0] = n + m*N;
            if(n+1<N){ //check for right neighbor
               vars[1] =  (n+1) + (m  )*N;
               OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered!
               //gm.addFactor(pottsid, vars, vars + 2);
            } 
            if(m+1<M){ //check for lower neighbor
               vars[1] =  (n  ) + (m+1)*N; 
               OPENGM_ASSERT(vars[0] < vars[1]); // variables need to be ordered!
               //gm.addFactor(pottsid, vars, vars + 2);
            }
         }
      }
   }

   // View some model information
   std::cout << "The model has " << gm.numberOfVariables() << " variables."<<std::endl;
   for(size_t i=0; i<gm.numberOfVariables(); ++i){
      std::cout << " * Variable " << i << " has "<< gm.numberOfLabels(i) << " labels."<<std::endl; 
   } 
   std::cout << "The model has " << gm.numberOfFactors() << " factors."<<std::endl;
   for(size_t f=0; f<gm.numberOfFactors(); ++f){
      std::cout << " * Factor " << f << " has order "<< gm[f].numberOfVariables() << "."<<std::endl; 
   }

   
   /*
   QPBO qpbo(gm);
   qpbo.infer();
   std::cout << "Test" << std::endl;
   std::cout << "value: " << qpbo.value() << std::endl;

   
   
   typedef opengm::external::MinSTCutKolmogorov<size_t, double> MinStCutType;
   typedef opengm::GraphCut<Model, opengm::Minimizer, MinStCutType> MinGraphCut;   

   
   //typedef opengm::MQPBO<Model, opengm::Adder > MQPBO;
   typedef opengm::AlphaExpansion<Model, MinGraphCut> MinAlphaExpansion;

   //MQPBO mqpbo(gm);
   
   MinAlphaExpansion ae(gm);
   ae.infer();
   std::cout << "value: " << ae.value() << std::endl;
   */
   
    std::cout << "  * Minimization/Adder ..." << std::endl;
    typedef opengm::GraphicalModel<double, opengm::Adder> GmType;
    typedef opengm::MQPBO<Model,opengm::Minimizer> MQPBOType;
    MQPBOType::Parameter para;
    para.useKovtunsMethod_=false;
    para.rounds_=1;
    std::cout << "... without probing ..."<<std::endl;
    //adderTester.test<MQPBOType > (para);
    MQPBOType t(gm);
    t.infer();
	
    std::cout << "value: " << t.value() << std::endl;
	std::vector<size_t> labeling(gm.numberOfVariables());
	t.arg(labeling);
	
	
  for(size_t variable = 0; variable < labeling.size(); ++variable) {
	  int id = graphicalModel.yVariables[variable].getId();
		std::cout << "y" << id << "=" << labeling[variable] << "\n";
   }

  return labeling;
}

