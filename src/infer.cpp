#include <iostream>
#include "GCoptimization.h"

void GeneralGraph_DArraySArraySpatVarying(int width,int height,int num_pixels,int num_labels)
{
	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*num_labels];
	for ( int i = 0; i < num_pixels; i++ )
		for (int l = 0; l < num_labels; l++ )
			if (i < 25 ){
				if(  l == 0 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
			else {
				if(  l == 2 ) data[i*num_labels+l] = 0;
				else data[i*num_labels+l] = 10;
			}
	// next set up the array for smooth costs
	int *smooth = new int[num_labels*num_labels];
	for ( int l1 = 0; l1 < num_labels; l1++ )
		for (int l2 = 0; l2 < num_labels; l2++ )
			smooth[l1+l2*num_labels] = (l1-l2)*(l1-l2) <= 4  ? (l1-l2)*(l1-l2):4;


	try{
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(num_pixels,num_labels);
		gc->setDataCost(data);
		gc->setSmoothCost(smooth);

		// now set up a grid neighborhood system
		// first set up horizontal neighbors
		for (int y = 0; y < height; y++ )
			for (int  x = 1; x < width; x++ ){
				int p1 = x-1+y*width;
				int p2 =x+y*width;
				gc->setNeighbors(p1,p2,p1+p2);
			}

		// next set up vertical neighbors
		for (int y = 1; y < height; y++ )
			for (int  x = 0; x < width; x++ ){
				int p1 = x+(y-1)*width;
				int p2 =x+y*width;
				gc->setNeighbors(p1,p2,p1*p2);
			}

		
		for ( int  i = 0; i < num_pixels; i++ ){
			result[i] = gc->whatLabel(i);
			std::cout << i << " : " << result[i] << std::endl;
		}
		
		printf("\nBefore optimization energy is %d",gc->compute_energy());
		gc->expansion(2);// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		printf("\nAfter optimization energy is %d",gc->compute_energy());

		for ( int  i = 0; i < num_pixels; i++ ){
			result[i] = gc->whatLabel(i);
			std::cout << i << " : " << result[i] << std::endl;
		}

		delete gc;
	}
	catch (GCException e){
		e.Report();
	}

	delete [] result;
	delete [] smooth;
	delete [] data;


}


int main(){

	int width = 10;
	int height = 10;
	int num_pixels = width*height;
	int num_labels = 5;

	std::cout << "Inference Test" << std::endl;
		// which actually is a grid. Also uses spatially varying terms
	GeneralGraph_DArraySArraySpatVarying(width,height,num_pixels,num_labels);

}
