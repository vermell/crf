#ifndef FEATUREFACTORY_H_INCLUDED /* any name uniquely mapped to file name */
#define FEATUREFACTORY_H_INCLUDED

#include <vector>
#include <iostream>
#include <map>
#include <string>
#include <iterator>

namespace pgm {

	typedef std::map<int,int> FeatureVector;
	

	class FeatureFactory {
		int featureNumber;
	public:
		FeatureFactory(int featureNumber): featureNumber(featureNumber){};

		FeatureVector generateFeatureVector(std::vector<int>& rawFeatures){
			//std::cout << rawFeatures << std::endl;
			/* Print path vector to console */
			std::copy(rawFeatures.begin(), rawFeatures.end(), std::ostream_iterator<int>(std::cout, " "));
			std::cout << std::endl;
			
			FeatureVector f;
			for (std::size_t i = 0; i != rawFeatures.size(); ++i) {
				if(rawFeatures[i] == 1) {
					f[i] = 1;
					std::cout << "F:" << rawFeatures[i] << std::endl;
				}
			}

			return f;
		};
	};
}
#endif
