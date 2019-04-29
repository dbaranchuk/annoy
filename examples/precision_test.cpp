/*
 * precision_test.cpp

 *
 *  Created on: Jul 13, 2016
 *      Author: Claudio Sanhueza
 *      Contact: csanhuezalobos@gmail.com
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include "../src/kissrandom.h"
#include "../src/annoylib.h"
#include <chrono>
#include <algorithm>
#include <map>
#include <random>


int precision(const int f=128, const int n=100000){
	std::chrono::high_resolution_clock::time_point t_start, t_end;
	size_t n_trees = 2;
	size_t search_k = 80;

	//******************************************************
	//Building the tree
	AnnoyIndex<int, float, Euclidean, Kiss32Random> t = AnnoyIndex<int, float, Euclidean, Kiss32Random>(f);

	std::cout << "Building index ... be patient !!" << std::endl;
	std::cout << "\"Trees that are slow to grow bear the best fruit\" (Moliere)" << std::endl;
	{
		float vec[f];
		std::ifstream base_input("../../rl_hnsw/notebooks/data/SIFT100K/sift_base.fvecs", std::ios::binary);
		uint32_t dim = 0;
		for (size_t i = 0; i < n; i++) {
			base_input.read((char *) &dim, sizeof(uint32_t));
			if (dim != f) {
				std::cout << "file error\n";
				exit(1);
			}
			base_input.read((char *) vec, dim * sizeof(float));
			t.add_item(i, vec);
			std::cout << "Loading objects ...\t object: " << i + 1 << "\tProgress:" << std::fixed
					  << std::setprecision(2) << (double) i / (double) (n + 1) * 100 << "%\r";
		}
	}
	std::cout << std::endl;
	std::cout << "Building index num_trees = ...";
	t_start = std::chrono::high_resolution_clock::now();
	t.build(n_trees);
	t_end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>( t_end - t_start ).count();
	std::cout << " Done in "<< duration << " secs." << std::endl;


	std::cout << "Saving index ...";
	t.save("precision.tree");
	std::cout << " Done" << std::endl;


	//******************************************************
	int K=1;
	int prec_n = 10000;

	std::vector<int> closest;

	float query[f*prec_n];
	{
		std::cout << " Load queries...\n";
		std::ifstream query_input("../../rl_hnsw/notebooks/data/SIFT100K/sift_query.fvecs", std::ios::binary);
		for (size_t i = 0; i < prec_n; i++){
			int dim = 0;
			query_input.read((char *) &dim, sizeof(uint32_t));
			if (dim != f) {
				std::cout << "file error\n";
				exit(1);
			}
			query_input.read((char *) (query + dim*i), dim * sizeof(float));
		}
	}
	uint32_t gt[prec_n];
	{
		std::cout << " Load groundtruths...\n";
		std::ifstream gt_input("../../rl_hnsw/notebooks/data/SIFT100K/test_gt.ivecs", std::ios::binary);
        uint32_t dim = 0;
		for (size_t i = 0; i < prec_n; i++){
			gt_input.read((char *) &dim, sizeof(uint32_t));
			if (dim != K) {
				std::cout << "file error\n";
				exit(1);
			}
			gt_input.read((char *) (gt + dim*i), dim * sizeof(uint32_t));
		}
	}

	// doing the work
	t_start = std::chrono::high_resolution_clock::now();
	int correct = 0;
	for(int i=0; i<prec_n; ++i){
		// getting the K closest
		t.get_nns_by_vector(query + f*i, K, search_k, &closest, nullptr);
		correct += (int) (closest[i] == gt[i]);
	}
	t_end = std::chrono::high_resolution_clock::now();
	std::cout << "Recall@1: " <<  correct << " Time: " << std::chrono::duration_cast<std::chrono::microseconds>( t_end - t_start ).count() / (float) prec_n << std::endl;
	std::cout << "\nDone" << std::endl;
	return 0;
}


void help(){
	std::cout << "Annoy Precision C++ example" << std::endl;
	std::cout << "Usage:" << std::endl;
	std::cout << "(default)		./precision" << std::endl;
	std::cout << "(using parameters)	./precision num_features num_nodes" << std::endl;
	std::cout << std::endl;
}

void feedback(int f, int n){
	std::cout<<"Runing precision example with:" << std::endl;
	std::cout<<"num. features: "<< f << std::endl;
	std::cout<<"num. nodes: "<< n << std::endl;
	std::cout << std::endl;
}


int main(int argc, char **argv) {
	int f, n;


	if(argc == 1){
		f = 128;
		n = 100000;

		feedback(f,n);

		precision(f, n);
	}
	else if(argc == 3){

		f = atoi(argv[1]);
		n = atoi(argv[2]);

		feedback(f,n);

		precision(f, n);
	}
	else {
		help();
		return EXIT_FAILURE;
	}


	return EXIT_SUCCESS;
}
