// Modified from REBEL

#include<stdio.h>
#include<stdlib.h>

#include<iostream>
#include<vector>
#include<string>
#include<fstream>
#include<math.h>

#include"Arguments.h"

using namespace std;

#define MAX_COUNT 2*8192
 

// Reads and stores data.
class Data{
public:
	Data(){}
	~Data(){
	}
	
	void init(){
		read_data();
		toArray();
	}
	
	void read_data(){
		int bufferSize = 10000000;
		ifstream ifs(Arguments::datafile, ios::in);
		if (!ifs){
			fprintf(stderr, " Cannot read file %s.\n", Arguments::datafile);
			exit(1);
		}
		fprintf(stderr, " Reading file %s...\n", Arguments::datafile);
		
		char* buffer= new char[bufferSize];
		ifs.getline(buffer, bufferSize);
		char* pch= strtok(buffer," \t");
		// http://www.cplusplus.com/ref/cstring/strtok.html
		while (pch != NULL){
			heads.push_back(atoi(pch));
			pch = strtok(NULL, " \t");
        }
		numattributes = heads.size();
	
		fprintf(stderr, " Heading read: %d columns.\n", numattributes);
	
		dm.clear();
		vector<float> temp;
		while (true){
			temp.clear();
			ifs.getline(buffer, bufferSize);
			pch = strtok(buffer," \t");
			while (pch != NULL){
				temp.push_back((float) atof(pch));
				pch = strtok(NULL, " \t");
			}
			if ((int)temp.size() != numattributes) break;
			
			dm.push_back(temp);
			
			if ((int)dm.size() >= atoi(Arguments::maxnumrecords)) break;
		}
		numrecords = dm.size();
		fprintf(stderr, " Data read: %d lines.\n", numrecords); 
		delete [] buffer;
	}

	void toArray() {
		scores = new float*[numrecords];
		for (int i = 0; i < numrecords; i++) {
			scores[i] = new float[numattributes];
			for (int j = 0; j < numattributes; j ++) {
				scores[i][j] = dm[i][j];
			}
		}
		dm.clear();
		feasible = new unsigned long int[numattributes];
		for (int j = 0; j < numattributes; j ++) {
			feasible[j] = (unsigned long int) heads[j];
		}
		heads.clear();
	}	
	unsigned long int* feasible;
	vector<int> arities;
	int numattributes;
	int numrecords;
	int maxarity;
	float** scores;
private:
	vector< vector<float> > dm;
	vector<int> heads;
};

