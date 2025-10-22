// Modified from REBEL

#ifndef ARGUMENTS_H_
#define ARGUMENTS_H_


#include<stdio.h>
#include<stdlib.h>

using namespace std;

class Arguments {
public:
	static const  char* datafile;
	static const char* maxindegree;
	static const char* pairs;
	static const char* maxnumrecords;
	static const char* size;
	
		static void init(int argc, char **args){
		for(int i = 1; i < argc; i ++){
			
			if(args[i][0]=='-' && args[i][1]=='d' && args[i][2]=='\0'){
				int j = i + 1;
				while (j < argc && args[j][0]!='-') {
					Arguments::datafile = args[j];
					j ++;
				} 
			}
			else if(args[i][0]=='-' && args[i][1]=='m' && args[i][2]=='\0'){
				int j = i + 1;
				while (j < argc && args[j][0]!='-') {
					Arguments::maxindegree = args[j]; 
					j ++;
				}
			}
			else if(args[i][0]=='-' && args[i][1]=='u' && args[i][2]=='\0'){
				int j = i + 1;
				while (j < argc && args[j][0]!='-') {
					Arguments::maxnumrecords = args[j]; 
					j ++;
				}
			}
			else if(args[i][0]=='-' && args[i][1]=='p' && args[i][2]=='\0'){
				int j = i + 1;
				while (j < argc && args[j][0]!='-') {
					Arguments::pairs = args[j]; 
					j ++;
				}
			}
			else if(args[i][0]=='-' && args[i][1]=='s' && args[i][2]=='\0'){
				int j = i + 1;
				while (j < argc && args[j][0]!='-') {
					Arguments::size = args[j]; 
					j ++;
				}
			}
		}
	
		print_arguments(stderr);
	}
	static void print_arguments(FILE *f){
		fprintf(f, "Inputs\n");
		fprintf(f, " -d Data file:\n");
		fprintf(f, "   %62s\n", Arguments::datafile);	
		fprintf(f, " -m Maximum indegree:\n");
		fprintf(f, "   %62s\n", Arguments::maxindegree);
		fprintf(f, " -p Number of Bucket Orders:\n");
		fprintf(f, "   %62s\n", Arguments::pairs);
		fprintf(f, " -s Bucket Size:\n");
		fprintf(f, "   %62s\n", Arguments::size);
	}
};

const char* Arguments::datafile = "testdata.dat";
const char* Arguments::maxindegree = "3";
const char* Arguments::pairs = "0";
const char* Arguments::maxnumrecords = "999999";
const char* Arguments::size = "2";

#endif

