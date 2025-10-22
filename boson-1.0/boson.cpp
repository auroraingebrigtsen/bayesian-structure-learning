/*
 *  BOSON: main program
 *  
 *  Copyright 2009, 2012 Pekka Parviainen <pekkapa(at)kth.se>
 *  
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */



#include <time.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <algorithm>
#include <limits>
#include <stdint.h>
#include "boson.h"
#include "Arguments.h"
#include "Data.h"
//#define WARNINGS
#define ALTERNATIVE


float getScore(float** scores, int i, int j);
int activeBit(uint64_t number, uint64_t bit);
int addZero(uint64_t number, uint64_t bit, uint64_t n);
int subtractZero(uint64_t number, uint64_t bit, uint64_t n);
int isFeasible(uint64_t number, uint64_t* partialOrder, int f, int size) ;
int computePartialOrder(uint64_t* po, int partialOrder, int f, int size);
uint64_t nextFeasible(uint64_t current, uint64_t* partialOrder, int f, int size, int n);
int getCompletePairs(uint64_t number, uint64_t* po, int f, int size);
int numberOfActiveBits(uint64_t number, int n) ;
int numberOfActiveBits(uint64_t number, int lower, int upper) ;
uint64_t getIndices(uint64_t number, uint64_t* po, int f, int size);
uint64_t getIndices(uint64_t number, int firstIndex, int lastIndex);
uint64_t smallestActiveBit(uint64_t number);
uint64_t biggestActiveBit(uint64_t number, int n);
uint64_t nextTailMembers(uint64_t current, int n, int nrOfOnes);
uint64_t factorial(int num);
uint64_t binom(int n, int k);

using namespace std;

int main(int argc, char* argv[]) {
	// Reading data
	Arguments::init(argc, argv);
	Data data;
	data.init();

	uint64_t n = data.numrecords; // Number of variables
	uint64_t p = (uint64_t) atoi(Arguments::pairs);
	uint64_t f; // Number of disjoint bucket orders
	uint64_t size =  (uint64_t) atoi(Arguments::size); // Size of each bucket order
	if (p < 0 || p > n/size) {
		f = n/size;
	}
	else {
		f = p;
	}

	// Compute the number of possible parent sets given maxindegree
	int maxindegree = atoi(Arguments::maxindegree);
	if (maxindegree < 1 || (uint64_t) maxindegree > n - 1) {
		maxindegree = n - 1;
	}
	uint64_t one = 1;
	int expectedSize = 1;
	for (int i = 0; i < maxindegree; i++) {
		expectedSize += (int) binom(n, i + 1);
	}
	if (maxindegree < (data.numrecords - 1) && expectedSize != data.numattributes) {
		cout << "Number of columns in input file doesn't match maxindegree" << endl;
		cout << "Columns: " << data.numattributes << " Expected columns: " << expectedSize << endl;
		return -1;
	}

	uint64_t base = (one<<((int) ceil((double) size/2))) + (one<<((int) floor(size/2))) - 1;
	uint64_t c = (uint64_t) pow(base, f)*(one<<(n - size*f));
	float minF = -1*numeric_limits<float>::max( );
	float score = minF;
	uint64_t* order = new uint64_t[n];
	uint64_t** dag = new uint64_t*[n];
	for (uint64_t j = 0; j < n; j++) {
		dag[j] = new uint64_t[n];
	}

	time_t mEnd, mMiddle, mStart;
	clock_t cEnd, cMiddle, cStart;
	mStart = time(NULL);
	cStart = clock();
	uint64_t r;
	uint64_t u;
	float s;
	uint64_t* pp;
	uint64_t ppp;
	uint64_t calls1;
	uint64_t calls2;
	uint64_t calls3;
	uint64_t calls4;
	calls1 = calls2 = 0;
	uint64_t** bps = new uint64_t*[n];
	float** bss = new float*[n];
	uint64_t* feasible = 0;
	feasible = new uint64_t[c];
	for (uint64_t j = 0; j < n; j++) {
		bps[j] = new uint64_t[c];
		bss[j] = new float[c];
	}
	uint64_t* prev = new uint64_t[c];
	float* ss = new float[c];

	uint64_t totalSize = n*c*(sizeof bps[0][0] + sizeof bss[0][0]) + c*(sizeof feasible[0] + sizeof prev[0] + sizeof ss[0]);
	uint64_t* po = new uint64_t[f];
	uint64_t orders;
	uint64_t base2 = (uint64_t) binom(size, size/2);
	if (f == 0) {
		orders = 1;
	}
	else {
		orders = (uint64_t) pow(base2, f);
	}

	cout << endl << "Analysis begins" << endl;
	//cout << "Maxindegree: " << maxindegree << endl;
	cout << f << " partially ordered " <<  size <<"-tuples -> " << orders << " partial orders" << endl;

	cout << endl << "Performance: " << endl;
	cout << "Space used: (" << totalSize << ") " << totalSize/(1024*1024) << "MB" << endl;
	cout << "Size of the input: (" << (sizeof data.scores[0][0])*n*data.numattributes << ") " <<  ((sizeof data.scores[0][0])*n*data.numattributes) /(1024*1024) << "MB" << endl;
	cout << "No. of ideals: " << c << endl;

	//int it = 10;
	//if (orders < it) {
	//	it = orders;
	//}
	//for (uint64_t i = 0; i < it; i++) {
	  
	 // i = partialOrder
	for (uint64_t i = 0; i < orders; i++) {
		calls1 = 0;
		calls2 = 0;
		calls3 = 0;
		calls4 = 0;
		mMiddle =  time(NULL);
		cMiddle = clock();
		cout << "Partial order: ";
		computePartialOrder(po, i, f, size);
		for (int j = 0; j < (int) f; ++j) {
			cout << po[j] << " ";
		}
		cout << endl;
		feasible[0] = 0;
		for (int j = 1; j < (int) c ; ++j) {
			feasible[j] =  nextFeasible(feasible[j - 1], po, f, size, n);
			#ifdef WARNINGS
			if (isFeasible(feasible[j], po, f, size) == 0 || feasible[j] <= feasible[j - 1]) {
				cout << "Function nextFeasible does not work properly!" << endl;
				return -1;
			}
			#endif
		}
		for (int j = 0; j <(int) n; ++j) {
			for (int k = 0; k < (int) c; ++k) {
				bss[j][k] = minF;
				bps[j][k] = -1;
			}
		}
		// Find the best parents for each variable and possible parent sets
		for (uint64_t j = 0; j < n; j++) {
			for (uint64_t k = 0; k < c; k++){
				if (activeBit(feasible[k], j) == 0) {
					if (numberOfActiveBits(feasible[k], n) <= maxindegree) {
						if ((uint64_t) maxindegree >= n - 1) {
							bss[j][k] = getScore(data.scores, j, feasible[k]);
							bps[j][k] = feasible[k];
							calls1++;
						}
						else {
							pp = lower_bound(data.feasible, data.feasible + data.numattributes, feasible[k]);
							r = (uint64_t) (pp - data.feasible);
							bss[j][k] = getScore(data.scores, j, r);
							bps[j][k] = feasible[k];
							calls1++;
						}
					}
					else {
						bss[j][k] =minF;
						bps[j][k] =0;
					}
					for (uint64_t l = 0; l < n; l++) {
						u =feasible[k] & ~(one<<l); // bit l is set to zero
						if (isFeasible(u, po, f, size) == 1) {
							pp = lower_bound(feasible, feasible + c, u);
							r = (uint64_t) (pp - feasible);
							if (bss[j][r] > bss[j][k]) {
								bss[j][k] = bss[j][r];
								bps[j][k] = bps[j][r];
							}
							calls3++;
						}
					}
					// Check the tail
					int p = getCompletePairs(feasible[k], po, f, size);
					int activeBits = numberOfActiveBits(feasible[k], n);
					if (p > 0 && activeBits - (int) size/2*p <= maxindegree) {
						uint64_t bits = getIndices(feasible[k], po, f, size);
						uint64_t* maskBits = new uint64_t[size/2*p];
						int ind2 = 0;
						for (uint64_t l = 0; l < n; l++) {
							if (activeBit(bits, l)) {
								maskBits[ind2] = (one<<l);
								ind2++;
							}
						}
						uint64_t mask;
						#ifdef ALTERNATIVE
						int minSize = max(1, activeBits - maxindegree);
						for (int ls = minSize; ls <= (int) (size/2*p); ls++) {
							uint64_t l = nextTailMembers(0, size/2*p, ls);
							int ok = 0;
							if (ls == 0) {
								ok = 1;
							}
							while(l != 0 || ok == 1) {
								if (ls == 0) {
									ok = 0;
								}
								#else
								for (uint64_t l = 0; l < (uint64_t) pow((one<<(size/2)), p) -1; l++) {
								#endif
									mask = 0;
									for (int ll = 0; ll < (int) size/2*p; ll++) {
										#ifdef ALTERNATIVE
										if (activeBit(l, ll)) {
										#else
										if (activeBit(l + 1, ll)) {
										#endif
											mask += maskBits[ll];
										}
									}
									#ifdef WARNINGS
									if (mask > bits && (l == pow((one<<(size/2)), p) -1 - 1 && mask < bits)) {
										cout << "Finding tail sets does not work properly " << endl;
										return -1;
									}
									#endif
									u = feasible[k] & ~mask;
									if (numberOfActiveBits(u, n) > maxindegree) {
										calls4++;
										continue;
									}

									if ((uint64_t) maxindegree >= n - 1) {
										s = getScore(data.scores, j, u);
										calls2++;
									}
									else {
										pp = lower_bound(data.feasible, data.feasible + data.numattributes, u);
										r = (uint64_t) (pp - data.feasible);
										s = getScore(data.scores, j, r);
										calls2++;
									}
									if (s > bss[j][k]) {
										bss[j][k] = s;
										bps[j][k] = u;
									}

								#ifdef ALTERNATIVE
								l = nextTailMembers(l, size/2*p, ls);
							}
							#endif
						}
						delete [] maskBits;
					}
				}
			}
		}
		// Find an optimal network compatible with the partial order
		for (uint64_t j = 0; j < c; j++) {
			prev[j] = 0;
			ss[j] = minF;
		}
		ss[0] = 0;

		for (uint64_t j = 0; j < c; j++) {
			for (unsigned int k = 0; k < n; k++) {
				u = feasible[j] & ~(one<<k);
				if (u != feasible[j] && isFeasible(u, po, f, size) == 1) {
					pp = lower_bound(feasible, feasible + c, u);
					r = (uint64_t) (pp - feasible);
					if (j == 0) {
						s =  bss[k][r];
					} else {
						s = ss[r] + bss[k][r];
					}
					if (s > ss[j]) {
						ss[j] = s;
						prev[j] = r;
					}
				}
			}
		}
		
		if (ss[c - 1] > score) {
			score = ss[c - 1];
			for (uint64_t j = 0; j < n; j++) {
				for (uint64_t k = 0; k < n; k++) {
					dag[j][k] = 0;
				}
			}

			u = c - 1;
			r = prev[c - 1];
			for (uint64_t j = 0; j < n - 1; j++) {
				order[n - j - 1] = smallestActiveBit(feasible[u] & ~feasible[r]);
				pp = lower_bound(feasible, feasible + c, feasible[r]);
				ppp = (uint64_t) (pp - feasible);
				for (uint64_t k = 0; k < n; k++) {
					if (activeBit(bps[order[n - j - 1]][ppp], k)) {
						dag[k][order[n - j - 1]] = 1;
					}
				}
				u = r;
				r = prev[r];
			}
			order[0] = smallestActiveBit(feasible[u] & ~feasible[r]);
		}

		cout << "Score: " << ss[c - 1] << endl;

		mEnd =  time(NULL);
		cEnd = clock();
		cout << "Partial order " << i << " analyzed. Elapsed time: " << (double) difftime(mEnd, mMiddle) << "s (" << (cEnd - cMiddle)/CLOCKS_PER_SEC << "s)" << endl;
		cout << "Number of calls - Regular: " << calls1 << " Tail: " << calls2 << " I(Y): " << calls3 << " Unnecessary: " << calls4 << endl;
		//cout << "Ratios: unnec/reg: " << calls4/calls1 << " (tail+unnec)/reg: " << (calls4 + calls2)/calls1 << endl;
	}
	delete[] prev;
	delete[] ss;
	delete[] feasible;
	for (uint64_t j = 0; j < n; j++) {
		delete [] bps[j];
		delete [] bss[j];
	}
	delete[] bps;
	delete[] bss;
	delete[] po;

	mEnd =  time(NULL);
	cEnd = clock();
	double mTime = (double) difftime(mEnd, mStart);

	cout << "Total time: " << mTime << "s (" << (cEnd - cStart)/CLOCKS_PER_SEC << "s)" << endl;
	cout << "Corrected time: " << mTime*orders/orders << "s (" << (cEnd - cStart)/CLOCKS_PER_SEC*orders/orders << "s)" << endl;

	cout << endl << "Results: " << endl;
	cout << "Score: " << score << endl;
	cout << "Order: ";
	for (uint64_t i = 0; i < n; i++) {
		cout << order[i] << " ";
	}
	cout << endl;
	delete [] order;
	cout << "Optimal DAG:" << endl;
	for (uint64_t i = 0; i < n; i++) {
	  for (uint64_t j = 0; j < n; j++) {
	    cout << dag[i][j] << " ";
	  }
	  cout << endl;
	}
	for (uint64_t i = 0; i < n; i++) {
		delete[] dag[i];
	}
	delete[] dag;
}

float getScore(float** scores, int i, int j) {
	return scores[i][j];
}

// Checks whether bit:th bit of number is one
int activeBit(uint64_t number, uint64_t bit)
{
	uint64_t one = 1;
    if((number & (one<<bit)) > 0) {
    	return 1;
    } else {
    	return 0;
    }
}

int addZero(uint64_t number, uint64_t bit, uint64_t n) {
	return ((number<<1) & ((1<<n) - ((1<<bit) + 1))) | (number & ((1<<bit) - 1));
}

int subtractZero(uint64_t number, uint64_t bit, uint64_t n) {
	return ((number & ((1<<n) - ((1<<bit) + 1)))>>1) | (number & ((1<<bit) - 1));
}

// Checks whether number describes a feasible set under partialOrder
int isFeasible(uint64_t number, uint64_t* po, int f, int size) {
	for (int i = 0; i < f; i++) {
		int prec = 1;
		int succ = 0;
		for (int j = 0; j < size; ++j) {
			if (activeBit(number, i*size + j) != activeBit(po[i], j) ) {
				if (activeBit(po[i], j) == 1) {
					prec = 0;
				}
				else {
					succ = 1;
				}
			}
		}
		if (succ >  prec) {
			return 0;
		}
	}
	return 1;
}

int computePartialOrder(uint64_t* po, int partialOrder, int f, int size) {
	int rest = partialOrder;
	for (int i = 0; i < f; ++i) {
		int temp2 =  rest % binom(size, size/2);
		int temp = (rest - temp2) / binom(size, size/2);
		rest = temp;
		int toBeAssigned = size/2;
		int p = 0;
		for (int j = 0; j < size; ++j) {
		#ifdef WARNINGS
			if (size - j < toBeAssigned) {
				cout << "Error: computePartialOrder1" << endl;
			}
		#endif
			if (toBeAssigned == 1) {
				p += (1<<(j + temp2));
				break;
			}
			else if (j + toBeAssigned == size) {
				for (int k = j; k < size; ++k) {
					p += (1<<k);
				}
				break;
			}
			else if (toBeAssigned > 1) {
				int left = binom(size - j, toBeAssigned) - binom(size - j - 1, toBeAssigned);
				if (temp2 < left) {
					p += (1<<j);
					--toBeAssigned;
				}
				else {
					temp2 -= left;
				}
			}
		}
		po[i] = p;
	#ifdef WARNINGS
		if (numberOfActiveBits(po[i], size) != size/2) {
			cout << "Error: computePartialOrder2 " << endl;
		}
	#endif
	}
	#ifdef WARNINGS
	if (rest > 0) {
		cout << "Error: computePartialOrder2" << endl;
		return -1;
	}
	#endif
	return 1;
}

uint64_t nextFeasible(uint64_t current, uint64_t* partialOrder, int f, int size, int n) {
	#ifdef WARNINGS
	if (current < 0 ) {
		cout << "current: " << current << endl;
	}
	#endif
	const uint64_t one = 1;
	uint64_t ret;// = current;
	uint64_t mTuple = 0;
	uint64_t mask = 0;
	for (int i = 0; i < size; ++i) {
		mask += (one<<(mTuple*size + i));
	}
	uint64_t tuple = current & mask;
	while (numberOfActiveBits(tuple, size*mTuple, size*(mTuple + 1)) == size) {
		++mTuple;
		mask = 0;
		for (int i = 0; i < size; ++i) {
			mask += (one<<(mTuple*size + i));
		}
		tuple = current & mask;
	}
	if (mTuple < (uint64_t) f) {
		ret = current & ~((one<<(mTuple*size)) - 1);
		uint64_t ind1 = getIndices(ret, mTuple*size, (mTuple + 1)*size);
		uint64_t ind2 = ind1 & partialOrder[mTuple];
		uint64_t ind3 = 0; // Backside indices
		uint64_t a = 0;
		if (ind2 >= (uint64_t) partialOrder[mTuple] ) {
			// Frontside is full
			for (int i = 0; i < size; ++i) {
				if (activeBit(partialOrder[mTuple], i) ==0) {
					if( activeBit(ind1, i) == 1) {
						ind3 += (one<<a);
					}
					++a;
				}
			}
			++ind3;
			int b = 0;
			for (int i = 0; i < size; ++i) {
				if (activeBit(ind3, b) == 1 && activeBit(partialOrder[mTuple], i) == 0) {
					if (activeBit(ret, mTuple*size + i) == 0) {
						ret += (one<<(mTuple*size + i));
					}
					++b;
				}
				else if (activeBit(ind3, b) == 0 && activeBit(partialOrder[mTuple], i) == 0) {
					if (activeBit(ret, mTuple*size + i) == 1) {
						ret -= (one<<(mTuple*size + i));
					}
					++b;
				}
			}
		}
		else {
			for (int i = 0; i < size; ++i) {
				if (activeBit(partialOrder[mTuple], i) ==1) {
					if( activeBit(ind1, i) == 1) {
						ind3 += (one<<a);
					}
					++a;
				}
			}
			++ind3;
			int b = 0;
			for (int i = 0; i < size; ++i) {
				if (activeBit(partialOrder[mTuple], i) == 1 && activeBit(ind3, b) == 1) {
					if (activeBit(ret, mTuple*size + i) == 0) {
						ret += (one<<(mTuple*size + i));
					}
					++b;
				}
				else if (activeBit(partialOrder[mTuple], i) == 1 && activeBit(ind3, b) == 0) {
					if (activeBit(ret, mTuple*size + i) == 1) {
						ret -= (one<<(mTuple*size + i));
					}
					++b;
				}
			}
		}
	}
	else {
		ret = 0;
		uint64_t ind1 = getIndices(current, mTuple*size, n);
		++ind1;
		for (int i = 0; i < n - f*size; ++i) {
			if (activeBit(ind1, i) == 1) {
				ret += (one<< (mTuple*size + i));
			}
		}
	}
	#ifdef WARNINGS
	if (ret <= current) {
		cout << "current: " << current << " ret: " << ret << " mTuple: " << mTuple << endl;
	}
	#endif
	return ret;
}

int getCompletePairs(uint64_t number, uint64_t* po, int f, int size) {
	int n = 0;
	for (int i = 0; i < f; i++) {
		int succ = 0;
		for (int j = 0; j < size; ++j) {
			if (activeBit(number, i*size + j) != activeBit(po[i], j) && activeBit(po[i], j) == 0) {
				succ = 1;
				break;
			}
		}
		if (succ == 1) {
			++n;
		}
	}
	return n;
}

int numberOfActiveBits(uint64_t number, int n) {
	int r = 0;
	for (int i = 0; i < n; i++) {
		if (activeBit(number, i) > 0) {
			r++;
		}
	}
	return r;
}

int numberOfActiveBits(uint64_t number, int lower, int upper) {
	int r = 0;
	for (int i = lower; i < upper; ++i) {
		if (activeBit(number, i) > 0) {
			r++;
		}
	}
	return r;
}

uint64_t getIndices(uint64_t number, uint64_t* po, int f, int size) {
	uint64_t n = 0;
	for (int i = 0; i < f; i++) {
		uint64_t r = 0;
		int ok = 0;
		for (int j = 0; j < size; ++j) {
			if (activeBit(number, i*size + j) == 1 && activeBit(po[i], j) == 1) {
				r += 1<<( i*size + j);
			}
			else if (activeBit(number, i*size + j) == 1 && activeBit(po[i], j) == 0) {
				ok = 1;
			}
		}
		if (ok == 1) {
			n += r;
		}
	}
	return n;
}

uint64_t getIndices(uint64_t number, int firstIndex, int lastIndex) {
	uint64_t n = 0;
	for (int i = firstIndex; i < lastIndex; i++) {
		if (activeBit(number, i) == 1) {
			n += (1<<(i - firstIndex));
		}
	}
	return n;
}

uint64_t smallestActiveBit(uint64_t number) {
	uint64_t s = 8* sizeof number;
	for (uint64_t i = 0; i < s; i++) {
		if (activeBit(number, i)) {
			return i;
		}
	}
	return s;
}

uint64_t biggestActiveBit(uint64_t number, int n) {
	uint64_t s = 8* sizeof number;
	for (uint64_t i = s - 1; i >= 0; i--) {
		if (activeBit(number, i)) {
			return i;
		}
	}
	return s;
}

// Lisää: Jos syötteekai annetaan 0, niin palauta ensimmäinen mahdollinen arvo, jos ei ole suurempia mahdollisia arvoja palauta 0
uint64_t nextTailMembers(uint64_t current, int n, int nrOfOnes) {
	if (nrOfOnes == 0) {
		return 0;
	}
	int* indices = new int[nrOfOnes];
	for (int i = 0; i < nrOfOnes; i++) {
		indices[i] = -1;
	}
	if (current == 0) {
		for (int i = 0; i < nrOfOnes; i++) {
			indices[i] = i;
		}
	}
	else {
		int ind = 0;
		for (int i = 0; i < n; i++) {
			if(activeBit(current, i)) {
				indices[ind] = i;
				ind++;
			}
		}
		if (ind != nrOfOnes) {
	 		return 0; // There's something wrong with the input
		}
		int movable = -1;
		for (int i = 0; i < nrOfOnes - 1; i++) {
			if (indices[i + 1] - indices[i] != 1 && movable == -1) {
				movable = i;
				continue;
			}
		}
		if (movable == -1) {
			if (indices[nrOfOnes - 1] < n - 1) {
				movable = nrOfOnes - 1;
			}
			else {
				return 0;
			}
		}
		indices[movable]++;
		for (int i = 0; i < movable; i++) {
			indices[i] = i;
		}
	}
	uint64_t r = 0;
	for (int i = 0; i < nrOfOnes; i++) {
		r += (uint64_t) pow(2, indices[i]);
	}
	delete[] indices;
	indices = 0;
	return r;
}

uint64_t factorial(int num) {
	int r = 1;
	for (int i = 0; i < num; i++) {
		r *= (i + 1);
	}
	return r;
}

uint64_t binom(int n, int k) {
	if (n == k || k == 0) {
		return 1;
	}
	else {
		return binom(n - 1, k - 1) + binom(n - 1, k);
	}
}
