#include "maxmul.h"

void maxmul(int k, int m, int n, int A[m][k], int B[k][n], int C[m][n]){
    int p, q, r;
    for(p=0; p<m; p++){
		for(q=0; q<k; q++){
			for(r=0; r<n; r++){
				C[p][r] += A[p][q] * B[q][r];
            }
        }
    }
}


