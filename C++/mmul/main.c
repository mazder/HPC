#include<stdio.h>

#include "maxmul.h"

int main(int argc, char* argv[]){

    int m=3,n=2,k=3;
    int i, j;
    int A[m][k], B[k][n], C[m][n];

    printf("%d %d %d\n\n",m,k,n);

    for(i=0; i<m; i++){
        for(j=0; j<k; j++){
            A[i][j]=2;
        }
    }
    for(i=0; i<k; i++){
        for(j=0; j<n; j++){
            B[i][j]=2;
        }
    }

    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            C[i][j]=0;
        }
    }

    printf("A===\n");
    for(i=0; i<m; i++){
        for(j=0; j<k; j++){
            printf("%d ", A[i][j]);
        }
        printf("\n");
    }

    printf("B===\n");
    for(i=0; i<k; i++){
        for(j=0; j<n; j++){
            printf("%d ", B[i][j]);
        }
        printf("\n");
    }
    printf("C===\n");
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    maxmul(k, m, n, A, B, C);

    printf("Results C===\n");
    for(i=0; i<m; i++){
        for(j=0; j<n; j++){
            printf("%d ", C[i][j]);
        }
        printf("\n");
    }

    return 0;

}
