//
//  main.c
//  GradientMethod
//
//  Created by MAAZ on 10/31/19.
//  Copyright © 2019 MAAZ. All rights reserved.

#include <stdio.h>
#include <Accelerate/Accelerate.h>
#include <time.h>
#include <math.h>
#include "timemeasure.h"
//#include <float.h>
#define BILLION  1000000000.0;
//#define DBL_MAX 1.7976931348623158e+308


void printArr(double * arr,int n){
    printf("\n\n");
    for (int i = 0; i < n; i++) printf("%f::",arr[i]);
    printf("\n\n");
    
}


double function (double x,double * ytilda,double * landa,int n){

    double temp = 0;
    
    for (int i=0;i < n; i++) temp+= (ytilda[i]/(x-(2*landa[i]))) * (ytilda[i]/(x-(2*landa[i])));
    
    return temp;
}
//
//double * ternarysearch(double left , double right,double * ytilda,double * landa, int n){
//
//
//    //printf("%f : %f \n",left,right);
//    double left_third;
//    double right_third;
//
//    double * point = (double *) malloc(2* sizeof(double));
//
//
//    for (int  i =0; i<20; i++){
//
//        left_third = ((2 * left) + right) / 3;
//        right_third = (left + (2 * right)) / 3;
//
//        if(function(left_third,ytilda,landa,n) > function(right_third,ytilda,landa,n)){
//
//            left = left_third;
//            //right = right;
//        }
//        else{
//            //left = left;
//            right = right_third;
//        }
//
//    }
//    point[0] = (left+right)/2;
//    point[1] = function(point[0],ytilda,landa,n);
//    return point;
//
//}

double * binarysearch(double left , double right ,int dir ,double * ytilda,double * landa,int n ){
    
    double *point = (double *) malloc(2* sizeof(double));
    double *  rtrn  ;
    
    
    double mid = 0 ;
    double value = 0 ;
    
    double leftinterval = left;
    double rightinterval = right;
    
    
    for (int i =0 ; i< 20 ;  i++){
        
        mid = (rightinterval + leftinterval) / 2 ;
        
        value  = function(mid,ytilda,landa,n);
        
        if(dir == 0){
            if (value < 1) leftinterval = mid;
            else rightinterval  = mid;
        }
        else if(dir == 1){
            if (value <1) rightinterval = mid;
            else leftinterval = mid;
        }
    }
    
    point[0]  = mid  ;
    point [1] = function(mid, ytilda, landa, n);

   
    rtrn =  point;
    //free(point);
    return point;
    
}

double * matrixdotvec(double * A , double * c , int n){
    
   
  
    double * rslt = (double *) malloc(n * sizeof(double));
    double  * rtrn ;

    
    double temp ;
    
    for (int i = 0 ; i<n ; i++){
        temp = 0;
        for  (int j = 0 ; j < n ;  j++) temp +=  A[(j*n) + i] * c[j];
            
        rslt[i]=temp;
    }
    
    rtrn  = rslt;
    //free(rslt);
    return rslt;
}

void swap (double * a ,double *b){

    double tmp = *(a);
    *(a) = *(b);
    *(b) = tmp;
    
}

void transpose(double * A ,int n){
    for (int i =0 ;  i<n  ; i++){
        for(int j = 0 ; j <  i ; j++){
            swap(A+(n*i)+j, A+(n*j)+i);
        }
    }
}

double vectordotvector(double * a, double * b , int n){
    double rslt =0;
    for (int i = 0 ; i<n ; i++){
        rslt+=a[i] * b[i];
    }
    return rslt;
}

void multiply(double *mat1, double * mat2,double * res,int N)
{
    int i, j, k;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            res[j*N + i] = 0;
            for (k = 0; k < N; k++)
                res[j*N + i] += mat1[k*N + i] * mat2[ N*j + k];
        }
    }
}

double model (double * x , double * A , double *y , int n){
   
    double * temp =  matrixdotvec(A, x, n);
    double temp1 = vectordotvector(temp, x, n);
    return temp1  + vectordotvector(x, y, n);
}

double  *  solution(double x , double * landa , double * ytilda,double * v , int n){
    
   
    
    double *term = (double *) malloc(n*n *sizeof(double));
    
    for (int i =0;i<n ; i++){
        for (int j = 0 ; j<n ; j++){
            if (j == i) term[i*n+j] = 1./(x- 2*landa[i]);
            else term[i*n + j] = 0;
        }
    }
   
    double * temp = matrixdotvec(term, ytilda, n);
    
    double * tmp = matrixdotvec(v, temp, n);
    double norm =  0;
    for (int i = 0 ; i <  n  ; i++){
        norm+=tmp[i] * tmp[i];
    }
    norm = sqrt(norm);
    
    for (int i = 0 ; i < n ; i++){
        tmp[i]/=norm;
    }
    return tmp;
}



void heapify(double arr[], int n, int i)
{
    int largest = i;
    int l = 2*i + 1;
    int r = 2*i + 2;
    
   
    if (l < n && arr[l] > arr[largest])
        largest = l;
    if (r < n && arr[r] > arr[largest])
        largest = r;
    
    if (largest != i)
    {
        swap(arr+i, arr+largest);
        heapify(arr, n, largest);
    }
}


void heapSort(double arr[], int n)
{

    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);
    for (int i=n-1; i>=0; i--)
    {
        
        swap(arr, arr+i);
        heapify(arr, i, 0);
    }
}

void get_eigs(double * a ,double * eigvecs, double * eigvals,int n){
    
    
    char lefteig = 'N';
    char righteig = 'V';
    int dim = n;
    int lwork = 5*dim;
    
    
    double *wi = (double *)malloc(dim * sizeof(double));
    
    double *vl = (double*) malloc(dim*dim * sizeof(double));
    
    
    double *work = (double*) malloc(lwork * sizeof(double));
    
    int info = -11;
    
    
    dgeev_(&lefteig, &righteig, &dim, a, &dim, eigvals, wi, vl, &dim, eigvecs, &dim, work, &lwork, &info); //seems only for not symm matrice
    
    free(wi);
    free(vl);
    free(work);
}


double  solve_model(double * A  , double * eigvecs,double * eigvals , double * y , int n, double * x,double *ytilda){

    double * eigvalsort = (double*) malloc(n*sizeof(double));
    
    for(int i =0 ; i< n ; i++){
        eigvalsort [i] = eigvals[i];
    }
    heapSort(eigvalsort, n);
    


    
    double * candidate_solutions = (double *)malloc(n * sizeof(double));
    
    //double * candidate_rslt   = (double *) malloc(n * sizeof(double));
    
 
    double * localmin  ;

   
   //printArr(eigvalsort, n);
    candidate_solutions[0] = binarysearch(-100,2*eigvalsort[0], 0, ytilda, eigvals, n)[0];
    
    candidate_solutions[1] = binarysearch(2*eigvalsort[n-1], 100, 1, ytilda, eigvals, n)[0];
    
    //printf("%f , %f", candidate_solutions[0],candidate_solutions[1]);
    
    double minrslt = DBL_MAX ;
   // double minrsltindex ;
    
    double modelrslt;
    double * modelsolution = (double  * )malloc(n*sizeof(double));
    
    int j = 2 ;
    
//    for (int i =1 ; i<n-2 ; i++ ){
//
//        localmin = ternarysearch(2 * eigvalsort[i],2 *  eigvalsort[i+1], ytilda, eigvals, n);  //i had forgotten to multiply eigvals by 2
//
//       //printf("local min , i : ( %f , %f )  , %d  \n",localmin[0],localmin[1],i);
//
//        if(localmin[1] == 1) candidate_solutions[j++] = localmin[0];
//
//        else if(localmin[1] < 1){
//            candidate_solutions[j++] =  binarysearch(eigvalsort[i] *2, localmin[0], 0, ytilda, eigvals, n)[0];
//            candidate_solutions[j++] = binarysearch(localmin[0], eigvalsort[i+1] * 2, 1, ytilda, eigvals, n)[0];
//        }
//
//    }
   // printArr(candidate_solutions, j);
   //printf("%d : ", j);
    double * thesolution;
    
    //printf("this is j : %d\n",j);
  
    for (int i =  0 ;  i<j ; i++){
        //printf("candidate %d : %f\n",i,candidate_solutions[i]);
        thesolution =  solution(candidate_solutions[i],eigvals, ytilda, eigvecs, n);
        //printArr(thesolution, n);
        modelrslt = model(thesolution, A, y, n);
        
        //printArr(thesolution, n);
        //printf("this is model rslt :%f\n",modelrslt);
        if(modelrslt < minrslt){
           // printf("this is i : %d\n",i);
            minrslt = modelrslt;
            modelsolution = thesolution;
          //  printArr(x, 4);
        }
        
    }
    
    for(int i =0; i< n ; i++){
        x[i]  = modelsolution[i];
    }
     
    free(candidate_solutions);
    free(eigvalsort);
    free(modelsolution);
    return minrslt;
}

double * loadDatasets(int * n){
    
    FILE * myFile = fopen("/Users/maaz/Desktop/optimization/GradientMethod/GradientMethod/dataset.txt", "r");
    
    fscanf(myFile, "%d", n);
    int N = (*n);
    
    
    double * A = (double *)malloc(((N*N)+N)*sizeof(double));
    
    for(int i = 0 ; i<(N*N) + N ; i++) fscanf(myFile, "%lf", A+i);
    
    
    fclose(myFile);
    
    return A;
}



int main(int argc, const char * argv[]) {                                  
    // insert code here..

    int n ;
    double * dataset  =loadDatasets(&n);
    
    double *  a =  (double *)malloc(n*n*sizeof(double));
    double *  Aeigs =  (double *)malloc(n*n*sizeof(double));
    double * y = (double *)malloc(n*sizeof(double));
    int  k = 0 , p = 0, h = 0;
    for ( ; k<n; k++) y[k] = dataset[k];
    for (; k < (n*n) + n ;k++ ) {
        Aeigs[p++] = dataset[k];
        a[h++]  = dataset[k];
    }
    
   // printArr(a, n*n);
   // printArr(y, n);
    
    double *  x = (double * )malloc(n*sizeof(double));
    double * ytilda ;
    double * eigenvals = (double *)malloc(n * sizeof(double));
    double * eigenvecs = (double *)malloc(n * n * sizeof(double));
    double timereport = 0;
    float rt;
    
        
    CCPAL_INIT_LIB
    CCPAL_START_MEASURING

    get_eigs(Aeigs,eigenvecs, eigenvals,n);
   // fprintf (stdout, "We have spent %lf seconds for eigval decompose.\n", elaps_s + ((double)elaps_ns) / 1.0e9 );
    transpose(eigenvecs, n);
    ytilda = matrixdotvec(eigenvecs, y, n);
    transpose(eigenvecs, n);
    
     rt = solve_model(a, eigenvecs, eigenvals, y, n, x,ytilda);
    
    CCPAL_STOP_MEASURING
    timereport+=elaps_s + ((double)elaps_ns) / 1.0e9;
    //fprintf (stdout, "%lf\n", elaps_s + ((double)elaps_ns) / 1.0e9 );


    double norm  = 0 ;
    for (int i =0  ; i<n ; i++) norm  +=   (x[i] * x[i]);
    norm = sqrt(norm);
    
    printf("%f\n",rt);
    
    
//    FILE * file = fopen("/Users/maaz/Desktop/diagramreport.txt", "a");
//    fprintf(file, "%d %f ",n,timereport);
//    fclose(file);
    
    
   /* struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    float rt = solve_model(a, eigenvecs, eigenvals, y, n, x,ytilda);
    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent = (end.tv_sec - start.tv_sec) +
    (end.tv_nsec - start.tv_nsec) / BILLION;
    printf("Time elpased is %f seconds\n", time_spent);
    */
    
    
    


    
    free(x);
    free(a);
    free(eigenvals);
    free(eigenvecs);
    free(ytilda);
    
    
    
    
    return(0);
}



