#include <math.h>
#include "SSM_user.h" /* needed by the program which calls SSM */
#include "timemeasure.h"
#include <time.h>
#define BILLION  1000000000.0;


void printArr(double * arr,int n){
    printf("\n\n");
    for (int i = 0; i < n; i++) printf("%f::",arr[i]);
    printf("\n\n");
    
}

double * loadDatasets(int * n){
    
    FILE * myFile = fopen("/Users/maaz/Desktop/optimization/SSM/SSM/dataset.txt", "r");
    
    fscanf(myFile, "%d", n);
    int N = (*n);
    
    
    double * A = (double *)malloc(((N*N)+N)*sizeof(double));
    
    for(int i = 0 ; i<(N*N) + N ; i++) fscanf(myFile, "%lf", A+i);
    
    
    fclose(myFile);
    
    return A;
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

double vectordotvector(double * a, double * b , int n){
    double rslt =0;
    for (int i = 0 ; i<n ; i++){
        rslt+=a[i] * b[i];
    }
    return rslt;
}


double model (double * x , double * A , double *y , int n){
    double * temp =  matrixdotvec(A, x, n);
    double temp1 = vectordotvector(temp, x, n);
    return temp1  + vectordotvector(x, y, n);
}






int main (void)
{
    int n, i, j, k, l, status, *Ai, *Ap ;
    double radius, rand_max, normb, bj, uj, yj, normu, t, c,*x, *b, *d, *u, *y, *Ax ;
    SSMParm Parm ;
    ssm_stat Stat ;
    
    //load data set from file
    double * dataset  =loadDatasets(&n);
    b  = malloc (n*sizeof (double)) ;
    Ax = malloc (n*n*sizeof (double)) ;
    int  kk = 0 , p = 0;
    for ( ; kk<n; kk++) b[kk] = dataset[kk];
    for (; kk < (n*n) + n ;kk++ ) Ax[p++] = dataset[kk];
    
    
    
    Ap = malloc ((n+1)*sizeof (int)) ;
    Ai = malloc (n*n*sizeof (int)) ;
    x  = malloc (n*sizeof (double)) ;
    d  = malloc (n*sizeof (double)) ;
    u  = malloc (n*sizeof (double)) ;
    y  = malloc (n*sizeof (double)) ;
    rand_max = (double) RAND_MAX ;
    normb = (double) 0 ;
    normu = (double) 0 ;
    for (j = 0; j < n; j++)
    {
        //bj = ((double) rand ()) / rand_max - (double) .5 ;
        //b [j] = bj ;
        normb += b[j]*b[j] ;
        d [j] = ((double) rand ()) / rand_max - (double) .5 ;
        uj = ((double) rand ()) / rand_max - (double) .5 ;
        u [j] = uj ;
        normu += uj*uj ;
    }
    t = sqrt(2)/normu ;
    c = (double) 0 ;
    for (j = 0; j < n; j++)
    {
        uj *= t ;
        u [j] = uj ;
        yj = y [j] = d [j]*uj ;
        c += yj*uj ;
        //b [j] /= normb ;
    }
    /* store the dense matrix (not very efficient) */
    k = 0 ;
    Ap [0] = 0 ;
    for (j = 0; j < n; j++)
    {
        /* A = diag(d) - y*u' - u*y' + (u*c)*u' ;*/
        l = k ;
        for (i = 0; i < n; i++)
        {
            //Ax [k] = (c*u [j] - y [j])*u [i]  - y [i]*u [j] ;
            Ai [k] = i ;
            k++ ;
        }
        //Ax [l+j] += d [j] ;
        Ap [j+1] = k ;
    }
    free (d) ;
    free (u) ;
    free (y) ;

    radius = (double) 1 ;

    /* make any changes to default parameter settings here */
    SSMdefault (&Parm) ;
    Parm.Lanczos_dim1 = 15 ;
    Parm.Lanczos_dim2 = 100 ;
    Parm.Lanczos_L1 = 15 ;
    Parm.Lanczos_L2 = 30 ;
    Parm.PrintLevel = 3 ;
    
    //printArr(b, n);
    //printArr(Ax,n*n);
    
    for(int i =0 ; i<n ; i++) b[i]  =  b[i]*0.5;
    
    /* call SSM */
    
    double timereport = 0;
    
  
    
    for (int i =0 ; i<100 ; i++){
        CCPAL_INIT_LIB
        CCPAL_START_MEASURING
        status = SSM (x, &Stat, n, Ax, Ai, Ap, b, radius, 1.e-6, NULL, &Parm) ;
        CCPAL_STOP_MEASURING
        timereport+=elaps_s + ((double)elaps_ns) / 1.0e9;
    }
    
    //timereport/=100;
    
    
    for(int i =0 ; i<n ; i++) b[i]  = b[i]*2;
    float rslt =  model(x, Ax, b, n);
    
    

    double norm  = 0 ;
    for (int i =0  ; i<n ; i++) norm  +=   (x[i] * x[i]);
    norm =  sqrt(norm);
    
    printf("%f\n",rslt);
//    FILE * file = fopen("/Users/maaz/Desktop/diagramreport.txt", "a");
//    fprintf(file,"%f\n",timereport);
//    fclose(file);
//
    
    
    free (Ax) ;
    free (Ai) ;
    free (Ap) ;
    free (x) ;
    free (b) ;
    
    
}

