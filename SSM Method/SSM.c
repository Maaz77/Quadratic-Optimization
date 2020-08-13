/* =========================================================================
   ================================ SSM ====================================
   =========================================================================
       ________________________________________________________________
      |    Solve a sphere constrained quadratic program of the form    |
      |                                                                |
      |           1                                                    |
      |       min - x'Ax + b'x  subject to ||x|| <= r or ||x|| = r     |
      |           2                                                    |
      |                                                                |
      |    using an implementation of the sequential subspace method   |
      |    which includes:                                             |
      |                                                                |
      |       1. SQP acceleration                                      |
      |       2. Solution of SQP system using MINRES and SSOR          |
      |          preconditioning                                       |
      |                                                                |
      |                  Version 1.1 (September 25, 2009)              |
      |                     Version 1.0 (May 5, 2009)                  |
      |                                                                |
      |                         William W. Hager                       |
      |                        hager@math.ufl.edu                      |
      |                     Department of Mathematics                  |
      |                       University of Florida                    |
      |                     Gainesville, Florida 32611                 |
      |                         352-392-0281 x 244                     |
      |                   http://www.math.ufl.edu/~hager               |
      |                                                                |
      |                   Copyright by William W. Hager                |
      |________________________________________________________________|
       ________________________________________________________________
      |This program is free software; you can redistribute it and/or   |
      |modify it under the terms of the GNU General Public License as  |
      |published by the Free Software Foundation; either version 2 of  |
      |the License, or (at your option) any later version.             |
      |This program is distributed in the hope that it will be useful, |
      |but WITHOUT ANY WARRANTY; without even the implied warranty of  |
      |MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the   |
      |GNU General Public License for more details.                    |
      |                                                                |
      |You should have received a copy of the GNU General Public       |
      |License along with this program; if not, write to the Free      |
      |Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, |
      |MA  02110-1301  USA                                             |
      |________________________________________________________________|*/

/* Contents:

    1.  SSM         - solve sphere constrained optimization problem
    2.  SSMdefault  - sets default parameter values in the SSMParm structure
    3.  SSMallocate - allocates the arrays needed in the SSM algorithm
    4.  SSMdestroy  - free the allocated memory
    5.  SSMinitProb - store problem, order rows in col, locate 1st super diag
    6.  SSMprint_parms - print the parameter array
    7.  SSMballdense- solve a dense sphere constrained QP
    8.  SSMsubspace - solve the SSM subspace problem
    9.  SSMorth     - generate orthonormal vectors using Householder
    10. SSMmineig   - eigenvector associated with smallest eigenvalue
    11. SSMdiag     - diagonalize a tridiagonal matrix */

#include "SSM.h"

/* PROTOTYPES */

PRIVATE void SSMallocate
(
    SSMcom    *Com, /* pointer to SSMcom structure */
    SSMProblem *PB, /* problem specification */
    SSMParm  *Parm, /* parameters, needed to determine allocation */
    int          n  /* problem dimension */
) ;

PRIVATE void SSMreallocate
(
    FLOAT       *x, /* current estimate of solution */
    SSMcom    *Com  /* SSMcom structure */
) ;

PRIVATE void SSMdestroy
(
    SSMcom *Com  /* SSMcom structure to free */
) ;

PRIVATE void SSMinitProb
(
    FLOAT      *Ax, /* numerical entries in matrix, packed array */
    int        *Ai, /* row indices for each column */
    INT        *Ap, /* points into Ax and Ai, Ap [0], ... Ap [n], packed */
    int          n, /* problem dimension */
    FLOAT       *b, /* linear term in objective function */
    FLOAT      rad, /* radius of sphere */
    SSMProblem *PB, /* problem structure */
    SSMParm  *Parm, /* parameter structure */
    SSMcom    *Com  /* SSMcom structure */
) ;

PRIVATE void SSMprint_parms
(
    SSMParm *Parm /* SSMparm structure to be printed */
) ;

PRIVATE int SSMballdense /*return 0 (error tolerance satisfied)
                                 -5 (failure of QR diagonalization)
                                 -6 (insufficient space in QR diagonalization)*/
(
    FLOAT        *x, /* n-by-1 solution vector (output) */
    SSMProblem  *PB, /* problem structure associated with A */
    SSMcom      *Com /* SSMcom structure */
) ;

PRIVATE int SSMsubspace /* return 0 (error tolerance satisfied)
                                -5 (failure of QR diagonalization)
                                -6 (insufficient space in QR diagonalization)*/
(
    FLOAT         *v1, /* subspace vector 1, solution is returned in v1 */
    FLOAT      normv1, /* norm of vector 1 */
    FLOAT        *Av1, /* A*v1 */
    FLOAT         *v2, /* subspace vector 2 */
    FLOAT         *v3, /* subspace vector 3 */
    FLOAT         *v4, /* subspace vector 4 */
    FLOAT         *v5, /* subspace vector 5 */
    int             m, /* number of vectors */
    int only_residual, /* only compute kkt error and eigenvector residual */
    SSMcom       *Com
) ;

PRIVATE void SSMorth
(
    FLOAT     *w, /* k-th orthonormal vector */
    FLOAT     *W, /* packed matrix storing Householder vectors */
    FLOAT     *x, /* k-th new vector */
    int        k,
    int        n  /* dimension of x */
) ;

PRIVATE void SSMmineig
(
    FLOAT    *emin, /* estimated smallest eigenvalue */
    FLOAT       *v, /* associated eigenvector */
    FLOAT       *w, /* work array of size n */
    SSMProblem *PB  /* problem specification */
) ;

PRIVATE int SSMdiag /* return: 0 if convergence tolerance satisfied
                              -5 (algorithm did not converge)
                              -6 (insufficient space allocated for Givens) */
(
    FLOAT  *din, /* diagonal of tridiagonal matrix */
    FLOAT  *uin, /* superdiagonal of tridiagonal matrix, use uin (0:n-2) */
    int       n, /* dimensional of tridiagonal matrix */
    SSMDiag *DT, /* structure for storing diagonalization */
    int    *wi1, /* work array of size n */
    FLOAT  *wx1, /* work array of size n */
    FLOAT  *wx2, /* work array of size n */
    FLOAT  *wx3  /* work array of size n */
) ;

PRIVATE int SSMtridiag /* return 0 (process completes)
                               -7 (starting Lanczos vector vanishes) */
(
    FLOAT       *x, /* current solution estimate, ignored in Householder */
    int         it, /* iteration number */
    SSMLanczos *LH,
    SSMProblem *PB,
    SSMcom    *Com
) ;

PRIVATE void SSMtriHouse
(
    FLOAT *A , /* n-by-n matrix dense symmetric matrix (input)
                                dense orthogonal matrix (output) */
    FLOAT *d , /* n-by-1 vector (output) */
    FLOAT *u , /* n-by-1 vector (output) */
    FLOAT *x , /* n-by-1 vector (workspace) */
    int n
) ;

PRIVATE int SSMtriLanczos /* return 0 (process completes)
                                  -7 (starting Lanczos vector vanishes) */
(
    FLOAT *start_vector, /* starting point */
    SSMLanczos      *LH, /* Lanczos structure */
    SSMProblem      *PB, /* Problem specification */
    SSMcom         *Com  /* SSMcom structure */
) ;

PRIVATE void SSMdiagopt
(
    FLOAT        *x,  /* n-by-1 solution vector (output) */
    FLOAT         r,  /* radius of sphere */
    int BndLessThan,  /* TRUE means ||x|| <= r, FALSE means ||x|| = r */
    SSMDiagOpt  *DO,  /* diagonal optimization structure */
    SSMcom     *Com
) ;

PRIVATE FLOAT SSMdiagF
(
    FLOAT       mu,    /* the multiplier */
    FLOAT      *f2,    /* f_i^2 */
    FLOAT       *d,
    FLOAT       rr,    /* radius of sphere squared */
    int          n     /* dimension */
) ;

PRIVATE FLOAT SSMdiagF1
(
    FLOAT       mu,    /* the multiplier */
    FLOAT      *f2,    /* f_i^2 */
    FLOAT       *d,
    int          n     /* dimension */
) ;

PRIVATE void SSMmult
(
    FLOAT    *p, /* output vector of size n */
    FLOAT    *x, /* input vector of size n */
    FLOAT   *Ax, /* numerical values in A excluding diagonal */
    FLOAT    *D, /* diagonal of matrix */
    int     *Ai, /* row indices for each column of A */
    INT     *Ap, /* Ap [j] = start of column j */
    int       n, /* dimension of matrix */
    SSMcom *Com
) ;

PRIVATE void SSMGivensMult
(
    FLOAT      *x,     /* the vector to which the rotations are applied */
    SSMDiag    *DT     /* diagonalization structure for tridiagonal matrix */
) ;

PRIVATE void SSMtGivensMult
(
    FLOAT      *x,     /* the vector to which the rotations are applied */
    SSMDiag    *DT     /* diagonalization structure for tridiagonal matrix */
) ;

PRIVATE void SSMDenseMult
(
    FLOAT  *y,     /* m by 1 product, output */
    FLOAT  *x,     /* n by 1 given vector */
    FLOAT  *V,     /* dense m by n matrix */
    int     m,     /* number of rows */
    int     n      /* number of columns */
) ;

PRIVATE void SSMtDenseMult
(
    FLOAT  *y,     /* n by 1 product, output */
    FLOAT  *x,     /* m by 1 given vector */
    FLOAT  *V,     /* dense m by n matrix */
    int     m,     /* number of rows */
    int     n      /* number of columns */
) ;

PRIVATE void SSMSSORmultP
(
    FLOAT       *y,  /* the resulting vector */
    FLOAT       *b,  /* vector to be multiplied by SSOR matrix */
    FLOAT      *wj,  /* the first half of the SSOR multiplication operation */
    FLOAT       *w,  /* w = x/||x|| */
    FLOAT      *aj,  /* aj = Awj, product of A with wj */
    FLOAT       mu,  /* multiplier */
    int    startup,  /* = 1 for starting multiplication, 0 otherwise */
    SSMProblem *PB,  /* problem specification */
    SSMcom    *Com   /* SSMcom structure */
) ;

PRIVATE void SSMSSORmult
(
    FLOAT       *y,  /* the resulting vector */
    FLOAT       *b,  /* vector to be multiplied by SSOR matrix */
    FLOAT      *wj,  /* the first half of the SSOR multiplication operation */
    FLOAT      *aj,  /* aj = Awj, product of A with wj */
    FLOAT       *d,  /* diagonal of matrix */
    FLOAT       *s,  /* square root of d */
    FLOAT       mu,  /* diagonal safeguard */
    int    startup,  /* = TRUE (starting multiplication), FALSE (otherwise) */
    SSMProblem *PB,  /* problem specification */
    SSMcom    *Com   /* SSMcom structure */
) ;

PRIVATE int SSMboundary /* return 0 (error tolerance satisfied)
                                -1 (min residual convergence failure in SQP)
                                -2 (SSM failed to converge)
                                -3 (error decay in SSM too slow)
                                -5 (failure of QR diagonalization)
                                -6 (insufficient space in QR method)
                                -8 (solution in interior ||x|| < r) */
(
    FLOAT       *x, /* estimated solution to ball problem, a'x = 0 */
    SSMcom    *Com  /* SSMcom structure */
) ;

PRIVATE int SSMinterior /* return 0 (error tolerance satisfied)
                                -1 (minimum residual algorithm did not converge)
                                -5 (failure of QR diagonalization)
                                -6 (insufficient space in QR method)
                                -9 (solution on boundary ||x|| = r) */
(
    FLOAT       *x, /* estimated solution to sphere constrained problem */
    SSMcom    *Com  /* SSMcom structure */
) ;

PRIVATE int SSMminresP /* return 1 if convergence tolerance met,
                                 0 if decay tolerance met
                                -1 for min residual convergence failure
                                   (too many iterations) */
(
    FLOAT      *xj, /* computed SQP iterate */
    FLOAT      *r0, /* b + Ax, cost gradient at starting point */
    FLOAT       *x, /* solution estimate */
    FLOAT      *Ax, /* A*x */
    FLOAT       *b,
    FLOAT       *X, /* x/||x|| */
    FLOAT       mu, /* multiplier for the constraint */
    FLOAT      rad, /* radius r */
    SSMProblem *PB, /* problem specification */
    SSMcom    *Com  /* pointer to SSMcom structure */
) ;

PRIVATE int SSMminres /* return 0 if ||Ax + b|| <= ball_tol
                               -1 for convergence failure (too many iterations)
                               -2 if ||x|| > r */
(
    FLOAT      *xj, /* computed SQP iterate */
    FLOAT      *r0, /* b + Ax, cost gradient at starting point */
    FLOAT       *x, /* solution estimate */
    FLOAT      rad, /* radius of sphere */
    FLOAT       mu, /* safeguard, (A + mu I) positive definite */
    int        IPM, /* TRUE (inverse power method), FALSE (interior point) */
    SSMProblem *PB, /* problem specification */
    SSMcom    *Com  /* pointer to SSMcom structure */
) ;

PUBLIC int SSM /* return 0 (error tolerance satisfied)
                        -1 (min residual convergence failure in SQP)
                        -2 (SSM failed to converge in specified iteration)
                        -3 (number of SSM restarts exceeded limit)
                        -4 (dimension <= 0)
                        -5 (failure of QR diagonalization)
                        -6 (insufficient space allocated for QR diagonalization)
                        -7 (starting Lanczos vector vanishes) */
(
/* output: */
    FLOAT       *x, /* solution (output) */
    ssm_stat *Stat, /* NULL means do not return statistics */

/* input: */
    int          n, /* size of x */
    FLOAT      *Ax, /* numerical entries in matrix, packed array */
    int        *Ai, /* row indices for each column */
    INT        *Ap, /* points into Ax and Ai, Ap [j] = start of column j */
    FLOAT       *b, /* linear term in objective function */
    FLOAT        r, /* radius of ball */
    FLOAT      tol, /* 2-norm KKT error tolerance for solution */
    FLOAT   *guess, /* starting, NULL means no guess given */
    SSMParm *Uparm/* user parameters, NULL means use default parameters */
)
{
    
    int j, it, status, Allocate, PrintLevel ;
    FLOAT normx, t, *xnew, *v, *vnew ;
    SSMDiagOpt *DO ;
    SSMDiag *DT ;
    SSMLanczos *LH ;
    SSMParm *Parm, ParmStruc ;
    SSMProblem PB ;
    SSMcom Com ;

    /*print_matrix (Ai, Ap, Ax, "A", n) ;*/
    /* initialize counters for statistics */
    Com.ssm_its = Com.minres_its = 0 ;
    Com.mults = SSMZERO ;
    Com.cost_old = INF ;
    Com.emin_old = INF ;
    Com.error = INF ;
    status = 0 ;
    it = 0 ;

    /* no memory has been allocated */
    Allocate = SSMFALSE ;
    if ( n <= 0 )
    {
        status = -4 ;
        goto Exit ;
    }

    /* ---------------------------------------------------------------------- */
    /* set parameter values */
    /* ---------------------------------------------------------------------- */
    Parm = Uparm ;
    if ( Uparm == NULL)
    {
        Parm = &ParmStruc ;
        SSMdefault (Parm) ;
    }
    PrintLevel = Parm->PrintLevel ;

    /* ---------------------------------------------------------------------- */
    /* if Parm->PrintParms is true, then print parameter values */
    /* ---------------------------------------------------------------------- */
    if ( Parm->PrintParms ) SSMprint_parms (Parm) ;

    /* trivial case n = 1 */
    if ( n == 1 )
    {
        if ( Parm->BndLessThan ) /* ||x|| <= r */
        {
            if ( Ap [0] )
            {
                t = -b [0]/Ax [0] ;
                if ( fabs (t) > r ) t = r*t/fabs (t) ;
            }
            else if ( b [0] ) t = -r*b [0]/fabs (b [0]) ;
            else              t = SSMZERO ;
        }
        else
        {
            if ( Ap [0] )
            {
                t = -b [0]/Ax [0] ;
                if ( t != SSMZERO ) t = r*t/fabs (t) ;
                else                t = r ;
            }
            else if ( b [0] ) t = -r*b [0]/fabs (b [0]) ;
            else              t = r ;
        }
        x [0] = t ;
        Com.error = SSMZERO ;
        goto Exit ;
    }

    /* check if x = 0 is a solution */
    if ( Parm->BndLessThan )
    {
        t = SSMZERO ;
        for (j = 0; j < n; j++) t += b [j]*b [j] ;
        t = sqrt (t) ;
        if ( t <= tol )
        {
            for (j = 0; j < n; j++) x [j] = SSMZERO ;
            Com.error = t ;
            goto Exit ;
        }
    }

    Com.tol = tol ; /* KKT error tolerance for solution */

    /* set srand, if requested (default is 1) */
    if ( Parm->seed != 0 )
    {
        srand (Parm->seed) ;
    }

    /* allocate memory */
    SSMallocate (&Com, &PB, Parm, n) ;
    /* memory has been allocated */
    Allocate = SSMTRUE ;
    v = Com.v ;
    vnew = Com.MINRES+(5*n) ;
    xnew = Com.MINRES+(6*n) ;
    /* NOTE: Lanczos starting guess after a restart is computed by
             SSMreallocate and stored in Com.MINRES+(8*n) */

    /* allocate memory for PB and setup problem structure
       (sort cols, extract diagonal D and last nonzero in each column) */
    SSMinitProb (Ax, Ai, Ap, n, b, r, &PB, Parm, &Com) ;
    if ( PB.Amax == SSMZERO ) /* the matrix is zero */
    {
        t = SSMZERO ;
        for (j = 0; j < n; j++) t += b [j]*b [j] ;
        if ( t != SSMZERO )
        {
            t = r/sqrt (t) ;
            for (j = 0; j < n; j++) x [j] = -t*b [j] ;
        }
        else /* anything is optimal */
        {
            for (j = 0; j < n; j++) x [j] = SSMZERO ;
            x [0] = r ;
        }
        goto Exit ;
    }

    DO = PB.DO ;
    DT = PB.DT ;
    LH = PB.LH ;

    /* if SSM does not converge, then restart the tridiagonalization
       process in order to obtain a better starting guess for SSM.
       Below the variable "it" is the number of times the
       tridiagonalization has been restarted */

    for (it = 0; it < Parm->nrestart; it++)
    {
        /* tridiagonalize the matrix A, populate LH structure */
        if ( it == 0 ) status = SSMtridiag (guess, it, LH, &PB, &Com) ;
        else           status = SSMtridiag (    x, it, LH, &PB, &Com) ;
        if ( status ) goto Exit ; /* Lanczos starting vector vanishes */
 
        /* diagonalize the tridiagonal matrix, populate the DT structure */
        status = SSMdiag (LH->d, LH->u, LH->ncols, DT,
                          Com.wi1, Com.wx1, Com.wx2, Com.wx3) ;
        if ( status ) goto Exit ; /* failure of QR diagonalization */

        /* copy eigenvalues and dimension into the SSMDiagOpt structure*/
        DO->n = LH->ncols ;
        for (j = 0; j < LH->ncols; j++) DO->d [j] = DT->e [j] ;

        /* Compute (b)'V and store it in the SSMDiagOpt structure */
        SSMtDenseMult (DO->f, b, LH->V, n, LH->ncols) ;

        /* Multiply DO->f by the Givens rotations x' G_1 G_2 ... */
        SSMGivensMult (DO->f, DT) ;

        /* Solve reduced problem */
        SSMdiagopt (Com.wx3, r, Parm->BndLessThan, DO, &Com) ;
        Com.mu = DO->mu ;
        if ( DO->mu > SSMZERO ) Com.Active = SSMTRUE ;
        else                    Com.Active = SSMFALSE ;

        /* Multiply solution of (WP) by rotations G_1 G_2 ... x */
        SSMtGivensMult (Com.wx3, DT) ;

        /* Multiply by V and solve the subspace problem */
        if ( it == 0 ) /* no prior eigenvector estimate exists */
        {
            /* evaluate starting estimate of solution to sphere problem */
            SSMDenseMult (x, Com.wx3, LH->V, n, LH->ncols) ;

            /* estimate smallest eigenvalue and associated eigenvector */
            SSMmineig (&Com.emin, v, Com.wx1, &PB) ;

            /* evaluate kkt error and eigenvector residual */
            status = SSMsubspace (x, t, Com.Ax, v, NULL, NULL, NULL, 2,
                                  SSMTRUE, &Com);
            if ( status ) goto Exit ; /* failure of QR diagonalization */
        }
        else           /* preserve prior eigenvector estimate */
        {
            SSMDenseMult (xnew, Com.wx3, LH->V, n, LH->ncols) ;
            /* new eigenvector estimate */
            SSMmineig (&Com.emin, vnew, Com.wx1, &PB) ;

            /* compute ||x|| */
            normx = SSMZERO ;
            for (j = 0; j < n; j++)
            {
                t = x [j] ;
                normx += t*t ;
            }
            normx = sqrt (normx) ;

            /* solve subspace problem, basis: x, xnew, v, vnew */
            status = SSMsubspace (x, normx, Com.Ax, xnew, v, vnew,
                                  NULL, 4, SSMFALSE, &Com) ;
            if ( status ) goto Exit ; /* QR method convergence failure */
        }
        if ( PrintLevel >= 1 )
        {
            //printf ("\nbegin restart #: %i ball_err: %e ball_tol: %e\n",it, Com.error, Com.tol) ;
            //printf ("emin: %e mu: %e eig_err: %e\n",Com.emin, Com.mu, Com.eig_error) ;
        }

        /* for a small problem, we stop immediately */
        status = 0 ;
        if ( n <= 5 ) goto Exit ;
        if ( Com.error <= tol ) goto Exit ;

        /* refine solution */
        if ( Parm->BndLessThan && !Com.Active ) status = -8 ;
        else                                    status = -9 ;
        while ( status <= -8 )
        {
            if ( (status == -9) || (PB.Dmin < SSMZERO) )
            {
                if ( PrintLevel >= 1 ) //printf ("\nBOUNDARY SOLUTION\n") ;
                status = SSMboundary (x, &Com) ;
            }
            else
            {
                if ( PrintLevel >= 1 ) //printf ("\nINTERIOR SOLUTION\n") ;
                status = SSMinterior (x, &Com) ;
            }
            if ( PrintLevel >= 1 )
            {
                if ( status > -8 ) 
                {
                    //printf ("\ndone with restart #: %i error: %e status: %i\n",it, Com.error, status) ;
                }
                else
                {
                    //printf ("\ncontinue restart #: %i error: %e status: %i\n",it, Com.error, status) ;
                }
            }
            /* check if SSM failed to converge or QR method failed */
            if ( (status == -2) || (status == -5) || (status == -6) ) goto Exit;
        }

        /* convergence tolerance satisfied */
        if ( status == 0 ) goto Exit ;

        /* else SSM failed to converge, generate a larger tridiagonal matrix */
        SSMreallocate (x, &Com) ;
    }

    /* convergence not achieved within specified number of restarts */
    status = -3 ;

    Exit:
    if ( Stat != NULL )
    {
        Stat->mults = Com.mults ;
        Stat->ssm_its = Com.ssm_its ;
        Stat->minres_its = Com.minres_its ;
        Stat->restart_its = it ;
        Stat->error = Com.error ;
    }
    if ( Parm->PrintFinal || (PrintLevel >= 1) )
    {
        //printf ("\n") ;
        if ( status == 0 )
        {
            //printf ("Convergence tolerance statisfied\n" ) ;
        }
        else if ( status == -1 )
        {
            //printf ("Minimum residual algorithm failed to converge.\n") ;
            //printf ("The iteration limit %i was reached\n",(int) Com.minres_limit) ;
        }
        else if ( status == -2 )
        {
            //printf ("SSM failed to converge in %i iterations\n",(int) Com.ssm_limit);
        }
        else if ( status == -3 )
        {
           // printf ("SSM failed to converge within %i restart\n",Parm->nrestart) ;
        }
        else if ( status == -4 )
        {
            //printf ("The specified problem dimension %i < 0\n", (int) n) ;
        }
        else if ( status == -5 )
        {
            //printf ("The QR method was unable to diagonalize the ""tridiagonal matrix\n") ;
        }
        else if ( status == -6 )
        {
            //printf ("In the QR method, there was not enough space to store\n") ;
            //printf ("the Givens rotations. The parameter Parm->qr_its\n") ;
           // printf ("should be increased so that more space will be ""allocated.\n") ;
        }
        else if ( status == -7 )
        {
           // printf ("Starting vector in the Lanczos process vanishes\n") ;
        }
       /* printf ("\n") ;
        printf ("Number multiplications by matrix:%10i\n", (int) Com.mults) ;
        printf ("SSM iterations .................:%10i\n", (int) Com.ssm_its);
        printf ("Minimum residual iterations ....:%10i\n",
                (int) Com.minres_its);
        printf ("Number of Lanczos restarts .....:%10i\n", it);
        printf ("2-norm of KKT error ............:%10.3e\n", Com.error);
        */
    }
    /* destroy allocated memory */
    if ( Allocate ) SSMdestroy (&Com) ;
    return (status) ;
}

/* ==========================================================================
   === SSMdefault =========================================================== 
   ========================================================================== 
    Set default parameter values.
   ========================================================================== */

PUBLIC void SSMdefault
(
    SSMParm  *Parm /* pointer to parameter structure */
)
{
    FLOAT t, eps ;

    /* random number generator seed for srand */
    Parm->seed = 1 ;

    /* compute machine epsilon for FLOAT */
    eps = SSMONE ;
    t = SSMONE ;
    while ( t > 0 )
    {
        eps /= 2. ;
        t = SSMONE + eps ;
        t -= SSMONE ;
    }
    eps *= 2 ;
    Parm->eps = eps ;

    /* the number of Lanczos iterations L is given by the formula:
       if      ( n <= dim1 ) L = n-1 ;
       else if ( n <= dim2 ) L = L1 ;
       else                  L = L1 + L2*(log10 (n/dim2)) ; */
    Parm->Lanczos_dim1 = 30 ;
    Parm->Lanczos_dim2 = 100 ;
    Parm->Lanczos_L1 = 30 ;
    Parm->Lanczos_L2 = 80 ;
    Parm->grow_Lanczos = 1.3e0;    /*factor to grow size of Lanczos startup*/
    Parm->Lanczos_bnd = 10000 ;    /* upper bound on number Lanczos iterations*/
    Parm->Lanczos_tol = eps ;      /* stop Lanczos when |u_j| <= tol */
    Parm->House_limit = 30 ;       /* use Householder when n <= limit */
    Parm->qr_its = 4 ;             /* mult qr_its by dim for max # qr steps*/
    Parm->qr_tol_rel = eps ;       /* relative eigenvalue convergence tol*/
    Parm->qr_tol_abs = eps*1.e-3  ;/* absolute eigenvalue convergence tol*/

    /* lower bound for qr scaling factor; if this is made large, then
       allocation for DT->gi and DT->gs should be increased so that
       -log_2(qr_lower) < the factor in these allocation */
    Parm->qr_lower = 1.e-9 ;

    Parm->diag_optim_tol = 1.e5*eps ;/* accuracy of mult. for diag. matrix */

    /* if |d [i]-d [j]| <= diag_eps * n * absolute maximum diagonal element
       in diagopt, then d [i] and d [j] considered equal */
    Parm->diag_eps = 100*eps ;

    Parm->check_tol = 1.e-8 ;  /* tolerance used when checking SSM structure */
    Parm->check_kkt = 1.e-6 ;  /* KKT error tolerance is check_kkt*n */
    Parm->eig_refine = 5.e-1 ; /* factor determines when to refine eig in SSM */

    /* immediately switch from interior routine to boundary routine when
       interior iterate has norm >= radius_flex * sphere radius */
    Parm->radius_flex = 1.1 ;

    /* Factor which determines when to stop SQP iteration. Error in SQP
       system <= sqp_decay times KKT error */
    Parm->sqp_decay = 1.e-2 ;

    /* Factor which determines when to stop inverse power iteration.
       Norm of the residual <= eig_decay */
    Parm->eig_decay = 1.e-2 ;

    /* eig_lower is an actual lower bound for the smallest eigenvalue,
       not an estimate. Such a lower bound is useful in the safeguarding
       process */
    Parm->eig_lower = -INF ;

    /* Factor which determines when to restart iteration. Error in SSM
       should decay by ssm_decay in one iteration, otherwise perform
       more Lanczos iterations */
    Parm->ssm_decay = 8.e-1 ;

    /* when optimal multiplier mu in diagopt is near zero, shrink to zero
       by factor shrink in each iteration */
    Parm->shrink = 1.e-1 ;     /* mu multiplied by shrink in diagopt */
    Parm->BndLessThan = SSMTRUE ;/* constraint is ||x|| <= r */

    /* TRUE  (use inverse power method to estimate smallest eigenvalue)
       FALSE (use SQP method to estimate smallest eigenvalue) */
    Parm->IPM = SSMFALSE ;

    Parm->minres_its_fac = 3 ;  /* max number MINRES iterations is fac*n */
    Parm->ssm_its_fac = 1 ;     /* max number SSM iterations is fac*n */
    Parm->nrestart = 40 ;       /* number of Lanczos restarts attempted */

    /* PrintLevel = 0, 1, 2, 3
       PrintLevel = 3 gives maximum printing of iteration data while
       PrintLevel = 0 gives no printing of iteration data */
    Parm->PrintLevel = 0 ;

    /* PrintParms = TRUE  means to print parameter values */
    Parm->PrintParms = SSMTRUE ;

    /* PrintFinal = TRUE means to print final status and statistics */
    Parm->PrintFinal = SSMTRUE ;
}

/* ==========================================================================
   === SSMallocate ==========================================================
   ==========================================================================
   Allocate memory for the SSM structures and copy parameter values into SSMcom 
   ==========================================================================*/

PRIVATE void SSMallocate
(
    SSMcom    *Com, /* pointer to SSMcom structure */
    SSMProblem *PB, /* problem specification */
    SSMParm  *Parm, /* parameters, needed to determine allocation */
    int          n  /* problem dimension */
)
{
    INT L, N, qr_its ;
    SSMDiag *DT ;
    SSMLanczos *LH ;
    SSMDiagOpt *DO ;
    SSMProblem *P ;

    /* malloc work arrays */
    Com->wi1 = malloc (n*sizeof (int)) ;
    Com->wi2 = malloc (n*sizeof (int)) ;
    Com->wx1 = malloc (3*n*sizeof (FLOAT)) ;
    Com->wx2 = (Com->wx1)+n ;
    Com->wx3 = (Com->wx2)+n ;
    Com->v = malloc (n*sizeof (FLOAT)) ;  /* estimate of eigenvector */
    Com->Av = malloc (n*sizeof (FLOAT)) ; /* A times v */
    Com->Ax = malloc (n*sizeof (FLOAT)) ; /* A times x */
    Com->r0 = malloc (n*sizeof (FLOAT)) ; /* r0 = Ax + b */

    /* create Lanczos/Householder structure */
    LH = malloc (sizeof (SSMLanczos)) ;
    PB->LH = LH ;
    /* max number of Lanczos iterations, needed for mallocs */
    if ( n <= Parm->House_limit ) L = n ; /* need space for full matrix */
    else if ( n <= Parm->Lanczos_dim1 ) L = n-1 ;
    else if ( n <= Parm->Lanczos_dim2 ) L = Parm->Lanczos_L1 ;
    else L = Parm->Lanczos_L1 + Parm->Lanczos_L2*
             (log10 ((double) n/ (double) Parm->Lanczos_dim2)) ;
    /* allocate space for Lanczos vectors/Householder vectors */
    LH->V = malloc (L*n*sizeof (FLOAT)) ;
    LH->d = malloc (L*sizeof (FLOAT)) ;   /* diagonal of tridiagonal matrix */
    LH->u = malloc (L*sizeof (FLOAT)) ;   /* superdiagonal of tridiag. matrix */

    /* create the diagonalization structure for tridiagonal matrix */
    DT = malloc (sizeof (SSMDiag)) ;
    PB->DT = DT ;
    DT->e = malloc (L*sizeof (FLOAT)) ;  /* eigenvalues */

    /* max number of qr iteration */
    qr_its = (Parm->qr_its)*L ;          /* max QR algorithm iterations */
    DT->nalloc = L ;
    DT->qr_lower = Parm->qr_lower ;
    DT->gx = malloc (qr_its*L*sizeof (FLOAT)) ;
    DT->gy = malloc (qr_its*L*sizeof (FLOAT)) ;
    DT->gz = malloc (qr_its*L*sizeof (short)) ;
    /* space for ending -1 in each iteration, the L final rescaling at end,
       and the in-iteration recaling which could occur at most every
       30 steps when qr_lower = 1.e-9 */
    DT->gs = malloc ((qr_its + L + 5 + L*(qr_its/30))*sizeof (FLOAT)) ;
    DT->gi = malloc ((qr_its + L + 5 + L*(qr_its/30))*sizeof (int)) ;
    DT->gj = malloc (qr_its*sizeof (INT)) ;
    DT->gk = malloc (qr_its*sizeof (int)) ;
    DT->gf = malloc (qr_its*sizeof (int)) ;

    /* create the diagonal matrix optimization structure */
    DO = malloc (sizeof (SSMDiagOpt)) ;
    PB->DO = DO ;
    DO->d = malloc (L*sizeof (FLOAT)) ;
    DO->f = malloc (L*sizeof (FLOAT)) ;
    DO->f2= malloc (L*sizeof (FLOAT)) ;
    DO->dshift = malloc (L*sizeof (FLOAT)) ;

    /* storage for d, s, p, q arrays used in SSOR iteration*/
    Com->SSOR = malloc (4*n*sizeof (FLOAT)) ;

    /*for AV (5n) or MINRES y, Ay, r, aj, vj, vj1, wj, pr, rj1, rj2, zj1, zj2*/
    Com->MINRES = malloc (12*n*sizeof (FLOAT)) ;

    Com->W = malloc (5*n*sizeof (FLOAT)) ;
    Com->V = malloc (5*n*sizeof (FLOAT)) ;
    Com->VAV = malloc (25*sizeof (FLOAT)) ;

    /* set up a dense problem structure for matrices of size up to 5 by 5 */
    P = malloc (sizeof (SSMProblem)) ;
    Com->PBdense = P ;
    N = 5 ;
    P->b = malloc (N*sizeof (FLOAT)) ;

    /* create Lanczos/Householder structure */
    LH = malloc (sizeof (SSMLanczos)) ;
    P->LH = LH ;
    /* default number of Lanczos iterations, needed for mallocs */
    LH->V = malloc (N*N*sizeof (FLOAT)) ;/* orthonormal vectors */
    LH->d = malloc (N*sizeof (FLOAT)) ;  /* diagonal of tridiagonal matrix*/
    LH->u = malloc (N*sizeof (FLOAT)) ;  /* superdiag. of tridiag. matrix */

    /* create the diagonalization structure for tridiagonal matrix */
    DT = malloc (sizeof (SSMDiag)) ;
    P->DT = DT ;
    DT->e = malloc (N*sizeof (FLOAT)) ;/* eigenvalues */
    qr_its = 30 ;
    DT->nalloc = N ;
    DT->qr_lower = Parm->qr_lower ;
    DT->gx = malloc (qr_its*N*sizeof (FLOAT)) ;
    DT->gy = malloc (qr_its*N*sizeof (FLOAT)) ;
    DT->gz = malloc (qr_its*N*sizeof (short)) ;
    DT->gs = malloc ((2*qr_its + N + 5 + N*(qr_its/30))*sizeof (FLOAT)) ;
    DT->gi = malloc ((2*qr_its + N + 5 + N*(qr_its/30))*sizeof (int)) ;
    DT->gj = malloc (qr_its*sizeof (INT)) ;
    DT->gk = malloc (qr_its*sizeof (int)) ;
    DT->gf = malloc (qr_its*sizeof (int)) ;

    /* create the diagonal matrix optimization structure */
    DO = malloc (sizeof (SSMDiagOpt)) ;
    P->DO = DO ;
    DO->d = malloc (N*sizeof (FLOAT)) ;
    DO->f = malloc (N*sizeof (FLOAT)) ;
    DO->f2= malloc (N*sizeof (FLOAT)) ;
    DO->dshift = malloc (N*sizeof (FLOAT)) ;
}

/* ==========================================================================
   === SSMreallocate ========================================================
   ==========================================================================
   SSM failed to converge. Tridiagonalize a larger matrix in order to
   obtain a better starting guess. Need to reallocate storage for the
   Lanczos process.
   ==========================================================================*/

PRIVATE void SSMreallocate
(
    FLOAT       *x, /* current estimate of solution */
    SSMcom    *Com  /* SSMcom structure */
)
{
    INT j, L, n, ncols, qr_its ;
    FLOAT mu, *Vj, *start_vector ;
    SSMDiag *DT ;
    SSMLanczos *LH ;
    SSMDiagOpt *DO ;
    SSMProblem *PB ;
    SSMParm *Parm ;

    PB = Com->PB ;
    Parm = Com->Parm ;
    n = PB->n ;
    LH = PB->LH ;
    DT = PB->DT ;
    DO = PB->DO ;
    L = DT->nalloc ;

    /* max number of Lanczos iterations, needed for mallocs */
    L *= Parm->grow_Lanczos ;
    if ( L >= n ) L = n ;
    else /* compute starting guess for Lanczos process */
    {
        L = SSMMIN (L, Parm->Lanczos_bnd) ;
        start_vector = Com->MINRES+(8*n) ;
        mu = Com->mu ;
        for (j = 0; j < n; j++) start_vector [j] = Com->r0 [j] + mu*x [j] ;
        /* remove projections on prior Lanczos vectors */
        ncols = LH->ncols ;
        Vj = LH->V ;

/*      for (j = 0; j < ncols; j++)
        {
            t = SSMZERO ;
            for (i = 0; i < n; i++) t += start_vector [i]*Vj [i] ;
            for (i = 0; i < n; i++) start_vector [i] -= t*Vj [i] ;
        } */
    }
    if ( DT->nalloc < L )
    {
        DT->nalloc = L ;
        /* free the prior Lanczo/Householder structure */
        free (LH->d) ;
        free (LH->u) ;
        free (LH->V) ;

        /* create new Lanczos structure */
        LH->V = malloc (L*n*sizeof (FLOAT)) ;/* orthonormal vectors */
        LH->d = malloc (L*sizeof (FLOAT)) ;/* diagonal of tridiagonal matrix */
        LH->u = malloc (L*sizeof (FLOAT)) ;/* superdiagonal of tridiag. matrix*/
        LH->max_its = L ;

        /* free the prior tridiagonalization structure */
        free (DT->e) ;
        free (DT->gx) ;
        free (DT->gy) ;
        free (DT->gz) ;
        free (DT->gs) ;
        free (DT->gi) ;
        free (DT->gj) ;
        free (DT->gk) ;
        free (DT->gf) ;
        /* create the diagonalization structure for tridiagonal matrix */
        DT->e = malloc (L*sizeof (FLOAT)) ;  /* eigenvalues */
        qr_its = (Parm->qr_its)*L ;          /* max QR algorithm iterations */
        DT->max_its = SSMMAX (30, qr_its*L) ;
        DT->gx = malloc (qr_its*L*sizeof (FLOAT)) ;
        DT->gy = malloc (qr_its*L*sizeof (FLOAT)) ;
        DT->gz = malloc (qr_its*L*sizeof (short)) ;
        DT->gs = malloc ((qr_its + L + 5 + L*(qr_its/30))*sizeof (FLOAT)) ;
        DT->gi = malloc ((qr_its + L + 5 + L*(qr_its/30))*sizeof (int)) ;
        DT->gj = malloc (qr_its*sizeof (INT)) ;
        DT->gk = malloc (qr_its*sizeof (int)) ;
        DT->gf = malloc (qr_its*sizeof (int)) ;

        /* free the prior diagonal optimization structure */
        free (DO->d) ;
        free (DO->f) ;
        free (DO->f2) ;
        free (DO->dshift) ;
        /* create the diagonal matrix optimization structure */
        DO->d = malloc (L*sizeof (FLOAT)) ;
        DO->f = malloc (L*sizeof (FLOAT)) ;
        DO->f2= malloc (L*sizeof (FLOAT)) ;
        DO->dshift = malloc (L*sizeof (FLOAT)) ;
    }
}

/* ==========================================================================
   === SSMdestroy ===========================================================
   ==========================================================================
    Free the storage memory
   ========================================================================== */
PRIVATE void SSMdestroy
(
    SSMcom *Com  /* SSMcom structure to free */
)
{
    SSMProblem *PB, *P ;

    free (Com->wi1) ;
    free (Com->wi2) ;
    free (Com->wx1) ;
    free (Com->v) ;
    free (Com->Av) ;
    free (Com->Ax) ;
    free (Com->r0) ;

    PB = Com->PB ;
    free (PB->D) ;
    free (PB->i) ;
    free (PB->p) ;
    free (PB->u) ;
    free (PB->x) ;
    free (PB->LH->V) ;
    free (PB->LH->d) ;
    free (PB->LH->u) ;
    free (PB->LH) ;

    free (PB->DT->e) ;
    free (PB->DT->gx) ;
    free (PB->DT->gy) ;
    free (PB->DT->gz) ;
    free (PB->DT->gs) ;
    free (PB->DT->gi) ;
    free (PB->DT->gj) ;
    free (PB->DT->gk) ;
    free (PB->DT->gf) ;
    free (PB->DT) ;

    free (PB->DO->d) ;
    free (PB->DO->f) ;
    free (PB->DO->f2) ;
    free (PB->DO->dshift) ;
    free (PB->DO) ;

    P = Com->PBdense ;
    free (P->b) ;
    /* destroy Lanczos/Householder structure */
    free (P->LH->V) ;
    free (P->LH->d) ;
    free (P->LH->u) ;
    free (P->LH) ;

    /* destroy the diagonalization structure for tridiagonal matrix */
    free (P->DT->e) ;
    free (P->DT->gx) ;
    free (P->DT->gy) ;
    free (P->DT->gz) ;
    free (P->DT->gs) ;
    free (P->DT->gi) ;
    free (P->DT->gj) ;
    free (P->DT->gk) ;
    free (P->DT->gf) ;
    free (P->DT) ;

    /* destroy the diagonal matrix optimization structure */
    free (P->DO->d) ;
    free (P->DO->f) ;
    free (P->DO->f2) ;
    free (P->DO->dshift) ;
    free (P->DO) ;
    free (P) ;

    free (Com->MINRES) ;
    free (Com->W) ;
    free (Com->V) ;
    free (Com->VAV) ;
    if ( Com->SSOR != NULL ) free (Com->SSOR) ;
}

/* ==========================================================================
   === SSMinitProb ==========================================================
   ==========================================================================
    Copy matrix, order column, extract diagonal, locate first element
    beneath diagonal
   ========================================================================== */
PRIVATE void SSMinitProb
(
    FLOAT      *Ax, /* numerical entries in matrix, packed array */
    int        *Ai, /* row indices for each column */
    INT        *Ap, /* points into Ax and Ai, Ap [0], ... Ap [n], packed */
    int          n, /* problem dimension */
    FLOAT       *b, /* linear term in objective function */
    FLOAT      rad, /* radius of sphere */
    SSMProblem *PB, /* problem structure */
    SSMParm  *Parm, /* parameter structure */
    SSMcom    *Com  /* SSMcom structure */
)
{
    INT k, l, nnz, p, *Ap1, *Bp, *Bp1, *Bu ;
    int ai, j, *Bi, *wi ;
    FLOAT ax, Amax, Dmin, dj, *Bx, *D ;
    SSMLanczos *LH ;
    SSMDiag *DT ;
    SSMDiagOpt *DO ;

    PB->n = n ;
    PB->rad = rad ;
    PB->b = b ;
    Com->minres_limit = (int) (((FLOAT) n)*Parm->minres_its_fac) ;
    Com->ssm_limit = (int) (((FLOAT) n)*Parm->ssm_its_fac) ;

    /* flag nonzero diagonal elements in matrix, diagonal stored in
       separate array PB->D */
    wi = Com->wi1 ;
    Ap1 = Ap+1 ;
    k = 0 ;
    for (j = 0; j < n; j++)
    {
        l = Ap1 [j] ;
        wi [j] = 0 ;
        for (; k < l; k++)
        {
            if ( Ai [k] == j )
            {
                wi [j] = 1 ;
                k = l ;
                break ;
            }
        }
    }

    Bp = PB->p = malloc ((n+1)*sizeof (INT)) ;
    Bp1 = Bp+1 ;
    p = 0 ;
    nnz = 0 ;
    for (j = 0; j < n; j++)
    {
        /* number of nonzero diagonal elements up to end of column j */
        p += wi [j] ;
        Bp [j] = nnz ;
        /* number nonzeros up to end of column j excluding diagonal elements*/
        nnz = Ap1 [j] - p ;
    }
    Bp [n] = nnz ;

    /* allocate memory for matrix storage */
    D  = PB->D = malloc (n*sizeof (FLOAT)) ;
    Bu = PB->u = malloc (n*sizeof (INT)) ;
    Bi = PB->i = malloc (nnz*sizeof (int)) ;
    Bx = PB->x = malloc (nnz*sizeof (FLOAT)) ;

    /* ======================================================================
       === B = A'============================================================
       ======================================================================
          - this transpose operation ensures that columns of matrix are sorted
          - store diagonal in PB->D
          - compute absolute largest element of A
       ====================================================================== */

    Amax = SSMZERO ;
    Dmin = INF ;
    k = 0 ;
    for (j = 0; j < n; j++)
    {
        l = Ap1 [j] ;
        dj = SSMZERO ;
        for (; k < l; k++)
        {
            ai = Ai [k] ;
            ax = Ax [k] ;
            if ( fabs (ax) > Amax ) Amax = fabs (ax) ;
            if ( ai == j )  /* store the diagonal in D, not  Bx */
            {
                dj = ax ;
            }
            else
            {
                p = Bp [ai]++ ;
                Bi [p] = j ;
                Bx [p] = ax ;
            }
        }
        D [j] = dj ;
        if ( dj < Dmin ) Dmin = dj ;
    }
    PB->Amax = Amax ;
    PB->Dmin = Dmin ;
    for (j = n; j > 0; j--) Bp [j] = Bp [j-1] ;
    Bp [0] = 0 ;

    /* Bu points just past last nonzero above diagonal in each column */
    Bu = PB->u ;
    k = 0 ;
    for (j = 0; j < n; j++)
    {
        l = Bp1 [j] ;
        Bu [j] = l ;
        for (; k < l; k++)
        {
            ai = Bi [k] ;
            if ( Bi [k] > j )
            {
                Bu [j] = k ;
                k = l ;
                break ;
            }
        }
    }

    /* store Parm and PB in Com */
    Com->Parm = Parm ;
    Com->PB = PB ;

    /* store Lanczos parameters in LH structure */
    LH = PB->LH ;
    LH->tol = Amax*Parm->Lanczos_tol ;   /*u_j termination tol in Lanczos iter*/
    LH->House_limit = Parm->House_limit ;
    LH->nrows = n ;

    /* store diagonalization and tridiagonalization parameters in DT structure*/
    DT = PB->DT ;
    LH->max_its = DT->nalloc ;
    DT->tol1 = Parm->qr_tol_rel ;        /* relative tolerance for QR method */
    DT->tol2 = Parm->qr_tol_abs ;        /* absolute tolerance for QR method */
    DT->max_its = SSMMAX (30, Parm->qr_its*DT->nalloc) ;

    /* store diagonal optimization parameter in DO structure*/
    DO = PB->DO ;
    DO->tol = Parm->diag_optim_tol ;

    /* now store parameters for the 5 by 5 matrices arising in refinement */
    DT = Com->PBdense->DT ;
    DT->tol1 = Parm->qr_tol_rel ;      /* relative tolerance for QR method*/
    DT->tol2 = Parm->qr_tol_abs ;      /* absolute tolerance for QR method*/
    DT->max_its = SSMMAX (30, Parm->qr_its*5) ;

    DO = Com->PBdense->DO ;
    DO->tol = Parm->diag_optim_tol ;
}

/* ==========================================================================
   === SSMprint_parms =======================================================
   ==========================================================================
   print values in SSMparm structure
   ========================================================================== */
PRIVATE void SSMprint_parms
(
    SSMParm *Parm /* SSMparm structure to be printed */
)
{
    /* numerical parameters
    printf ("print level (0 = none, 3 = maximum) .......... PrintLevel: %i\n",Parm->PrintLevel) ;
    printf ("machine epsilon ..................................... eps: %e\n",
             Parm->eps) ;
    printf ("subdiagonal tolerance in Lanczos iteration .. Lanczos_tol: %e\n",
             Parm->Lanczos_tol) ;
    printf ("change from Householder to Lanczos at dim ... House_limit: %i\n",
             (int) Parm->House_limit) ;
    printf ("relative eigenvalue convergence tol .......... qr_tol_rel: %e\n",
             Parm->qr_tol_rel) ;
    printf ("absolute eigenvalue convergence tol .......... qr_tol_abs: %e\n",
             Parm->qr_tol_abs) ;
    printf ("lower bound for QR scaling factor .............. qr_lower: %e\n",
             Parm->qr_lower) ;
    printf ("max number QR algorithm iterations qr_its * dim .. qr_its: %i\n",
             (int) Parm->qr_its) ;
    printf ("multiplier accuracy for diagonal matrix .. diag_optim_tol: %e\n",
             Parm->diag_optim_tol) ;
    printf ("accuracy of diag in tridiagonal matrix .... diag_diag_eps: %e\n",
             Parm->diag_eps) ;
    printf ("tolerance used when checking cost decay ....... check_tol: %e\n",
             Parm->check_tol) ;
    printf ("tolerance used when checking KKT error ........ check_kkt: %e\n",
             Parm->check_kkt) ;
    printf ("radius_flex times radius ok in interior ..... radius_flex: %e\n",
             Parm->radius_flex) ;
    printf ("error decay factor in SQP iteration ........... sqp_decay: %e\n",
             Parm->sqp_decay) ;
    printf ("error decay factor in inverse power iteration . eig_decay: %e\n",
             Parm->eig_decay) ;
    printf ("all eigenvalues of matrix are >= .............. eig_lower: %e\n",
             Parm->eig_lower) ;
    printf ("error decay factor in SSM iteration ........... ssm_decay: %e\n",
             Parm->ssm_decay) ;
    printf ("mu multiplication factor if mu = 0 possible ...... shrink: %e\n",
             Parm->shrink) ;
    printf ("max number MINRES iterations is fac*n .... minres_its_fac: %e\n",
             Parm->minres_its_fac) ;
    printf ("max number SSM iterations is fac*n .......... ssm_its_fac: %e\n",
             Parm->ssm_its_fac) ;
    printf ("upper bound on number of Lanczos iterations . Lanczos_bnd: %i\n",
             (int) Parm->Lanczos_bnd) ;
    printf ("restart Lanczos, iterations grow by factor . grow_Lanczos: %e\n",
             Parm->grow_Lanczos) ;
    printf ("number of Lanczos restarts attempted ........... nrestart: %i\n",
             Parm->nrestart) ;

    printf ("\n") ;
    printf ("The number of Lanczos iterations L is given by the formula:\n"
            "    if      ( n <= dim1 ) L = n-1\n"
            "    else if ( n <= dim2 ) L = L1\n"
            "    else                  L = L1 + L2*(log10 (n/dim2)\n") ;
    printf ("...................................................  dim1: %i\n",
             (int) Parm->Lanczos_dim1) ;
    printf ("...................................................  dim2: %i\n",
             (int) Parm->Lanczos_dim2) ;
    printf ("...................................................    L1: %i\n",
             (int) Parm->Lanczos_L1) ;
    printf ("...................................................    L2: %i\n",
             (int) Parm->Lanczos_L2) ;

    // logical parameters
    printf ("\nLogical parameters:\n") ;
    if ( Parm->BndLessThan == SSMTRUE )
        printf ("    Constraint is ||x|| <= r\n") ;
    else
        printf ("    Constraint is ||x|| = r\n") ;
    if ( Parm->PrintParms == SSMTRUE )
        printf ("    Print the parameter structure\n") ;
    else
        printf ("    Do not print parameter structure\n") ;
    if ( Parm->PrintFinal == SSMTRUE )
        printf ("    Print final status and statistics\n") ;
    else
        printf ("    Do not print final status and statistics\n") ;
    if ( Parm->IPM == SSMTRUE )
        printf ("    Use inverse power method to estimate smallest "
                     "eigenvalue\n") ;
    else
        printf ("    Use SQP method to estimate smallest eigenvalue\n") ;
     */
}

/* ==========================================================================
   === SSMballdense ==========================================================
   ==========================================================================

   Solve a dense sphere constrained quadratic program of the form

       min  x'Ax + 2b'x  subject to ||x|| = r
   ========================================================================== */

PRIVATE int SSMballdense /*return 0 (error tolerance satisfied)
                                 -5 (failure of QR diagonalization)
                                 -6 (insufficient space in QR diagonalization)*/
(
    FLOAT        *x, /* n-by-1 solution vector (output) */
    SSMProblem  *PB, /* problem structure associated with A */
    SSMcom      *Com /* SSMcom structure */
)
{
    int j, n, status ;
    FLOAT  *b, *A ;
    SSMDiag  *DT ;
    SSMDiagOpt *DO ;
    SSMLanczos *LH ;

    DT = PB->DT ;
    DO = PB->DO ;
    LH = PB->LH ;
    b = PB->b ;
    A = LH->V ;
    n = PB->n ;
    SSMtriHouse (A, LH->d, LH->u, Com->wx1, n) ;
    LH->ncols = n ;
    status = SSMdiag (LH->d, LH->u, n, DT,
                      Com->wi1, Com->wx1, Com->wx2, Com->wx3) ;

    /* copy the eigenvalues and the dimension into the SSMDiagOpt structure*/
    DO->n = n ;
    for (j = 0; j < n; j++) DO->d [j] = DT->e [j] ;
    SSMtDenseMult (DO->f, b, LH->V, n, n) ;
    SSMGivensMult (DO->f, DT) ;
    SSMdiagopt (Com->wx3, Com->PB->rad, Com->Parm->BndLessThan, DO, Com) ;
    Com->Active = SSMTRUE ;
    if ( Com->Parm->BndLessThan && (DO->mu <= SSMZERO) ) Com->Active = SSMFALSE;
    SSMtGivensMult (Com->wx3, DT) ;
    SSMDenseMult (x, Com->wx3, LH->V, n, n) ;
    return (status) ;
}

/* ==========================================================================
   === SSMsubspace ==========================================================
   ==========================================================================
   Minimize the quadratic over a subspace spanned by m vectors, m <= 5.
   Since ||v1|| and the product A*v1 have generally been computed already,
   they are provided as input parameters to avoid their recomputation.
   NOTE: if only_residual is TRUE, it is assumed that v1 contains x
   ========================================================================== */
PUBLIC int SSMsubspace /* return 0 (error tolerance satisfied)
                                -5 (failure of QR diagonalization)
                                -6 (insufficient space in QR diagonalization)*/
(
    FLOAT         *v1, /* subspace vector 1, solution is returned in v1 */
    FLOAT      normv1, /* norm of vector 1 */
    FLOAT        *Av1, /* A*v1 */
    FLOAT         *v2, /* subspace vector 2 */
    FLOAT         *v3, /* subspace vector 3 */
    FLOAT         *v4, /* subspace vector 4 */
    FLOAT         *v5, /* subspace vector 5 */
    int             m, /* number of vectors */
    int only_residual, /* only compute kkt error and eigenvector residual */
    SSMcom       *Com
)
{
    int BndLessThan ;
    int i, j, n, status ;
    FLOAT rad, r, s, t, u, X [5], evec [5], mu, ball_err, normx,
          *V, *W, *AV, *VAV, *VAVsave, *b, *v, *Av, *Ax, *r0, *emin, *x ;
    SSMProblem *PB, *P ;

    PB = Com->PB ;
    b = PB->b ;
    n = PB->n ;
    rad = PB->rad ;
    P = Com->PBdense ;
    P->n = m ;
    V = Com->V ;        /* storage for orthogonal vectors */
    W = Com->W ;        /* storage for Householder vectors */
    AV= Com->MINRES ;   /* storage for A*V */
    VAV= P->LH->V ;     /* storage for V'AV */
    VAVsave = Com->VAV ;/* save copy of V'AV in case eigenvector refined */
    Ax = Com->Ax ;
    Av = Com->Av ;
    v  = Com->v ;
    r0 = Com->r0 ;
    emin = &Com->emin ;
    BndLessThan = Com->Parm->BndLessThan ;
    x = v1 ;            /* solution is return in array x = v1 */
    if ( only_residual == SSMTRUE )
    {
        /* if constraint active or ||x|| > rad, then normalize x
           NOTE: x = v1 */
        normx = SSMZERO ;
        for (j = 0; j < n; j++)
        {
            t = x [j] ;
            normx += t*t ;
        }
        normx = sqrt (normx) ;
        if ( (PB->DO->mu != SSMZERO) || !BndLessThan || (normx > rad) )
        {
            if ( normx != SSMZERO )
            {
                t = rad/normx ;
                for (j = 0; j < n; j++) x [j] *= t ;
            }
        }
        SSMmult (Ax, x, PB->x, PB->D, PB->i, PB->p, n, Com) ;
        SSMmult (Av, v, PB->x, PB->D, PB->i, PB->p, n, Com) ;
        P->DO->mu = PB->DO->mu ;
        goto Residual_computation ;
    }
    if ( m < 5 )
    {

        /* compute orthonormal basis for given basis vectors */
        t = SSMONE/sqrt (normv1*(normv1 + fabs (v1 [0]))) ;
        if ( v1 [0] > SSMZERO ) W [0] = t*(v1 [0] + normv1) ;
        else                    W [0] = t*(v1 [0] - normv1) ;
        for (j = 1; j < n; j++) W [j] = v1 [j]*t ; /* H = I - w*w' */
        if ( m >= 2 ) SSMorth (V+(1*n), W, v2, 1, n) ;
        if ( m >= 3 ) SSMorth (V+(2*n), W, v3, 2, n) ;
        if ( m >= 4 ) SSMorth (V+(3*n), W, v4, 3, n) ;

        /* normalize v1 and A*v1, they are first column of V and AV */
        t = SSMONE/normv1 ;
        if ( t != SSMONE )
        {
            for (j = 0; j < n; j++) AV [j] = t*Av1 [j] ;
            for (j = 0; j < n; j++)  V [j] = t* v1 [j] ;
        }
        else
        {
            for (j = 0; j < n; j++) AV [j] =   Av1 [j] ;
            for (j = 0; j < n; j++)  V [j] =    v1 [j] ;
        }

        /* multiply remaining columns of V by A */
        for (i = 1; i < m; i++)
            SSMmult (AV+(i*n), V+(i*n), PB->x, PB->D, PB->i, PB->p, n, Com) ;
    
        /* compute elements of V'AV on or below diagonal */
        for (i = 0; i < m; i++)
        {
            SSMtDenseMult (VAV+(i*m + i), V+(i*n), AV+(i*n), n, m-i) ;
        }
    
        /* elements above diagonal known from symmetry */
        for (j = 0; j < m-1; j++)
        {
            for (i = j+1; i < m; i++)
            {
                /* (i,j) element inserted in location (j,i) above diagonal */
                VAV [m*i+j] = VAV [m*j+i] ;
            }
        }
    
        /* save VAV in case we also compute a refined eigenvector */
        for (i = 0; i < m*m; i++) VAVsave [i] = VAV [i] ;
    
        /* compute linear term */
        SSMtDenseMult (P->b, b, V, n, m) ; /* linear term P->b = V'b */
    
        /* subproblem dimension is m */
        P->n = m ;
    
        /* X = solution of dense QP */
        status = SSMballdense (X, P, Com) ;
        if ( status ) return (status) ;
        /*x = VX, solution x is returned in array v1 (x = v1)*/
        SSMDenseMult (x, X, V, n, m) ;
        /* compute Ax */
        SSMmult (Ax, x, PB->x, PB->D, PB->i, PB->p, n, Com) ;
    
        /* smallest eigenpair of V'AV */
        SSMmineig (emin, evec, Com->wx1, P) ;
        SSMDenseMult (v, evec, V, n, m) ;/* v = V*evec */
        SSMmult (Av, v, PB->x, PB->D, PB->i, PB->p, n, Com) ; /* Av */
    }
    else /* special case where the eigenvector was refined */
    {
        /* vector v5 used to generate 5th orthonormal column of V */
        SSMorth (V+(4*n), W, v5, 4, n) ;
        /* multiply 5th column of V by A to obtain 5th column of AV */
        SSMmult (AV+(4*n), V+(4*n), PB->x, PB->D, PB->i, PB->p, n, Com) ;
        /* scatter 4 by 4 matrix in VAVsave into 5 by 5 matrix VAV */
        for (j = 3; j >= 0; j--)
        {
            for (i = 3; i >= 0; i--) VAV [5*j+i] = VAVsave [4*j+i] ;
        }
        /* compute 5th column of 5 by 5 matrix of VAV */
        SSMtDenseMult (VAV+20, AV+(4*n), V, n, 5) ;
    
        /* use symmetry to generate 5th row of VAV */
        VAV [4]  = VAV [20] ; VAV [9]  = VAV [21] ;
        VAV [14] = VAV [22] ; VAV [19] = VAV [23] ;

        /* due to the 5th column of V, there is a 5th component of P->b */
        SSMtDenseMult ((P->b)+4, b, V+(4*n), n, 1) ;
        P->n = 5 ;
        /* X = solution of dense QP */
        status = SSMballdense (X, P, Com) ;
        if ( status ) return (status) ;

        /* x = VX, return solution in v1 (x = v1) */
        SSMDenseMult (x, X, V, n, 5) ;
        /* compute Ax */
        SSMmult (Ax, x, PB->x, PB->D, PB->i, PB->p, n, Com) ;
    
        /* smallest eigenpair of V'AV */
        SSMmineig (emin, evec, Com->wx1, P) ;
        /* v = V*evec */
        SSMDenseMult (v, evec, V, n, m) ;
        SSMmult (Com->Av, v, PB->x, PB->D, PB->i, PB->p, n, Com) ; /* Av */
    }

    Residual_computation:

    /* compute residual r0 and estimate multiplier for the
       original problem if subproblem mu != 0 */
    if ( (P->DO->mu != SSMZERO) || !BndLessThan )
    {
        mu = SSMZERO ;
        for (j = 0; j < n; j++)
        {
            t = b [j] + Ax [j] ;
            mu -= t*x [j] ;
            r0 [j] = t ;
        }
        mu /= rad*rad ;  /* mu = r0'*x/rad^2 */
        Com->mu = mu ;

        /* estimate error */
        ball_err = SSMZERO ;
        for (j = 0; j < n; j++)
        {
            t = r0 [j] + x [j]*mu ;
            ball_err += t*t ;
        }
        ball_err = sqrt (ball_err) ;
        if ( (mu < SSMZERO) && BndLessThan )
        {
            ball_err = SSMMAX (ball_err, fabs (mu)*rad) ;
        }
        Com->error = ball_err ;
    }
    else
    {
        /* compute residual */
        Com->mu = SSMZERO ;
        ball_err = SSMZERO ;
        for (j = 0; j < n; j++)
        {
            t = b [j] + Ax [j] ;
            r0 [j] = t ;
            ball_err += t*t ;
        }
        ball_err = sqrt (ball_err) ;
        Com->error = ball_err ;
    }

    /* compute eigenvector residual */
    t = SSMZERO ;
    u = SSMZERO ;
    for (j = 0; j < n; j++)
    {
        r = v [j] ;
        s = Av [j] - *emin*r ;    /* residual */
        t += s*s ;               /* norm of residual */
        u += r*r ;               /* norm of v */
    }
    Com->eig_error = sqrt (t/u) ;/* norm (residual)/norm (eigenvector) */
    return (0) ;
}

/* ==========================================================================
   === SSMorth ==============================================================
   ==========================================================================
    One step in the Householder orthogonalization of a collection of vectors
    or equivalently, one step in the QR factorization of the matrix whose
    columns are the vectors to be orthogonalized.  W stores the Householder
    vectors h0, h1, ... , hk-1 in its columns. The columns are packed
    to squeeze out the zeros.  Given a new vector x,
    the goal is to generate the next orthonormal vector. As in the QR
    factorization of a matrix, we first compute
    y = (I - hk-1 hk-1') ... (I - h1 h1')(I - h0 h0')x
    Then we evaluate the Householder vector hk which annihilates components
    k+1 through n-1 in y. Finally, we multiply the product
    (I - h0 h0') ... (I - hk hk') by the k-th column of the identity to
    obtain the desired orthonormal vector w
   ========================================================================== */

PRIVATE void SSMorth
(
    FLOAT     *w, /* k-th orthonormal vector */
    FLOAT     *W, /* packed matrix storing Householder vectors */
    FLOAT     *x, /* k-th new vector */
    int        k,
    int        n  /* dimension of x */
)
{
    int i, j, jp1, K ;
    FLOAT xk, s, t, u, *Wp ;
    K = k ;
    if ( k == n-1 )
    {
        Wp = W+((n*(n-1) - 2)/2) ;
        t = Wp [k] ;
        w [k] = SSMONE - t*t ;
        K-- ;
        w [K] = -Wp [K]*t ;
        goto compute_w ;
    }

    /* compute y = (I - hk-1 hk-1') ... (I - h1 h1')(I - h0 h0')x, store in w */
    for (j = 0; j < n; j++) w [j] = x [j] ;
    Wp = W ;
    for (j = 0; j < k;)
    {
        s = SSMZERO ;
        for (i = j; i < n; i++) s += Wp [i]*w [i] ;
        for (i = j; i < n; i++) w [i] -= Wp [i]*s ;
        j++ ;
        Wp += n - j ;  /* advance to next column, columns are packed */
    }

    u = SSMZERO ;
    for (j = k; j < n; j++) u = SSMMAX (u, fabs (w [j])) ;
    if ( u == SSMZERO )
    {
        for (j = k; j < n; j++) Wp [j] = SSMZERO ;
        goto init_w ;
    }
    s = SSMZERO ;
    for (j = k; j < n; j++)
    {
        t = w [j]/u ;
        Wp [j] = t ;
        s += t*t ;
    }
    xk = Wp [k] ;
    s = sqrt (s) ;
    if ( xk >= SSMZERO ) Wp [k] = xk + s ;
    else              Wp [k] = xk - s ;
    t = SSMONE/sqrt (s*(s + fabs (xk))) ;
    for (j = k; j < n; j++) Wp [j] *= t ;

    /* compute column k of the product (I - h0 h0') ... (I - hk hk') */
    init_w:
    t = Wp [k] ;
    for (i = k; i < n; i++) w [i] = -t*Wp [i] ;
    w [k] += SSMONE ;

    compute_w:
    for (j = K-1; j >= 0; j--)
    {
        jp1 = j + 1 ;
        Wp -= n - jp1 ;
        s = SSMZERO ;
        for (i = jp1; i < n; i++) s += w [i]*Wp [i] ;
        w [j] = -s*Wp [j] ;
        for (i = jp1; i < n; i++) w [i] -= s*Wp [i] ;
    }
}

/* ==========================================================================
   === SSMmineig ============================================================
   ==========================================================================
    Estimate a smallest eigenvalue and associated eigenvector for the
    matrix associated with PB. It is assumed that the diagonal optimization
    structure, the diagonlization, and the Lanczos/Householder
    tridiagonalization of the matrix have already been performed and
    stored in PB->DO, PB->DT, PB->LH
   ==========================================================================*/

PUBLIC void SSMmineig
(
    FLOAT    *emin, /* estimated smallest eigenvalue */
    FLOAT       *v, /* associated eigenvector */
    FLOAT       *w, /* work array of size n */
    SSMProblem *PB  /* problem specification */
)
{
    int imin, j, n ;
    SSMDiagOpt *DO ;
    SSMDiag *DT ;
    SSMLanczos *LH ;

    DO = PB->DO ;
    DT = PB->DT ;
    LH = PB->LH ;
    n = PB->n ;
    *emin = DO->dmin ;  /* smallest eigenvalue */
    imin  = DO->imin ;  /* index of smallest eigenvalue */
    for (j = 0; j < n; j++) w [j] = SSMZERO ;
    w [imin] = SSMONE ;

    /* Multiply by rotations ... G_2' G_1' diagonalizing tridiagonal matrix*/
    SSMtGivensMult (w, DT) ;

    /* Multiply by V */
    SSMDenseMult (v, w, LH->V, n, LH->ncols) ; /* v = estimated eigenvector */
    return ;
}

/* ========================================================================== */
/* === SSMdiag ============================================================== */
/* ========================================================================== */
/*
        Input: An n-dimensional tridiagonal matrix stored in arrays d and u

       Output: The eigenvalues along with the Givens rotations needed to
               diagonalize the matrix

    Algorithm: Inverse shifted QR algorithm with fast Givens rotations.
               The number of Givens rotations needed for the diagonalization
               is related to the number of QR iterations. This algorithm
               is an explicit implicit version of the QR method. It is
               explicit since the rotations are computed as in a
               QR factorization. It is implicit since each rotation
               is applied to both the right and left side of the matrix
               simultaneously, as is done with an implicit implementation
               of the QR method. An advantage of this explicit implicit
               algorithm is that the loss of shift phenomenon associated
               with the implicit QR algorithm does not occur since the
               rotations are computed from the explicit QR factorization.
               The speed associated with the implicit QR method is
               retained since each iteration entails a single pass over
               the matrix from upper left to lower right corner. In
               the fast Givens implementation, the matrix is stored as
               a product DTD where D is diagonal and T is tridiagonal.
               In the QR algorithm, the Givens rotations are applied to
               DTD - sigma I = D(T - sigma/D^2)D. In the explicit QR
               algorithm, the partially factored matrix has the form:

                         x  x  f  0  0  0  0
                         0  x  x  f  0  0  0
                         0  0  p  x  0  0  0
                         0  0  q  d  u  0  0
                         0  0  0  u  d  u  0
                         0  0  0  0  u  d  u

               Here f denotes fill, d and u are the original diagonal,
               subdiagonal, and superdiagonal elements, and p and q
               are the elements on the diagonal and subdiagonal which
               are used to generate the next Given rotation G. The
               rotation is designed to annihilate q while replacing
               p by sqrt(p*p + q*q). In performing the left and right
               multiplications by G, one needs to recall the structure
               of the matrix that is operated on:

                         d  u  0  0  0
                         u  d  u  a  0
                         0  u  d  u  0
                         0  a  u  d  u
                         0  0  0  u  d

               The matrix is tridiagonal with a bulge corresponding to
               the element a. In the code below, the iteration starts
               below the comment "perform one pass". The variables p, q,
               and a inside the subsequent code are identical to the
               p, q, and a above.

               An advantage of the fast Givens rotation is that several
               multiplications associated with the usual Givens rotation
               are eliminated as well as the square root. For details
               concerning fast Givens, see W. W. Hager, Applied Numerical
               Linear Algebra, pages 221-223. In the code that follows, the
               squares of the diagonal of the diagonal scaling matrix D are
               stored in the array sc (scaling array). As the algorithm
               progresses, the elements of sc tend to 0 since they are being
               multiplied by factors between 1/2 and 1.  When sc[i] drops
               below qr_lower, we multiply row i of T by sc[i] and reset
               sc[i] = 1. The code below rectifies an unflow problem
               which could occur in the analogous NAPACK code tdg.f.
               In tdg.f, we check sc at the start of each qr iteration.
               However, for a matrix with 5000 or more rows, an underflow
               could potentially occur within a QR iteration. Hence, in
               the new code, we check for small scales within the iteration.
               The code also checks to see if the matrix can be split due
               to a small off-diagonal element.
   ========================================================================== */
PRIVATE int SSMdiag /* return: 0 (convergence tolerance satisfied)
                              -5 (algorithm did not converge)
                              -6 (insufficient space allocated for Givens) */
(
    FLOAT  *din, /* diagonal of tridiagonal matrix */
    FLOAT  *uin, /* superdiagonal of tridiagonal matrix, use uin (0:n-2) */
    int       n, /* dimensional of tridiagonal matrix */
    SSMDiag *DT, /* structure for storing diagonalization */
    int    *wi1, /* work array of size n */
    FLOAT  *wx1, /* work array of size n */
    FLOAT  *wx2, /* work array of size n */
    FLOAT  *wx3  /* work array of size n */
)
{
    FLOAT a, b, c, delta, lb, tol1, tol2, break_tol,
         *gx, *gy, *gs, *e, *d, *u, *sc ;
    FLOAT p, q, r, s, scale, sigma, t, w, x, y, z0, z1 ;
    INT *gj ;
    int i, ip1, im1, k, l, l1, nm1, istart, nstart,
       *gi, *gk, *gf, *start_index ;
    short *gz ;
    INT ngivens,        /* number of fast Givens */
        nscale ;        /* number of scalings performed */
    int npass,          /* number of passes */
        max_npass ;

    /* output, not defined on input: */
    DT->n = n ;
    e  = DT->e,         /* eigenvalues */
    gx = DT->gx ;       /* fast Givens x factor, FLOAT */
    gy = DT->gy ;       /* fast Givens y factor, FLOAT */
    gz = DT->gz ;       /* rotation type 1 or 2, short */
    gs = DT->gs ;       /* scaling factors, FLOAT */
    gi = DT->gi ;       /* row indices of scale factors, int */
    gj = DT->gj ;       /* indexed by QR pass, points into Givens factors, INT */
    gk = DT->gk ;       /* index of last diagonal element in QR iteration, int */
    gf = DT->gf ;       /* index of first diagonal element in QR iteration */

    /* input, not modified: */
    nm1 = n - 1 ;
    tol1 = DT->tol1 ;       /* stop when u_{k-1}^2 <= tol1*|d_k| */
    tol2 = DT->tol2 ;       /* stop when |u_{k-1}| <= tol2 */
    max_npass = DT->max_its ;/* maximum number of QR algorithm passes */

    /* workspace vectors of size n */
    d = wx1 ;
    u = wx2 ;
    sc = wx3 ;
    start_index = wi1 ;
    nstart = 0 ;
    start_index [0] = 0 ;

    /* ---------------------------------------------------------------------- */

    DT->ngivens = 0 ;      /* number of fast Givens */
    DT->its = 0 ;          /* number of QR iterations */
    DT->nscale = 0 ;       /* number of scalings */

    /* ---------------------------------------------------------------------- */
    /* check for 1-by-1 case */
    /* ---------------------------------------------------------------------- */

    if (n == 1)
    {
        e [0] = din [0] ;
        return (0) ;
    }
    /* ---------------------------------------------------------------------- */
    /* determine the problem scaling */
    /* ---------------------------------------------------------------------- */
    scale = fabs (din [nm1]) ;
    for (i = 0 ; i < nm1 ; i++)
    {
        scale = SSMMAX (scale, fabs (din [i])) ;
        t = uin [i] ;
        scale = SSMMAX (scale, fabs (t)) ;
        if ( t == SSMZERO )
        {
            nstart++ ;
            start_index [nstart] = i + 1 ;
        }
    }
    if (scale == 0 )
    {
        for (i = 0 ; i < n ; i++) e[i] = 0 ;
        return (0) ;
    }
    scale = 1/scale ;
    /* ---------------------------------------------------------------------- */
    /* make a scaled copy of the input problem */
    /* ---------------------------------------------------------------------- */
    for (i = 0; i < nm1; i++)
    {
        d [i] = scale*din [i] ; /* scaled diagonal is needed in next QR phase */
        u [i] = scale*uin [i] ;
        sc[i] = 1. ;
    }
    t = scale*din [nm1] ;
    d [nm1] = t ; /* scaled diagonal is needed in next QR phase */
    sc[nm1] = 1. ;
    u [nm1] = 0 ;
    /* ---------------------------------------------------------------------- */
    /* inverse shifted QR method */
    /* ---------------------------------------------------------------------- */
    lb = DT->qr_lower ; /* threshold in fast Givens for rescaling of rows */
    break_tol = lb*tol2 ;
    ngivens = 0 ;
    npass = 0 ;
    nscale = 1 ;
    gi [0] = 0 ;
    k = n ;
    l = n-1 ;

    while (k > 1)
    {
        l1 = 0 ;
        k = l ;
        l-- ;
        while ( k == start_index [nstart] )
        {
            k = l ;
            l-- ;
            nstart-- ;
            if ( k == 0 ) break ;
        }
        if ( k == 0 ) break ;

        while ( 1 )
        {
            if ( l1 >= 30 ) return (-5) ;
            t = sqrt(sc [l]*sc [k])*fabs(u [l]) ;
            if ( t <= tol2 ) break ;
            if ( t <= tol1*fabs(sc [k]*d [k]) ) break ;
            /* -------------------------------------------------------------- */
            /* log the start of a pass */
            /* -------------------------------------------------------------- */
            l1++ ;
            /* check if enough space was allocated for Givens rotations,
               if not, need to increase Parm->qr_its */
            if (npass >= max_npass) return (-6) ;
            gk [npass] = k ;
            istart = gf [npass] = start_index [nstart] ;
            gj [npass] = ngivens ;

            npass++ ;
            /* -------------------------------------------------------------- */
            /* compute shift sigma */
            /* -------------------------------------------------------------- */
            /* note that a, b, c, t, and delta */
            /* are not used outside this scope */
            a = d [k]*sc [k] ;
            b = d [l]*sc [l] ;
            c = sqrt(sc [l]*sc [k])*fabs(u [l]) ;
            delta = .5*(b-a) ;
            if (delta >= 0)
            {
                if (delta <= c)
                {
                    t = delta/c ;
                    sigma = a - c/(t+sqrt(t*t+1)) ;
                }
                else
                {
                    t = c/delta ;
                    sigma = a - c*t/(1+sqrt(t*t+1)) ;
                }
            }
            else
            {
                if (-delta <= c)
                {
                    t = -delta/c ;
                    sigma = a + c/(t+sqrt(t*t+1)) ;
                }
                else
                {
                    t = -c/delta ;
                    sigma = a + c*t/(1+sqrt(t*t+1)) ;
                }
            }

            /* -------------------------------------------------------------- */
            /* compute the initial p and q */
            /* -------------------------------------------------------------- */

            p = d [istart] - sigma/sc [istart] ;
            q = u [istart] ;
            /* -------------------------------------------------------------- */
            /* perform one pass */
            /* -------------------------------------------------------------- */
            for (i = istart; i < k; i++)
            { 
                /* test for break up of matrix */
                if ( i > istart )
                {
                    if ( fabs (u [i-1])*z0*z1 < break_tol )
                    {
                        nstart++ ;
                        start_index [nstart] = i ;
                    }
                }
	        /* ---------------------------------------------------------- */
	        /* scale the problem, if necessary */
	        /* ---------------------------------------------------------- */
                im1 = i - 1 ;
                z0 = sc [i] ;
                if ( z0 < lb )
                {
                    gi [nscale] = (i+1) ;
                    d [i] *= z0 ;
                    z0 = sqrt (z0) ;
                    gs [nscale] = z0 ;
                    nscale++ ;
                    p *= z0 ;
                    u [i] *= z0 ;
                    if ( i > 0 ) u [im1] *= z0 ;
                    z0 = SSMONE ;
                    sc [i] = SSMONE ;
                }
                ip1 = i + 1 ;
                z1 = sc [ip1] ;
                if ( z1 < lb )
                {
                    gi [nscale] = -(i+1) ;
                    d [ip1] *= z1 ;
                    z1 = sqrt (z1) ;
                    gs [nscale] = z1 ;
                    nscale++ ;
                    q *= z1 ;
                    u [i] *= z1 ;
                    u [ip1] *= z1 ;
                    if ( i > 0 ) a *= z1 ;
                    z1 = SSMONE ;
                    sc [ip1] = SSMONE ;
                }

                if (fabs (p) > fabs (q))
                {
                    r = z1/z0 ;
                    s = q/p ;
                    t = r*s*s ;

                    if (t <= 1)
                    {
                        t = 1/(1+t) ;
                        sc [i] = z0*t ;
                        sc [ip1]=z1*t ;
                        y = s ;
                        x = r*s ;
                        gx [ngivens] = x ;
                        gy [ngivens] = y ;
                        gz [ngivens] = 1 ;
                        ngivens++ ;

                        if (i > 0) u [im1] = u [im1] + x*a ;
                        t = d [i] ;
                        p = u [i] ;
                        q = t + x*p ;
                        r = p - y*t ;
                        w = d [ip1] - y*p ;
                        s = p + x*d [ip1] ;
                        u [i] = s - y*q ;
                        d [i] = q + x*s ;
                        d [ip1] = w - r*y ;
                        q = u [ip1] ;
                        a = x*q ;
                        p = w - sigma/z1 ;
                    }
                    else
                    {

                        t = t/(1+t) ;
                        r = z0/z1 ;
                        s = p/q ;
                        sc [i] = z1*t ;
                        sc [ip1] = z0*t ;
                        y = s ;
                        x = r*s ;
                        gx [ngivens] = x ;
                        gy [ngivens] = y ;
                        gz [ngivens] = 2 ;
                        ngivens++ ;

                        if (i > 0) u [im1] = x*u [im1] + a ;
                        t = d [i] ;
                        p = u [i] ;
                        q = x*t+p ;
                        r = y*p-t ;
                        w = y*d [ip1]-p ;
                        s = x*p+d [ip1] ;
                        u [i] = y*s-q ;
                        d [i] = x*q+s ;
                        d [ip1] = y*w-r ;
                        a = u [ip1] ;
                        q = a ;
                        u [ip1] = y*a ;
                        p = w - y*sigma/z1 ;
                    }

                }
                else
                {
                    if (q == 0)
                    {
                        gx [ngivens] = 0 ;
                        gy [ngivens] = 0 ;
                        gz [ngivens] = 1 ;
                        ngivens++ ;

                        p = d [ip1] - sigma/z1 ;
                        q = u [ip1] ;
                    }
                    else
                    {
                        r = z0/z1 ;
                        s = p/q ;
                        t = r*s*s ;

                        if (t < 1)
                        {
                            t = 1/(1+t) ;
                            sc [i] = z1*t ;
                            sc [ip1] = z0*t ;
                            y = s ;
                            x = r*s ;
                            gx [ngivens] = x ;
                            gy [ngivens] = y ;
                            gz [ngivens] = 2 ;
                            ngivens++ ;

                            if (i > 0) u [im1] = x*u [im1] + a ;
                            t = d [i] ;
                            p = u [i] ;
                            q = x*t+p ;
                            r = y*p-t ;
                            w = y*d [ip1] - p ;
                            s = x*p + d [ip1] ;
                            u [i] = y*s - q ;
                            d [i] = x*q + s ;
                            d [ip1] = y*w - r ;
                            a = u [ip1] ;
                            q = a ;
                            u [ip1] = y*a ;
                            p = w - y*sigma/z1 ;
                        }
                        else
                        {
                            t = t/(1+t) ;
                            r = z1/z0 ;
                            s = q/p ;
                            sc [i] = z0*t ;
                            sc [ip1] = z1*t ;
                            y = s ;
                            x = r*s ;
                            gx [ngivens] = x ;
                            gy [ngivens] = y ;
                            gz [ngivens] = 1 ;
                            ngivens++ ;

                            if (i > 0) u [im1] = u [im1] + x*a ;
                            t = d [i] ;
                            p = u [i] ;
                            q = t + x*p ;
                            r = p - y*t ;
                            w = d [ip1] - y*p ;
                            s = p + x*d [ip1] ;
                            u [i] = s - y*q ;
                            d [i] = q + x*s ;
                            d [ip1] = w - r*y ;
                            q = u [ip1] ;
                            a = x*q ;
                            p = w - sigma/z1 ;
                        }
                    }
                }
            }
            gi [nscale] = 0 ;
            nscale++ ;
        }
    }

    /* ---------------------------------------------------------------------- */

    scale = 1/scale ;
    for (i = 0 ; i < n ; i++ )
    {
        e [i] = sc [i]*d [i]*scale ;
    }
    DT->nscale = nscale ;

    if ( ngivens > 0 )
    {
        for (i = 0 ; i < n ; i++)
        {
            gs [nscale] = sqrt (sc [i]) ;

            nscale++ ;
        }
    }

    DT->ngivens = ngivens ;
    DT->its = npass ;
    return (0) ;
}
/* Contents:

   1. SSMtridiag    - reduce A to tridiagonal matrix by orthog. similarity
   2. SSMtriHouse   - full Householder reduction to tridiagonal form
   3. SSMtriLanczos - partial Lanczos tridiagonalization

   ==========================================================================
   === SSMtridiag ===========================================================
   ==========================================================================

    Given a sparse symmetric matrix A, contruct a matrix V with
    orthonormal columns such that V'AV is tridiagonal:
                           _
                          | d_1  u_1   0    0    .  .  .
                          |
            -V'AV = T =   | u_1  d_2   u_2   0   .  .  .
                          |
                          |  0   u_2   d_3   0   .  .  .
                          |
                          |  .    .     .    .   .  .  .

    The "-" sign is needed since the matrix is -(A+D).
    If the dimension of A is <= House_limit, then the Householder
    process is used to obtain a square orthogonal matrix V. Otherwise,
    the Lanczos process is employed until either a small u_i is
    encountered or max_its iterations are performed, which ever
    occurs first. The matrix P is an orthogonal projection into the space
    orthogonal to a given vector "a". If "a" is NULL, then
    we assume that it is the vector of all ones. If the numerical
    entries of A are null, then we assume that all the elements are one.
    In the Lanczos process, the starting column of V is a vector of
    the form Px where x is chosen randomly.  This choice ensures that
    the columns of V are all contained in the range of P.
   ========================================================================== */

PRIVATE int SSMtridiag /* return 0 (process completes)
                               -7 (starting Lanczos vector vanishes) */
(
    FLOAT       *x, /* current solution estimate, ignored in Householder */
    int         it, /* iteration number */
    SSMLanczos *LH,
    SSMProblem *PB,
    SSMcom    *Com
)
{
    INT k, l, n2, *Ap, *Ap1 ;
    int j, n, status, *Ai ;
    FLOAT t, nnz, rand_max, *b, *Ax, *V, *D, *start_vector ;
    SSMParm *Parm ;

    Parm = Com->Parm ;
    n = PB->n ;
    b = PB->b ;
    Ax = PB->x ;
    Ai = PB->i ;
    Ap = PB->p ;
    D  = PB->D ;
    Ap1= Ap+1 ;
    V = LH->V ;
    /* use Householder */
    nnz = (FLOAT) Ap [n] ;
    t = (FLOAT) n ;
    if ( (n <= LH->House_limit)
         || ((t*t*t < 20*nnz*PB->DT->nalloc) && (PB->DT->nalloc == n)) )
    {
        /* create dense matrix */
        n2 = n*n ;
        for (k = 0; k < n2; k++) V [k] = SSMZERO ;
        k = 0 ;
        for (j = 0; j < n; j++)
        {
             l = Ap1 [j] ;
             for (; k < l; k++) V [Ai [k]] = Ax [k] ;
             V [j] = D [j] ;
             V = V+n ;
        }

        SSMtriHouse (LH->V, LH->d, LH->u, Com->wx1, n) ;
        LH->ncols = n ;
        status = 0 ;
        /* set bound on number of qr its */
    }
    else
    {
        /* Compute Lanczos starting point.  For starting iteration,
           starting guess is
           1. user guess is x if x not NULL
           2. b if b is nonzero
           3. random otherwise */
        start_vector = LH->V ;
        if ( it == 0 )
        {
            if ( x == NULL )
            {
                for (j = 0; j < n; j++)
                {
                    if ( b [j] != SSMZERO ) break ;
                }
                if ( j < n ) start_vector = b ;
                else
                {
                    rand_max = (FLOAT) RAND_MAX ;
                    for (j = 0; j < n; j++)
                    {
                        start_vector [j] = (((FLOAT) rand ())/rand_max)-SSMHALF;
                    }
                }
            }
            else start_vector = x ;
        }
        /* after the first iteration, start_vector is the projected
           KKT residual which is stored in Com->MINRES+(8*n) */
        else start_vector = Com->MINRES+(8*n) ;
        status = SSMtriLanczos (start_vector, LH, PB, Com) ;
        if ( Parm->PrintLevel >= 1 )
        {
           // printf ("number Lanczos: %i upper limit: %i\n",(int) LH->ncols, (int) LH->max_its) ;
        }
    }
    return (status) ;
}

/* ==========================================================================
   === SSMtriHouse ==========================================================
   ==========================================================================
  
    Reduce A to tridiagonal form by Householder orthogonal similarity
    transformations:
                                 _
                                | d_1  u_1   0    0    .  .  .
                                |
             V'AV = T =         | u_1  d_2   u_2   0   .  .  .
                                |
                                |  0   u_2   d_3   0   .  .  .
                                |
                                |  .    .     .    .   .  .  .

   where V is computed as a product of Householder matrices. A is
   stored as a dense symmetric matrix, and V is stored as a dense matrix.
   The output matrix V overwrites the input matrix A.

n = size (A, 1) ;
d = zeros(n,1) ;
u = zeros(n,1) ;

if ( n == 1 ) % {

    d(1) = A(1,1) ;
    V(1) = 1 ;

elseif ( n == 2 ) % } {

    d(1) = A(1,1) ;
    d(2) = A(2,2) ;
    u(1) = A(2,1) ;
    V = zeros(2) ;
    V(1,1) = 1 ;
    V(2,2) = 2 ;

else % } {

    W = zeros(n,n) ; Householder vectors (stored in A)
    for j = 1:n-2 % {

        %-----------------------------------------------------------------------
        % compute v
        %-----------------------------------------------------------------------

        % accesses only the strictly lower triangular part of A:
        v = house (A(:,j),j+1) ;
        % v (1:j) is zero

        %-----------------------------------------------------------------------
        % compute column j+1 of W
        %-----------------------------------------------------------------------

        % column 1 of W is zero.  W is strictly lower triangular
        W(:,j+1) = v ;

        %-----------------------------------------------------------------------
        % compute x
        %-----------------------------------------------------------------------

        % since v (j) is zero, we can skip column j of A
        % also note that A (1:(j-1),(j+1):n) is zero
        % accesses just lower part of A to compute this
        x = zeros (n,1) ;
        x (j:n) = A (j:n,(j+1):n) * v ((j+1):n) ;

        %-----------------------------------------------------------------------
        % compute a
        %-----------------------------------------------------------------------

        % x (1:(j-1)) is zero and v (1:j) is zero
        a = .5 * ( (v ((j+1):n))' * x ((j+1):n)) ;

        %-----------------------------------------------------------------------
        % compute b
        %-----------------------------------------------------------------------

        b = zeros (n,1) ;
        b (j:n) = a * v (j:n) - x (j:n) ;

        %-----------------------------------------------------------------------
        % update A
        %-----------------------------------------------------------------------

        % just update the lower triangular part of A
        A(j:n,j:n) = A(j:n,j:n) + v(j:n)*b(j:n)' + b(j:n)*v(j:n)' ;

        %-----------------------------------------------------------------------
        % save diagonal and off-diagonal
        %-----------------------------------------------------------------------

        d(j) = A(j,j) ;
        u(j) = A(j+1,j) ;

    end % }
    d(n-1) = A(n-1,n-1) ;
    u(n-1) = A(n,n-1) ;
    d(n) = A(n,n) ;

    %---------------------------------------------------------------------------
    % project W to V
    %---------------------------------------------------------------------------

    V = zeros(n,n) ;
    for j = 1:n % {
        V(:,j) = qj (W, j) ;
    end % }

end % }

   ========================================================================== */

PRIVATE void SSMtriHouse
(
    FLOAT *A , /* n-by-n matrix dense symmetric matrix (input)
                                dense orthogonal matrix (output) */
    FLOAT *d , /* n-by-1 vector (output) */
    FLOAT *u , /* n-by-1 vector (output) */
    FLOAT *x , /* n-by-1 vector (workspace) */
    int n
)
{
    FLOAT a, hj, y, s, t, *Ak, *Aj ;
    int i, j, jp1, k ;

    if (n == 1)
    {

        d [0] = A [0] ;
        u [0] = 0 ;
        A [0] = 1 ;
        return ;

    }
    else if (n == 2)
    {

        d [0] = A [0] ;
        d [1] = A [3] ;
        u [0] = A [1] ;
        A [0] = 1 ;
        A [1] = 0 ;
        A [2] = 0 ;
        A [3] = 1 ;
        return ;

    }

     /* start Householder reduction for matrix of dimension >= 3 */
     /* initialize current column of A */
    Aj = A ;
    for (j = 0 ; j < n-2 ; j++)
    {
        d [j] = Aj [j] ;
        s = SSMZERO ;
        jp1 = j+1 ;
        for (i = jp1; i < n; i++) s += Aj [i]*Aj [i] ;
        if ( s == SSMZERO )
        {
            u [j] = SSMZERO ;
            Aj += n ;
            continue ;   /* the Householder vector is 0 as is subcolumn of A */
        }

        /* scaling factor for Household matrix */
        s = sqrt (s) ;
        hj = Aj [jp1] ;
        t = 1/sqrt (s*(s + fabs (hj))) ;
        if ( hj >= 0 )
        {
            hj += s ;    /* hj is (j+1)st element of Household matrix */
            y = -s ;     /* y = u_j = (j+1)st element of (Householder * Aj) */
        }
        else
        {
            hj -= s ;
            y = s ;
        }
        u [j] = y ;

        /* ----------------------------------------------------------- */
        /* overwrite column j of A with Householder vector */
        /* ----------------------------------------------------------- */

        Aj [jp1] = hj ;
        for (i = jp1; i < n; i++) Aj [i] *= t ;
    
        /* x ((j+1):n) = A ((j+1):n,(j+1):n) * house ((j+1):n) */
        for (i = jp1; i < n; i++) x [i] = SSMZERO ;
        Ak = Aj ;
        for (k = jp1 ; k < n ; k++)
        {
    	    /* Ak is column k of A */
            Ak += n ;
            /* diagonal */
            {
                FLOAT vk = Aj [k] ;
                FLOAT xk = Ak [k] * vk ;
    
                /* dot product with row k of A and saxpy with column k of A */
                for (i = k+1 ; i < n ; i++)
                {
                    FLOAT aki = Ak [i] ;
                    xk += aki * Aj [i] ;
                    x [i] += aki * vk ;
                }
                x [k] += xk ;
            }
        }

        /* -------------------------------------------------------------- */
        /* a = .5 * ( (v ((j+1):n))' * x ((j+1):n)) ; */
        /* -------------------------------------------------------------- */

        a = 0 ;
        for (i = jp1; i < n; i++) a += Aj [i] * x [i] ;
        a /= 2 ;

        /* ----------------------------------------------------- */
        /* b (j:n) = a * v (j:n) - x (j:n) ; write b back into x */
        /* ----------------------------------------------------- */

        for (i = jp1; i < n; i++)
        {
            x [i] = a*Aj [i] - x [i] ;
        }

        /* ----------------------------------------------------------------- */
        /* A(j+1:n,j+1:n) = A(j+1:n,j+1:n)
                            + v(j+1:n)*b(j+1:n)' + b(j+1:n)*v(j+1:n)' */
        /* ----------------------------------------------------------------- */

        /* note that x holds b */
        /* update only the lower triangular part of A */
        Ak = Aj ;
        for (k = jp1 ; k < n ; k++)
        {
            Ak += n ;
            {
                FLOAT vk = Aj [k] ;
                FLOAT xk = x [k] ;
                Ak [k] += 2.*(vk * xk) ;
                for (i = k+1; i < n; i++)
                {
                    Ak [i] += (Aj [i] * xk) + (x [i] * vk) ;
                }
            }
        }
        Aj += n ;
    }

    /* ------------------------------------------------------------------ */
    /* d(n-1) = A(n-1,n-1) ; u(n-1) = A(n,n-1) ; u(n) = 0 ; d(n) = A(n,n) */
    /* ------------------------------------------------------------------ */

    d [n-2] = Aj [n-2] ;
    u [n-2] = Aj [n-1] ;
    Aj += n ;           /* Aj = last column of matrix */
    d [n-1] = Aj [n-1] ;
    u [n-1] = 0 ;

    /* ------------------------------------------------------------------ */
    /* compute columns of V and store them in A */
    /* start with last column and work to first column */
    /* ------------------------------------------------------------------ */

    /* last column is special */
    Ak = Aj-(n+n) ;    /* first column with a Householder vector */
    t = Ak [n-1] ;
    Aj [n-2] =   - t*Ak [n-2] ;
    Aj [n-1] = 1 - t*Ak [n-1] ;
    for (j = n-4; j >= 0; j--)
    {
        /* compute the new element in Aj */
        Ak -= n ;
        t = SSMZERO ;
        for (i = j+2; i < n; i++)
        {
            t += Aj [i]*Ak [i] ;
        }
        Aj [j+1] = -Ak [j+1]*t ;

        /* update the remaining elements in Aj */
        for (i = j+2; i < n; i++)
        {
            Aj [i] -= Ak [i]*t ;
        }
    }
    Aj [0] = 0 ;

    /* compute next to last through 2nd column of V */
    for (j = n-2; j > 0; j--)
    {
        Aj -= n ;
        /* startup, column of identity times preceding Householder */
        Ak = Aj - n ;
        t = Ak [j] ;
        for (i = j; i < n; i++) Aj [i] = -Ak [i]*t ;
        Aj [j]++ ;
        for (k = j-2; k >= 0; k--)
        {
            /* compute the new element in Aj */
            Ak -= n ;
            t = SSMZERO ;
            for (i = k+2; i < n; i++)
            {
                t += Aj [i]*Ak [i] ;
            }
            Aj [k+1] = -Ak [k+1]*t ;

            /* update the remaining elements in Aj */
            for (i = k+2; i < n; i++)
            {
                Aj [i] -= Ak [i]*t ;
            }
        }
        Aj [0] = SSMZERO ;
    }
    /* store the first columns of V = first column of identity */
    Aj -= n ;
    for (i = 0; i < n; i++) Aj [i] = 0 ;
    Aj [0] = 1 ;
}

/* ========================================================================== */
/* === SSMtriLanczos ======================================================== */
/* ========================================================================== 
    Partially reduce P(A+D)P to tridiagonal form by the Lanczos process.
    The maximum number of Lanczos iterations is Com->Lanczos_max_its.  The
    Lanczos process is terminated when there is a small
    off-diagonal element (|u_j| <= Amax*Parm->Lanczos_tol, where Amax is the
    absolute largest element in A + D).
                           _
                          | d_1  u_1   0    0    .  .  .
                          |
            -V'AV = T =   | u_1  d_2   u_2   0   .  .  .
                          |
                          |  0   u_2   d_3   0   .  .  .
                          |
                          |  .    .     .    .   .  .  .

   ========================================================================== */

PRIVATE int SSMtriLanczos /* return 0 (process completes)
                                  -7 (starting Lanczos vector vanishes) */
(
    FLOAT *start_vector, /* starting point */
    SSMLanczos      *LH, /* Lanczos structure */
    SSMProblem      *PB, /* Problem specification */
    SSMcom         *Com  /* SSMcom structure */
)
{
    INT *Ap ;
    int i, j, n, *Ai ;
    FLOAT s, t, uj, *Ax, *d, *u, *r, *q, *Vj, *Vjprior ;

    /* ---------------------------------------------------------------------- */
    /* Read in the needed arrays */
    /* ---------------------------------------------------------------------- */

    /* output */
    Vj= LH->V ;    /* dense FLOAT matrix of orthonormal vectors */
    d = LH->d ;    /* FLOAT array containing diagonal of tridiagonal matrix */
    u = LH->u ;    /* FLOAT array containing subdiagonal of tridiagonal matrix*/

    /* problem specification */
    n  = PB->n ;  /* problem dimension */
    Ax = PB->x ;  /* numerical values for edge weights */
    Ai = PB->i ;  /* adjacent vertices for each node */
    Ap = PB->p ;  /* points into Ax or Ai */

    /* work arrays */
    r = Com->wx1 ;
    q = Com->wx2 ;

/* Lanczos iteration (orthogonal vectors stored in q_1, q_2, ... )

    q_0 = 0, r = q_1 (starting vector)
    for j = 1:n
        u_{j-1} = ||r||                  (u_0 is discarded)
        q_j = r/u_{j-1}                  (normalize vector)
        d_j = q_j'Aq_j                   (diagonal element)
        r = (A-d_jI)q_j - u_{j-1}q_{j-1} (subtract projection on prior vectors)
    end
*/

    s = SSMZERO ;
    for (i = 0; i < n; i++)
    {
        t = start_vector [i] ;
        s += t*t ;
    }
    if ( s == SSMZERO )
    {
        if ( Com->Parm->PrintLevel > 0 )
        {
           // printf ("starting vector in SSMLanczos vanishes!!\n") ;
            LH->ncols = 0 ;
            return (-7) ;
        }
    }
    s = sqrt (s) ;

    /* normalize starting vector */
    t = SSMONE/s ;
    for (i = 0; i < n; i++) Vj [i] = t*start_vector [i] ;

    SSMmult (r, Vj, Ax, PB->D, Ai, Ap, n, Com) ; /* startup: compute r = A*Vj */

    /* compute d [0] */
    s = SSMZERO ;
    for (i = 0; i < n; i++) s += r [i]*Vj [i] ;
    d [0] = s ;
    for (i = 0; i < n; i++) r [i] -= s* Vj [i] ; /* 1st iteration r computed */

    /* the main Lanczos iteration */
    j = 0 ;
    while ( 1 )
    {
        s = 0;
        for (i = 0; i < n; i++) s += r [i]*r [i] ;
        uj = sqrt (s) ;
        u [j] = uj ;
        if ( uj < LH->tol ) break ;
        Vjprior = Vj ;
        Vj += n ;
        for (i = 0; i < n; i++) Vj [i] = r[i]/uj ;

        SSMmult (r, Vj, Ax, PB->D, Ai, Ap, n, Com) ; /* compute q = Ar */

        /* compute d_j */
        s = SSMZERO ;
        for (i = 0; i < n; i++) s += r [i]*Vj [i] ;
        j++ ;
        d [j] = s ;
        if ( j+1 == LH->max_its ) break ; /* max number of iteration */

        /* compute new residual r */

        for (i = 0; i < n; i++) r [i] -= s*Vj [i] + uj*Vjprior [i] ;

    }
    u [j] = SSMZERO ;
    LH->ncols = j+1 ;                        /* number of columns in V */
    return (0) ;
}
/* Contents:

    1. SSMdiagopt      -  main routine to optimize diagonal QP with ||x|| = r.
    2. SSMdiagF        -  evaluate F (see below)
    3. SSMdiagF1       -  evaluate F'
   ==========================================================================
   === SSMdiagopt ===========================================================
   ==========================================================================
    Find a global minimizer for the diagonalized QP

    (DQP)     minimize sum d_i x_i^2 + 2 f_i x_i
              subject to ||x|| <= r or ||x|| = r

    If the constraint is ||x|| <= r, then we first check to see if the
    constraint is active. The constraint is inactive if:

        1. d_i >= 0 for all i
        2. f_i =  0 if d_i = 0
        3. sum { (f_i/d_i)^2: d_i > 0 } <= r

    If the constraint is inactive, we store the solution and exit.
    Otherwise, we search for values of mu for which the KKT conditions hold:

        (d_i+mu)x_i + f_i = 0  or x_i = -f_i/(d_i+mu)

    and ||x|| = r. In other words,
                          _        _
                         |  f_i^2   |
    (*)     F (mu) = sum | ---------| - r^2 = 0 .
                         |(d_i+mu)^2|
                          -        -
    Note that mu >= -dmin = - min (d).  We define d_shift = d - dmin
    and we replace d by d_shift. Instead of mu >= -dmin, the
    constraint is mu >= 0.  We start with upper and
    lower bounds for the root of (*) and perform a Newton step
    on the left side and a secant step on the right side. The
    Newton step should produce an iterate on the left of the root
    and the secant iterate should yield a point on the right of
    the root. The initial bounds on the root are obtained as follows:
    Neglecting those i for which d_shift_i > 0 yields

              mu >= sqrt (sum (f_i^2: d_shift_i = 0)) / r

    Replacing all d_shift_i by 0 gives

              mu <= ||f||/ r

    The algorithm for computing the root mu is as follows in MATLAB notation:

    dmin = min (d) ;
    dshift = d - dmin ;            % dshift >= 0
    dzero = find (dshift <= 0) ;   % dzero = list of all (now zero) entries
    dpositive = find (d >  0) ;    % dpositive = list of all pos. entries

    x = zeros (n, 1) ;
    if ( isempty(dpositive) )      % all elements of d are identical
        normf = norm(f) ;
        if ( norm(f) > 0 )         % f is nonzero and d is a multiple of 1
            x = -(r/normf)*f ;
        else
            x(1) = r ;             % all elements of f are zero, x can be
        end                        % any norm r vector
        A = (normf/r) - emin ;
        return
    end

    x (dpositive) = -f (dpositive) ./ e (dpositive) ;
    normx = norm (x) ;
    f0 = norm (f (dzero)) ;        % part of f corresponding to dshift = 0
    f2 = f.^2 ;                    % square of f

    if ( (f0 == 0) & (normx <= r) )% mu = -dmin satisfies KKT conditions
                                   % x for dpositive as above, rest of x
                                   % chosen to satisfy norm constraint
        x (dzero (1)) = sqrt (r^2 - normx^2) ;
        mu = -dmin ;

    else                           % mu > -dmin

        A = f0 / r ;               % A <= root (root lower bound) corresponds
                                   % to discarding dpositive terms in (*)
        if ( A == 0 ) % shift A positive to avoid pole in (*) -- see code
        Fl = sum ((f./(A+d)).^2) - r^2 ; %Fl = F (A)

        if ( Fl < 0 )              % F not positive => mu = A

            x = -f./(d+A) ;
            mu = A - dmin ;

        else
            B = norm (f) / r ;    % right of root (upper bound) corresponds
                                   % replacing dshift by zero
            Fr = sum ((f./(B+d)).^2) - r^2 ; % F2 = F (B)
            while (abs ((B - A) / A) > tol)

                st = .5*Fl / sum (f2./((A+d).^3));% Newton step
                A = A + st ;                      % left side of root
                Fl = sum ((f./(A+d)).^2) - r^2 ;
                B = A - Fl*(B-A)/(Fr-Fl) ;     % secant step right side
                Fr = sum ((f./(B+d)).^2) - r^2 ;
                if ( Fl < 0 )
                    break       % impossible, mu = A
                elseif ( Fr > 0 )
                    mu = B ;    % impossible, mu = B
                    break
                end
            end
            x = -f ./ (A + d) ;
            mu = A - emin ;
        end
    end
   ========================================================================== */

PRIVATE void SSMdiagopt
(
    FLOAT        *x,  /* n-by-1 solution vector (output) */
    FLOAT         r,  /* radius of sphere */
    int BndLessThan,  /* TRUE means ||x|| <= r, FALSE means ||x|| = r */
    SSMDiagOpt  *DO,  /* diagonal optimization structure */
    SSMcom     *Com
)
{
    int i, imin, n ;
    FLOAT dmin, dmax, fmax, f0, Fl, Fr, Ft, A, B, Aold, Flold, normf, normx2,
          rr, s, t, fi, tol, width, deps, feps, *d, *f, *f2, *dshift ;

    rr = r*r ;                        /* rr = radius squared */
    n = DO->n ;
    d = DO->d ;
    /* find dmin */
    dmin = INF ;
    dmax = SSMZERO ;
    for (i = 0; i < n; i++)
    {
        t = d [i] ;
        if ( t < dmin )
        {
            dmin = t ;
            imin = i ;
        }
        if ( fabs (t) > dmax ) dmax = fabs (t) ;
    }
    DO->dmin = dmin ;
    DO->imin = imin ;

    fmax = SSMZERO ;
    f = DO->f ;
    for (i = 0; i < n; i++) if ( fabs (f [i]) > fmax ) fmax = fabs (f [i]) ;

    /* Treat components of d near dmin as equal to dmin, the corresponding
       components of f that are essentially zero are set to zero.
       Also, compute dshift = d - dmin, f2 = f^2, normf = ||f||,
       f0 = ||{f [i]: d [i] = dmin}|| */
    t = Com->Parm->diag_eps*n ;
    deps = dmax*t ;
    feps = fmax*t ;

    /* if problem is essentially positive semidefinite,
       check if optimal solution satisfies ||x|| <= r */
    if ( (dmin >= -deps) && BndLessThan )
    {
        s = SSMZERO ;
        for (i = 0; i < n; i++)
        {
            if ( fabs (d [i]) > deps )
            {
                t = -f [i]/d [i] ;
                s += t*t ;
                x [i] = t ;
            }
            else
            {
                /* if d_i essentially vanishes but not f_i, treat as ||x|| = r*/
                if ( fabs (f [i]) > feps )
                {
                    if ( d [i] > SSMZERO )
                    {
                        t = -f [i]/d [i] ;
                        s += t*t ;
                        x [i] = t ;
                    }
                }
                /* if d_i and f_i both vanish to within tolerance, set x_i = 0*/
                else x [i] = SSMZERO ; 
            }
        }
        if ( (s <= rr) && (i == n) )
        {
            DO->mu = SSMZERO ;
            return ;
        }
    }

    /* ||x|| = r */
    normx2 = SSMZERO ;
    f0 = SSMZERO ;
    normf = SSMZERO ;
    f2 = DO->f2 ;
    dshift = DO->dshift ;
    for (i = 0; i < n; i++)
    {
        fi = f [i] ;
        s = fi*fi ;
        f2 [i] = s ;
        normf += s ;
        t = d [i] - dmin ;
        dshift [i] = t ;
        if ( t == SSMZERO ) f0 += s ;
        else { t = fi/t ; normx2 += t*t ; }
    }
    f0 = sqrt (f0) ;        /* norm of f for indices with dshift_i = 0*/
    normf = sqrt (normf) ;  /* ||f|| */

    /* trivial case, x [0] = +- r */
    if ( n == 1 )
    {
        if ( f [0] > SSMZERO ) x [0] = -r ;
        else                   x [0] =  r ;
        DO->mu = -(d [0] + f [0]/x [0]) ;
        return ;
    }

    /* Compute the global minimizer */

    tol = DO->tol ;       /* relative accuracy of solution */

    /*------------------------------------------------------------------------*/
    /* solve the problem when dshift is nonzero */
    /*------------------------------------------------------------------------*/

    B = normf/r ;                     /* B is upper bound for mu */
    /*PRINTF (("f0: %e B: %e r: %e\n", f0, B, r) ; */
    if ( f0 == SSMZERO )              /* search for positive lower bound A */
    {
        i = 0 ;
        if ( normx2 > rr )            /* mu > 0 */
        {
            A = B ;                   /* start right of root */
            for (i = 0; i < 20; i++)
            {
                A = A*Com->Parm->shrink ;/* multiply upper bound by shrink */
                Fl = SSMdiagF (A, f2, dshift, rr, n) ;
                if ( Fl > SSMZERO ) break ; /* F (A) > 0 => A is left of root */
                B = A ;
            }
        }
        if ( (i == 20) || (normx2 <= rr) ) /* mu = 0 */
        {
            for (i = 0; i < n; i++)
            {
                if ( dshift [i] > SSMZERO )
                {
                    x [i] = -f [i]/dshift [i] ;
                }
                else
                {
                    x [i] = SSMZERO ;
                }
            }
            x [imin] = sqrt (rr - normx2) ;
            DO->mu = -dmin ;
            return ;
        }
    }
    else                       /* f0 > 0 */
    {
        A = f0/r ;             /* A stores lower bound for mu */
        Fl = SSMdiagF (A, f2, dshift, rr, n) ;
        if ( Fl < SSMZERO ) goto Exit ; /* Fl should be > 0, Fl < 0 => mu = A */
    }

    /*------------------------------------------------------------------------*/
    /* the root is bracketed, 0 < A < mu < B */
    /*------------------------------------------------------------------------*/
    Fr = SSMdiagF (B, f2, dshift, rr, n) ;
    if ( Fr > SSMZERO )        /* Fr should be < 0, Fr > 0 => mu = B */
    {
        A = B ;
        goto Exit ;
    }

    /*------------------------------------------------------------------------*/

    width = ((FLOAT) 2)*fabs (B-A) ;
/*  printf ("A: %30.15e B: %30.15e\n", A, B) ;*/
    while ( SSMTRUE )
    {
        Aold = A ;
        A -= Fl/SSMdiagF1 (A, f2, dshift, n) ;   /* Newton step */
        Flold = Fl ;
        Fl = SSMdiagF (A, f2, dshift, rr, n) ;   /* function value at A */
/*      printf ("A: %30.15e B: %30.15e\n", A, B) ; */
/*      printf ("Fl: %30.15e Fr: %30.15e\n", Fl, Fr) ; */
        if ( ((B-A)/A <= tol) || (B-A <= tol*tol) ) break ;
        if ( Fl < SSMZERO ) /* with perfect precision, this would not happen */
        {
            B = A ;
            Fr = Fl ;
            A = Aold ;
            Fl = Flold ;
        }
        if ( Fr == Fl ) break ;
        B = A - Fl*(B-A)/(Fr-Fl) ;             /* secant step */
        Fr = SSMdiagF (B, f2, dshift, rr, n) ; /* function value at B */
        if ( (Fr >= SSMZERO) || ((B-A)/A <= tol) )
        {
            A = B ;                            /* B = best estimate of mu */
            break ;
        }
        if ( B - A > width ) /* bisection step when slow decay of width */
        {
            t = SSMHALF*(A+B) ;
            if ( (t <= A) || (t >= B) ) break ;
            Ft = SSMdiagF (t, f2, dshift, rr, n) ;/* function value at t */
            if ( Ft > SSMZERO )
            {
                A = t ;
                Fl = Ft ;
            }
            else
            {
                B = t ;
                Fr = Ft ;
            }
        }
        width *= SSMHALF ;
    }

    /* best estimate for mu is stored in A */
    Exit:
    for (i = 0; i < n; i++) x [i] = -f [i]/(A + dshift [i]) ;
    DO->mu = A - dmin ;
}

/* ==========================================================================
   Evaluate the function  (*):
                          _        _
                         |  f_i^2   |
            F (mu) = sum |----------| - r^2
                         |(d_i+mu)^2|
                          -        -
   ========================================================================== */
PRIVATE FLOAT SSMdiagF
(
    FLOAT       mu,    /* the multiplier */
    FLOAT      *f2,    /* f_i^2 */
    FLOAT       *d,
    FLOAT       rr,    /* radius of sphere squared */
    int          n     /* dimension */
)
{
    int i ;
    FLOAT F, t ;
    F = -rr ;
    for (i = 0; i < n; i++)
    {
        t = mu + d [i] ;
        F += f2 [i]/(t*t) ;
    }
    return (F) ;
}

/* ==========================================================================
   Evaluate the function F':
                               _        _
                              |   f_i^2  |
            F' (mu) = - 2 sum | ---------| .
                              |(d_i+mu)^3|
                               -        -
   ========================================================================== */
PRIVATE FLOAT SSMdiagF1
(
    FLOAT       mu,    /* the multiplier */
    FLOAT      *f2,    /* f_i^2 */
    FLOAT       *d,
    int          n     /* dimension */
)
{
    int i ;
    FLOAT F, t ;
    F = 0 ;
    for (i = 0; i < n; i++)
    {
        t = mu + d [i] ;
        F += f2 [i]/(t*t*t) ;
    }
    F = -2.*F ;
    return (F) ;
}
/* Contents:

    1. SSMmult        - compute (A+D)x by rows
    2. SSMGivensMult  - compute x'G_1 G_2 ..., G_i = ith rotation in
                        diagonalization of a tridiagonal matrix
    3. SSMtGivensMult - compute G_1 G_2 ... x
    4. SSMDenseMult   - multiply dense matrix and vector 
    5. SSMtDenseMult  - multiply dense matrix transpose and vector
    6. SSMSSORmultP   - SSOR preconditioning operation with projection
    7. SSMSSORmult    - SSOR preconditioning operation */

/* ========================================================================== */
/* === SSMmult ============================================================== */
/* ========================================================================== */

/* Evaluate p = (A+D)x */

PRIVATE void SSMmult
(
    FLOAT    *p, /* output vector of size n */
    FLOAT    *x, /* input vector of size n */
    FLOAT   *Ax, /* numerical values in A excluding diagonal */
    FLOAT    *D, /* diagonal of matrix */
    int     *Ai, /* row indices for each column of A */
    INT     *Ap, /* Ap [j] = start of column j */
    int       n, /* dimension of matrix */
    SSMcom *Com
)
{
    INT k, l ;
    int j ;
    FLOAT t ;
    k = 0 ;
    for (j = 0; j < n; j++)
    {
        l = Ap [j+1] ;
        t = D [j]*x [j] ;
        for (; k < l; k++)
        {
            t += Ax [k]*x [Ai [k]] ;
        }
        p [j] = t ;
    }
    Com->mults++ ;
}

/* ========================================================================== */
/* === SSMGivensMult ======================================================== */
/* ========================================================================== */

/* Multiply row vector x on the left by the Givens rotations generated
   during the QR algorithm.  The rotations are applied in the same order
   that they were generated during the QR algorithm. */

PRIVATE void SSMGivensMult
(
    FLOAT      *x,     /* the vector to which the rotations are applied */
    SSMDiag    *DT     /* diagonalization structure for tridiagonal matrix */
)
{   

    INT ngivens, nscale ;
    int i, j, k, f, n, pass, npass, *gk, *gf, *gi ;
    short *gz, *Gz ;
    FLOAT xxt, xxs, *gx, *gy, *gs, *Gx, *Gy, *Gs ;

    /* ---------------------------------------------------------------------- */
    /* Read in the needed arrays */
    /* ---------------------------------------------------------------------- */
    
    /* input */
    npass = DT->its ;   /* number of QR iterations */
    if ( npass == 0 ) return ; /* diagonal matrix */
    n = DT->n ;
    gx = DT->gx ;       /* fast Givens x factor */
    gy = DT->gy ;       /* fast Givens y factor */
    gz = DT->gz ;       /* rotation type 1 or 2 */
    gs = DT->gs ;       /* scaling factors */
    gi = DT->gi ;       /* row indices for scaling factors */
    gk = DT->gk ;       /* index of last diagonal element in QR iteration */
    gf = DT->gf ;       /* index of first diagonal element in QR iteration */

    ngivens = 0 ;
    nscale = 0 ;
    for (pass = 0; pass < npass; pass++)
    {
        k = gk [pass] ;
        f = gf [pass] ;

        /* apply Givens rotation */
        Gx = gx+(ngivens-f) ;
        Gy = gy+(ngivens-f) ;
        Gz = gz+(ngivens-f) ;
        nscale++ ;
        i = abs (gi [nscale]) - 1 ;
        for (j = gf [pass]; j < k; j++)
        {
            if ( i == j )
            {
                if ( gi [nscale] > 0 )
                {
                    x [j] *= gs [nscale] ;
                    nscale++ ;
                    i = abs (gi [nscale]) - 1 ;
                    if ( i == j )
                    {
                        x [j+1] *= gs [nscale] ;
                        nscale++ ;
                        i = abs (gi [nscale]) - 1 ;
                    }
                }
                else
                {
                    x [j+1] *= gs [nscale] ;
                    nscale++ ;
                    i = abs (gi [nscale]) - 1 ;
                }
            }
            xxt = x [j] ;
            xxs = x [j+1] ;
            if ( Gz [j] == 1 )
            {
                x [j]   = xxt + Gx [j] * xxs ;
                x [j+1] = xxs - Gy [j] * xxt ;
            }
            else
            {
                x [j]   = Gx [j] * xxt + xxs ;
                x [j+1] = Gy [j] * xxs - xxt ;
            }
        }
        ngivens += (k-f) ;
    }

    /* final scaling */
    nscale++ ;
    Gs = gs+nscale ;
    for (i = 0; i < n ; i++) x [i] *= Gs [i] ;
}

/* ==========================================================================
   === SSMtGivensMult =======================================================
   ==========================================================================

   Multiply a column vector x on the right by the Givens rotations generated
   during the QR algorithm.  The rotations are applied in the reverse order
   that they were generated during the QR algorithm. */

PRIVATE void SSMtGivensMult
(
    FLOAT      *x,     /* the vector to which the rotations are applied */
    SSMDiag    *DT     /* diagonalization structure for tridiagonal matrix */
)
{
    INT jj, nscale, *gj ;
    int npass, i, j, k, kk, kf, n, *gi, *gk, *gf ;
    short *gz, *Gz ;
    FLOAT *gx, *gy, *gs, *Gx, *Gy, *Gs ;

    /* input */
    npass = DT->its ;   /* number of QR iterations */
    if ( npass == 0 ) return ; /* diagonal matrix */
    gx = DT->gx ;       /* fast Givens x factor */
    gy = DT->gy ;       /* fast Givens y factor */
    gz = DT->gz ;       /* rotation type 1 or 2 */
    gs = DT->gs ;       /* scaling factors */
    gi = DT->gi ;       /* indexed by QR pass, points into scale factors */
    gj = DT->gj ;       /* indexed by QR pass, points into Givens factors */
    gk = DT->gk ;       /* index of last diagonal element in QR iteration */
    gf = DT->gf ;       /* index of first diagonal element in QR iteration */
    nscale = DT->nscale ; /* number of scaling operations */
    n = DT->n ;

    Gs = gs+nscale ;
    for (j = 0 ; j < n ; j++)
    {
        x [j] *= Gs [j] ;
    }
    nscale-- ;

    for (k = npass-1 ; k >= 0 ; k--)
    {
        jj = gj [k] ;
        kk = gk [k] -1 ;
        kf = gf [k] ;
        Gx = gx+(jj-kf) ;
        Gy = gy+(jj-kf) ;
        Gz = gz+(jj-kf) ;
        nscale-- ;
        i = abs (gi [nscale]) - 1 ;

        for (j = kk; j >= kf; j--)
        {
            double xt = x [j] ;
            double xs = x [j+1] ;

            if (Gz [j] == 1)
            {
                x [j]   = xt - Gy [j] * xs ;
    	        x [j+1] = xs + Gx [j] * xt ;
            }
            else
            {
                x [j]   = Gx [j] * xt - xs ;
                x [j+1] = Gy [j] * xs + xt ;
            }
            /* scale */
            if ( j == i )
            {
                if ( gi [nscale] < 0 )
                {
                    x [j+1] *= gs [nscale] ;
                    nscale-- ;
                    i = abs (gi [nscale]) - 1 ;
                    if ( i == j )
                    {
                        x [j] *= gs [nscale] ;
                        nscale-- ;
                        i = abs (gi [nscale]) - 1 ;
                    }
                }
                else
                {
                    x [j] *= gs [nscale] ;
                    nscale-- ;
                    i = abs (gi [nscale]) - 1 ;
                }
            }
        }
    }
}

/* ==========================================================================
   === SSMDenseMult =========================================================
   ==========================================================================

   Compute y = Vx where V is m by n */

PRIVATE void SSMDenseMult
(
    FLOAT  *y,     /* m by 1 product, output */
    FLOAT  *x,     /* n by 1 given vector */
    FLOAT  *V,     /* dense m by n matrix */
    int     m,     /* number of rows */
    int     n      /* number of columns */
)
{
    int i, j ;
    FLOAT *Vp, t ;

    if ( n < 1 ) return ;
    t = x [0] ;
    for (i = 0; i < m; i++) y [i] = V [i]*t ;
    Vp = V+m ;
    for (j = 1; j < n; j++)
    {
        t = x [j] ;
        for (i = 0; i < m; i++) y [i] += Vp [i]*t ;
        Vp += m ;
    }
}

/* ==========================================================================
   === SSMtDenseMult ==========================================================
   ==========================================================================

   Compute y' = x'V where V is m by n */

PRIVATE void SSMtDenseMult
(
    FLOAT  *y,     /* n by 1 product, output */
    FLOAT  *x,     /* m by 1 given vector */
    FLOAT  *V,     /* dense m by n matrix */
    int     m,     /* number of rows */
    int     n      /* number of columns */
)
{
    int i, j ;
    FLOAT *Vp, t ;

    Vp = V ;
    for (j = 0; j < n; j++)
    {
        t = 0. ;
        for (i = 0; i < m; i++)
        {
            t += Vp [i]*x [i] ;
        }
        y [j] = t ;
        Vp += m ;
    }
}

/* ==========================================================================
   === SSMSSORmultP =========================================================
   ==========================================================================

    Multiply b by the matrix associated with SSOR
    preconditioning for a linear system with matrix P(A + mu I)P
    where P projects a vector into the space orthogonal to x.
    See the documentation for the SSMSSORmult algorithm where we
    give the SSOR preconditioner for a matrix A. In order to handle
    the projection, we use the strategy explained on page 203 of the
    following paper (see Algorithms 2 and 3):
    W. W. Hager, Minimizing a quadratic over a sphere, SIAM Journal on
    Optimization, 12 (2001), pp. 188-208. Below d = diag (A) + mu,
    s = sqrt(d), p and q are vectors associated with the
    projection, and w = x/||x||. */

PRIVATE void SSMSSORmultP
(
    FLOAT       *y,  /* the resulting vector */
    FLOAT       *b,  /* vector to be multiplied by SSOR matrix */
    FLOAT      *wj,  /* the first half of the SSOR multiplication operation */
    FLOAT       *w,  /* w = x/||x|| */
    FLOAT      *aj,  /* aj = Awj, product of A with wj */
    FLOAT       mu,  /* multiplier */
    int    startup,  /* = 1 for starting multiplication, 0 otherwise */
    SSMProblem *PB,  /* problem specification */
    SSMcom    *Com   /* SSMcom structure */
)
{
    INT k, l, *Ap, *Ap1, *Au ;
    int j, n, n1, *Ai ;
    FLOAT r, t, u, xj, yj, *Ax, *d, *p, *q, *s ;
    n  = PB->n ;  /* dimension of A */
    Ax = PB->x ;  /* numerical values in A (input) */
    Ai = PB->i ;  /* row indices for each column of A */
    Ap = PB->p ;  /* Ap [j] = start of column j */
    Au = PB->u ;  /* Au [j] = location right after last nonzero above diagonal*/
    Ap1= Ap+1 ;
    d = Com->SSOR ; /* d  = diag (C), C = (A + mu*I) */
    s = d+(1*n) ; /* s = sqrt (d) */
    p = d+(2*n) ; /* p = A*w - (w'*Aw)w */
    q = d+(3*n) ; /* q = (A + mu*I)w */
    n1 = n - 1 ;
/*  s = 0 ;
    t = 0 ;
    for i = n : -1 : 2
        y(i) = ( y(i) + w(i)*s + p(i)*t ) / d(i) ;
        s = s + q(i)*y(i) ;
        t = t + w(i)*y(i) ;
%       y(1:i-1) = y(1:i-1) - A(1:i-1,i)*y(i) ;
        y = y - U (:,i)*y(i) ; U = upper triangle of A
    end
    y(1) = ( y(1) + w(1)*s + p(1)*t ) / d(1) ; */

    if ( startup )
    {
        for (j = 0; j < n; j++) y [j] = b [j] ;
        goto StartUp ;
    }

    for (j = 0; j < n; j++) y [j] = s [j]*b [j] ;
    u = SSMZERO ;
    t = SSMZERO ;
    for (j = n1; ; j--)
    {
        xj = w [j] ;
        yj = (y [j] + xj*u + q [j]*t)/d [j] ;
        y [j] = yj ;
        if ( j <= 0 ) break ;

        t += (xj   * yj) ;
        u += p [j] * yj ;

        /* y = y - U (:,j) * yj */
        l = Au [j] ;
        for (k = Ap [j]; k < l; k++)
        {
            y [Ai [k]] -= Ax [k]*yj ;
        }
    }
    Com->mults += SSMHALF ;

    r = SSMZERO ;
    for (j = 0; j < n; j++) r += y [j]*w [j] ;
    /* note: aj and wj below are returned arrays */
    for (j = 0; j < n; j++)
    {
        t = y [j] - r*w [j] ;
        y [j] = t ;
        wj [j] = t ;
    }

    SSMmult (aj, wj, Ax, PB->D, Ai, Ap, n, Com) ; /* aj = A*wj */
    r = SSMZERO ;
    for (j = 0; j < n; j++)
    {
        t = aj [j] + y [j]*mu ;
        y [j] = t ;
        r += w [j]*t ;
    }
    for (j = 0; j < n; j++) y [j] -= r*w [j] ;

/*  s = 0 ;
    t = 0 ;
    for i = 1:n-1
        y(i) = ( y(i) + w(i)*s + q(i)*t) / d(i) ;
        s = s + p(i)*y(i) ;
        t = t + w(i)*y(i) ;
%       y(i+1:n) = y(i+1:n) - A(i+1:n,i)*y(i) ;
        y = y - L (: ,i)*y(i) ;
    end
    y(n) = ( y(n) + w(n)*s + q(n)*t ) / d(n) ; */

    StartUp: /* compute sqrt (d)*lower triangular SSOR stuff */
    u = SSMZERO ;
    t = SSMZERO ;
    for (j = 0; ; j++)
    {
        xj = w [j] ;

        yj = (y [j] + xj*u + p [j]*t) / d [j] ;
        y [j] = s [j]*yj ;
        if ( j >= n1 ) break ;

        t += xj    * yj ;
        u += q [j] * yj ;

        /* y = y - L (:,j) * yj */
        l = Ap1 [j] ;
        for (k = Au [j]; k < l ; k++)
        {
            y [Ai [k]] -= Ax [k]*yj ;
        }
    }
    Com->mults += SSMHALF ;
}

/* ==========================================================================
   === SSMSSORmult ==========================================================
   ==========================================================================

    Multiply b by the matrix associated with SSOR
    preconditioning for a linear system with matrix A.
    This code is based on formula (6.3) in the following paper:
 ***W. W. Hager, Iterative methods for nearly singular systems,
    SIAM Journal on Scientific Computing, 22 (2000), pp. 747-766.
    If this formula is combined with (6.2), we see that in the
    minimal residual algorithm, each iteration involves multiplication
    by the matrix

                    - sqrt(D) inv(L) A inv(L)' sqrt(D)

    Here D is the diagonal of A and L is the lower triangular
    matrix formed by elements of A on the diagonal and below the diagonal.
    In the code below we ignore the leading "-" sign. To correct for this,
    we add the correction term to x in the calling routine SSMSSOR
    (instead of subtract the term). See Algorithm 3 in the paper ***. */

PRIVATE void SSMSSORmult
(
    FLOAT       *y,  /* the resulting vector */
    FLOAT       *b,  /* vector to be multiplied by SSOR matrix */
    FLOAT      *wj,  /* the first half of the SSOR multiplication operation */
    FLOAT      *aj,  /* aj = Awj, product of A with wj */
    FLOAT       *d,  /* diagonal of matrix */
    FLOAT       *s,  /* square root of d */
    FLOAT       mu,  /* diagonal safeguard */
    int    startup,  /* = TRUE (starting multiplication), FALSE (otherwise) */
    SSMProblem *PB,  /* problem specification */
    SSMcom    *Com   /* SSMcom structure */
)
{
    INT k, l, *Ap, *Ap1, *Au ;
    int j, n, n1, *Ai ;
    FLOAT yj, *Ax ;
    n  = PB->n ;  /* dimension of A */
    Ax = PB->x ;  /* numerical values in A (input) */
    Ai = PB->i ;  /* row indices for each column of A */
    Ap = PB->p ;  /* Ap [j] = start of column j */
    Au = PB->u ;  /* Au [j] = location right after last nonzero above diagonal*/
    Ap1= Ap+1 ;
    n1 = n - 1 ;
/*  s = 0 ;
    t = 0 ;
    for i = n : -1 : 2
        y(i) = y(i) / d(i) ;
        y = y - U (:,i)*y(i) ; U = upper triangle of A
    end
    y(1) = y(1) / d(1) ; */

    if ( startup == SSMTRUE )
    {
        for (j = 0; j < n; j++) y [j] = b [j] ;
        goto StartUp ;
    }

    /* multiply by sqrt (D) */
    for (j = 0; j < n; j++) y [j] = s [j]*b [j] ;
    
    /* multiply by inv(L)' */
    for (j = n1; ; j--)
    {
        yj = y [j]/d [j] ;
        y [j] = yj ;
        if ( j <= 0 ) break ;

        /* y = y - U (:,j) * yj */
        l = Au [j] ;
        for (k = Ap [j]; k < l; k++)
        {
            y [Ai [k]] -= Ax [k]*yj ;
        }
    }
    Com->mults += SSMHALF ;

    /* note: aj and wj below are returned arrays */
    for (j = 0; j < n; j++) wj [j] = y [j] ;

    /* multiply by A, aj = A*wj */
    SSMmult (aj, wj, Ax, PB->D, Ai, Ap, n, Com) ;
    if ( mu == SSMZERO ) for (j = 0; j < n; j++) y [j] = aj [j] ;
    else                 for (j = 0; j < n; j++) y [j] = aj [j] + y [j]*mu ;

/*  for i = 1:n-1
        y(i) = y(i) / d(i) ;
        y = y - L (: ,i)*y(i) ;
    end
    y(n) = y(n) / d(n) ; */

    StartUp: /* compute sqrt (d)*lower triangular SSOR stuff */
    /* multiply by sqrt(D) inv(L) */
    for (j = 0; ; j++)
    {
        yj = y [j] / d [j] ;
        y [j] = s [j]*yj ;            /* y_j multiplied by sqrt(D_j) */
        if ( j >= n1 ) break ;

        /* y = y - L (:,j) * yj */
        l = Ap1 [j] ;
        for (k = Au [j]; k < l ; k++)
        {
            y [Ai [k]] -= Ax [k]*yj ;
        }
    }
    Com->mults += SSMHALF ;
}
/* Contents:

    1. SSMboundary - sequential subspace method for ||x|| = r
    2. SSMinterior - SSM for ||x|| < r, solve Ax = -b
    3. SSMminresP  - minimum residual algorithm applied to SQP system
    4. SSMminres   - minimum residual algorithm applied to Ax + b = 0 */

/* ==========================================================================
   === SSMboundary ==========================================================
   ==========================================================================
    Starting with a guess for the solution of the following problem,
    use SSM to refine the solution estimate until the error tolerance
    is fulfilled:

         Minimize x'Ax + 2x'b subject to x'x = r^2

    For the SQP step in SSM we need to solve a linear system of the form

 (SQP)  P(A + mu I)P y = -P(b + (A+ muI)x)

    where x = prior iterate, P = I - X*X', X = x/||x||
    The updated iterate and multiplier are

            x_new = x + y
            mu_new = rho (x_new), rho (x) = - (b+Ax)'x/||x||^2.
                            
    We use preconditioned MINRES to solve (SQP). Since we
    know that A + mu I should be positive semidefinite at the optimal
    mu, we choose mu large enough to ensure that A + mu I
    is positive definite. As a consequence, diag (A) + mu I is positive.
    To apply SSOR preconditioning, we need the diagonal of P(A + mu I)P
    positive. Since A + mu I is positive definite, x'P(A + mu I)Px = 0
    if and only if x is a multiple of w.  The diagonal of P(A + mu I)P
    has a zero if and only if ei'P(A + mu I)P*ei = 0 for some column
    ei of the identity. Hence, diag (P(A + mu I)P) is positive if and
    only if w is not a column of the identity */

PRIVATE int SSMboundary /* return 0 (error tolerance satisfied)
                                -1 (min residual convergence failure in SQP)
                                -2 (SSM failed to converge)
                                -3 (error decay in SSM too slow)
                                -5 (failure of QR diagonalization)
                                -6 (insufficient space in QR method)
                                -8 (solution in interior ||x|| < r) */
(
    FLOAT       *x, /* estimated solution to ball problem, a'x = 0 */
    SSMcom    *Com  /* SSMcom structure */
)
{
    int j, n, ssm_limit, done, status, BndLessThan, PrintLevel ;
    FLOAT s, t, fj, vj, ball_tol, mu, mulow, rad, prior_err,
          *eig_res, *eig_guess, *xj, *r0, *Ax, *Av, *b, *V, *v, *y ;
    SSMParm *Parm ; /* parameter structure */
    SSMProblem *PB ; /* problem structures  */

    PB = Com->PB ;
    n = PB->n ;
    rad = PB->rad ;
    b = PB->b ;
    Parm = Com->Parm ;
    BndLessThan = Parm->BndLessThan ;
    PrintLevel = Parm->PrintLevel ;
    ball_tol = Com->tol ;
    ssm_limit = Com->ssm_limit ;
    xj = Com->wx1 ;
    eig_res = Com->wx2 ;
    eig_guess = Com->wx3 ;
    V = Com->V ;
    v = Com->v ;
    Av = Com->Av ;
    Ax = Com->Ax ;
    r0 = Com->r0 ;
    y = Com->MINRES ;

    while (Com->error > ball_tol)
    {
        Restart:
        prior_err = Com->error ;
        Com->ssm_its++ ;
        if ( Com->ssm_its > ssm_limit )
        {
            if ( PrintLevel >= 1 )
            {
               // printf ("SSM refinement did not converge within %i \n""iterations as required by Parm->ssm_its_fac %e\n",(int) ssm_limit, Parm->ssm_its_fac) ;
            }
            return (-2) ;
        }
        /* convert x to unit vector to facilitate projection computation */
        t = SSMONE/rad ;
        for (j = 0; j < n; j++) V [j] = t*x [j] ;

        /* A + mulow I should be positive definite */
        mulow = SSMMIN (Com->eig_error - Com->emin, -Parm->eig_lower) ;
        if ( mulow < -PB->Dmin ) mulow = -PB->Dmin + Parm->eps*PB->Amax ;
        mu = Com->mu ;
        if ( mulow > mu ) mu = mulow + (mulow-mu) ;/* increase mu if necessary*/

        done = SSMminresP (xj, r0, x, Ax, b, V, mu, rad, PB, Com) ;
        /* solve subspace problem, optimize over x, v, r0, xj */
        status = SSMsubspace (x, rad, Ax, v, r0, xj, NULL, 4, SSMFALSE, Com) ;
        if ( PrintLevel >= 2 )
        {
           // printf ("\nSSM iteration: %i ball_err: %e ball_tol: %e\n",(int) Com->ssm_its, Com->error, ball_tol) ;
           // printf ("emin: %e mu: %e eig_err: %e\n",Com->emin, Com->mu, Com->eig_error) ;
        }
        if ( status ) return (status) ; /* failure of QR diagonalization */
        mu = Com->mu ;
        if ( done == 1 )
        {
            /* check if constraint inactive */
            if ( !Com->Active && (SSMMAX(-rad*Com->mu, SSMZERO) > ball_tol) )
                return (-8) ;
            /* otherwise constraint is active. Check to see if there
               was excessive growth of error in the evaluation of
               ball_err in minresP */
            if ( Com->error > ball_tol ) goto Restart ;
            return (0) ;
        }
        if ( done < 0 ) return (done) ;


        /* exit if constraint inactive */
        if ( BndLessThan && !Com->Active) return (-8) ;

        mu = Com->mu ;
        mulow = SSMMIN (Com->eig_error - Com->emin, -Parm->eig_lower) ;
        if ( mulow < -PB->Dmin ) mulow = -PB->Dmin + Parm->eps*PB->Amax ;
        /* check if eigenvalue should be refined when mulow > mu */
        if ( mulow > mu )
        {
            if ( PrintLevel >= 2 )
            {
               // printf ("mulow %e > mu %e eig_err: %e ball_err: %e\n",mulow, mu, Com->eig_error, Com->error) ;
            }
            mu = mulow + (mulow - mu) ;
            /* refine eigenvalue estimate when eigen error > x error */
            if ( Com->eig_error > Parm->eig_refine*Com->error/rad )
            {
                if ( PrintLevel >= 2 ) //printf ("refine eigenpair\n") ;
                /* be sure that A + mu I is positive definite */
                mu += ((FLOAT) 100)*Parm->eps*PB->Amax ;

                if ( Parm->IPM == SSMTRUE ) /* apply inverse power method */
                {
                    /* prepare for inverse power method iteration applied to
                       (A + mu I)x = v, current estimate for eigenvector.
                       Initial guess is x = s v where s is chosen to
                       minimize the residual ||(A + mu I)(s v) - v||.
                       Defining f = (A + mu I)v, we have s = f'v/f'f.
                       We now compute s, the residual (A + mu I)(s v) - v,
                       and sv the starting guess for the inverse power scheme */
                    s = SSMZERO ;
                    t = SSMZERO ;
                    for (j = 0; j < n; j++)
                    {
                        vj = v [j] ;
                        fj = Av [j] + mu*vj ;
                        s += fj*vj ;
                        t += fj*fj ;
                    }
                    s = s/t ;
                    t = mu*s - SSMONE ;
                    for (j = 0; j < n; j++)
                    {
                        vj = v [j] ;
                        /* starting guess */
                        eig_guess [j] = s*vj ;
                        /* = residual (A + mu I)(s v) - v */
                        eig_res [j] = s*Av [j] + t*vj ;
                    }
                    /* MINRES applied to (A + mu I)x = v */
                    SSMminres (xj, eig_res, eig_guess, 1.e20, mu,
                               SSMTRUE, PB, Com);
                    /* minimize over subspace spanned by previous vectors and
                       xj = solution - v */
                }
                else         /* use SQP method to improve eigenvalue estimate */
                {
                    /* The smallest eigenvalue is the solution to the
                       problem: min x'Ax subject to ||x|| = 1. The
                       multiplier associated with the constraint is the
                       eigenvalue. We apply one iteration of the SQP
                       method to the SQP system using the current
                       eigenvalue and eigenvector estimates as the
                       linearization points. We solve the SQP system
                       using SSMminresP (MINRES with projection) */
               
                    s = SSMZERO ;
                    for (j = 0; j < n; j++)
                    {
                        t = v [j] ;
                        s += t*t ;
                    }
                    s = sqrt (s) ;
                    if ( s > SSMZERO ) s = SSMONE/s ;
                    for (j = 0; j < n; j++) v [j] *= s ;
                    for (j = 0; j < n; j++) Av [j] *= s ;
                    SSMminresP (xj, Av, v, Av, NULL, v, mu, SSMONE, PB, Com);
                }
                /* add the new eigenvector estimate to the subspace and
                   reoptimize */
                status = SSMsubspace (x, rad, Ax, v, r0, xj, xj, 5, SSMFALSE,
                                      Com) ;
                if ( PrintLevel >= 2 )
                {
                   /* printf ("\nSSM iteration: %i ball_err: %e ball_tol: %e\n",
                             (int) Com->ssm_its, Com->error, ball_tol) ;
                    printf ("emin: %e mu: %e eig_err: %e\n",
                             Com->emin, Com->mu, Com->eig_error) ;*/
                }
                if ( status ) return (status) ; /* QR diagonalization failure */
            }
        }
        if ( Com->error <= ball_tol )
        {
            /* check if constraint inactive */
            if ( !Com->Active && (SSMMAX(-rad*mu, SSMZERO) > ball_tol) )
                return (-8) ;
            /* otherwise constraint is active and error tolerance satisfied */
            return (0) ;
        }

        /* return if convergence is slow */
        if ( Com->error > prior_err*Parm->ssm_decay ) return (-3) ;
    }
    return (0) ;
}

/* ==========================================================================
   === SSMinterior ==========================================================
   ==========================================================================
    Starting with an interior guess for the solution of the following problem,
    apply precondition MINRES to Ax = -b.

         Minimize x'Ax + 2x'b subject to x'x <= r^2

    If the iterate exceeds the norm constraint, then switch to
    SSMboundary. Otherwise, continue to apply MINRES iterations
    until convergence is achieved. If the convergence criterion is not
    satisfied, then we return to the Lanczos process and try to compute
    a better starting guess */

PRIVATE int SSMinterior /* return 0 (error tolerance satisfied)
                                -1 (minimum residual algorithm did not converge)
                                -5 (failure of QR diagonalization)
                                -6 (insufficient space in QR method)
                                -9 (solution on boundary ||x|| = r) */
(
    FLOAT       *x, /* estimated solution to sphere constrained problem */
    SSMcom    *Com  /* SSMcom structure */
)
{
    int j, n, done, status, PrintLevel ;
    FLOAT rad, t, normx, ball_tol, *xj, *y ;
    SSMParm *Parm ; /* parameter structure */
    SSMProblem *PB ;

    PB = Com->PB ;
    n = PB->n ;
    rad = PB->rad ;
    Parm = Com->Parm ;
    PrintLevel = Parm->PrintLevel ;
    ball_tol = Com->tol ;
    xj = Com->wx1 ;
    y = Com->MINRES ;

    if ( PB->Dmin == SSMZERO )
    {
        t = Parm->eps*PB->Amax ;
        for (j = 0; j < n; j++)
        {
            if ( PB->D [j] == SSMZERO ) PB->D [j] = t ;
        }
        PB->Dmin = t ;
    }
    Com->mu = SSMZERO ;

    done = SSMminres (xj, Com->r0, x, rad, SSMZERO, SSMFALSE, PB, Com) ;
    normx = Com->normx ;
    if ( done == 0  )               /* error tolerance achieved */
    {
        if ( normx <= rad )
        {
            for (j = 0; j < n; j++) x [j] = y [j] ;
            return (0) ;
        }
        else if ( normx - rad <= ball_tol )
        {
            t = rad/normx ;
            for (j = 0; j < n; j++) x [j] = t*y [j] ;
            return (0) ;
        }
        /* else treat as boundary solution */
        Com->error = SSMMAX (Com->error, normx - rad) ;
    }
    if ( done == -1 ) return (-1) ; /* MINRES convergence failure */

    /* solution lies on boundary - to ensure convergence, minimize objective
       over subspace, then return to boundary routine */
    normx = SSMZERO ;
    for (j = 0; j < n; j++)
    {
        t = x [j] ;
        normx += t*t ;
    }
    normx = sqrt (normx) ;
    status = SSMsubspace (x, normx, Com->Ax, Com->v, Com->r0, xj,
                          NULL, 4, SSMFALSE,Com);
    if ( PrintLevel >= 1 ) //printf ("interior solution reaches boundary\n") ;

    if ( status ) return (status) ;  /* QR method convergence failure */

    /* normalize x if constraint not active */
    if ( !Com->Active )
    {
        normx = SSMZERO ;
        for (j = 0; j < n; j++)
        {
            t = x [j] ;
            normx += t*t ;
        }
        normx = sqrt (normx) ;
        if ( normx > SSMZERO )
        {
            t = rad/normx ;
            for (j = 0; j < n; j++) x [j] *= t ;
        }
    }
    return (-9) ;                             /* solution on boundary */
}
 
/* ==========================================================================
   === SSMminresP (minimum residual algorithm with projection) =============
   ==========================================================================
    Apply MINRES to the SQP system:

       (A+muI)y + (x) nu = -[(A+muI)x + b]
         x'y             = 0

    Define X = x/||x|| and P = I - XX'. Make the change of variables
    y = Pz. Hence, the SQP system is equivalent to solving
   
            P(A + mu I)Pz = -P r0 where r0 = (A+muI)x + b.

    We solve this system using Algorithm 3 in the paper
    W. W. Hager, Iterative methods for nearly singular linear systems,
    SIAM Journal on Scientific Computing, 22 (2000), pp. 747-766.
   ==========================================================================*/

PRIVATE int SSMminresP /* return 1 if convergence tolerance met,
                                 0 if decay tolerance met
                                -1 for min residual convergence failure
                                   (too many iterations) */
(
    FLOAT      *xj, /* computed SQP iterate */
    FLOAT      *r0, /* b + Ax, cost gradient at starting point */
    FLOAT       *x, /* solution estimate */
    FLOAT      *Ax, /* A*x */
    FLOAT       *b,
    FLOAT       *X, /* x/||x|| */
    FLOAT       mu, /* multiplier for the constraint */
    FLOAT      rad, /* radius r */
    SSMProblem *PB, /* problem specification */
    SSMcom    *Com  /* pointer to SSMcom structure */
)
{
    int j, n, it, it_limit, PrintLevel ;
    FLOAT ball_tol, ball_err, sqp_err, s, t, tj,
          u, uj, uj1, ej, c, z1, nu, dj,
          *Ay, *r, *aj, *vj, *vj1, *wj, *pr, *rj1, *rj2, *zj1, *zj2,
          *d, *D, *ds, *p, *q, *y, Qj1 [4], Qj2 [2] ;
    SSMParm *Parm ;
    Parm = Com->Parm ;
    PrintLevel = Parm->PrintLevel ;
    n = PB->n ;
    D = PB->D ;
    ball_tol = Com->tol ;
    y = Com->MINRES ;
    Ay = y+(1*n) ;
    r  = y+(2*n) ;
    aj = y+(3*n) ;
    vj = y+(4*n) ;
    vj1= y+(5*n) ;
    wj = y+(6*n) ;
    pr = y+(7*n) ;
    rj1= y+(8*n) ;
    rj2= y+(9*n) ;
    zj1= y+(10*n) ;
    zj2= y+(11*n) ;

    d = Com->SSOR ; /* d  = diag (A + mu*I) */
    ds= d+(1*n) ;
    p = d+(2*n) ;
    q = d+(3*n) ; /* q = (A + mu*I)w */

    s = SSMZERO ;
    for (j = 0; j < n; j++) s += X [j]*r0 [j] ;
    for (j = 0; j < n; j++) r [j] = s*X [j] - r0 [j] ;
    t = SSMZERO ;
    for (j = 0; j < n; j++) t += X [j]*Ax [j] ;
    t /= rad ;
    u = SSMONE/rad ;
    for (j = 0; j < n; j++)
    {
        s = u*Ax [j] ;
        p [j]  = s - X [j]*t ;
        q [j]  = s + X [j]*mu ;
    }
    /* compute diagonal of SSOR matrix */
    s = INF ;
    for (j = 0; j < n; j++)
    {
        t = mu + D [j] - (q [j] + p [j])*X [j] ;
        if ( t < s ) s = t ;
        d [j] = t ;
    }

    if ( s <= SSMZERO ) /* try to make diagonal positive */
    {
        /* if mu is increased by t, the diagonal increases by t(1-Xj^2) */
        u = SSMZERO ;
        c = (FLOAT) 100 * Parm->eps * PB->Amax ; /* min diagonal element */
        for (j = 0; j < n; j++)
        {
            if ( d [j] <= c )
            {
                t = SSMONE - X [j]*X [j] ;
                if ( t > SSMZERO )
                {
                    t = (c-d [j])/t ;
                    if ( t > u ) u = t ;
                }
                else
                {
                    u = -SSMONE ;
                    break ;
                }
            }
        }
        if ( u < SSMZERO ) /* the matrix cannot be fixed, force d > 0 */
        {
            s = c - s ;
            for (j = 0; j < n; j++) d [j] += s ; /* make diagonal >= c */
        }
        else
        {
            for (j = 0; j < n; j++)
            {
                t = X [j] ;
                q [j] += t*u ;
                d [j] += u*(SSMONE-t*t) ;
            }
        }
    }
    for (j = 0; j < n; j++) ds [j] = sqrt (d [j]) ;

    SSMSSORmultP (vj, r, wj, X, aj, mu, (int) 1, PB, Com) ;
    uj = SSMZERO ;
    ej = SSMZERO ;
    for (j = 0; j < n; j++)
    {
        ej += vj [j]*vj [j] ;
        xj [j] = SSMZERO ;
        Ay [j] = SSMZERO ;
        rj1 [j] = SSMZERO ;
        rj2 [j] = SSMZERO ;
        zj1 [j] = SSMZERO ;
        zj2 [j] = SSMZERO ;
        vj1 [j] = SSMZERO ;
    }
    ej = sqrt (ej) ;
    for (j = 0; j < n; j++) vj [j] /= ej ;
    Qj1 [0] = SSMONE ;
    Qj1 [1] = SSMZERO ;
    Qj1 [2] = SSMZERO ;
    Qj1 [3] = SSMONE ;
    Qj2 [0] = SSMZERO ;
    Qj2 [1] = SSMZERO ;
    /* do not perform more iterations than was done in Lanczos process */
    if ( PB->DT->nalloc == Parm->Lanczos_bnd ) it_limit = Com->minres_limit ;
    else               it_limit = SSMMIN (Com->minres_limit, 2*PB->DT->nalloc) ;
    for (it = 0; it < it_limit; it++)
    {
        Com->minres_its++ ;
        SSMSSORmultP (pr, vj, wj, X, aj, mu, (int) 0, PB, Com) ;
        dj = SSMZERO ;
        for (j = 0; j < n; j++)
        {
            dj += pr [j]*vj [j] ;
        }
        uj1 = uj ;
        uj = SSMZERO ;
        for (j = 0; j < n; j++)
        {
            t = pr [j] - vj [j]*dj - vj1 [j]*uj1 ;
            pr [j] = t ;
            uj += t*t ;
        }
        uj = sqrt (uj) ;
        for (j = 0; j < n; j++)
        {
            vj1 [j] = vj [j] ;
            vj [j] = pr [j]/uj ;
        }
        tj = Qj2 [0]*uj1 ;
        t  = Qj2 [1]*uj1 ;
        uj1 = Qj1 [0]*t + Qj1 [2]*dj ;
        dj  = Qj1 [1]*t + Qj1 [3]*dj ;
        Qj2 [0] = Qj1 [2] ;
        Qj2 [1] = Qj1 [3] ;
        t = sqrt (dj*dj + uj*uj) ;
        if ( t != SSMZERO )
        {
            c = dj/t ;
            s = uj/t ;
        }
        else
        {
            c = SSMONE ;
            s = SSMZERO ;
        }
        Qj1 [0] = c ;
        Qj1 [1] =-s ;
        Qj1 [2] = s ;
        Qj1 [3] = c ;
        z1 = ej*c ;
        ej = -s*ej ;
        dj = c*dj + s*uj ;
        u = SSMZERO ;
        for (j = 0; j < n; j++)
        {
            t = (wj [j] - zj1 [j]*uj1 - zj2 [j]*tj)/dj ;
            zj2 [j] = zj1 [j] ;
            zj1 [j] = t ;
            s = xj [j] + t*z1 ;
            xj [j] = s ;
            t = x [j] + s ;
            y [j] = t ;
            u += t*t ;
            t = (aj [j] - rj1 [j]*uj1 - rj2 [j]*tj)/dj ;
            rj2 [j] = rj1 [j] ;
            rj1 [j] = t ;
            Ay [j] += t*z1 ;
        }

        u = rad/sqrt (u) ;
        /* y = x_old + xj, the updated SQP iteration, u is the scale
           factor to ensure that y has the correct norm */
        for (j = 0; j < n; j++)
        {
            y [j] *= u ;
            wj [j] = u*(Ax [j] + Ay [j]) ;
        }
        t = SSMZERO ;
        if ( b == NULL )
        {
            for (j = 0; j < n; j++) t += wj [j]*y[j] ;
        }
        else
        {
            for (j = 0; j < n; j++)
            {
                s = b [j] + wj [j] ; /* wj = Ay */
                wj [j] = s ;
                t += s*y[j] ;
            }
        }
        nu = -t/(rad*rad) ;
        Com->mu = nu ;

        /* ball_err = ||b + Ay + nu*y||, nu chosen to minimize residual */
        ball_err = SSMZERO ;
        for (j = 0; j < n; j++)
        {
            t = y [j]*nu + wj [j] ;
            ball_err += t*t ;
        }
        ball_err = sqrt (ball_err) ;
        Com->error = ball_err ;
        /* include factor .5 in ball_tol since ball_err polluted by error */
        if ( ball_err <= .5*ball_tol ) return (1) ;
        s = SSMZERO ;
        for (j = 0; j < n; j++)
        {
            t = r0 [j] + Ay [j] + xj [j]*mu ;
            /* r = b + A(x+xj) + mu*xj = r0 + (A + mu I)xj, xj orthogonal to x*/
            r [j] = t ;
            s += t*X [j] ;
        }

        /* the error in the the SQP linear system is sqp_err = ||Pr||,
           which we compute below.  The error along x can be set to zero
           by an appropriate choice of nu in the SQP system
           (A + mu I)xj + nu x = -r0  */
        sqp_err = SSMZERO ;
        for (j = 0; j < n; j++)
        {
            t = r [j] - X [j]*s ;
            sqp_err += t*t ;
        }
        sqp_err = sqrt (sqp_err) ;
        if ( PrintLevel >= 3 )
        {
            //printf ("minres it: %i err: %e ball_err: %e ball_tol: %e\n",
                  //   (int) it, sqp_err, ball_err, ball_tol) ;
        }
        if ( sqp_err <= Parm->sqp_decay*ball_err ) return (0) ;
    }
    if ( PrintLevel >= 2 )
    {
        //printf ("after %i minres iterations, convergence not achieved\n",
                // (int) it) ;
    }
    return (-1) ;
}
/* ==========================================================================
   === SSMminres ===========================================================
   ==========================================================================
    Solve the unconstrained optimization problem

         Minimize x'(A + mu I)x + 2x'b

   This differs from SSMminresP in that there is no projection.
   If this is routine called from the interior point routine,
   then mu is zero. If this is called by the boundary routine,
   then we are refining the estimate for the eigenvector associated with the
   smallest eigenvalue. In the refinement process, we apply one iteration
   of the inverse power method. When called from the interior point
   routine, convergence is achieved either when the requested error
   tolerance has been satisfied, or when the solution norm exceeds
   the constraint radius. For the inverse power method, convergence is
   achieved when the relative residual for the eigenvector estimate
   decreases by the factor eig_decay.
   ==========================================================================*/

PRIVATE int SSMminres /* return 0 if ||Ax + b|| <= ball_tol
                               -1 for convergence failure (too many iterations)
                               -2 if ||x|| > r */
(
    FLOAT      *xj, /* computed SQP iterate */
    FLOAT      *r0, /* b + Ax, cost gradient at starting point */
    FLOAT       *x, /* solution estimate */
    FLOAT      rad, /* radius of sphere */
    FLOAT       mu, /* safeguard, (A + mu I) positive definite */
    int        IPM, /* TRUE (inverse power method), FALSE (interior point) */
    SSMProblem *PB, /* problem specification */
    SSMcom    *Com  /* pointer to SSMcom structure */
)
{
    int PrintLevel ;
    int j, n, it, it_limit ;
    FLOAT ball_tol, ball_err, eig_err, eig_tol, normx, s, t, tj,
          uj, uj1, ej, c, z1, dj,
          *Ay, *r, *aj, *vj, *vj1, *wj, *pr, *rj1, *rj2, *zj1, *zj2,
          *d, *D, *ds, *y, Qj1 [4], Qj2 [2] ;
    SSMParm *Parm ;
    Parm = Com->Parm ;
    eig_tol = Parm->eig_decay ;
    PrintLevel = Parm->PrintLevel ;
    n = PB->n ;
    D = PB->D ;
    ball_tol = Com->tol ;
    y = Com->MINRES ;
    Ay = y+(1*n) ;
    r  = y+(2*n) ;
    aj = y+(3*n) ;
    vj = y+(4*n) ;
    vj1= y+(5*n) ;
    wj = y+(6*n) ;
    pr = y+(7*n) ;
    rj1= y+(8*n) ;
    rj2= y+(9*n) ;
    zj1= y+(10*n) ;
    zj2= y+(11*n) ;

    d = Com->SSOR ; /* d  = diag (A + mu*I) */
    ds= d+(1*n) ;

    for (j = 0; j < n; j++) r [j] = -(r0 [j] + mu*x [j]) ;
    for (j = 0; j < n; j++)
    {
        t = D [j] + mu ;
        ds [j] = sqrt (t) ;
        d [j] = t ;
    }

    SSMSSORmult (vj, r, wj, aj, d, ds, mu, SSMTRUE, PB, Com) ;
    uj = SSMZERO ;
    ej = SSMZERO ;
    for (j = 0; j < n; j++)
    {
        ej += vj [j]*vj [j] ;
        xj [j] = SSMZERO ;
        Ay [j] = SSMZERO ;
        rj1 [j] = SSMZERO ;
        rj2 [j] = SSMZERO ;
        zj1 [j] = SSMZERO ;
        zj2 [j] = SSMZERO ;
        vj1 [j] = SSMZERO ;
    }
    ej = sqrt (ej) ;
    for (j = 0; j < n; j++) vj [j] /= ej ;
    Qj1 [0] = SSMONE ;
    Qj1 [1] = SSMZERO ;
    Qj1 [2] = SSMZERO ;
    Qj1 [3] = SSMONE ;
    Qj2 [0] = SSMZERO ;
    Qj2 [1] = SSMZERO ;
    it_limit = Com->minres_limit ;
    for (it = 0; it < it_limit; it++)
    {
        Com->minres_its++ ;
        SSMSSORmult (pr, vj, wj, aj, d, ds, mu, SSMFALSE, PB, Com) ;
        dj = SSMZERO ;
        for (j = 0; j < n; j++)
        {
            dj += pr [j]*vj [j] ;
        }
        uj1 = uj ;
        uj = SSMZERO ;
        for (j = 0; j < n; j++)
        {
            t = pr [j] - vj [j]*dj - vj1 [j]*uj1 ;
            pr [j] = t ;
            uj += t*t ;
        }
        uj = sqrt (uj) ;
        for (j = 0; j < n; j++)
        {
            vj1 [j] = vj [j] ;
            vj [j] = pr [j]/uj ;
        }
        tj = Qj2 [0]*uj1 ;
        t  = Qj2 [1]*uj1 ;
        uj1 = Qj1 [0]*t + Qj1 [2]*dj ;
        dj  = Qj1 [1]*t + Qj1 [3]*dj ;
        Qj2 [0] = Qj1 [2] ;
        Qj2 [1] = Qj1 [3] ;
        t = sqrt (dj*dj + uj*uj) ;
        if ( t != SSMZERO )
        {
            c = dj/t ;
            s = uj/t ;
        }
        else
        {
            c = SSMONE ;
            s = SSMZERO ;
        }
        Qj1 [0] = c ;
        Qj1 [1] =-s ;
        Qj1 [2] = s ;
        Qj1 [3] = c ;
        z1 = ej*c ;
        ej = -s*ej ;
        dj = c*dj + s*uj ;
        for (j = 0; j < n; j++)
        {
            t = (wj [j] - zj1 [j]*uj1 - zj2 [j]*tj)/dj ;
            zj2 [j] = zj1 [j] ;
            zj1 [j] = t ;
            s = xj [j] + t*z1 ;
            xj [j] = s ;
            t = x [j] + s ;
            y [j] = t ;
            t = (aj [j] - rj1 [j]*uj1 - rj2 [j]*tj)/dj ;
            rj2 [j] = rj1 [j] ;
            rj1 [j] = t ;
            Ay [j] += t*z1 ;
        }
        /* compute ball_err = ||b + A(x+xj)|| = ||r0 + Axj|| if b != NULL
                            = ||A(x+xj)||                    otherwise
                   sqp_err  = ||r0 + mu(x+xj)||              if b != NULL
                            = ||A(x+xj) + mu(x+xj)||         otherwise */

        /* termination conditions for Ax = b and for the inverse power method
           are slightly different. For Ax = b, we continue until the residual
           satisfies the convergence criterion. For the inverse power
           method, we require that the residual norm is less than eig_tol */

        if ( IPM == SSMTRUE )
        {
            eig_err = SSMZERO ;
            normx = SSMZERO ;
            for (j = 0; j < n; j++)
            {
                s = y [j] ;
                t = r0 [j] + Ay [j] + mu*s ;   /* Ay = Axj */
                eig_err += t*t ;
                normx += s*s ;
            }
            normx = sqrt (normx) ;
            eig_err = sqrt (eig_err)/SSMMAX (SSMONE, normx) ;
            if ( PrintLevel >= 3 )
            {
               // printf ("EIGRES it: %i eig_res: %e eig_tol: %e\n",
                    // (int) it, eig_err, eig_tol) ;
            }
            if ( (normx > rad) || (eig_err <= eig_tol) ) return (0) ;
            /* return when constraint is violated
               or the relative residual <= eig_tol */
        }
        else
        {
            ball_err = SSMZERO ;
            normx = SSMZERO ;
            for (j = 0; j < n; j++)
            {
                t = r0 [j] + Ay [j] ;   /* Ay = Axj */
                ball_err += t*t ;
                s = y [j] ;
                normx += s*s ;
            }
            ball_err = sqrt (ball_err) ;
            normx = sqrt (normx) ;
            Com->normx = normx ;
            Com->error = ball_err ;
            if ( PrintLevel >= 3 )
            {
               // printf ("MINRES it: %i ball_err: %e ball_tol: %e\n",
                        // (int) it, ball_err, ball_tol) ;
            }
            /* include factor .5 since ball_err polluted by errors */
            if ( ball_err <= .5*ball_tol ) return (0) ;
            /* return when constraint is sufficiently violated */
            if ( normx > rad*Parm->radius_flex ) return (-2) ;
        }
    }
    if ( PrintLevel >= 2 )
    {
        //printf ("after %i iterations, MINRES has not converged\n", it) ;
    }
    return (-1) ;
}
/*
Version 1.1 Change:
    1. In SSMdiagopt, use a separate block of code to treat the interior
       solution case instead of trying to do both interior and boundary
       solution at the same time.
    2. In SSMdiagopt, when solving for the multiplier, retain the
       prior iterate in case it is more accurate than the current iterate.
*/
