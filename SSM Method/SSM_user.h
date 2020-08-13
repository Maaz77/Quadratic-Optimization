/* ========================================================================== */
/* === SSM_user.h =========================================================== */
/* ========================================================================== */
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#define FLOAT double
#define INT int
#ifndef NULL
#define NULL 0
#endif

/* User code should include SSM_user.h and nothing else */

typedef struct SSMParm      /* parameters */
{
    unsigned int seed ;     /* seed to start srand; do not use srand if zero */

    FLOAT            eps ; /* machine epsilon for FLOAT  */
    int     Lanczos_dim1 ; /* Lanczos iterations L:           */
    int     Lanczos_dim2 ; /*   if      ( n <= dim1 ) L = n-1 */
    int       Lanczos_L1 ; /*   else if ( n <= dim2 ) L = L1  */
    int       Lanczos_L2 ; /*   else             L = L1 + L2*(log10 (n/dim2)) */
    int      Lanczos_bnd ; /* upper bound on number of Lanczos iterations */
    FLOAT    Lanczos_tol ; /* stop Lanczos when |u_j|/max |A| <= tol */
    int      House_limit ; /* use Householder when n <= limit */
    int           qr_its ; /* max number QR algorithm iterations*/
    FLOAT     qr_tol_rel ; /* relative eigenvalue convergence tol*/
    FLOAT     qr_tol_abs ; /* absolute eigenvalue convergence tol*/
    FLOAT       qr_lower ; /* lower bound for qr scaling factor */
    FLOAT diag_optim_tol ; /* accuracy of multiplier for diagonal matrix */
    FLOAT       diag_eps ; /* bound on error in diagonal of tridiag matrix */
    FLOAT      check_tol ; /* tolerance used when checking QP structure */
    FLOAT      check_kkt ; /* tolerance used when checking KKT error */
    FLOAT     eig_refine ; /* factor determines when to refine eig in SSM */
    FLOAT    radius_flex ; /* flex factor for radius in interior routines */
    FLOAT      sqp_decay ; /* factor determines when to stop SQP iteration */
    FLOAT      eig_decay ; /* factor determines when to stop inverse iteration*/
    FLOAT      eig_lower ; /* lower bound smallest eigenvalue (-INF default) */
    int              IPM ; /* TRUE (eigenvalue estimate using inverse power
                              method), FALSE (... using SQP method) */
    FLOAT      ssm_decay ; /* required error decay factor in SSM, else restart*/
    FLOAT         shrink ; /* mu multiplied by shrink in diagopt */
    FLOAT minres_its_fac ; /* max number minres iterations in SSM is fac*n */
    FLOAT    ssm_its_fac ; /* max number SSM iterations is fac*n */
    FLOAT   grow_Lanczos ; /* factor to grow number of Lanczos iterations */
    int         nrestart ; /* number of Lanczos restarts attempted */
    int       PrintLevel ; /* Level 0  = no print, ... , Level 2 = max print*/
    int       PrintParms ; /* TRUE means print parameter values */
    int       PrintFinal ; /* TRUE means print status and statistics */
    int      BndLessThan ; /* TRUE means ||x|| <= r, FALSE means ||x|| = r */
} SSMParm ;

typedef struct ssm_stat_struct /* statistics returned to user */
{
    int              mults ; /* number of matrix/vector multiplications */
    int            ssm_its ; /* number of SSM iterations */
    int         minres_its ; /* number of minimum residual iterations */
    int        restart_its ; /* number of times Lanczos restarted */
    FLOAT            error ; /* 2-norm of KKT error */
} ssm_stat ;

/* prototypes */

int SSM /* return 0 (error tolerance satisfied)
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
    int          n, /* size of x, constraint lo <= a'x <= hi */
    FLOAT      *Ax, /* numerical entries in matrix, packed array */
    int        *Ai, /* row indices for each column */
    INT        *Ap, /* points into Ax and Ai, Ap [j] = start of column j */
    FLOAT       *b, /* linear term in objective function */
    FLOAT        r, /* radius of ball */
    FLOAT      tol, /* KKT error tolerance for solution */
    FLOAT   *guess, /* starting, NULL means no guess given */
    SSMParm *UParm/* user parameters, NULL means use default parameters */
) ;

void SSMdefault
(
    SSMParm  *Parm /* pointer to parameter structure */
) ;
