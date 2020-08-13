#include <math.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

/* include the user-visible definitions for SSM.h */
#include "SSM_user.h"

/* ========================================================================== */
/* === macros =============================================================== */
/* ========================================================================== */

#define PRIVATE static
#define PUBLIC
#define SSMFALSE 0
#define SSMTRUE 1
#define SSMONE ((FLOAT) 1)
#define SSMZERO ((FLOAT) 0)
#define SSMHALF ((FLOAT) .5)

#define SSMMIN(a,b) (((a) < (b)) ? (a) : (b))
#define SSMMAX(a,b) (((a) > (b)) ? (a) : (b))

#define INF DBL_MAX

/* ========================================================================== */
/* === structures =========================================================== */
/* ========================================================================== */


typedef struct SSMDiag     /* diagonalization of tridiagonal matrix */
{
    int                n ; /* dimension of matrix */
    FLOAT           tol1 ; /* stop when u_{k-1}^2 <= qr_tol1*|d_k| */
    FLOAT           tol2 ; /* stop when |u_{k-1}| <= qr_tol2 */
    int          max_its ; /* max QR iterations (default: 4n) */
    int              its ; /* actual number of QR iterations */
    INT           nscale ; /* number of scaling operations */
    INT          ngivens ; /* number of Givens rotation */
    FLOAT             *e ; /* eigenvalues of tridiagonal matrix, size nalloc */
    int           nalloc ; /* space allocated for diagonal, must be >= n */
    FLOAT       qr_lower ; /* lower bound for scaling in qr method */
    FLOAT            *gx ; /* first Givens factor, size=max_its*nalloc */
    FLOAT            *gy ; /* second Givens factors, same size as gx */
    short            *gz ; /* 1 or 2, type of Givens transform, size of gx */
    FLOAT            *gs ; /* fast Givens scale factors, same size as gx */
    int              *gi ; /* row indices associated with scale factors */
    INT              *gj ; /* indexed by QR pass, points into Givens factors*/
    int              *gk ; /* index of last diagonal element in QR iteration */
    int              *gf ; /* index of first diagonal element in QR iteration */
} SSMDiag ;

typedef struct SSMLanczos  /* Lanczos/Householder tridiagonalization structure*/
{
    int          max_its ; /* upper bound on the number of Lanczos iterations */
    FLOAT            tol ; /* termination tolerance for u_j in Lanczos iter */
    int      House_limit ; /* Householder tridiag if n <= House_limit */
    FLOAT             *V ; /* orthonormal vectors, V'AV = T is tridiagonal */
    int            ncols ; /* number of cols in the Lanczos orthogonal matrix*/
    int            nrows ; /* number of rows in the Lanczos orthogonal matrix*/
    FLOAT             *d ; /* diagonal of tridiagonal T */
    FLOAT             *u ; /* upper diagonal of tridiagonal T */
} SSMLanczos ;

typedef struct SSMDiagOpt  /* optimization of diagonal quadratic */
{
    int                n ; /* dimension */
    FLOAT             *d ; /* diagonal matrix */
    FLOAT             *f ; /* linear term in cost function */
    FLOAT            *f2 ; /* square of linear term in cost function */
    FLOAT        *dshift ; /* d - min (d) */
    FLOAT           dmin ; /* dmin = min (d) */
    FLOAT           imin ; /* index of minimal component of d */
    FLOAT            tol ; /* relative error tolerance for optimal multiplier */
    FLOAT             mu ; /* multiplier associated with minimizer */
} SSMDiagOpt ;

typedef struct SSMProblem  /* problem specification */
{
    int                n ; /* problem dimension */
    FLOAT             *x ; /* edge weights */
    int               *i ; /* adjacent vertices for each node */
    INT               *u ; /* location last super diag in col (used in SSOR) */
    INT               *p ; /* points into x and i, p [0], ..., p [n] */
    FLOAT             *D ; /* diagonal of objective matrix */
    FLOAT             *b ; /* linear term in objective function */
    FLOAT            rad ; /* radius of sphere */
    FLOAT           Amax ; /* absolute largest element in matrix */
    FLOAT           Dmin ; /* smallest diagonal element */
    SSMDiag          *DT ; /* diagonalization structure for tridiagonal matrix*/
    SSMDiagOpt       *DO ; /* diagonal optimization structure */
    SSMLanczos       *LH ; /* Lanczos/Householder structure */
} SSMProblem ;

typedef struct SSMcom
{
    int             *wi1 ; /* work array of size n */
    int             *wi2 ; /* work array of size n */
    FLOAT           *wx1 ; /* work array of size n, wx1 */
    FLOAT           *wx2 ; /* work array of size n, wx2 = wx1+n */
    FLOAT           *wx3 ; /* work array of size n, wx3 = wx2+2n */
    FLOAT          *SSOR ; /* work array of size 4n for SSOR d, s, p, q */
    FLOAT        *MINRES ; /* work array of size 12n needed for MINRES */
    FLOAT             *W ; /* work array of size 5n for Householder vectors */
    FLOAT             *V ; /* work array of size 5n for orthogonal vectors */
    FLOAT           *VAV ; /* work array of size 25 for storing V'AV */
    FLOAT             *v ; /* current eigenvector estimate */
    FLOAT            *Av ; /* A times v */
    FLOAT            *Ax ; /* A times x */
    FLOAT            *r0 ; /* Ax + b */
    int           Active ; /* TRUE (||x|| = r), FALSE (||x|| < r) */
    FLOAT           emin ; /* current minimum eigenvalue estimate */
    FLOAT             mu ; /* current multiplier estimate */
    FLOAT          normx ; /* norm of x */
    FLOAT            tol ; /* KKT error tolerance for solution */
    SSMProblem       *PB ; /* problem specification */
    SSMProblem  *PBdense ; /* structure for dense subproblems */
    SSMParm        *Parm ; /* parameters */
    double         mults ; /* number of matrix/vector products */
    int          ssm_its ; /* number of SSM iterations */
    int        ssm_limit ; /* limit on number of SSM iterations */
    int       minres_its ; /* number of minimum residual iterations */
    int     minres_limit ; /* limit on minres iterations */
    FLOAT          error ; /* 2-norm of KKT error */
    FLOAT      eig_error ; /* relative 2-norm of eigenvector error */

/* check info */
    FLOAT       cost_old ; /* cost previous time objective function checked */
    FLOAT       emin_old ; /* prior eigenvalue estimate */
} SSMcom ;

