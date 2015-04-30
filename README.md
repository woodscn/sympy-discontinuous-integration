# sympy-discontinuous-integration

Symbolic preprocessing for numerical evaluation of multidimensional, discontinuous integrals.

Obtaining accurate results for the numerical evaluation of discontinuous integrals requires that the user intelligently subdivide the integral into segments wherein the integrand is smooth. This process is well-understood and straightforward for one-dimensional integrals, however it becomes much more difficult when dealing with integrals in two or more dimensions. This problem has traditionally been solved using adaptive integrators, but these can be slow and inaccurate for some problems. It is possible to leverage prior knowledge of the discontinuities inherent in the integrand function to exactly fit the form of discontinuities for multidimensional integrals using the algorithm described in: 

Woods, C Nathan, and Ryan P Starkey. “Verification of Fluid - Dynamic Codes in the Presence of Shocks and Other Discontinuities.” Journal of Computational Physics 294 (2015): 312–28, available until June 2, 2015 at http://authors.elsevier.com/a/1QsPV508HRs4u. 

This package implements this algorithm using Sympy and SciPy tools. 
