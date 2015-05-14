from ..integration import *

from nose.tools import assert_equal
from nose.tools import assert_not_equal
from nose.tools import assert_raises
from nose.tools import raises

import sympy
import random


#This is lame nose testing, but it'll do for now.
def test_integration():
    # Test handling of symbolic discontinuities
    x, y, z = [sympy.Symbol(var, real=True) for var in ['x', 'y', 'z']]
    ranges = [[x, -.25, 1.25], [y, -.25, 1.25], [z, -.25, 1.25]]
    sym_disc = x ** 2 + y ** 2 + z ** 2 - 1
    disc = Discontinuity(sym_disc, ranges)
    eqns = [- sympy.sqrt(1 - y ** 2 - z ** 2), sympy.sqrt(1 - y ** 2 - z ** 2)]
    sym_sols = list(disc.sym_sols)
    for eqn in eqns:
        for ind, sym_sol in enumerate(sym_sols):
            if sympy.Eq(eqn, sym_sol):
                sym_sols.pop(ind)
                break
        else:
            raise DiscontinuityError(
                "Discontinuity test returned incorrect symbolic solutions!")
    yrand, zrand = [.5 * random.random() - .25 for ind in [0, 1]]
    lambda_sol = disc._lambdified(yrand, zrand)
    subs_sol = [sym_sol.subs({y: yrand, z: zrand})
                for sym_sol in disc.sym_sols]
    err = [((lambda_sol[ind] - subs_sol[ind]) ** 2) ** .5
           for ind in range(len(subs_sol))]
    if max(err) > 1 * 10 ** - 13:
        raise DiscontinuityError(
            "Lambdified solution does not match symbolic solution!")
    test_children = [y ** 2 + z ** 2 - 1, y ** 2 + z ** 2 - 15. / 16]
    for test in test_children:
        for child in disc.children:
            if sympy.Equivalent(test, child._disc):
                break
        else:
            raise DiscontinuityError(
                "Direct children do not match!")
    test_discs = [[disc_._disc for disc_ in level] for level in
                  Discontinuities([sym_disc], ranges).leveled_discs]
    sol_discs = [[x ** 2 + y ** 2 + z ** 2 - 1],
                 [y ** 2 + z ** 2 - 1, y ** 2 + z ** 2 - 15. / 16],
                 [z ** 2 - 1, z ** 2 - 15. / 16, z ** 2 - 7. / 8]]
    for inda in range(len(test_discs)):
        if not set(test_discs[ind]) == set(sol_discs[ind]):
            raise DiscontinuityError("Levelled discontinuities do not match!")
    test = Discontinuities([sym_disc], ranges)
    test2 = test.nquad_disc_functions
    args_list = [[.5 * random.random() - .25 for ind in range(inds)]
                 for inds in [2, 1, 0]]
    vars_list = [[y, z], [z], []]
    for ind, level in enumerate(test.leveled_discs):
        subs_points = [sym_sol.subs(dict(zip(vars_list[ind], args_list[ind])))
                       for disc_ in level for sym_sol in disc_.sym_sols]
        subs_points.sort()
        lambda_points = test2[ind](*args_list[ind])
        lambda_points.sort()
        err = [((subs_points[ind] - lambda_points[ind]) ** 2) ** .5 < 10 ** -13
               for ind in range(len(subs_points))]
        if [item for item in err if not item]:
            raise DiscontinuityError(
                "Lambdified functions do not match symbolic functions!")

    # Test integration of functions
    test_discs = [x ** 2 + y ** 2 - 1]
    # This takes too long at present. Only integrate over x, y.
    test4 = IntegrableFunction(H(test_discs[0]),
                               ranges[: - 1], test_discs)
    # Check against evaluation in Mathematica.
    if ((test4.integrate()[0] - 0.907360054182972) ** 2) ** .5 > 10 ** - 7:
        raise IntegrationError(
            "Incorrect integration result for 2-D integration!")

