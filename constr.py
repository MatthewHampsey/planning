import numpy as np
import matplotlib.pyplot as plt

def interiorPoint(initx, f, gradf, hessian, eqConstraint, eqConstraintLen, eqConstraint_j, ineqConstraint, ineqConstraintLen, ineqConstraint_j, t_mult, k_upper, verbose=False):
    x = initx
    lambd = 20.0*np.ones(ineqConstraintLen)
    mu = 10.0*np.ones(eqConstraintLen)
    k = 1
    t = 0.001

    beta = 0.8
    alpha = 0.1

    def residual(x, lambd, mu, t):
        return np.concatenate((gradf(x) + ineqConstraint_j(x).T.dot(lambd) + eqConstraint_j(x).T.dot(mu), 
                    -np.diag(lambd).dot(ineqConstraint(x)) - (1.0/t)*np.ones(ineqConstraintLen), eqConstraint(x)))

    while(True):

        hessx = hessian(x)
        eqConstrJac = eqConstraint_j(x)
        ineqConstrJac = ineqConstraint_j(x)
        lambdDiag = np.diag(lambd)
        fDiag = np.diag(ineqConstraint(x))

        transition_mat = np.concatenate(
            (np.concatenate(
                (hessx, ineqConstrJac.T, eqConstrJac.T), axis=1),
             np.concatenate(
                (-lambdDiag.dot(ineqConstrJac), -fDiag, np.zeros(shape=(ineqConstraintLen, eqConstraintLen))), axis=1),
             np.concatenate(
                (eqConstrJac, np.zeros(shape=(eqConstraintLen, ineqConstraintLen)), np.zeros(shape=(eqConstraintLen, eqConstraintLen))), axis=1))
            )

        sol = np.linalg.solve(transition_mat, -residual(x, lambd, mu, t))

        x_delta     = sol[:len(x)]
        lambd_delta = sol[len(x):len(x)+ineqConstraintLen]
        mu_delta    = sol[len(x)+ineqConstraintLen:]

        s = 1.0

        while(s > 0.0):
            temp_lambd = lambd + s*lambd_delta
            if (temp_lambd >= 0).all():
                break
            s -= 0.1
        if s <= 0.0 + 0.000001:
            s = 0.0

        while (ineqConstraint(x + s*x_delta) > 0.0 ).any() :
            s *= beta

        while( np.linalg.norm(residual(x + s*x_delta, 
                                       lambd + s*lambd_delta, 
                                       mu + s*mu_delta, 
                                       t))**2 > (1 - alpha*s)*np.linalg.norm(residual(x, lambd, mu, t))**2):
            s *= beta

        x = x + s*x_delta
        lambd = lambd + s*lambd_delta
        mu = mu + s*mu_delta

        res = residual(x, lambd, mu, t)

        if verbose:
            print("residuals: ", np.linalg.norm(res[:len(x)]), " ", np.linalg.norm(res[len(x):len(x) + ineqConstraintLen]), " ", np.linalg.norm(res[len(x) + ineqConstraintLen:len(x) + ineqConstraintLen + eqConstraintLen]))
            
        surr_dual_gap = -ineqConstraint(x).dot(lambd)
        if np.linalg.norm(residual(x, lambd, mu, t))**2 < 0.0002 and surr_dual_gap < 0.0002:
            break
        
        if k >= k_upper:
            print("solution not found")
            break

        t = t_mult*ineqConstraintLen/surr_dual_gap
        if verbose:
            print("1/t: ", 1/t)
        k += 1
    if verbose:
        print("k: ", k)
    return x