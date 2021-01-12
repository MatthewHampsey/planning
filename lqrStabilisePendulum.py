import constr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy.optimize
import sys
import planPendulum
#import lqr

if __name__ == '__main__':
    np.set_printoptions(linewidth=160)
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(formatter={'float': lambda x: "{0: 0.2f}".format(x)})


    planLength = 250
    dimx = 4
    dimu = 1
    dimtot = dimx+dimu
    
    x0 = np.array([-2.0, 0.0, 0.0, 0.0])
    
    x1 = np.array([0.0, 3.14159265, 0.0, 0.0])
    dt = 0.02

    m1 = 0.3
    m2 = 0.02
    l = 3.2
    g = 9.81
    pendulum_params = {
        'm1': m1,
        'm2': m2,
        'l': l,
        'g': g
    }

    plan = planPendulum.planPendulumTrajectory(planLength, dt, dimx, dimu, x0, x1, pendulum_params)

    def computeQdot(state, u):
        x = state[0:2]
        x_dot = state[2:4]
        M_q = np.array([[m1 + m2, m2*l*np.cos(x[1])],[m2*l*np.cos(x[1]), m2*l*l]])
        C_q = np.array([[0.0, -m2*l*x_dot[1]*np.sin(x[1])],[0.0, 0.0]])
        B = np.array([1, 0])
        tau = np.array([0, -m2*g*l*np.sin(x[1])])
        
        x_ddot = np.linalg.solve(M_q, -C_q.dot(x_dot) + B.dot(u) + tau)

        return np.append(x_dot, x_ddot)

    def stateJac(state, u):
        x = state[0]
        xdot = state[2]
        theta = state[1]
        thetadot = state[3]

        cos = np.cos(theta)
        sin = np.sin(theta)

        jac = np.zeros(shape=(dimx, dimx))

        jac[0][2] = 1
        jac[1][3] = 1
        
        #theta
        jac[2][1] = (-m2*g*sin**2 + m2*g*cos**2 + m2*l*thetadot**2*cos)/(m1+m2*sin**2) - (2*m2*sin*cos*(m2*g*sin*cos + m2*l*thetadot**2*sin + u))/((m1+m2*sin**2)**2)
        #thetadot
        jac[2][3] = m2*sin*l*2*thetadot/(m1 + m2*sin**2)
        #theta
        jac[3][1] = (u*sin + m2*l*thetadot**2*sin**2 - m2*l*thetadot**2*cos**2 - g*(m1+m2)*cos)/(l*(m1+m2*sin**2)) - (2*m2*sin*cos*(-u*cos - m2*l*thetadot**2*cos*sin - (m1 + m2)*g*sin))/(l*(m1+m2*sin**2)**2)
        #thetadot
        jac[3][3] = (-m2*cos*sin*l*2*thetadot)/(l*(m1 + m2*sin**2))

        return jac

    def controlJac(state, u):
        theta = state[1]
        sin = np.sin(theta)
        cos = np.cos(theta)
        jac = np.zeros(shape=(dimx, 1))
        jac[0] = 0
        jac[1] = 0
        jac[2] = 1.0/(m1 + m2*sin**2)
        jac[3] = -cos/(l*(m1 + m2*sin**2))
        return jac

    def riccati(A, B, Q, R, K):
        return -A.transpose().dot(K) - K.dot(A) - Q + K.dot(B).dot(np.linalg.inv(R)).dot(B.transpose()).dot(K)

    #quadratic costs
    H = 1.0*np.eye(4)
    Q = 0.5*np.eye(dimx)
    R = 0.5*np.eye(dimu)

    q = np.zeros(shape=(planLength, dimx))

    naive_q = np.zeros(shape=(planLength, dimx))

    #perturb initial conditions to test stability of trajectory
    q[0] = np.array([x0[0]-0.3, x0[1]-0.2, x0[2], x0[3]])
    naive_q[0] = q[0].copy()
    K = []
    for i in range(planLength):
        K.append(np.zeros(shape=(4,4)))

    K[planLength-1] = H.copy()
    for index in range(0, planLength-1):
        i = planLength-index-1
        state = plan[dimtot*i:dimtot*i+dimx]
        if i < planLength-1:
            control = plan[dimtot*i+dimx:dimtot*i+dimx+dimu][0]
        else:
            control = 0
        Kdot = riccati(stateJac(state, control), controlJac(state, control), Q, R, K[i])
        K[i-1] = K[i] - dt*Kdot
        #iterate to improve derivative estimate
        for j in range(0, 30):
            state = plan[dimtot*(i-1):dimtot*(i-1)+dimx]
            control = plan[dimtot*(i-1)+dimx:dimtot*(i-1)+dimx+dimu][0]
            Kdot = riccati(stateJac(state, control), controlJac(state, control), Q, R, K[i-1])
            K[i-1] = K[i] - dt*Kdot

    #Make sure K satisfies Riccati eq
    fwd_K = K[0]
    for i in range(0, planLength-1):
        diff = abs(sum(sum(fwd_K - K[i])))
        if not np.allclose(fwd_K, K[i]):# diff > 0.001:
            print("diff between fwd and back, index: ", i, " ", diff)
        state = plan[dimtot*i:dimtot*i+dimx]
        control = plan[dimtot*i+dimx:dimtot*i+dimx+dimu][0]
        Kdot = riccati(stateJac(state, control), controlJac(state, control), Q, R, K[i])
        fwd_K += dt*Kdot
    diff = abs(sum(sum(fwd_K - K[planLength-1])))
    if diff > 0.001:
        print("diff between fwd and back, index: ", planLength-1, " ", diff)

    u_plan = plan[dimx::dimtot]
    u_corrected = []
    for i in range(1, planLength):
        #Linearise along trajectory
        B = controlJac(plan[dimtot*(i-1):dimtot*(i-1)+dimx], plan[dimtot*(i-1)+dimx:dimtot*(i-1)+dimx+dimu])
        u_delta = -np.linalg.inv(R).dot(B.transpose()).dot(K[i-1]).dot(q[i-1] - plan[dimtot*(i-1):dimtot*(i-1)+dimx])[0]
        qdot = computeQdot(q[i-1], u_plan[i-1] + u_delta)
        u_corrected.append(u_plan[i-1] + u_delta)
        naive_qdot = computeQdot(naive_q[i-1], u_plan[i-1])
                
        q[i] = q[i-1] + dt*qdot
        naive_q[i] = naive_q[i-1] + dt*naive_qdot

    # fig = plt.figure()
    # ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
    # line, = ax.plot([], [], lw=2)
    
    # def init():
    #     line.set_data([], [])
    #     return line,

    # def animate(i):        
    #     st = naive_q
    #     line.set_data([st[i][0]-0.3, st[i][0], st[i][0] + l*np.sin(st[i][1]), st[i][0], st[i][0]+0.3], [0.0, 0.0, -l*np.cos(st[i][1]), 0.0, 0.0])

    #     return line,

    # anim = animation.FuncAnimation(fig, animate, init_func=init,
    #                            frames=planLength, interval=10, blit=True)

    # plt.show()

    t = np.arange(0, dt*planLength, dt)

    lqr_x = [x[0] for x in q]
    unfiltered_x = [x[0] for x in naive_q]
    target_x = plan[0::dimtot]

    lqr_theta = [theta[1] for theta in q]
    unfiltered_theta = [theta[1] for theta in naive_q]
    target_theta = plan[1::dimtot]

    lqr_xdot = [xdot[2] for xdot in q]
    unfiltered_xdot = [xdot[2] for xdot in naive_q]
    target_xdot = plan[2::dimtot]

    lqr_thetadot = [thetadot[3] for thetadot in q]
    unfiltered_thetadot = [thetadot[3] for thetadot in naive_q]
    target_thetadot = plan[3::dimtot]

    f1 = plt.figure()
    ax = f1.add_subplot(111)
    leg1, = ax.plot(t, lqr_x, label="LQR stabilised x(t)")
    leg2, = ax.plot(t, unfiltered_x, label="unstabilised x(t)")
    leg3, = ax.plot(t, target_x, label="target x(t)")
    plt.legend(handles=[leg1, leg2, leg3])
    ax.set(xlabel='t', ylabel='x(t)')
    ax.grid()

    f2 = plt.figure()
    ax = f2.add_subplot(111)

    leg4, = ax.plot(t, lqr_theta, label="LQR stabilised theta(t)")
    leg5, = ax.plot(t, unfiltered_theta, label="unstabilised theta(t)")
    leg6, = ax.plot(t, target_theta, label="target theta(t)")
    plt.legend(handles=[leg4, leg5, leg6])
    ax.set(xlabel='t', ylabel='theta(t)')
    ax.grid()

    f3 = plt.figure()
    ax = f3.add_subplot(111)

    leg7, = ax.plot(t, lqr_xdot, label="LQR stabilised dx/dt(t)")
    leg8, = ax.plot(t, unfiltered_xdot, label="unstabilised dx/dt(t)")
    leg9, = ax.plot(t, target_xdot, label="target dx/dt(t)")
    plt.legend(handles=[leg7, leg8, leg9])
    ax.set(xlabel='t', ylabel='dx/dt(t)')
    ax.grid()

    f4 = plt.figure()
    ax = f4.add_subplot(111)

    leg10, = ax.plot(t, lqr_thetadot, label="LQR stabilised dtheta/dt(t)")
    leg11, = ax.plot(t, unfiltered_thetadot, label="unstabilised dtheta/dt(t)")
    leg12, = ax.plot(t, target_thetadot, label="target dtheta/dt(t)")
    plt.legend(handles=[leg10, leg11, leg12])
    ax.set(xlabel='t', ylabel='dtheta/dt(t)')
    ax.grid()

    plt.show()
