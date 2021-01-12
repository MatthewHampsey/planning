import constr
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import animation

def planPendulumTrajectory(N, dt, dimx, dimu, x0, x1, pendulum_params):
    dimtot = (dimx + dimu)

    NN = (N-1)*dimtot + dimx

    def ctrlgrad(x):
        result = np.zeros(NN)
        for i in range(0,N-1):
          result[dimtot*i + dimx] = 2*dt*x[dimtot*i + dimx]
        return result

    def ctrlhessian(x):
        result = np.zeros(shape=(NN, NN))
        for i in range(0, N-1):
            result[dimtot*i + dimx, dimtot*i + dimx] = 2*dt
        return result

    def cntrlconstVec(x):
        result = np.zeros(2*N-2)
        for i in range(0, N-1):
            result[i] = -x[dimtot*i + dimx] +10.0
            result[i + N-1] = x[dimtot*i + dimx] +10.0
        return result

    def cntrlconstJac(x):
        result = np.zeros(shape=(2*N-2, NN))
        for i in range(0, N-1):
            result[i, dimtot*i + dimx] = -1
            result[i+N-1, dimtot*i + dimx] = 1
        return result

    rows = N+1

    m1 = pendulum_params['m1']
    m2 = pendulum_params['m2']
    l = pendulum_params['l']
    g = pendulum_params['g']


    def equalityConstraint(x, x0, x1):
        constraint = np.zeros(rows*dimx)
        constraint[0:dimx] = x[0:dimx] - x0
        constraint[dimx:2*dimx] = x[(N-1)*dimtot:(N-1)*dimtot+dimx] - x1
        for i in range(0, N-1):
            y_offset = 2*dimx + i*dimx
            x_offset = dimtot*i

            x_i = x[x_offset]
            x_i_1 = x[x_offset+dimtot]
            xdot_i = x[x_offset+2]
            xdot_i_1 = x[x_offset+dimtot+2]

            theta_i = x[x_offset+1]
            cos_i = np.cos(theta_i)
            sin_i = np.sin(theta_i)
            theta_i_1 = x[x_offset+dimtot + 1]
            thetadot_i = x[x_offset+3]
            thetadot_i_1 = x[x_offset+dimtot+3]

            u = x[x_offset+4]
            # x_i - x_{i-1} - dt*xdot_{i-1} = 0
            constraint[y_offset] = x_i_1 - x_i- dt*xdot_i
            # t_i - t_{i-1} - dt*tdot_{i-1} = 0
            constraint[y_offset+1] = theta_i_1 - theta_i - dt*thetadot_i
            #xdot_i - xdot_{i-1} - dt*xdotdot_{i-1} = 0
            xdotdot = (u + m2*sin_i*(g*cos_i + l*thetadot_i**2))/(m1 + m2*sin_i**2)
            constraint[y_offset+2] = xdot_i_1 - xdot_i - dt*xdotdot
            #tdot_i - tdot_{i-1} - dt*tdotdot_{i-1} = 0
            thetadotdot = (-u*cos_i - m2*l*thetadot_i**2*cos_i*sin_i - (m1 + m2)*g*sin_i)/(l*(m1+m2*sin_i**2))
            constraint[y_offset+3] = thetadot_i_1 - thetadot_i - dt*thetadotdot

        return constraint

    def equalityConstraintJac(x):
        constraint = np.zeros(shape=(rows*dimx, NN))
        constraint[0:dimx, 0:dimx] = np.eye(dimx)
        constraint[dimx:2*dimx, (N-1)*dimtot:(N-1)*dimtot+dimx] = np.eye(dimx)
        for i in range(0, N-1):
            y_offset = 2*dimx + i*dimx
            x_offset = dimtot*i

            x_i = x[x_offset]
            x_i_1 = x[x_offset+dimtot]
            xdot_i = x[x_offset+2]
            xdot_i_1 = x[x_offset+dimtot+2]

            theta_i = x[x_offset+1]
            cos_i = np.cos(theta_i)
            sin_i = np.sin(theta_i)
            theta_i_1 = x[x_offset+dimtot + 1]
            thetadot_i = x[x_offset+3]
            thetadot_i_1 = x[x_offset+dimtot+3]
            u_i = x[x_offset+4]
            
            constraint[y_offset, x_offset] = -1.0
            constraint[y_offset, x_offset+2] = -dt
            constraint[y_offset, x_offset+dimtot] = 1.0

            constraint[y_offset+1, x_offset+1] = -1.0
            constraint[y_offset+1, x_offset+3] = -dt
            constraint[y_offset+1, x_offset+dimtot+1] = 1.0

            constraint[y_offset+2, x_offset+2] = -1.0
            constraint[y_offset+2, x_offset+dimtot+2] = 1.0
            #theta
            constraint[y_offset+2, x_offset+1] = -dt*((-m2*g*sin_i**2 + m2*g*cos_i**2 + m2*l*thetadot_i**2*cos_i)/(m1+m2*sin_i**2) - (2*m2*sin_i*cos_i*(m2*g*sin_i*cos_i + m2*l*thetadot_i**2*sin_i + u_i))/((m1+m2*sin_i**2)**2))
            #thetadot
            constraint[y_offset+2, x_offset+3] = -dt*m2*sin_i*l*2*thetadot_i/(m1 + m2*sin_i**2)
            #u
            constraint[y_offset+2, x_offset+4] = -dt/(m1 + m2*sin_i**2)

            constraint[y_offset+3, x_offset+dimtot+3] = 1.0
            #theta
            constraint[y_offset+3, x_offset+1] = -dt*((u_i*sin_i + m2*l*thetadot_i**2*sin_i**2 - m2*l*thetadot_i**2*cos_i**2 - g*(m1+m2)*cos_i)/(l*(m1+m2*sin_i**2)) - (2*m2*sin_i*cos_i*(-u_i*cos_i - m2*l*thetadot_i**2*cos_i*sin_i - (m1 + m2)*g*sin_i))/(l*(m1+m2*sin_i**2)**2))
            #thetadot
            constraint[y_offset+3, x_offset+3] = -1.0 -dt*(-m2*cos_i*sin_i*l*2*thetadot_i)/(l*(m1 + m2*sin_i**2))
            #u
            constraint[y_offset+3, x_offset+4] = -dt*(-cos_i)/(l*(m1 + m2*sin_i**2))
        return constraint
    
    initx = np.zeros(NN)
    for i in range(0, N-1):
        initx[dimtot*i] = (x1[0] - x0[0])*i/N
    

    return constr.interiorPoint(initx, lambda x: dt*sum(x[dimx::dimtot]), ctrlgrad, ctrlhessian, 
                    lambda x: equalityConstraint(x, x0.copy(), x1.copy()), rows*dimx, equalityConstraintJac, 
                    lambda x : -cntrlconstVec(x), 2*N-2, lambda x: -cntrlconstJac(x), 1.8, 200, verbose=True)

if __name__ == '__main__':
    np.set_printoptions(linewidth=160)
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(formatter={'float': lambda x: "{0: 0.2f}".format(x)})

    N = 250
    dimx = 4
    dimu = 1

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

    plan = planPendulumTrajectory(N, dt, dimx, dimu, x0, x1, pendulum_params)

    u = plan[dimx::(dimx + dimu)]
    x = np.zeros(shape=(N, 2))
    x[0] = np.array([x0[0], x0[1]])
    x_dot = np.zeros(shape=(N, 2))
    x_dot[0] = np.array([x0[2], x0[3]])

    for i in range(1, N):
        M_q = np.array([[m1 + m2, m2*l*np.cos(x[i-1][1])],[m2*l*np.cos(x[i-1][1]), m2*l*l]])
        C_q = np.array([[0.0, -m2*l*x_dot[i-1][1]*np.sin(x[i-1][1])],[0.0, 0.0]])
        B = np.array([1, 0])
        tau = np.array([0, -m2*g*l*np.sin(x[i-1][1])])
        x_ddot = np.linalg.solve(M_q, -C_q.dot(x_dot[i-1]) + B.dot(u[i-1]) + tau)
        x_dot[i] = x_dot[i-1] + dt*x_ddot
        x[i] = x[i-1] + dt*x_dot[i-1]

    fig = plt.figure()
    ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
    line, = ax.plot([], [], lw=2)
    
    def init():
        line.set_data([], [])
        return line,

    def animate(i):        
        line.set_data([x[i][0]-0.3, x[i][0], x[i][0] + l*np.sin(x[i][1]), x[i][0], x[i][0]+0.3], [0.0, 0.0, -l*np.cos(x[i][1]), 0.0, 0.0])

        return line,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=N, interval=10, blit=True)

    plt.show()