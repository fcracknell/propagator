
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def solve_CR3BP(t, Y, config):

    # unpack state and config
    x, y, z, vx, vy, vz = Y
    m1, m2 = config

    # define state derivative vector
    Ydot = np.zeros_like(Y)
    Ydot[:3] = [vx, vy, vz]

    pi2 = m2 / (m1 + m2)

    sigma = np.sqrt( (x + pi2)**2 + y**2 + z**2 )

    psi = np.sqrt( (x - 1 + pi2)**2 + y**2 + z**2 )

    ax = 2*vy + x - ((1 - pi2)/sigma**3)*(x + pi2) - (pi2/psi**3)*(x - 1 + pi2)
    ay = -2*vx + y - ((1 - pi2)/sigma**3)*y - (pi2/psi**3)*y
    az = -((1 - pi2)/sigma**3)*z - (pi2/psi**3)*z

    Ydot[3:] = [ax, ay, az]

    return Ydot

if __name__ == "__main__":

    m1 = 5.974e24
    m2 = 7.348e22

    pi_2 = m2 / (m1 + m2)

    x_0 = 1 - pi_2
    y_0 = 0.0455
    z_0 = 0
    vx_0 = -0.5
    vy_0 = 0.5
    vz_0 = 0
    
    y0 = np.hstack((x_0, y_0, z_0, vx_0, vy_0, vz_0))

    t_0 = 0
    t_f = 20
    t_points = np.linspace(t_0, t_f, 1000)

    sol = solve_ivp(solve_CR3BP, (t_0, t_f), y0, args=([m1, m2],), atol=1e-9, rtol=1e-6, t_eval=t_points)

    Y = sol.y.T
    r = Y[:, :3]  # nondimensional distance
    v = Y[:, 3:]  # nondimensional velocity

    x_2 = (1 - pi_2) * np.cos(np.linspace(0, np.pi, 100))
    y_2 = (1 - pi_2) * np.sin(np.linspace(0, np.pi, 100))
    fig, ax = plt.subplots(figsize=(5, 5), dpi=96)

    # Plot the orbits
    ax.plot(r[:, 0], r[:, 1], "r", label="Trajectory")
    ax.axhline(0, color="k")
    ax.plot(np.hstack((x_2, x_2[::-1])), np.hstack((y_2, -y_2[::-1])))
    ax.plot(-pi_2, 0, "bo", label="$m_1$")
    ax.plot(1 - pi_2, 0, "go", label="$m_2$")
    ax.plot(x_0, y_0, "ro")
    ax.set_aspect("equal")
    plt.show()

    speed_sq = np.sum(np.square(v), axis=1)

    r[:, 0] += pi_2
    sigma = np.sqrt(np.sum(np.square(r), axis=1))
    r[:, 0] -= 1.0
    psi = np.sqrt(np.sum(np.square(r), axis=1))
    r[:, 0] = r[:, 0] + 1.0 - pi_2

    J = (
        0.5 * speed_sq
        - (1 - pi_2) / sigma
        - pi_2 / psi
        - 0.5 * ((1 - pi_2) * sigma**2 + pi_2 * psi**2)
    )

    fig, ax = plt.subplots(figsize=(8, 6), dpi=96)
    ax.plot(sol.t, J, label="Jacobi Constant")
    ax.axhline(J[0], color="C1", label="Initial Jacobi Constant")
    ax.legend(loc="center left")
    ax.set_xlabel("$\\tau$")
    ax.set_ylabel("$J$")
    plt.show()