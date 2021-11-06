import numpy as np
from   real.actuator import *


class Pendulum:
    def __init__(self, m=0.125, l=0.155):
        self.k_m = 0.06499285#0.0512#0.0306
        self.m = m
        self.l = l
        self.g = 9.8
        m_ad = 0.46
        self.mgl = self.m*self.g*self.l

        self.w = 0.41
        self.h = 0.02
        self.I = m*(self.w**2 + self.h**2)/12 + self.m*self.l**2
        print(self.I)
        # self.I = 0.0045 + m_ad*self.l**2 + m_ad*0.055**2/2  # self.I
        self.k_x = 0.00109

        self.Kp = 1.1
        self.Kd = 0.11

        self.u_max = 3

    def ff_control(self, X_des):
        u = (self.mgl*np.sin(X_des[0]) + self.I *
             X_des[2] + self.k_x*X_des[1])/self.k_m
        return u

    def cont_pd(self, X_des, X, kp=1, kd=0.11):
        K = np.array([[kp, kd]])
        u = np.dot(K, X_des[0:2]-X)
        u = np.clip(u, -self.u_max, self.u_max)
        return u[0][0]

    def control_pd(self, X_des, X, kp=1, kd=0.11):
        K = np.array([[kp, kd]])
        u = self.ff_control(X_des) + np.dot(K, X_des[0:2]-X)/self.k_m
        u = np.clip(u, -self.u_max, self.u_max)
        return u[0][0]

    def control_pd_g(self, X_des, X, kp=1, kd=0.2):
        K = np.array([[kp, kd]])
        u = (self.mgl*np.sin(X[0]) + self.k_x*X[1]) / \
            self.k_m + np.dot(K, X_des[0:2]-X)/self.k_m
        u = np.clip(u, -self.u_max, self.u_max)
        return u[0][0]

    def control_sl_mod(self, X_des, X):
        K = np.array([[self.Kp, self.Kd]])
        e = X_des[0:2]-X

        k_sl = 0.1 + 0.01*abs(e[1])
        a = 50
        s = a*e[0] + e[1]
        u_sl = -k_sl*s
        u_sl = np.clip(u_sl, -0.1, 0.1)
        u = self.ff_control(X_des) + \
            np.dot(K, X_des[0:2]-X)/self.k_m - u_sl/self.k_m
        u = np.clip(u, -self.u_max, self.u_max)
        return u[0][0]

    def control(self, X_des, X):
        K = np.array([[self.Kp, self.Kd]])
        u = (self.mgl*np.sin(X[0]) + self.I*X_des[2] +
             self.k_x*X[1] + np.dot(K, X_des[0:2]-X))/self.k_m
        u = np.clip(u, -self.u_max, self.u_max)
        # print(u)
        return u[0][0]


def chirp_traj(a0, a, w0, nu, t):
    omega = 2*np.pi*(w0 + nu*t)
    X_des = np.array([a0 + a*np.sin(omega*t),
                      a*(omega + 2*np.pi*nu*t)*np.cos(omega*t),
                      4*a*nu*np.pi*np.cos(omega*t) - 4*a*np.pi**2*np.sin(omega*t)*(w0+2*nu*t)**2])
    return X_des


def sin_traj(a0, a, w, t):
    omega = 2*np.pi*w
    X_des = np.array([a0 + a*np.sin(omega*t),
                      a*omega*np.cos(omega*t),
                      -a*omega**2*np.sin(omega*t)])
    return X_des


def ident_traj(t):
    #a = [1.54, 0.0081, 0.0078, 0.24]
    a = [2.96, -0.96, 0.35, 0.06]
    f0 = 2*np.pi*0.2
    X_des = np.array([0*t,
                      0*t,
                      0*t])
    #X_des = np.zeros((3,len(t)))
    for i in range(4):
        X_des[0] += a[i]*np.sin((i+1)*f0*t)
        X_des[1] += a[i]*f0*(i+1)*np.cos((i+1)*f0*t)
        X_des[2] += -a[i]*f0**2*(i+1)**2*np.sin((i+1)*f0*t)
    return X_des


def step_traj(t):
    if t % 80 < 40:
        X_des = np.array([[-np.pi/2 + int((t / 2) % 20)*np.pi/20],
                          [0],
                          [0]])
    if t % 80 > 40:
        X_des = np.array([[np.pi/2 - int((t / 2) % 20)*np.pi/20],
                          [0],
                          [0]])
    return X_des
