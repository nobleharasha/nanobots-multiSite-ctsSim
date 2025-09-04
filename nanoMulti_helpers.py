import numpy as np
import time




arrangement="a"

if arrangement=="a":
    c = 2
    P_M = [25., 25.]
    y = np.array([[.0015, .0025], [.0035, .0025]])
elif arrangement=="b":
    c = 2
    P_M = [15., 35.]
    y = np.array([[.0015, .0025], [.0035, .0025]])
elif arrangement=="c":
    c=5
    P_M = [10.,10.,10.,10.,10.]
    y = np.array([[.0025,.0025],[.001,.004],[.001,.001],[.004,.004],[.004,.001]])
elif arrangement=="d":
    c=5
    P_M = [10.,10.,10.,10.,10.]
    y = np.array([[.0008,.0027],[.0012,.0023],[.0008,.0023],[.0012,.0027],[.004,.0025]])
else:   # if arrangement=="e":
    c=3
    P_M=[46.,2.,2.]
    y=np.array([[.0025,.0025],[.001,.002],[.004,.003]])





def l2_dist(x1, x2):
    return np.linalg.norm(x2 - x1)



def gamma_M(x):
    return sum([P_M[j] * np.exp(-1.*l2_dist(x, y[j])**2 / m) for j in range(c)]) / (np.pi * m)

def GRAD_gamma_M(x):
    out1 = 2. * sum([P_M[j] * np.exp(-1. * l2_dist(x, y[j])**2 / m) * (y[j][0] - x[0]) for j in range(c)]) / (np.pi * m**2)
    out2 = 2. * sum([P_M[j] * np.exp(-1. * l2_dist(x, y[j])**2 / m) * (y[j][1] - x[1]) for j in range(c)]) / (np.pi * m**2)
    return np.array([out1, out2])



# A is implicitly at time t/T*
def gamma_A(x, t, A):
    outs = []
    for j in range(c):
        tmp = 0
        for tstar in A[j]:
            if t-tstar <= 0:
                continue
            elif np.array_equal(np.array(x), y[j]):
                tmp += 1. / (t-tstar)
            else:
                tmp += np.exp(-1.*l2_dist(x,y[j])**2. / (4*D_A*(t-tstar))) / (t-tstar)
        tmp *= P_A / (4.*np.pi*D_A)
        outs.append(tmp)

    return sum(outs)

# A is implicitly at time t/T*
def GRAD_gamma_A(x, t, A):
    out1 = (P_A / (8*np.pi*D_A**2)) * sum([sum([np.exp(-1.*l2_dist(x, y[j])**2 / (4*D_A*(t-tstar))) * (y[j][0] - x[0]) / (t-tstar)**2 for tstar in A[j] if (t-tstar)>0]) for j in range(c)])
    out2 = (P_A / (8*np.pi*D_A**2)) * sum([sum([np.exp(-1.*l2_dist(x, y[j])**2 / (4*D_A*(t-tstar))) * (y[j][1] - x[1]) / (t-tstar)**2 for tstar in A[j] if (t-tstar)>0]) for j in range(c)])
    return np.array([out1, out2])



# R is implicitly at time t/T*
def gamma_R(x, t, R):
    return (P_R / (4*np.pi*D_R)) * sum([sum([np.exp(-1.*l2_dist(x, y[j])**2 / (4*D_R*(t-tstar))) / (t-tstar) for tstar in R[j] if (t-tstar)>0]) for j in range(c)])

# R is implicitly at time t/T*
def GRAD_gamma_R(x, t, R):
    out1 = (P_R / (8*np.pi*D_R**2)) * sum([sum([np.exp(-1.*l2_dist(x, y[j])**2 / (4*D_R*(t-tstar))) * (y[j][0] - x[0]) / (t-tstar)**2 for tstar in R[j] if (t-tstar)>0]) for j in range(c)])
    out2 = (P_R / (8*np.pi*D_R**2)) * sum([sum([np.exp(-1.*l2_dist(x, y[j])**2 / (4*D_R*(t-tstar))) * (y[j][1] - x[1]) / (t-tstar)**2 for tstar in R[j] if (t-tstar)>0]) for j in range(c)])
    return np.array([out1, out2])



# K is implicitly at time T*
def metric(K):
    return sum([min(K[j] / r_KM, P_M[j]) for j in range(c)]) / sum(P_M)




def ori(x, t, A, R, b, alg="KMAR"):
    ct=0
    x_new = np.array([phi_max+1., phi_max+1.])
    while x_new[0] > phi_max or x_new[1] > phi_max or x_new[0] < 0 or x_new[1] < 0:

        if alg == "RW":
            beta = np.random.uniform(-1.*np.pi,np.pi)
            mu = np.array([1.,0.])
        elif alg == "KM":
            if gamma_M(x) == 0:
                beta = np.random.uniform(-1.*np.pi,np.pi)
                mu = np.array([1.,0.])
            else:
                mu = GRAD_gamma_M(x)
                var = 1. / ( b * np.linalg.norm(mu) )
                beta = np.random.normal(loc=np.pi, scale=var, size=1)[0] % (2.*np.pi) - np.pi
        elif alg == "KMA":
            if gamma_M(x) + gamma_A(x,t,A) == 0:
                beta = np.random.uniform(-1.*np.pi,np.pi)
                mu = np.array([1.,0.])
            else:
                mu = GRAD_gamma_M(x) + GRAD_gamma_A(x,t,A)
                var = 1. / ( b * np.linalg.norm(mu) )
                beta = np.random.normal(loc=np.pi, scale=var, size=1)[0] % (2.*np.pi) - np.pi
        else:   # if alg == "KMAR"
            if (gamma_M(x) + gamma_A(x,t,A) - gamma_R(x,t,R) == 0):
                beta = np.random.uniform(-1.*np.pi,np.pi)
                mu = np.array([1.,0.])
            else:
                if gamma_A(x,t,A)==0 and gamma_R(x,t,R)==0:
                    mu = GRAD_gamma_M(x)
                elif gamma_A(x,t,A)==0:
                    mu = GRAD_gamma_M(x) - GRAD_gamma_R(x,t,R)
                elif gamma_B(x,t,B)==0:
                    mu = GRAD_gamma_M(x) + GRAD_gamma_A(x,t,A)
                else:
                    mu = GRAD_gamma_M(x) + GRAD_gamma_A(x,t,A) - GRAD_gamma_R(x,t,R)
                var = 1. / ( b * np.linalg.norm(mu) )
                beta = np.random.normal(loc=np.pi, scale=var, size=1)[0] % (2.*np.pi) - np.pi

        theta = np.array([[np.cos(beta), -1.*np.sin(beta)], [np.sin(beta), np.cos(beta)]]) @ mu.T
        x_new = x + alpha * theta / np.linalg.norm(theta)
        ct+=1
        if ct>=10:
            return x
    return x_new




def run(max_Tstar, metric_step, b, alg="KMAR"):
    xs = np.zeros((n, 2))

    terminated = [False for _ in range(n)]
    for i in range(n):
        xs[i][0], xs[i][1] = np.random.uniform(0, phi_max), np.random.uniform(0, phi_max)

    K = [0 for _ in range(c)]
    A = [[] for _ in range(c)]
    R = [[] for _ in range(c)]
    metric_hist = [0]

    for t in range(1, int(max_Tstar)):
        time.sleep(0.00001)
        for i in range(n):
            if not terminated[i]:
                xs[i] = ori(xs[i], t, A, R, b, alg)

                for jjj in range(c):
                    if l2_dist(xs[i], y[jjj]) <= epsilon:
                        K[jjj] += 1

                        if alg == "KMA":
                            A[jjj].append(t)
                        if alg == "KMAR":
                            if gamma_A(y[jjj],t,A, prnt=False) / P_M[jjj] < r_AM:
                                A[jjj].append(t)
                            else:
                                R[jjj].append(t)

                        terminated[i] = True
        if t % metric_step == 0:
            metric_hist.append(metric(K))

    print("K:", K, "A:", A, "B:", B)

    return metric_hist
