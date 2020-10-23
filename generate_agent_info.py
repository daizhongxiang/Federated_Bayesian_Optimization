import GPy
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pickle


dim = 2
ls = 0.01
v_kernel = 1.0
obs_noise = 1e-6

M = 100

s = np.random.multivariate_normal(np.zeros(dim), 1 / (ls**2) * np.identity(dim), M)
b = np.random.uniform(0, 2 * np.pi, M)

random_features = {"M":M, "length_scale":ls, "s":s, "b":b, "obs_noise":obs_noise, "v_kernel":v_kernel}
pickle.dump(random_features, open("saved_random_features/RF_M_" + str(M) + ".pkl", "wb"))

mine_list = np.arange(29)

tns = 50
policy = "ts"

all_info_full_dist = []
all_info_ts = []
all_inc_taf = []
for l in mine_list:
    itr = 0
    log_file_name = "results_standard_bo/" + policy + "_field_" + str(l) + "_iter_" + str(itr) + "_no_agent_info.p"
    res = pickle.load(open(log_file_name, "rb"))
    ys = np.array(res["all"]["values"]).reshape(-1, 1)
    params = np.array(res["all"]["params"])
    xs = np.array(params)

    xs = xs[:tns]
    ys = ys[:tns]
    
    all_inc_taf.append(np.max(ys))

    Phi = np.zeros((xs.shape[0], M))
    for i, x in enumerate(xs):
        x = np.squeeze(x).reshape(1, -1)
        features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)

        features = features / np.sqrt(np.inner(features, features))
        features = np.sqrt(v_kernel) * features

        Phi[i, :] = features

    Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M)
    Sigma_t_inv = np.linalg.inv(Sigma_t)
    nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), ys)

    w_samples = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)

    all_info_full_dist.append([Sigma_t_inv, nu_t])
    all_info_ts.append(w_samples)

# "all_info_full_dist" contains information to fully characterize the enture surrogate GP posterior of every agent
pickle.dump(all_info_full_dist, open("saved_agent_info/agents_M_" + str(M) + "_full_dist.pkl", "wb"))
# "all_info_ts" contains \omega_n from each agent, i.e., the information to be passed from every agent
pickle.dump(all_info_ts, open("saved_agent_info/agents_M_" + str(M) + "_ts.pkl", "wb"))
# "all_inc_taf" contains the incumbent (i.e., the currently observed maximum) of every agent
pickle.dump(all_inc_taf, open("saved_agent_info/agents_M_" + str(M) + "_inc_TAF.pkl", "wb"))

