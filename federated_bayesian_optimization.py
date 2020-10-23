# -*- coding: utf-8 -*-
"""
This script implements the FTS algorithm, as well as the RGPE and TAF algorithms adapted for the FBO setting.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
import GPy
from federated_helper_funcs import UtilityFunction, unique_rows, PrintLog, acq_max
import pickle
from tqdm import tqdm
import itertools
import time

class BayesianOptimization(object):

    def __init__(self, f, pbounds, gp_opt_schedule, gp_model, ARD=False, \
                 use_init=False, gp_mcmc=False, log_file=None, save_init=False, save_init_file=None, fix_gp_hypers=None, \
                 policy="ts", M=50, N=50, random_features=None, info_full_dist=None, info_ts=None, \
                 pt=None, P_N=None, M_target=100, \
                 all_inc_taf=None, no_weight_learning=False, S_rgpe=100, kernel_rho=0.5, verbose=1):
        """
            f: the objective function of the target agent
            gp_opt_schedule: optimize the GP hyperparameters after every gp_opt_schedule iterations
            policy: the BO algorithm to run, can take values: ["TS", "RGPE", and "TAF"]
            M: the number of random features for RFF approximation
            N: the number of other agents
            random_features: the saved random features, which is shared among all agents
            info_full_dist: the information received from each agent for the RGPE and TAF algorithm; for each agent, the information includes \nu_n and \Sigma_n
            info_ts: the information received from each agent for the FTS algorithm; for each agent, the information includes a sampled \omega_n
            pt: the sequence p_t; to run the standard TS algorithm (without using information from other agents), just set pt to all ones
            P_N: the distribution over agents used by the FTS algorithm
            M_target: the number of random features used by both TS and FTS to draw samples from the GP posterior of the target agent
            all_inc_taf: the incumbent of the BO observations of each agent, required by the TAF algorithm
            no_weight_learning: if set to False, the RGPE and TAF algorithms run as usual; if set to True, the RGPE and TAF algorithms don't learn the task similarity
            S_rgpe: a parameter for the RGPE algorithm
            kernel_rho: a parameter for the TAF algorithm
        """

        self.all_inc_taf = all_inc_taf
        self.no_weight_learning = no_weight_learning
                
        self.policy = policy
        self.M = M
        self.N = N
        self.random_features = random_features
        self.info_full_dist = info_full_dist
        self.info_ts = info_ts

        self.M_target = M_target
        
        self.agents_used_flag = np.ones(self.N)
        
        self.pt = pt
        self.ws = P_N

        self.use_init = use_init
        
        self.time_started = 0

        self.ARD = ARD
        self.fix_gp_hypers = fix_gp_hypers
    
        self.log_file = log_file
        
        self.pbounds = pbounds
        
        self.incumbent = None
        
        self.keys = list(pbounds.keys())

        self.dim = len(pbounds)

        self.bounds = []
        for key in self.pbounds.keys():
            self.bounds.append(self.pbounds[key])
        self.bounds = np.asarray(self.bounds)
        
        self.f = f

        self.initialized = False

        self.init_points = []
        self.x_init = []
        self.y_init = []

        self.X = np.array([]).reshape(-1, 1)
        self.Y = np.array([])
        
        self.i = 0

        self.S = S_rgpe # this is for RGPE
        self.kernel_rho = kernel_rho # this is for TAF
        
        self.gp_mcmc = gp_mcmc
        
        self.gp_model = gp_model
        self.gp = None
        self.gp_params = None
        self.gp_opt_schedule = gp_opt_schedule

        self.federated_gps = []

        self.util = None

        self.plog = PrintLog(self.keys)
        
        self.save_init = save_init
        self.save_init_file = save_init_file

        self.res = {}
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values':[], 'params':[], 'init_values':[], 'init_params':[], 'init':[], \
                          'time_started':0, 'timestamps':[]}

        self.verbose = verbose
        
        
    def init(self, init_points):
        """
            random initializations
        """

        l = [np.random.uniform(x[0], x[1], size=init_points) for x in self.bounds]

        self.init_points += list(map(list, zip(*l)))
        y_init = []
        for x in self.init_points:
            print("[init point]: ", x)
            curr_y = self.f(x)
            
            y_init.append(curr_y)
            self.res['all']['init_values'].append(curr_y)
            self.res['all']['init_params'].append(dict(zip(self.keys, x)))

        self.X = np.asarray(self.init_points)
        self.Y = np.asarray(y_init)

        self.incumbent = np.max(y_init)
        self.initialized = True
        
        init = {"X":self.X, "Y":self.Y}
        self.res['all']['init'] = init

        if self.save_init:
            pickle.dump(init, open(self.save_init_file, "wb"))


    def maximize(self, init_points=5, n_iter=25):
        """
            init_points: the number of random initializations
            n_iter: the number of BO iterations
        """

        xi=0.0
        # Reset timer
        self.plog.reset_timer()

        self.time_started = time.time()
        self.res['all']['time_started'] = self.time_started

        self.util_ts = UtilityFunction(kind="ts")
        self.util_rgpe = UtilityFunction(kind="rgpe")
        self.util_taf = UtilityFunction(kind="taf")

        # get random initializations
        if not self.initialized:
            if self.use_init != None:
                init = pickle.load(open(self.use_init, "rb"))

                print("[loaded init: {0}; {1}]".format(init["X"], init["Y"]))

                self.X, self.Y = init["X"], init["Y"]
                self.incumbent = np.max(self.Y)
                self.initialized = True
                self.res['all']['init'] = init
                self.res['all']['init_values'] = list(self.Y)
                
                print("Using pre-existing initializations with {0} points".format(len(self.Y)))
            else:
                if init_points > 0:
                    self.init(init_points)

        y_max = np.max(self.Y)
        ur = unique_rows(self.X)

        if self.fix_gp_hypers is None:
            self.gp = GPy.models.GPRegression(self.X[ur], self.Y[ur].reshape(-1, 1), \
                    GPy.kern.RBF(input_dim=self.X.shape[1], lengthscale=1.0, variance=0.1, ARD=self.ARD))
            self.gp["Gaussian_noise.variance"][0] = 1e-4
        else:
            self.gp = GPy.models.GPRegression(self.X[ur], self.Y[ur].reshape(-1, 1), \
                    GPy.kern.RBF(input_dim=self.X.shape[1], lengthscale=self.fix_gp_hypers, ARD=self.ARD))

        if init_points > 1:
            if self.fix_gp_hypers is None:
                if self.gp_mcmc:
                    self.gp.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                    self.gp.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                    self.gp.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                    print("[Running MCMC for GP hyper-parameters]")
                    hmc = GPy.inference.mcmc.HMC(self.gp, stepsize=5e-2)
                    gp_samples = hmc.sample(num_samples=500)[-300:] # Burnin

                    gp_samples_mean = np.mean(gp_samples, axis=0)
                    print("Mean of MCMC hypers: {0}".format(gp_samples_mean))

                    self.gp.kern.variance.fix(gp_samples_mean[0])
                    self.gp.kern.lengthscale.fix(gp_samples_mean[1])
                    self.gp.likelihood.variance.fix(gp_samples_mean[2])

                    self.gp_params = self.gp.parameters
                else:
                    self.gp.optimize_restarts(num_restarts = 10, messages=False)
                    self.gp_params = self.gp.parameters

                    gp_samples = None
                    print("---Optimized hyper: ", self.gp)


            if self.policy == "rgpe" and not self.no_weight_learning:
                s_rgpe = self.random_features["s"]
                b_rgpe = self.random_features["b"]
                obs_noise_rgpe = self.random_features["obs_noise"]
                v_kernel_rgpe = self.random_features["v_kernel"]

                ### here we update the weights for RGPE
                N_t = self.X.shape[0]
                losses_meta = np.zeros((self.N+1, self.S))
                for ind_1 in np.arange(N_t):
                    mask = np.ones(self.X.shape[0], dtype=bool)
                    mask[ind_1] = 0
                    X_ = self.X[mask, :]
                    Y_ = self.Y[mask]

                    gp_ = self.gp.copy()
                    gp_.set_XY(X=X_, Y=Y_.reshape(-1, 1))
                    post_samples_target = gp_.posterior_samples_f(self.X, full_cov=True, size=self.S)
                    post_samples_target = np.squeeze(post_samples_target) # shape: N_t * self.S

                    for ind_2 in np.arange(N_t):
                        if ind_1 != ind_2:
                            for m in range(self.N):
                                info = self.info_full_dist[m]
                                Sigma_t_inv, nu_t = info[0], info[1]

                                x_ = np.squeeze(self.X[ind_1]).reshape(1, -1)
                                features = np.sqrt(2 / self.M) * np.cos(np.squeeze(np.dot(x_, s_rgpe.T)) + b_rgpe)
                                features = features.reshape(-1, 1)
                                features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
                                features = np.sqrt(v_kernel_rgpe) * features
                                mean_n = np.squeeze(np.dot(features.T, nu_t))
                                var_n = obs_noise_rgpe * np.squeeze(np.dot(np.dot(features.T, Sigma_t_inv), features))
                                post_samples_m_1 = np.random.normal(mean_n, np.sqrt(var_n), self.S)

                                x_ = np.squeeze(self.X[ind_2]).reshape(1, -1)
                                features = np.sqrt(2 / self.M) * np.cos(np.squeeze(np.dot(x_, s_rgpe.T)) + b_rgpe)
                                features = features.reshape(-1, 1)
                                features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
                                features = np.sqrt(v_kernel_rgpe) * features
                                mean_n = np.squeeze(np.dot(features.T, nu_t))
                                var_n = obs_noise_rgpe * np.squeeze(np.dot(np.dot(features.T, Sigma_t_inv), features))
                                post_samples_m_2 = np.random.normal(mean_n, np.sqrt(var_n), self.S)


                                for s in range(self.S):
                                    losses_meta[m, s] += np.bitwise_xor(self.Y[ind_1] < self.Y[ind_2], post_samples_m_1[s] < post_samples_m_2[s])

                            for s in range(self.S):
                                losses_meta[-1, s] += np.bitwise_xor(self.Y[ind_1] < self.Y[ind_2], post_samples_target[ind_1, s] < post_samples_target[ind_2, s])

                for m in range(self.N):
                    if np.median(losses_meta[m, :]) > np.percentile(losses_meta[-1, :], 95):
                        losses_meta[m, :] = 1e5

                best_ind = np.argmin(losses_meta, axis=0)
                ws = np.zeros(self.N+1)
                for s in range(self.S):
                    equ_inds = np.nonzero(losses_meta[:, s] == losses_meta[best_ind[s], s])[0]
                    if len(equ_inds) == 1:
                        ws[best_ind[s]] += 1 / self.S
                    else:
                        if self.N in equ_inds:
                            ws[self.N] += 1 / self.S
                        else:
                            ws[best_ind[s]] += 1 / self.S
                print("ws: ", ws)
                self.ws = ws

            if self.policy == "taf" and not self.no_weight_learning:
                s_rgpe = self.random_features["s"]
                b_rgpe = self.random_features["b"]
                obs_noise_rgpe = self.random_features["obs_noise"]
                v_kernel_rgpe = self.random_features["v_kernel"]

                N_t = self.X.shape[0]
                all_meta_features = np.zeros((self.N, N_t * (N_t-1)))
                target_features = np.zeros(N_t * (N_t-1))
                count = 0
                for ind_1 in np.arange(N_t):
                    for ind_2 in np.arange(N_t):
                        if ind_1 != ind_2:
                            for m in range(self.N):
                                info = self.info_full_dist[m]
                                Sigma_t_inv, nu_t = info[0], info[1]

                                x_ = np.squeeze(self.X[ind_1]).reshape(1, -1)
                                features = np.sqrt(2 / self.M) * np.cos(np.squeeze(np.dot(x_, s_rgpe.T)) + b_rgpe)
                                features = features.reshape(-1, 1)
                                features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
                                features = np.sqrt(v_kernel_rgpe) * features
                                mean_m_1 = np.squeeze(np.dot(features.T, nu_t))

                                x_ = np.squeeze(self.X[ind_2]).reshape(1, -1)
                                features = np.sqrt(2 / self.M) * np.cos(np.squeeze(np.dot(x_, s_rgpe.T)) + b_rgpe)
                                features = features.reshape(-1, 1)
                                features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
                                features = np.sqrt(v_kernel_rgpe) * features
                                mean_m_2 = np.squeeze(np.dot(features.T, nu_t))

                                if mean_m_1 > mean_m_2:
                                    all_meta_features[m, count] = 1.0 / (N_t * (N_t - 1))

                            if self.Y[ind_1] > self.Y[ind_2]:
                                target_features[count] = 1.0 / (N_t * (N_t - 1))

                            count += 1

                ws = np.zeros(self.N+1)
                for m in range(self.N):
                    dist = np.sqrt(np.sum((target_features - all_meta_features[m, :]) ** 2))
                    tmp = dist / self.kernel_rho
                    if tmp <= 1:
                        ws[m] = 3 / 4 * (1 - tmp ** 2)
                    else:
                        ws[m] = 0
                ws[-1] = 3 / 4
                print("ws: ", ws / np.sum(ws))
                self.ws = ws


        if self.policy == "rgpe":
            x_max, all_ucb = acq_max(ac=self.util_rgpe.utility,
                            gp=self.gp,
                            M=self.M, N=self.N, random_features=self.random_features, \
                            info_full_dist=self.info_full_dist, info_ts=self.info_ts, pt=self.pt, ws=self.ws, use_target_label=False, 
                            w_sample=None, \
                            y_max=y_max, bounds=self.bounds, iteration=1, gp_samples=None, all_inc_taf=self.all_inc_taf, \
                            no_weight_learning=self.no_weight_learning)
        elif self.policy == "taf":
            x_max, all_ucb = acq_max(ac=self.util_taf.utility,
                            gp=self.gp,
                            M=self.M, N=self.N, random_features=self.random_features, \
                            info_full_dist=self.info_full_dist, info_ts=self.info_ts, pt=self.pt, ws=self.ws, use_target_label=False, 
                            w_sample=None, \
                            y_max=y_max, bounds=self.bounds, iteration=1, gp_samples=None, all_inc_taf=self.all_inc_taf, \
                            no_weight_learning=self.no_weight_learning)
        elif self.policy == "ts":
            if np.random.random() < self.pt[0]:
                M_target = self.M_target

                if self.fix_gp_hypers is None:
                    ls_target = self.gp["rbf.lengthscale"][0]
                    v_kernel = self.gp["rbf.variance"][0]
                    obs_noise = self.gp["Gaussian_noise.variance"][0]
                else:
                    ls_target = self.fix_gp_hypers
                    v_kernel = 1.0
                    obs_noise = 1e-2

                s = np.random.multivariate_normal(np.zeros(self.dim), 1 / (ls_target**2) * np.identity(self.dim), M_target)
                b = np.random.uniform(0, 2 * np.pi, M_target)

                random_features_target = {"M":M_target, "length_scale":ls_target, "s":s, "b":b, "obs_noise":obs_noise, "v_kernel":v_kernel}

                Phi = np.zeros((self.X.shape[0], M_target))
                for i, x in enumerate(self.X):
                    x = np.squeeze(x).reshape(1, -1)
                    features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)

                    features = features / np.sqrt(np.inner(features, features))
                    features = np.sqrt(v_kernel) * features

                    Phi[i, :] = features

                Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M_target)
                Sigma_t_inv = np.linalg.inv(Sigma_t)
                nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), self.Y.reshape(-1, 1))

                w_sample = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)

                x_max, all_ucb = acq_max(ac=self.util_ts.utility, gp=self.gp,
                            M=M_target, N=self.N, random_features=random_features_target, \
                            info_full_dist=self.info_full_dist, info_ts=self.info_ts, pt=self.pt, ws=self.ws, use_target_label=True, 
                            w_sample=w_sample, \
                            y_max=y_max, bounds=self.bounds, iteration=1, gp_samples=None, all_inc_taf=self.all_inc_taf, \
                            no_weight_learning=self.no_weight_learning)
            else:
                use_target_label = False

                agent_ind = np.arange(self.N)
                random_agent_n = np.random.choice(agent_ind, 1, p=self.ws)[0]
                w_sample = self.info_ts[random_agent_n]
                
                x_max, all_ucb = acq_max(ac=self.util_ts.utility, gp=self.gp,
                            M=self.M, N=self.N, random_features=self.random_features, \
                            info_full_dist=self.info_full_dist, info_ts=self.info_ts, pt=self.pt, ws=self.ws, use_target_label=False, 
                            w_sample=w_sample, \
                            y_max=y_max, bounds=self.bounds, iteration=1, gp_samples=None, all_inc_taf=self.all_inc_taf, \
                            no_weight_learning=self.no_weight_learning)

        if self.verbose:
            self.plog.print_header(initialization=False)

        for i in range(n_iter):
            if not self.X.shape[0] == 0:
                if np.any(np.all(self.X - x_max == 0, axis=1)):
                    x_max = np.random.uniform(self.bounds[:, 0],
                                          self.bounds[:, 1],
                                          size=self.bounds.shape[0])

            curr_y = self.f(x_max)

            self.res["all"]["timestamps"].append(time.time())

            self.Y = np.append(self.Y, curr_y)
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))

            if self.Y[-1] > y_max:
                y_max = self.Y[-1]
                self.incumbent = self.Y[-1]

            ur = unique_rows(self.X)
            if self.gp_model == 'sklearn':
                pass
            elif self.gp_model == 'gpflow':
                pass
            elif self.gp_model == 'gpy':
                self.gp.set_XY(X=self.X[ur], Y=self.Y[ur].reshape(-1, 1))

                if i >= self.gp_opt_schedule and i % self.gp_opt_schedule == 0:
                    if self.gp_mcmc:
                        self.gp.kern.lengthscale.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                        self.gp.kern.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                        self.gp.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(1.,10.))
                        print("[Running MCMC for GP hyper-parameters]")
                        hmc = GPy.inference.mcmc.HMC(self.gp, stepsize=5e-2)
                        gp_samples = hmc.sample(num_samples=500)[-300:] # Burnin

                        gp_samples_mean = np.mean(gp_samples, axis=0)
                        print("Mean of MCMC hypers: {0}".format(gp_samples_mean))

                        self.gp.kern.variance.fix(gp_samples_mean[0])
                        self.gp.kern.lengthscale.fix(gp_samples_mean[1])
                        self.gp.likelihood.variance.fix(gp_samples_mean[2])
                    else:
                        self.gp.optimize_restarts(num_restarts = 10, messages=False)
                        self.gp_params = self.gp.parameters

                        gp_samples = None
                        print("---Optimized hyper: ", self.gp)

                if self.policy == "rgpe" and not self.no_weight_learning:
                    s_rgpe = self.random_features["s"]
                    b_rgpe = self.random_features["b"]
                    obs_noise_rgpe = self.random_features["obs_noise"]
                    v_kernel_rgpe = self.random_features["v_kernel"]

                    ### here we update the weights for RGPE
                    N_t = self.X.shape[0]
                    losses_meta = np.zeros((self.N+1, self.S))
                    for ind_1 in np.arange(N_t):
                        mask = np.ones(self.X.shape[0], dtype=bool)
                        mask[ind_1] = 0
                        X_ = self.X[mask, :]
                        Y_ = self.Y[mask]

                        gp_ = self.gp.copy()
                        gp_.set_XY(X=X_, Y=Y_.reshape(-1, 1))
                        post_samples_target = gp_.posterior_samples_f(self.X, full_cov=True, size=self.S)
                        post_samples_target = np.squeeze(post_samples_target) # shape: N_t * self.S

                        for ind_2 in np.arange(N_t):
                            if ind_1 != ind_2:
                                for m in range(self.N):
                                    info = self.info_full_dist[m]
                                    Sigma_t_inv, nu_t = info[0], info[1]

                                    x_ = np.squeeze(self.X[ind_1]).reshape(1, -1)
                                    features = np.sqrt(2 / self.M) * np.cos(np.squeeze(np.dot(x_, s_rgpe.T)) + b_rgpe)
                                    features = features.reshape(-1, 1)
                                    features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
                                    features = np.sqrt(v_kernel_rgpe) * features
                                    mean_n = np.squeeze(np.dot(features.T, nu_t))
                                    var_n = obs_noise_rgpe * np.squeeze(np.dot(np.dot(features.T, Sigma_t_inv), features))
                                    post_samples_m_1 = np.random.normal(mean_n, np.sqrt(var_n), self.S)

                                    x_ = np.squeeze(self.X[ind_2]).reshape(1, -1)
                                    features = np.sqrt(2 / self.M) * np.cos(np.squeeze(np.dot(x_, s_rgpe.T)) + b_rgpe)
                                    features = features.reshape(-1, 1)
                                    features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
                                    features = np.sqrt(v_kernel_rgpe) * features
                                    mean_n = np.squeeze(np.dot(features.T, nu_t))
                                    var_n = obs_noise_rgpe * np.squeeze(np.dot(np.dot(features.T, Sigma_t_inv), features))
                                    post_samples_m_2 = np.random.normal(mean_n, np.sqrt(var_n), self.S)

                                    for s in range(self.S):
                                        losses_meta[m, s] += np.bitwise_xor(self.Y[ind_1] < self.Y[ind_2], post_samples_m_1[s] < post_samples_m_2[s])

                                # calculate the ranking loss for the target function
                                for s in range(self.S):
                                    losses_meta[-1, s] += np.bitwise_xor(self.Y[ind_1] < self.Y[ind_2], post_samples_target[ind_1, s] < post_samples_target[ind_2, s])

                    for m in range(self.N):
                        if np.median(losses_meta[m, :]) > np.percentile(losses_meta[-1, :], 95):
                            losses_meta[m, :] = 1e5

                    best_ind = np.argmin(losses_meta, axis=0)
                    ws = np.zeros(self.N+1)
                    for s in range(self.S):
                        equ_inds = np.nonzero(losses_meta[:, s] == losses_meta[best_ind[s], s])[0]
                        if len(equ_inds) == 1:
                            ws[best_ind[s]] += 1 / self.S
                        else:
                            if self.N in equ_inds:
                                ws[self.N] += 1 / self.S
                            else:
                                ws[best_ind[s]] += 1 / self.S
                    self.ws = ws


                if self.policy == "taf" and not self.no_weight_learning:
                    s_rgpe = self.random_features["s"]
                    b_rgpe = self.random_features["b"]
                    obs_noise_rgpe = self.random_features["obs_noise"]
                    v_kernel_rgpe = self.random_features["v_kernel"]

                    #### derive the weights
                    N_t = self.X.shape[0]
                    all_meta_features = np.zeros((self.N, N_t * (N_t-1)))
                    target_features = np.zeros(N_t * (N_t-1))
                    count = 0
                    for ind_1 in np.arange(N_t):
                        for ind_2 in np.arange(N_t):
                            if ind_1 != ind_2:
                                for m in range(self.N):
                                    info = self.info_full_dist[m]
                                    Sigma_t_inv, nu_t = info[0], info[1]

                                    x_ = np.squeeze(self.X[ind_1]).reshape(1, -1)
                                    features = np.sqrt(2 / self.M) * np.cos(np.squeeze(np.dot(x_, s_rgpe.T)) + b_rgpe)
                                    features = features.reshape(-1, 1)
                                    features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
                                    features = np.sqrt(v_kernel_rgpe) * features
                                    mean_m_1 = np.squeeze(np.dot(features.T, nu_t))

                                    x_ = np.squeeze(self.X[ind_2]).reshape(1, -1)
                                    features = np.sqrt(2 / self.M) * np.cos(np.squeeze(np.dot(x_, s_rgpe.T)) + b_rgpe)
                                    features = features.reshape(-1, 1)
                                    features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
                                    features = np.sqrt(v_kernel_rgpe) * features
                                    mean_m_2 = np.squeeze(np.dot(features.T, nu_t))
                                    
                                    if mean_m_1 > mean_m_2:
                                        all_meta_features[m, count] = 1.0 / (N_t * (N_t - 1))

                                if self.Y[ind_1] > self.Y[ind_2]:
                                    target_features[count] = 1.0 / (N_t * (N_t - 1))

                                count += 1

                    ws = np.zeros(self.N+1)
                    for m in range(self.N):
                        dist = np.sqrt(np.sum((target_features - all_meta_features[m, :]) ** 2))
                        tmp = dist / self.kernel_rho
                        if tmp <= 1:
                            ws[m] = 3 / 4 * (1 - tmp ** 2)
                        else:
                            ws[m] = 0
                    ws[-1] = 3 / 4
                    self.ws = ws

            if self.policy == "rgpe":
                x_max, all_ucb = acq_max(ac=self.util_rgpe.utility,
                                gp=self.gp,
                                M=self.M, N=self.N, random_features=self.random_features, \
                                info_full_dist=self.info_full_dist, info_ts=self.info_ts, pt=self.pt, ws=self.ws, use_target_label=None, \
                                w_sample=None, \
                                y_max=y_max, bounds=self.bounds, iteration=i+2, gp_samples=None, all_inc_taf=self.all_inc_taf, \
                            no_weight_learning=self.no_weight_learning)
            elif self.policy == "taf":
                x_max, all_ucb = acq_max(ac=self.util_taf.utility,
                                gp=self.gp,
                                M=self.M, N=self.N, random_features=self.random_features, \
                                info_full_dist=self.info_full_dist, info_ts=self.info_ts, pt=self.pt, ws=self.ws, use_target_label=None, \
                                w_sample=None, \
                                y_max=y_max, bounds=self.bounds, iteration=i+2, gp_samples=None, all_inc_taf=self.all_inc_taf, \
                            no_weight_learning=self.no_weight_learning)
            elif self.policy == "ts":
                if np.random.random() < self.pt[i+1]:
                    M_target = self.M_target

                    if self.fix_gp_hypers is None:
                        ls_target = self.gp["rbf.lengthscale"][0]
                        v_kernel = self.gp["rbf.variance"][0]
                        obs_noise = self.gp["Gaussian_noise.variance"][0]
                    else:
                        ls_target = self.fix_gp_hypers
                        v_kernel = 1.0
                        obs_noise = 1e-2

                    s = np.random.multivariate_normal(np.zeros(self.dim), 1 / (ls_target**2) * np.identity(self.dim), M_target)
                    b = np.random.uniform(0, 2 * np.pi, M_target)

                    random_features_target = {"M":M_target, "length_scale":ls_target, "s":s, "b":b, "obs_noise":obs_noise, "v_kernel":v_kernel}

                    Phi = np.zeros((self.X.shape[0], M_target))
                    for i, x in enumerate(self.X):
                        x = np.squeeze(x).reshape(1, -1)
                        features = np.sqrt(2 / M_target) * np.cos(np.squeeze(np.dot(x, s.T)) + b)

                        features = features / np.sqrt(np.inner(features, features))
                        features = np.sqrt(v_kernel) * features

                        Phi[i, :] = features

                    Sigma_t = np.dot(Phi.T, Phi) + obs_noise * np.identity(M_target)
                    Sigma_t_inv = np.linalg.inv(Sigma_t)
                    nu_t = np.dot(np.dot(Sigma_t_inv, Phi.T), self.Y.reshape(-1, 1))

                    w_sample = np.random.multivariate_normal(np.squeeze(nu_t), obs_noise * Sigma_t_inv, 1)

                    x_max, all_ucb = acq_max(ac=self.util_ts.utility, gp=self.gp,
                                M=M_target, N=self.N, random_features=random_features_target, \
                                info_full_dist=self.info_full_dist, info_ts=self.info_ts, pt=self.pt, ws=self.ws, use_target_label=True, 
                                w_sample=w_sample, \
                                y_max=y_max, bounds=self.bounds, iteration=i+2, gp_samples=None, all_inc_taf=self.all_inc_taf, \
                            no_weight_learning=self.no_weight_learning)
                    
                else:
                    #### we only sample from those unused agents
                    curr_unused_inds = np.nonzero(self.agents_used_flag)[0]
                    curr_ws = self.ws[curr_unused_inds]
                    curr_ws = curr_ws / np.sum(curr_ws)
                    
                    unused_agent_inds = np.arange(len(curr_unused_inds))
                    random_agent_n_ind = np.random.choice(unused_agent_inds, 1, p=curr_ws)[0]
                    random_agent_n = curr_unused_inds[random_agent_n_ind]
                    self.agents_used_flag[random_agent_n] = 0
                    
                    w_sample = self.info_ts[random_agent_n]
                    
                    x_max, all_ucb = acq_max(ac=self.util_ts.utility, gp=self.gp,
                                M=self.M, N=self.N, random_features=self.random_features, \
                                info_full_dist=self.info_full_dist, info_ts=self.info_ts, pt=self.pt, ws=self.ws, use_target_label=False, 
                                w_sample=w_sample, \
                                y_max=y_max, bounds=self.bounds, iteration=i+2, gp_samples=None, all_inc_taf=self.all_inc_taf, \
                            no_weight_learning=self.no_weight_learning)
                    
            if self.verbose:
                self.plog.print_step(x_max, self.Y[-1], warning=False)

            self.i += 1

            x_max_param = self.X[self.Y.argmax(), :-1]

            self.res['max'] = {'max_val': self.Y.max(), 'max_params': dict(zip(self.keys, x_max_param))}
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(self.X[-1])
            self.res['all']['params'].append(self.X[-1])

            if self.log_file is not None:
                pickle.dump(self.res, open(self.log_file, "wb"))

        if self.verbose:
            self.plog.print_summary()

