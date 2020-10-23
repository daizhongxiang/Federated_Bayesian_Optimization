import GPy
from federated_bayesian_optimization import BayesianOptimization
import pickle

import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn import svm
from sklearn.metrics import roc_auc_score


max_iter = 50

M = 100

ls = 0.01
v_kernel = 1.0
obs_noise = 1e-6
random_features = pickle.load(open("saved_random_features/RF_M_" + str(M) + ".pkl", "rb"))

M_target = 200

### we've implemented 3 federated BO algorithms: our FTS, and RGPE and TAF that are modified for the FBO setting (Section 3.2 of the main paper)
policy = "ts"
# policy = "rgpe"
# policy = "taf"


if policy == "taf":
    all_inc_taf = pickle.load(open("saved_agent_info/agents_M_" + str(M) + "_inc_TAF.pkl", "rb"))
else:
    all_inc_taf = None

S_rgpe = 100

# the number of other agents
N = 28

pt = 1 - 1 / (np.arange(max_iter+5) + 1) ** (2.0)
pt[0] = pt[1]
P_N = np.ones(N) / N

landmine_data = pickle.load(open("landmine_formated_data.pkl", "rb"))
all_X_train, all_Y_train, all_X_test, all_Y_test = landmine_data["all_X_train"], landmine_data["all_Y_train"], \
        landmine_data["all_X_test"], landmine_data["all_Y_test"]
N_land = len(all_X_train)

mine_list = np.arange(0, 6)
run_list = np.arange(0, 5)

for l in mine_list:
    print("[landmine field {}]".format(l))
    
    info_full_dist = pickle.load(open("saved_agent_info/agents_M_" + str(M) + "_full_dist.pkl", "rb"))
    info_ts = pickle.load(open("saved_agent_info/agents_M_" + str(M) + "_ts.pkl", "rb"))
    ### remove the lth element and use the other landmines as the other agents
    info_full_dist.pop(l)
    info_ts.pop(l)

    X_train = all_X_train[l]
    Y_train = np.squeeze(all_Y_train[l])
    X_test = all_X_test[l]
    Y_test = np.squeeze(all_Y_test[l])

    def obj_func_landmine(param):
        parameter_range = [[1e-4, 10.0], [1e-2, 10.0]]
        C_ = param[0]
        C = C_ * (parameter_range[0][1] - parameter_range[0][0]) + parameter_range[0][0]
        gam_ = param[1]
        gam = gam_ * (parameter_range[1][1] - parameter_range[1][0]) + parameter_range[1][0]
        
        clf = svm.SVC(kernel="rbf", C=C, gamma=gam, probability=True)
        clf.fit(X_train, Y_train)
        pred = clf.predict_proba(X_test)
        score = roc_auc_score(Y_test, pred[:, 1])

        return score

    for itr in run_list:
        if policy == "ts":
            log_file_name = "results_fts/" + policy + "_M_" + str(M) + "_subject_" + str(l) + "_iter_" + str(itr) + ".p"
        elif policy == "rgpe":
            log_file_name = "results_rgpe/" + policy + "_M_" + str(M) + "_subject_" + str(l) + "_iter_" + str(itr) + ".p"
        elif policy == "taf":
            log_file_name = "results_taf/" + policy + "_M_" + str(M) + "_subject_" + str(l) + "_iter_" + str(itr) +".p"

        FBO = BayesianOptimization(f=obj_func_landmine, pbounds={'x1':(0, 1), 'x2':(0, 1)}, gp_opt_schedule=5, \
                gp_model='gpy', use_init="inits/init_field_" + str(l) + "_iter_" + str(itr) + ".p", \
                gp_mcmc=False, log_file=log_file_name, save_init=False, save_init_file=None, fix_gp_hypers=None, \
                policy=policy, M=M, N=N, random_features=random_features, info_full_dist=info_full_dist, info_ts=info_ts, \
                pt=pt, P_N=P_N, M_target=M_target, all_inc_taf=all_inc_taf, S_rgpe=S_rgpe)

        FBO.maximize(n_iter=max_iter, init_points=3)

