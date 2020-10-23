import GPy
from federated_bayesian_optimization import BayesianOptimization
import pickle
import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import roc_auc_score

max_iter = 50

# The Thompson sampling policy
policy = "ts"

# the number of other agents
N = 28

# here we set all p_t=1, which means we don't use any information from the other agents, i.e., we run standard BO
pt = np.ones(max_iter+5) # if not using information from the other agents
# the distribution used by FTS to sample agents, set to the uniform distribution
P_N = np.ones(N) / N

# load the formated landmine detection data
landmine_data = pickle.load(open("landmine_formated_data.pkl", "rb"))
all_X_train, all_Y_train, all_X_test, all_Y_test = landmine_data["all_X_train"], landmine_data["all_Y_train"], \
        landmine_data["all_X_test"], landmine_data["all_Y_test"]
N_land = len(all_X_train)

### we run a separate BO task for every agent; this is to generate the BO observations of every agent to be passed to the target agent
# mine_list = np.arange(0, N_land)
# run_list = np.arange(0, 1)

### We run standard BO for the first 6 agents, and average over 5 random initializations for each agent; this is for comparison with other BO algorithms
mine_list = np.arange(0, 6)
run_list = np.arange(0, 5)

for l in mine_list:
    print("[landmine field {}]".format(l))

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
        log_file_name = "results_standard_bo/" + policy + "_field_" + str(l) + "_iter_" + str(itr) + "_no_agent_info.p"

        BO = BayesianOptimization(f=obj_func_landmine, pbounds={'x1':(0, 1), 'x2':(0, 1)}, gp_opt_schedule=5, \
                gp_model='gpy', use_init=None, gp_mcmc=False, log_file=log_file_name, save_init=True, \
                save_init_file="inits/init_field_" + str(l) + "_iter_" + str(itr) + ".p", fix_gp_hypers=None, \
                policy=policy, random_features=None, pt=pt, P_N=None, M_target=200)
        BO.maximize(n_iter=max_iter, init_points=3)

