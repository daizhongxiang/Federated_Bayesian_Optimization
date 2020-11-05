"""
This script contains some helper functions used by the FTS algorithm, as well as the RGPE and TAF algorithms adapted for the FBO setting.
"""

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipydirect import minimize as mini_direct
import pickle

# USE_DIRECT_OPTIMIZER = True
USE_DIRECT_OPTIMIZER = False

def acq_max(ac, gp, M, N, random_features, info_full_dist, info_ts, pt, ws, use_target_label, w_sample, \
            y_max, bounds, iteration, gp_samples=None, all_inc_taf=None, no_weight_learning=False):
    para_dict={"gp":gp, "M":M, "N":N, "random_features":random_features, "info_gp_ucb":info_gp_ucb, \
               "info_ts":info_ts, "pt":pt, "ws":ws, "use_target_label":use_target_label, \
               "tmp_ucb":None, "w_sample":w_sample, "federated_gps":federated_gps, "y_max":y_max, \
               "iteration":iteration, "gp_samples":gp_samples, "all_inc_taf":all_inc_taf, "no_weight_learning":no_weight_learning}

    if not USE_DIRECT_OPTIMIZER:
        x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                     size=(10000, bounds.shape[0]))

        ys = []
        for x in x_tries:
            ys.append(ac(x.reshape(1, -1), para_dict))
        ys = np.array(ys)
        
        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()

        x_seeds = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                    size=(1, bounds.shape[0]))
        for x_try in x_seeds:
            res = minimize(lambda x: -ac(x.reshape(1, -1), para_dict),
                           x_try.reshape(1, -1),
                           bounds=bounds,
                           method="L-BFGS-B")

            if max_acq is None or -res.fun >= max_acq:
                x_max = res.x
                max_acq = -res.fun
    
    else:
        print("[Running the direct optimizer]")

        bound_list = []
        for b in bounds:
            bound_list.append(tuple(b))

        res = mini_direct(ac, bound_list, para_dict=para_dict)
        x_max = res["x"]

    return x_max, None

class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind):
        if kind not in ['ts', 'rgpe', 'taf']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose ucb or ts.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    # This function is defined to work with the DIRECT optimizer
    def utility(self, x, para_dict):
        gp, M, N, random_features, info_full_dist, info_ts, pt, ws, use_target_label, w_sample, \
            y_max, iteration, gp_samples, all_inc_taf, no_weight_learning = \
                para_dict["gp"], para_dict["M"], para_dict["N"], \
                para_dict["random_features"], para_dict["info_full_dist"], para_dict["info_ts"], para_dict["pt"], para_dict["ws"], \
                para_dict["use_target_label"], para_dict["w_sample"], para_dict["y_max"], \
                para_dict["iteration"], para_dict["gp_samples"], para_dict["all_inc_taf"], para_dict["no_weight_learning"]

        if self.kind == 'ts':
            return self._ts(x, gp, M, N, random_features, info_ts, use_target_label, w_sample, iteration)
        elif self.kind == 'rgpe':
            return self._rgpe(x, gp, M, N, random_features, info_full_dist, pt, ws, iteration, gp_samples, \
                              y_max, no_weight_learning)
        elif self.kind == 'taf':
            return self._taf(x, gp, M, N, random_features, info_full_dist, pt, ws, iteration, gp_samples, \
                             y_max, all_inc_taf, no_weight_learning)

    @staticmethod
    def _ts(x, gp, M, N, random_features, info_ts, use_target_label, w_sample, iteration):
        if USE_DIRECT_OPTIMIZER:
            x = x.reshape(1, -1)
        d = x.shape[1]
        
        s = random_features["s"]
        b = random_features["b"]
        obs_noise = random_features["obs_noise"]
        v_kernel = random_features["v_kernel"]

        x = np.squeeze(x).reshape(1, -1)
        features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x, s.T)) + b)
        features = features.reshape(-1, 1)

        features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
        features = np.sqrt(v_kernel) * features # v_kernel is set to be 1 here in the synthetic experiments

        f_value = np.squeeze(np.dot(w_sample, features))

        if USE_DIRECT_OPTIMIZER:
            optimizer_flag = -1
        else:
            optimizer_flag = 1
        
        return optimizer_flag * f_value

    @staticmethod
    def _rgpe(x, gp, M, N, random_features, info_full_dist, pt, ws, iteration, gp_samples, y_max, no_weight_learning):
        if USE_DIRECT_OPTIMIZER:
            x = x.reshape(1, -1)
        d = x.shape[1]
        
        post_mean, post_var = 0, 0

        s = random_features["s"]
        b = random_features["b"]
        obs_noise = random_features["obs_noise"]
        v_kernel = random_features["v_kernel"]

        for n in range(N):
            info = info_full_dist[n]
            Sigma_t_inv, nu_t = info[0], info[1]

            x_ = np.squeeze(x).reshape(1, -1)
            features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x_, s.T)) + b)
            features = features.reshape(-1, 1)

            features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
            features = np.sqrt(v_kernel) * features # v_kernel is set to be 1 here in the synthetic experiments

            mean_n = np.squeeze(np.dot(features.T, nu_t))
            var_n = obs_noise * np.squeeze(np.dot(np.dot(features.T, Sigma_t_inv), features))
            std_n = np.sqrt(var_n)

            if not no_weight_learning:
                post_mean += mean_n * ws[n]
                post_var += var_n * (ws[n]**2)
            else:
                post_mean += mean_n * ws[n] * (1-pt[iteration-1])
                post_var += var_n * ((ws[n] * (1-pt[iteration-1]))**2)

        mean, var = gp.predict(x)

        if not no_weight_learning:
            post_mean += mean * ws[-1]
            post_var += var * (ws[-1]**2)
        else:
            post_mean += mean * pt[iteration-1]
            post_var += var * (pt[iteration-1]**2)

        post_std = np.sqrt(post_var)

        if USE_DIRECT_OPTIMIZER:
            optimizer_flag = -1
        else:
            optimizer_flag = 1

        z = (post_mean - y_max - xi)/post_std
        return optimizer_flag * ((post_mean - y_max - xi) * norm.cdf(z) + post_std * norm.pdf(z))

    @staticmethod
    def _taf(x, gp, M, N, random_features, info_full_dist, pt, ws, iteration, gp_samples, y_max, all_inc_taf, no_weight_learning):
        if USE_DIRECT_OPTIMIZER:
            x = x.reshape(1, -1)
        d = x.shape[1]
        
        s = random_features["s"]
        b = random_features["b"]
        obs_noise = random_features["obs_noise"]
        v_kernel = random_features["v_kernel"]
        
        ei_overall = 0

        for n in range(N):
            info = info_full_dist[n]
            Sigma_t_inv, nu_t = info[0], info[1]

            x_ = np.squeeze(x).reshape(1, -1)
            features = np.sqrt(2 / M) * np.cos(np.squeeze(np.dot(x_, s.T)) + b)
            features = features.reshape(-1, 1)

            features = features / np.sqrt(np.inner(np.squeeze(features), np.squeeze(features)))
            features = np.sqrt(v_kernel) * features # v_kernel is set to be 1 here in the synthetic experiments

            mean_n = np.squeeze(np.dot(features.T, nu_t))
            y_max_m = all_inc_taf[n]

            var_n = obs_noise * np.squeeze(np.dot(np.dot(features.T, Sigma_t_inv), features))
            std_n = np.sqrt(var_n)
            pred = np.random.normal(mean_n, std_n, 1)[0]

            if not no_weight_learning:
                ei_overall += np.max([0, (pred - y_max_m - xi)]) * ws[n]
            else:
                ei_overall += np.max([0, (pred - y_max_m - xi)]) * ws[n] * (1-pt[iteration-1])
        
        mean, var = gp.predict(x)
        std = np.sqrt(var)
        z = (mean - y_max - xi) / std
        
        if not no_weight_learning:
            ei_overall += ((mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)) * ws[-1]
            ei_overall = ei_overall / np.sum(ws)
        else:
            ei_overall += ((mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)) * pt[iteration-1]

        if USE_DIRECT_OPTIMIZER:
            optimizer_flag = -1
        else:
            optimizer_flag = 1

        return optimizer_flag * ei_overall


def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]


class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class PrintLog(object):

    def __init__(self, params):

        self.ymax = None
        self.xmax = None
        self.params = params
        self.ite = 1

        self.start_time = datetime.now()
        self.last_round = datetime.now()

        # sizes of parameters name and all
        self.sizes = [max(len(ps), 7) for ps in params]

        # Sorted indexes to access parameters
        self.sorti = sorted(range(len(self.params)),
                            key=self.params.__getitem__)

    def reset_timer(self):
        self.start_time = datetime.now()
        self.last_round = datetime.now()

    def print_header(self, initialization=True):

        if initialization:
            print("{}Initialization{}".format(BColours.RED,
                                              BColours.ENDC))
        else:
            print("{}Bayesian Optimization{}".format(BColours.RED,
                                                     BColours.ENDC))

        print(BColours.BLUE + "-" * (29 + sum([s + 5 for s in self.sizes])) +
            BColours.ENDC)

        print("{0:>{1}}".format("Step", 5), end=" | ")
        print("{0:>{1}}".format("Time", 6), end=" | ")
        print("{0:>{1}}".format("Value", 10), end=" | ")

        for index in self.sorti:
            print("{0:>{1}}".format(self.params[index],
                                    self.sizes[index] + 2),
                  end=" | ")
        print('')

    def print_step(self, x, y, warning=False):

        print("{:>5d}".format(self.ite), end=" | ")

        m, s = divmod((datetime.now() - self.last_round).total_seconds(), 60)
        print("{:>02d}m{:>02d}s".format(int(m), int(s)), end=" | ")

        if self.ymax is None or self.ymax < y:
            self.ymax = y
            self.xmax = x
            print("{0}{2: >10.5f}{1}".format(BColours.MAGENTA,
                                             BColours.ENDC,
                                             y),
                  end=" | ")

            for index in self.sorti:
                print("{0}{2: >{3}.{4}f}{1}".format(
                            BColours.GREEN, BColours.ENDC,
                            x[index],
                            self.sizes[index] + 2,
                            min(self.sizes[index] - 3, 6 - 2)
                        ),
                      end=" | ")
        else:
            print("{: >10.5f}".format(y), end=" | ")
            for index in self.sorti:
                print("{0: >{1}.{2}f}".format(x[index],
                                              self.sizes[index] + 2,
                                              min(self.sizes[index] - 3, 6 - 2)),
                      end=" | ")

        if warning:
            print("{}Warning: Test point chose at "
                  "random due to repeated sample.{}".format(BColours.RED,
                                                            BColours.ENDC))

        print()

        self.last_round = datetime.now()
        self.ite += 1

    def print_summary(self):
        pass
