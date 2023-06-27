import numpy as np
from tqdm import trange, tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad
from sklearn.metrics import pairwise_distances
import math
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

SAVE_SIZE = (18, 14)
SAVE_NAME = "log/defense_results.jpg"

def plot_gradients_2d(gradients, attacker_indices):
    fig = plt.figure()

    for (worker_id, gradient) in gradients:
        if worker_id in attacker_indices:
            plt.scatter(gradient[0], gradient[1], color="blue", marker="x", s=250, linewidth=2)
        else:
            plt.scatter(gradient[0], gradient[1], color="orange", marker="+", s=100, linewidth=1)

    fig.set_size_inches(SAVE_SIZE, forward=False)
    plt.grid(False)
    plt.margins(0,0)
    plt.savefig(SAVE_NAME, bbox_inches='tight', pad_inches=0.1)

def apply_standard_scaler(gradients):
    scaler = StandardScaler()

    return scaler.fit_transform(gradients)
    
def calculate_pca_of_gradients(gradients, num_components):
    pca = PCA(n_components=num_components)

    print("Computing {}-component PCA of gradients".format(num_components))

    return pca.fit_transform(gradients)

def magnitude(x):
    return math.sqrt(sum(i ** 2 for i in x))

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.rng = np.random.default_rng()
        self.clients_len = 0

    def train(self):
        '''Train using Federated Average'''
        print('Training with {} workers ---'.format(self.clients_per_round))

        test_accuracies = []
        acc_10quant = []
        acc_20quant = []
        test_acc_var = []
        train_accuracies = []
        train_losses = []
        num_sampled = []
        client_sets_all = np.zeros([self.num_rounds, self.clients_per_round], dtype=int)
        diff_grad = np.zeros([self.num_rounds, len(self.clients)])
        param_diff = []
        worker_ids = []
    
        attackers_round = 0
        attacker_probability = 0.5
        attacker_indices = []
        selected_attackers = []
        attacker_indices, selected_attackers = self.select_attackers(self.num_attackers, num_clients=self.num_attackers)
        if self.num_attackers > 0 :
            tqdm.write('AttackerSet ：{}'.format(list(np.array(attacker_indices))))

        f = open("log/GradientSolution.log", "w")

        self.clients_len = len(self.clients)
        
        for i in range(self.num_rounds):
            
            allcl_models = []
            for cc in self.clients:
                clmodel = cc.get_params()
                allcl_models.append(clmodel)

            # test model
            if i % self.eval_every == 0:
                stats = self.test()  # have set the latest model for all clients
                stats_train = self.train_error_and_loss()
                
                test_accuracies.append(np.sum(stats[3]) * 1.0 / np.sum(stats[2]))
                acc_10quant.append(np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.1))
                acc_20quant.append(np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.2))
                test_acc_var.append(np.var([i/j for i,j in zip(stats[3], stats[2])]))
                train_accuracies.append(np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2]))
                train_losses.append(np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2]))

                tqdm.write('At round {} per-client-accuracy: {}'.format(i, list(np.around(np.array([i/j for i,j in zip(stats[3], stats[2])]),2))))
                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))  # testing accuracy
                tqdm.write('At round {} acc. variance: {}'.format(i, np.var([i/j for i,j in zip(stats[3], stats[2])])))  # testing accuracy variance
                tqdm.write('At round {} acc. 10th: {}'.format(i, np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.1)))  # testing accuracy variance
                tqdm.write('At round {} acc. 20th: {}'.format(i, np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.2)))  # testing accuracy variance
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])))

            model_len = process_grad(self.latest_model).size
            global_grads = np.zeros(model_len)
            client_grads = np.zeros(model_len)
            num_samples = []
            local_grads = []

            for c in self.clients:
                num, client_grad = c.get_grads(model_len)  # get client_grad and operate on it
                local_grads.append(client_grad)
                num_samples.append(num)
                global_grads = np.add(global_grads, client_grad * num)
            global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))

            difference = 0
            for idx in range(self.clients_len):
                difference += np.sum(np.square(global_grads - local_grads[idx]))
            difference = difference * 1.0 / self.clients_len
            if i % self.eval_every == 0:
                tqdm.write('gradient difference: {}'.format(difference))

            if self.clientsel_algo == 'divfl_lazy':
                #if i % self.m_interval == 0: # Moved the condition inside the function
                if i == 0 or self.eval_every == 1:  # at the first iteration or when m=1, collect gradients from all clients
                    self.all_grads = np.asarray(self.show_grads()[:-1])
                    self.norm_diff = pairwise_distances(self.all_grads, metric="euclidean")
                    np.fill_diagonal(self.norm_diff, 0)
                indices, selected_clients, all_grad = self.select_cl_submod(i, num_clients=self.clients_per_round, stochastic_greedy = False)
                active_clients = selected_clients # Dropping clients don't apply in this case
                if i == 0:
                    diff_grad[i] = np.zeros(len(all_grad))
                else:
                    diff_grad[i] = np.linalg.norm(all_grad - old_grad, axis=1)
                old_grad = all_grad.copy()
            elif self.clientsel_algo == 'divfl_stochastic':
                #if i % self.m_interval == 0: # Moved the condition inside the function
                if i == 0 or self.eval_every == 1:  # at the first iteration or when m=1, collect gradients from all clients
                    self.all_grads = np.asarray(self.show_grads()[:-1])
                    self.norm_diff = pairwise_distances(self.all_grads, metric="euclidean")
                    np.fill_diagonal(self.norm_diff, 0)
                indices, selected_clients, all_grad = self.select_cl_submod(i, num_clients=self.clients_per_round, stochastic_greedy = True)
                active_clients = selected_clients # Dropping clients don't apply in this case
                if i == 0:
                    diff_grad[i] = np.zeros(len(all_grad))
                else:
                    diff_grad[i] = np.linalg.norm(all_grad - old_grad, axis=1)
                old_grad = all_grad.copy()
            elif self.clientsel_algo == 'powerchoice':
                if i % self.m_interval == 0:
                    lprob = stats_train[2]/np.sum(stats_train[2], axis=0)
                    #d=100
                    subsample = 0.1
                    #d = max(self.clients_per_round, int(subsample * self.clients_len))
                    d = self.clients_len
                    lvals = self.rng.choice(stats_train[4], size=d, replace = False, p=lprob)
                    Mlist = [np.where(stats_train[4] == i)[0][0] for i in lvals]
                    lossvals = np.asarray(stats_train[4]) #All loss values
                    sel_losses = lossvals[Mlist]
                    idx_losses = np.argpartition(sel_losses, -self.clients_per_round)
                    values = sel_losses[idx_losses[-math.ceil(self.clients_per_round * (1-self.drop_percent)):]]
                    
                    listvalues = values.tolist()
                    listlossvals = lossvals.tolist()
                    indices = [listlossvals.index(i) for i in listvalues] 
                selected_clients = np.asarray(self.clients)[indices]
                np.random.seed(i)
                active_clients = selected_clients # np.random.choice(selected_clients, math.ceil(self.clients_per_round * (1-self.drop_percent)), replace=False)
            else:
                indices, selected_clients = self.select_clients(i, num_clients=math.ceil(self.clients_per_round * (1-self.drop_percent)))  # uniform sampling
                np.random.seed(i)
                active_clients = selected_clients # np.random.choice(selected_clients, math.ceil(self.clients_per_round * (1-self.drop_percent)), replace=False)
            
            tqdm.write('Client set ：{}'.format(list(np.array(indices))))
            print('Selected Client set length ', len(selected_clients))
            print('Active Client set length ', len(active_clients))
            client_sets_all[i] = indices
            tqdm.write('Num. clients sampled: {}'.format(i, len(indices)))
            num_sampled.append(len(indices))

            csolns = []  # buffer for receiving client solutions
            
            glob_copy = np.append(self.latest_model[0].flatten(), self.latest_model[1])

            attackers = []
            np.random.seed(i)
            if( attackers_round > 0 ) :
                attacker_probability = 0.50
            elif ( attacker_probability < 0.9 ) :
                attacker_probability += 0.1
            random_attack = np.random.binomial(n=1, p=min(1.0,attacker_probability), size=[20])
            #tqdm.write('RandomAttack: {}'.format(list(np.array(random_attack))))
            attackers_round = 0
            
            for idx, c in enumerate(active_clients.tolist()):  # simply drop the slow devices
                # communicate the latest model
                c.set_params(self.latest_model)

                if indices[idx] in attacker_indices :
                    if( random_attack[idx%20] ) :
                        doattacked = 1
                        attackers.append(indices[idx])
                        attackers_round += 1 
                    else :
                        doattacked = 0
                else :
                    doattacked = 0

                # solve minimization locally
                soln, stats, grads = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size, attacked=doattacked)
                #print("Shape of grads", np.shape(grads))
                
                if( (idx>1000) and (indices[idx] in attacker_indices or indices[idx] < 3) ):
                    f.write('At round {} GradMag of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, magnitude(np.array(grads)), len(grads) ))
                    f.write('At round {} GradSets of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, list(np.around(np.array(grads),2)), len(grads) ))
                    f.write('At round {} GradShape of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, np.shape(grads), len(grads) ))
                    f.write('At round {} GradType of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, type(grads), len(grads) ))
                    f.write('At round {} SoluShape11 of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, np.shape(soln[1][0]), len(soln[1][0]) ))
                    f.write('At round {} SoluType100 of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, type(soln[1][0]), len(soln[1][0]) ))
                    f.write('At round {} SoluType100 of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, type(soln[1][1]), len(soln[1][1]) ))
                    a = soln[0]
                    f.write('At round {} SoluSets00 of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, type(a), a ))
                    a = np.array(soln[1][0])
                    f.write('At round {} SoluMag10 of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, magnitude(np.hstack(a)), len(a) ))
                    f.write('At round {} SoluSets10 of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, list(np.around(np.hstack(a),2)), len(a) ))
                    a = np.array(soln[1][1])
                    f.write('At round {} SoluMag11 of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, magnitude(np.hstack(a)), len(a) ))
                    f.write('At round {} SoluSets11 of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, list(np.around(np.hstack(a),2)), len(a) ))
                    a = soln[1][0].flatten()
                    f.write('At round {} SoluMag10F of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, magnitude(np.hstack(a)), len(a) ))
                    f.write('At round {} SoluSets10F of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, list(np.around(np.hstack(a),2)), len(a) ))                
                    f.write('At round {} SoluSets10F of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, list(np.around(np.hstack(a),2)), len(a) ))
                    print(type(soln[0]))
                    print(type(soln[1]))
                    flat_list = np.concatenate(soln[1][0])
                    f.write('At round {} SoluSets1F of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, list(np.around(flat_list,2)), len(flat_list) ))
                    print(type(soln[1][0]))
                    print(type(soln[1][1]))

                # gather solutions from client
                csolns.append(soln)
                
                if( i < 100 ) :
                    if( doattacked > 0 ):
                        #param_diff.append(grads)
                        #param_diff.append(soln[1][0].flatten())
                        param_diff.append(soln[1][1].flatten())
                        worker_ids.append(indices[idx])
                    else :
                        #param_diff.append(grads)
                        #param_diff.append(soln[1][0].flatten())
                        param_diff.append(soln[1][1].flatten())
                        #param_diff.append(np.zeros(len(soln[1][0].flatten())))
                        worker_ids.append(indices[idx])

                if 'divfl' in self.clientsel_algo :
                    self.all_grads[indices[idx]] = grads
                
                # Update clients' models (only for the selected clients)
                #c.updatevec = (glob_copy - np.append(c.get_params()[0].flatten(), c.get_params()[1]))*0.01
                c.updatevec = np.append(c.get_params()[0].flatten(), c.get_params()[1])

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            if self.num_attackers > 0 :
                tqdm.write('ActiveAttacker ：{}'.format(list(np.array(attackers))))
                #tqdm.write('Worker Sets ：{}'.format(list(np.array(worker_ids[-self.clients_len:]))))

            # update models
            if 'divfl' in self.clientsel_algo :
                self.norm_diff[indices] = pairwise_distances(self.all_grads[indices], self.all_grads, metric="euclidean")
                self.norm_diff[:, indices] = self.norm_diff[indices].T
                self.latest_model = self.aggregate(csolns)
            elif self.clientsel_algo == 'powerchoice':
                self.latest_model = self.aggregate_simple(csolns)
            else:
                self.latest_model = self.aggregate(csolns)

        f.close()
        
        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} per-client-accuracy: {}'.format(i, [i/j for i,j in zip(stats[3], stats[2])]))
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} acc. variance: {}'.format(self.num_rounds, np.var([i/j for i,j in zip(stats[3], stats[2])])))
        tqdm.write('At round {} acc. 10th: {}'.format(self.num_rounds, np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.1)))
        tqdm.write('At round {} acc. 20th: {}'.format(self.num_rounds, np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.2)))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
        tqdm.write('At round {} training loss: {}'.format(self.num_rounds, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])))

        #if 'divfl' in self.clientsel_algo:
        #    np.save('./results/sent140/psubmod_select_client_sets_all_%s_epoch%d_numclient%d_m%d.npy' % (self.clientsel_algo, self.num_epochs, self.clients_per_round, self.m_interval), client_sets_all)
        #    np.save('./results/sent140/psubmod_client_diff_grad_all_%s_epoch%d_numclient%d_m%d.npy' % (self.clientsel_algo, self.num_epochs, self.clients_per_round, self.m_interval), diff_grad)
        #elif self.clientsel_algo == 'powerchoice':
        #    np.save('./results/sent140/powerofchoice_select_client_sets_all_%s_epoch%d_numclient%d_m%d.npy' % (self.clientsel_algo, self.num_epochs, self.clients_per_round, self.m_interval), client_sets_all)

        print('Number of samples', stats_train[2])

        # save_dir = "./results/"
        # result_path = os.path.join(save_dir,'divfl.csv')
        # print('Writing Statistics to file')
        # with open(result_path, 'wb') as f:
        #     np.savetxt(f, np.c_[test_accuracies, train_accuracies, train_losses, num_sampled], delimiter=",")
        
        print("Gradients shape: ({}, {})".format(len(param_diff), param_diff[0].shape[0]))
        #print("Prescaled gradients: {}".format(str(param_diff)))
        scaled_param_diff = apply_standard_scaler(param_diff)
        print("Postscaled gradients: {}".format(str(scaled_param_diff)))
        dim_reduced_gradients = calculate_pca_of_gradients(scaled_param_diff, min(len(param_diff), param_diff[0].shape[0]))
        print("PCA reduced gradients: {}".format(str(dim_reduced_gradients)))
        print("Dimensionally-reduced gradients shape: ({}, {})".format(len(dim_reduced_gradients), dim_reduced_gradients[0].shape[0]))
        plot_gradients_2d(zip(worker_ids, dim_reduced_gradients), attacker_indices)