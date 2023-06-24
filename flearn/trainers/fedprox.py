import numpy as np
from tqdm import trange, tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad
import math

def magnitude(x):
    return math.sqrt(sum(i ** 2 for i in x))

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train')
        self.inner_opt = PerturbedGradientDescent(params['learning_rate'], params['mu'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))

        train_accuracies = []
        num_sampled = []
        client_sets_all = np.zeros([self.num_rounds, self.clients_per_round], dtype=int)

        attackers_round = 0
        attacker_probability = 0.5
        attacker_indices = []
        selected_attackers = []
        attacker_indices, selected_attackers = self.select_attackers(self.num_attackers, num_clients=self.num_attackers)
        if self.num_attackers > 0 :
            tqdm.write('AttackerSet ：{}'.format(list(np.array(attacker_indices))))

        f = open("log/GradientSolution.log", "w")

        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats = self.test() # have set the latest model for all clients
                stats_train = self.train_error_and_loss()

                tqdm.write('At round {} per-client-accuracy: {}'.format(i, list(np.around(np.array([i/j for i,j in zip(stats[3], stats[2])]),2))))
                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3])*1.0/np.sum(stats[2])))  # testing accuracy
                tqdm.write('At round {} acc. variance: {}'.format(i, np.var([i/j for i,j in zip(stats[3], stats[2])])))  # testing accuracy variance
                tqdm.write('At round {} acc. 10th: {}'.format(i, np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.1)))  # testing accuracy variance
                tqdm.write('At round {} acc. 20th: {}'.format(i, np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.2)))  # testing accuracy variance
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])))

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
            for idx in range(len(self.clients)):
                difference += np.sum(np.square(global_grads - local_grads[idx]))
            difference = difference * 1.0 / len(self.clients)
            if i % self.eval_every == 0:
                tqdm.write('gradient difference: {}'.format(difference))

            indices, selected_clients = self.select_clients(i, num_clients=math.ceil(self.clients_per_round * (1 - self.drop_percent)))  # uniform sampling
            np.random.seed(i)  # make sure that the stragglers are the same for FedProx and FedAvg
            active_clients = selected_clients # np.random.choice(selected_clients, math.ceil(self.clients_per_round * (1 - self.drop_percent)), replace=False)

            tqdm.write('Client set ：{}'.format(list(np.array(indices))))
            print('Selected Client set length ', len(selected_clients))
            print('Active Client set length ', len(active_clients))
            client_sets_all[i] = indices
            tqdm.write('At round {} num. clients sampled: {}'.format(i, len(indices)))
            num_sampled.append(len(indices))

            csolns = [] # buffer for receiving client solutions

            self.inner_opt.set_params(self.latest_model, self.client_model)

            attackers = []
            np.random.seed(i)
            if( attackers_round > 0 ) :
                attacker_probability = 0.5
            elif ( attacker_probability < 0.9 ) :
                attacker_probability += 0.1
            random_attack = np.random.binomial(n=1, p=min(1.0,attacker_probability), size=[20])
            tqdm.write('RandomAttack: {}'.format(list(np.array(random_attack))))
            attackers_round = 0
            
            for idx, c in enumerate(selected_clients.tolist()):
                # communicate the latest model
                c.set_params(self.latest_model)

                total_iters = int(self.num_epochs * c.num_samples / self.batch_size)+2 # randint(low,high)=[low,high)

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
                if c in active_clients:
                    soln, stats, grads = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size, attacked=doattacked)
                else:
                    #soln, stats = c.solve_iters(num_iters=np.random.randint(low=1, high=total_iters), batch_size=self.batch_size, attacked=doattacked)
                    #print("num_epochs:", self.num_epochs)
                    if self.num_epochs == 1:
                        soln, stats, grads = c.solve_inner(num_epochs=1, batch_size=self.batch_size, attacked=doattacked)
                    else:
                        soln, stats, grads = c.solve_inner(num_epochs=np.random.randint(low=1, high=self.num_epochs), batch_size=self.batch_size, attacked=doattacked)

                if indices[idx] in attacker_indices or indices[idx] < 3 :
                    f.write('At round {} GradMag of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, magnitude(np.array(grads)), len(grads) ))
                    f.write('At round {} GradSets of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, list(np.around(np.array(grads),2)), len(grads) ))
                    a = np.array(soln[1][0])
                    f.write('At round {} SoluMag of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, magnitude(np.hstack(a)), len(a) ))
                    f.write('At round {} SoluSets of Client {} ：{} ：{} ：{}\n'.format(i, c.id, doattacked, list(np.around(np.hstack(a),2)), len(a) ))
                
                # gather solutions from client
                csolns.append(soln)
        
                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            if self.num_attackers > 0 :
                tqdm.write('At round {} ActiveAttacker ：{}'.format(i, list(np.array(attackers))))

            # update models
            self.latest_model = self.aggregate(csolns)
            self.client_model.set_params(self.latest_model)

        f.close()
        
        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} per-client-accuracy: {}'.format(i, [i/j for i,j in zip(stats[3], stats[2])]))
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3])*1.0/np.sum(stats[2])))
        tqdm.write('At round {} acc. variance: {}'.format(self.num_rounds, np.var([i/j for i,j in zip(stats[3], stats[2])])))
        tqdm.write('At round {} acc. 10th: {}'.format(self.num_rounds, np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.1)))
        tqdm.write('At round {} acc. 20th: {}'.format(self.num_rounds, np.quantile([i/j for i,j in zip(stats[3], stats[2])], 0.2)))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
        tqdm.write('At round {} training loss: {}'.format(self.num_rounds, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])))
                
        print('Number of samples', stats_train[2])
        