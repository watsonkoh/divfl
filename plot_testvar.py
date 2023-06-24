import os, sys
import re
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import rcParams
from mpl_toolkits.axisartist.axislines import Subplot

matplotlib.rc('xtick', labelsize=17)
matplotlib.rc('ytick', labelsize=17)


filename="gradient"
#dataset = ["synthetic_iid", "synthetic_1_1", "mnist", "nist", "shakespeare", "sent140"]
#titles = ["IID", "Synthetic", "MNIST", "NIST", "Shakespeare", "Sent140"]
#rounds = [500, 500, 200, 500, 40, 800]
#sampling_rate = [1, 1, 1, 2, 1, 10]
#mus=[1, 1, 1, 1, 0.001, 0.01]

dataset = ["synthetic_iid", "synthetic_0_0", "synthetic_1_1", "mnist", "nist"]
titles = ["synthetic_IID", "synthetic_0_0", "synthetic_1_1", "MNIST", "NIST"]
rounds = [500, 500, 500, 500, 500, 500]
sampling_rate = [1, 1, 1, 1, 1, 1]
mus = [1, 1, 1, 1, 1, 1]

testvar=( "divfl_lazy", "divfl_stochastic", "powerchoice", "fedavg", "fedprox0" ) #clientsel=#
#testvar=( 1, 2, 3, 4, 5 ) #epochs eval
#testvar=( 5, 10, 15, 20, 30 ) #batch cpr
#testvar=( 0.001, 0.01, 0.05, 0.1, 0.5 ) #learn_rate

drop_rates=[0, 0.5, 0.9]
labels = ['FedAvg', r'FedProx ($\mu$=0)', r'FedProx ($\mu$>0)']
totalcolumn=len(testvar)
lastcolumnidx=totalcolumn-1
improv = 0

log = ["", "", "", "", "", "", "", "", "", "", "", ""]

idx=int(sys.argv[3]) if len(sys.argv) > 3 else int(0)
titles=sys.argv[2] if len(sys.argv) > 2 else 'clientsel'
plotselect = sys.argv[4] if len(sys.argv) > 4 else 'AvgProx0Prox1'


def parse_log(file_name):
    rounds = []
    accu = []
    loss = []
    sim = []

    for line in open(file_name, 'r'):

        search_train_accu = re.search(r'At round (.*) training accuracy: (.*)', line, re.M | re.I)
        if search_train_accu:
            rounds.append(int(search_train_accu.group(1)))
        else:
            search_test_accu = re.search(r'At round (.*) accuracy: (.*)', line, re.M | re.I)
            if search_test_accu:
                accu.append(float(search_test_accu.group(2)))

        search_loss = re.search(r'At round (.*) training loss: (.*)', line, re.M | re.I)
        if search_loss:
            loss.append(float(search_loss.group(2)))

        search_grad = re.search(r'gradient difference: (.*)', line, re.M | re.I)
        if search_grad:
            sim.append(float(search_grad.group(1)))

    return rounds, sim, loss, accu


f = plt.figure(figsize=[23, 10])


for pltrowidx in range(3):
    for pltidx in range(totalcolumn):

        ax = plt.subplot(3, 5, 5*(pltrowidx)+pltidx+1)
        
        
        if pltidx < 4 :
            logdataset = str(testvar[pltidx])+"/"+dataset[idx]
        else:
            logdataset = str(testvar[0])+"/"+dataset[idx]

        if 'Avg' in plotselect and pltidx < 4:
            rounds1, sim1, losses1, test_accuracies1 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop"+str(drop_rates[pltrowidx]))
        else :
            rounds1, sim1, losses1, test_accuracies1 = [], [], [], []

        if 'Prox0' in plotselect and pltidx >= 4:
            rounds2, sim2, losses2, test_accuracies2 = parse_log("logs/" + titles + "/" + logdataset + "/fedprox_drop"+str(drop_rates[pltrowidx])+"_mu0")
        else :
            rounds2, sim2, losses2, test_accuracies2 = [], [], [], []

        if 'Prox1' in plotselect and pltidx >= 4 :
            rounds3, sim3, losses3, test_accuracies3 = parse_log("logs/" + titles + "/" + logdataset + "/fedprox_drop"+str(drop_rates[pltrowidx])+"_mu" + str(mus[idx]))
        else :
            rounds3, sim3, losses3, test_accuracies3 = [], [], [], []
        
        # #ff7f0e -> #014421
        if sys.argv[1] == 'loss':
            if pltrowidx == 2 and pltidx == lastcolumnidx:
                plt.plot(np.asarray(rounds1[:len(losses1):sampling_rate[idx]]), np.asarray(losses1)[::sampling_rate[idx]], ":", linewidth=3.0, label=labels[0], color="#014421")
                plt.plot(np.asarray(rounds2[:len(losses2):sampling_rate[idx]]), np.asarray(losses2)[::sampling_rate[idx]], '--', linewidth=2.0, label=labels[1], color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(losses3):sampling_rate[idx]]), np.asarray(losses3)[::sampling_rate[idx]], ".", linewidth=2.0, label=labels[2], color="#17becf")
            else:
                plt.plot(np.asarray(rounds1[:len(losses1):sampling_rate[idx]]), np.asarray(losses1)[::sampling_rate[idx]], ":", linewidth=3.0, color="#014421")
                plt.plot(np.asarray(rounds2[:len(losses2):sampling_rate[idx]]), np.asarray(losses2)[::sampling_rate[idx]], '--', linewidth=2.0, color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(losses3):sampling_rate[idx]]), np.asarray(losses3)[::sampling_rate[idx]], ".", linewidth=2.0, color="#17becf")

        elif sys.argv[1] == 'accuracy':
            if pltrowidx == 2 and pltidx == lastcolumnidx:
                plt.plot(np.asarray(rounds1[:len(test_accuracies1):sampling_rate[idx]]), np.asarray(test_accuracies1)[::sampling_rate[idx]], ":", linewidth=3.0, label=labels[0], color="#014421")
                plt.plot(np.asarray(rounds2[:len(test_accuracies2):sampling_rate[idx]]), np.asarray(test_accuracies2)[::sampling_rate[idx]], '--', linewidth=2.0, label=labels[1], color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(test_accuracies3):sampling_rate[idx]]), np.asarray(test_accuracies3)[::sampling_rate[idx]], ".", linewidth=2.0, label=labels[2], color="#17becf")
            else:
                plt.plot(np.asarray(rounds1[:len(test_accuracies1):sampling_rate[idx]]), np.asarray(test_accuracies1)[::sampling_rate[idx]], ":", linewidth=3.0, color="#014421")
                plt.plot(np.asarray(rounds2[:len(test_accuracies2):sampling_rate[idx]]), np.asarray(test_accuracies2)[::sampling_rate[idx]], '--', linewidth=2.0, color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(test_accuracies3):sampling_rate[idx]]), np.asarray(test_accuracies3)[::sampling_rate[idx]], ".", linewidth=2.0, color="#17becf")

        else:
            if pltrowidx == 2 and pltidx == lastcolumnidx:
                plt.plot(np.asarray(rounds1[:len(sim1):sampling_rate[idx]]), np.asarray(sim1)[::sampling_rate[idx]], ":", linewidth=3.0, label=labels[0], color="#014421")
                plt.plot(np.asarray(rounds2[:len(sim2):sampling_rate[idx]]), np.asarray(sim2)[::sampling_rate[idx]], '--', linewidth=2.0, label=labels[1], color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(sim3):sampling_rate[idx]]), np.asarray(sim3)[::sampling_rate[idx]], ".", linewidth=2.0, label=labels[2], color="#17becf")
            else:
                plt.plot(np.asarray(rounds1[:len(sim1):sampling_rate[idx]]), np.asarray(sim1)[::sampling_rate[idx]], ":", linewidth=3.0, color="#014421")
                plt.plot(np.asarray(rounds2[:len(sim2):sampling_rate[idx]]), np.asarray(sim2)[::sampling_rate[idx]], '--', linewidth=2.0, color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(sim3):sampling_rate[idx]]), np.asarray(sim3)[::sampling_rate[idx]], ".", linewidth=2.0, color="#17becf")

        plt.xlabel("# Rounds", fontsize=16)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)

        if pltidx == 0:
            if sys.argv[1] == 'loss':
                plt.ylabel('Training Loss', fontsize=22)
            elif sys.argv[1] == 'accuracy':
                plt.ylabel('Testing Accuracy', fontsize=22)
            else:
                plt.ylabel('Gradient Variance', fontsize=22)

        if sys.argv[1] == 'loss':
                plt.ylim(0.2, 3.0)
                filename="loss"
        elif sys.argv[1] == 'accuracy':
                plt.ylim(0.0, 1.0)
                filename="accuracy"
        else:
                filename="gradient"

        if pltrowidx == 0:
            plt.title(testvar[pltidx], fontsize=22, fontweight='bold')

        ax.tick_params(color='#dddddd')
        ax.spines['bottom'].set_color('#dddddd')
        ax.spines['top'].set_color('#dddddd')
        ax.spines['right'].set_color('#dddddd')
        ax.spines['left'].set_color('#dddddd')
        ax.set_xlim(0, rounds[idx])


f.legend(frameon=False, loc='lower center', ncol=3, prop=dict(weight='bold'), borderaxespad=-0.3, fontsize=26)  # note: different from plt.legend
plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

if plotselect == 'AvgProx0Prox1' :
    f.savefig(filename + "_" + dataset[idx] + "_" + titles + ".pdf")
else :
    f.savefig(filename + "_" + dataset[idx] + "_" + titles + "_" + plotselect + ".pdf")
