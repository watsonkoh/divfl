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

testvar=( "divfl_lazy", "divfl_stochastic", "powerchoice", "fedavg" ) #clientsel=#
#testvar=( 1, 2, 3, 4, 5 ) #epochs eval
#testvar=( 5, 10, 15, 20, 30 ) #batch cpr
#testvar=( 0.001, 0.01, 0.05, 0.1, 0.5 ) #learn_rate

drop_rates=[0, 0.5, 0.9]
labels = ['FedAvg', r'FedProx ($\mu$=0)', r'FedProx ($\mu$>0)']
plotcolor=["#014421", "#e377c2", "#17becf"]
plotrow=["accuracy", "loss", "gradient"] 
plotcolumn=testvar
totalcolumn=len(plotcolumn)
lastcolumnidx=totalcolumn-1
improv = 0

log = ["", "", "", "", "", "", "", "", "", "", "", ""]

idx=int(sys.argv[3]) if len(sys.argv) > 3 else int(0)
titles=sys.argv[2] if len(sys.argv) > 2 else 'clientsel'
plotselect = sys.argv[4] if len(sys.argv) > 4 else 'a'

def parse_log(file_name):
    rounds = []
    accu = []
    loss = []
    gradient = []

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
            gradient.append(float(search_grad.group(1)))

    return rounds, gradient, loss, accu


f = plt.figure(figsize=[23, 10])

for pltrowidx in range(3):
    for pltidx in range(totalcolumn):

        ax = plt.subplot(3, 5, 5*(pltrowidx)+pltidx+1)
        
        logdataset = str(testvar[pltidx])+"/"+dataset[idx]

        rounds1, gradient1, losses1, test_accuracies1 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop"+str(drop_rates[pltrowidx]))
        rounds2, gradient2, losses2, test_accuracies2 = parse_log("logs/" + titles + "/" + logdataset + "/fedprox_drop"+str(drop_rates[pltrowidx])+"_mu0")
#        rounds2, gradient2, losses2, test_accuracies2 = [], [], [], []
        rounds3, gradient3, losses3, test_accuracies3 = [], [], [], []

        plotdataround1 = rounds1
        plotdataround2 = rounds2
        plotdataround3 = rounds3

        if pltrowidx == 0 :
            plotdata1 = test_accuracies1
            plotdata2 = test_accuracies2
            plotdata3 = test_accuracies3
            plt.ylim(0.0, 1.0)
        elif pltrowidx == 1 :
            plotdata1 = losses1
            plotdata2 = losses2
            plotdata3 = losses3
            plt.ylim(0.2, 3.0)
        elif pltrowidx == 2 :
            plotdata1 = gradient1
            plotdata2 = gradient2
            plotdata3 = gradient3

        
        if pltrowidx == 2 and pltidx == lastcolumnidx:
                plt.plot(np.asarray(plotdataround1[:len(plotdata1):sampling_rate[idx]]), np.asarray(plotdata1)[::sampling_rate[idx]], ":", linewidth=3.0, label=labels[0], color="#014421")
                plt.plot(np.asarray(plotdataround2[:len(plotdata2):sampling_rate[idx]]), np.asarray(plotdata2)[::sampling_rate[idx]], '--', linewidth=1.0, label=labels[1], color="#e377c2")
                plt.plot(np.asarray(plotdataround3[:len(plotdata3):sampling_rate[idx]]), np.asarray(plotdata3)[::sampling_rate[idx]], ".", linewidth=1.0, label=labels[2], color="#17becf")
        else:
                plt.plot(np.asarray(plotdataround1[:len(plotdata1):sampling_rate[idx]]), np.asarray(plotdata1)[::sampling_rate[idx]], ":", linewidth=3.0, color="#014421")
                plt.plot(np.asarray(plotdataround2[:len(plotdata2):sampling_rate[idx]]), np.asarray(plotdata2)[::sampling_rate[idx]], '--', linewidth=1.0, color="#e377c2")
                plt.plot(np.asarray(plotdataround3[:len(plotdata3):sampling_rate[idx]]), np.asarray(plotdata3)[::sampling_rate[idx]], ".", linewidth=1.0, color="#17becf")


        plt.xlabel("# Rounds", fontsize=16)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)

        if pltidx == 0:
            if pltrowidx == 0 :
                plt.ylabel('Testing Accuracy', fontsize=22)
            elif pltrowidx == 1 :
                plt.ylabel('Training Loss', fontsize=22)
            elif pltrowidx == 2 :
                plt.ylabel('Gradient Variance', fontsize=22)

        filename="alg"

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
f.savefig(filename + "_" + dataset[idx] + "_" + titles + ".pdf")
