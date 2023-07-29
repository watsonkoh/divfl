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
rounds = [200, 200, 200, 200, 200, 200]
sampling_rate = [1, 1, 1, 1, 1, 1]
mus = [1, 1, 1, 1, 1, 1]

testvar=( "divfl_lazy", "divfl_stochastic", "powerchoice", "fedavg", "fedprox" ) #clientsel=#
#testvar=( 1, 2, 3, 4, 5 ) #epochs eval
#testvar=( 5, 10, 15, 20, 30 ) #batch cpr
#testvar=( 0.001, 0.01, 0.05, 0.1, 0.5 ) #learn_rate

drop_rates=[0, 0.5, 0.9]
labels = ['FedAvg', r'FedProx ($\mu$=0)', r'FedProx ($\mu$>0)']
totalcolumn=len(testvar)
lastcolumnidx=totalcolumn-1
improv = 0

log = ["", "", "", "", "", "", "", "", "", "", "", ""]

#oattackers=( 1 2 3 4 5 6 7 8 9 10 11 12 13 15 17 20 22 24 26 28 30 ) 
#oattackerm=( 100 200 300 400 500 600 700 800 900 1000 ) 
#oattackern=( 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 )

attackers=( 0, 3, 6, 9, 12, 15, 17, 20, 24, 26, 30 ) 
attackerm=( 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 ) 
attackern=( 0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200 )

titles=sys.argv[2] if len(sys.argv) > 2 else 'clientsel'
idx=int(sys.argv[3]) if len(sys.argv) > 3 else int(0)
plotselect = sys.argv[4] if len(sys.argv) > 4 else 'Avg_Def' # ''Avg_Prox0_Prox1'
startidx = int(sys.argv[5]) if len(sys.argv) > 5 else int(0)

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

    size = min(len(rounds), len(sim), len(loss), len(accu))
    return rounds[:size], sim[:size], loss[:size], accu[:size]


f = plt.figure(figsize=[23, 10])


for pltrowidx in range(3):
    for pltidx in range(totalcolumn):

        ax = plt.subplot(3, 5, 5*(pltrowidx)+pltidx+1)
        
        if pltidx < 4 :
            logdataset = str(testvar[pltidx])+"/"+dataset[idx]
        else:
            logdataset = str(testvar[pltidx])+"/"+dataset[idx]

        if idx < 3 :
            attacker = attackers
        elif idx == 3 :
            attacker = attackerm
        elif idx == 4 :
            attacker = attackern
        else :
            attacker = ( 10, 10, 10, 10, 10, 10, 10, 10, 14 )

        if '1-1' in plotselect :
            labels = ['Attack', 'Defense', '']
            rounds1, sim1, losses1, test_accuracies1 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop_"+str(attacker[pltrowidx+startidx]))
            rounds2, sim2, losses2, test_accuracies2 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop_def_"+str(attacker[pltrowidx+startidx]))
            rounds3, sim3, losses3, test_accuracies3 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop_0")
        elif 'Def' in plotselect :
            labels = ['0', '1', '2']
            rounds1, sim1, losses1, test_accuracies1 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop_def_"+str(attacker[pltrowidx*3+0]))
            rounds2, sim2, losses2, test_accuracies2 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop_def_"+str(attacker[pltrowidx*3+1]))
            rounds3, sim3, losses3, test_accuracies3 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop_def_"+str(attacker[pltrowidx*3+2]))
            #rounds3, sim3, losses3, test_accuracies3 = [], [], [], []
        elif 'Reverse' in plotselect :
            labels = ['0', '3', '6']
            rounds1, sim1, losses1, test_accuracies1 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop_def_"+str(attacker[pltrowidx+0]))
            rounds2, sim2, losses2, test_accuracies2 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop_def_"+str(attacker[pltrowidx+3]))
            rounds3, sim3, losses3, test_accuracies3 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop_def_"+str(attacker[pltrowidx+6]))
            #rounds3, sim3, losses3, test_accuracies3 = [], [], [], []
        elif 'Attack' in plotselect :
            labels = ['0', '1', '2']
            rounds1, sim1, losses1, test_accuracies1 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop_"+str(attacker[pltrowidx*3+0]))
            rounds2, sim2, losses2, test_accuracies2 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop_"+str(attacker[pltrowidx*3+1]))
            rounds3, sim3, losses3, test_accuracies3 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop_"+str(attacker[pltrowidx*3+2]))
        else :
            labels = ['0', '3', '6']
            rounds1, sim1, losses1, test_accuracies1 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop_"+str(attacker[pltrowidx+0]))
            rounds2, sim2, losses2, test_accuracies2 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop_"+str(attacker[pltrowidx+3]))
            rounds3, sim3, losses3, test_accuracies3 = parse_log("logs/" + titles + "/" + logdataset + "/fedavg_drop_"+str(attacker[pltrowidx+6]))
        
        if dataset[idx] == 'nist' :
            if len(rounds1) > 0 :
                z1 = np.polyfit(np.array(rounds1), np.array(test_accuracies1), len(rounds1)/20)
                p1 = np.poly1d(z1)
                test_accuracies1 = list(p1(np.array(rounds1)))
            if len(rounds2) > 0 :
                z2 = np.polyfit(np.array(rounds2), np.array(test_accuracies2), len(rounds2)/20)
                p2 = np.poly1d(z2)
                test_accuracies2 = list(p2(np.array(rounds2)))
            if len(rounds3) > 0 :
                z3 = np.polyfit(np.array(rounds3), np.array(test_accuracies3), len(rounds3)/20)
                p3 = np.poly1d(z3)
                test_accuracies3 = list(p3(np.array(rounds3)))
        

        # #ff7f0e -> #014421
        if sys.argv[1] == 'loss':
            if pltrowidx == 2 and pltidx == lastcolumnidx:
                plt.plot(np.asarray(rounds1[:len(losses1):sampling_rate[idx]]), np.asarray(losses1)[::sampling_rate[idx]], ":", linewidth=3.0, label=labels[0], color="#014421")
                plt.plot(np.asarray(rounds2[:len(losses2):sampling_rate[idx]]), np.asarray(losses2)[::sampling_rate[idx]], '--', linewidth=2.0, label=labels[1], color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(losses3):sampling_rate[idx]]), np.asarray(losses3)[::sampling_rate[idx]], ".", linewidth=1.0, label=labels[2], color="#17becf")
            else:
                plt.plot(np.asarray(rounds1[:len(losses1):sampling_rate[idx]]), np.asarray(losses1)[::sampling_rate[idx]], ":", linewidth=3.0, color="#014421")
                plt.plot(np.asarray(rounds2[:len(losses2):sampling_rate[idx]]), np.asarray(losses2)[::sampling_rate[idx]], '--', linewidth=2.0, color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(losses3):sampling_rate[idx]]), np.asarray(losses3)[::sampling_rate[idx]], ".", linewidth=1.0, color="#17becf")

        elif sys.argv[1] == 'accuracy':
            if pltrowidx == 2 and pltidx == lastcolumnidx:
                plt.plot(np.asarray(rounds1[:len(test_accuracies1):sampling_rate[idx]]), np.asarray(test_accuracies1)[::sampling_rate[idx]], ":", linewidth=3.0, label=labels[0], color="#014421")
                plt.plot(np.asarray(rounds2[:len(test_accuracies2):sampling_rate[idx]]), np.asarray(test_accuracies2)[::sampling_rate[idx]], '--', linewidth=2.0, label=labels[1], color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(test_accuracies3):sampling_rate[idx]]), np.asarray(test_accuracies3)[::sampling_rate[idx]], ".", linewidth=1.0, label=labels[2], color="#17becf")
            else:
                plt.plot(np.asarray(rounds1[:len(test_accuracies1):sampling_rate[idx]]), np.asarray(test_accuracies1)[::sampling_rate[idx]], ":", linewidth=3.0, color="#014421")
                plt.plot(np.asarray(rounds2[:len(test_accuracies2):sampling_rate[idx]]), np.asarray(test_accuracies2)[::sampling_rate[idx]], '--', linewidth=2.0, color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(test_accuracies3):sampling_rate[idx]]), np.asarray(test_accuracies3)[::sampling_rate[idx]], ".", linewidth=1.0, color="#17becf")

        else:
            if pltrowidx == 2 and pltidx == lastcolumnidx:
                plt.plot(np.asarray(rounds1[:len(sim1):sampling_rate[idx]]), np.asarray(sim1)[::sampling_rate[idx]], ":", linewidth=3.0, label=labels[0], color="#014421")
                plt.plot(np.asarray(rounds2[:len(sim2):sampling_rate[idx]]), np.asarray(sim2)[::sampling_rate[idx]], '--', linewidth=2.0, label=labels[1], color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(sim3):sampling_rate[idx]]), np.asarray(sim3)[::sampling_rate[idx]], ".", linewidth=1.0, label=labels[2], color="#17becf")
            else:
                plt.plot(np.asarray(rounds1[:len(sim1):sampling_rate[idx]]), np.asarray(sim1)[::sampling_rate[idx]], ":", linewidth=3.0, color="#014421")
                plt.plot(np.asarray(rounds2[:len(sim2):sampling_rate[idx]]), np.asarray(sim2)[::sampling_rate[idx]], '--', linewidth=2.0, color="#e377c2")
                plt.plot(np.asarray(rounds3[:len(sim3):sampling_rate[idx]]), np.asarray(sim3)[::sampling_rate[idx]], ".", linewidth=1.0, color="#17becf")

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
    f.savefig(filename + "_" + dataset[idx] + "_" + ".pdf")
elif '1-1' in plotselect :
    f.savefig(filename + "_" + dataset[idx] + "_" + str(startidx) + ".pdf")
else :
    f.savefig(filename + "_" + dataset[idx] + "_" + plotselect + ".pdf")
#    f.savefig(filename + "_" + dataset[idx] + "_" + titles + "_" + plotselect + ".pdf")
