#!/usr/bin/env bash
mkdir pca
mkdir pca/divfl_lazy
mkdir pca/divfl_stochastic
mkdir pca/powerchoice
mkdir pca/fedprox
mkdir pca/fedavg

grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" divfl_lazy/mnist | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/divfl_lazy/mnist.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" divfl_lazy/nist | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/divfl_lazy/nist.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" divfl_lazy/synthetic_0_0 | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/divfl_lazy/synthetic_0_0.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" divfl_lazy/synthetic_1_1 | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/divfl_lazy/synthetic_1_1.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" divfl_lazy/synthetic_iid | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/divfl_lazy/synthetic_iid.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" divfl_stochastic/mnist | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/divfl_stochastic/mnist.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" divfl_stochastic/nist | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/divfl_stochastic/nist.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" divfl_stochastic/synthetic_0_0 | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/divfl_stochastic/synthetic_0_0.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" divfl_stochastic/synthetic_1_1 | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/divfl_stochastic/synthetic_1_1.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" divfl_stochastic/synthetic_iid | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/divfl_stochastic/synthetic_iid.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" powerchoice/mnist | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/powerchoice/mnist.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" powerchoice/nist | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/powerchoice/nist.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" powerchoice/synthetic_0_0 | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/powerchoice/synthetic_0_0.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" powerchoice/synthetic_1_1 | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/powerchoice/synthetic_1_1.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" powerchoice/synthetic_iid | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/powerchoice/synthetic_iid.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" fedprox/mnist | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/fedprox/mnist.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" fedprox/nist | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/fedprox/nist.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" fedprox/synthetic_0_0 | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/fedprox/synthetic_0_0.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" fedprox/synthetic_1_1 | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/fedprox/synthetic_1_1.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" fedprox/synthetic_iid | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/fedprox/synthetic_iid.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" fedavg/mnist | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/fedavg/mnist.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" fedavg/nist | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/fedavg/nist.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" fedavg/synthetic_0_0 | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/fedavg/synthetic_0_0.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" fedavg/synthetic_1_1 | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/fedavg/synthetic_1_1.csv
grep -R "_algo\|dataset\|num_attackers\|AttackerSet\|Threshold" fedavg/synthetic_iid | sed 's/:AttackerSet:/:AttackerSet: List /g' > pca/fedavg/synthetic_iid.csv

grep -R "Suspected_AttackerSet:" pca/f*/ pca/p*/ pca/d*/ | sed 's/\//:/g' | sed 's/fedavg_drop_def_//g' | sed 's/:/ /g' | sed 's/List/Suspect/g' > pca/confusion.csv
