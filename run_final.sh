#array=( "synthetic_1_1" "mnist" "nist" "shakespeare" "sent140" )
#array=( "synthetic_iid" "synthetic_1_1" "mnist" "nist" )
array=( "synthetic_iid" "synthetic_0_0" "synthetic_1_1" "mnist" "nist" )
mus=( 1 1 1 1 1 1 )
r1attackers=( 2 2 2 20 5 )
r2attackers=( 4 4 4 40 10 )
#testvar=( 0.001 0.01 0.05 0.1 0.5 ) #rate
#testvar=( 5 10 15 20 30 ) #cpr batch
#testvar=( 1 2 3 4 5 ) #epochs eval
testvar=( "divfl_lazy" "divfl_stochastic" "powerchoice" "fedavg" ) #clientsel=
testgroup="clientsel"   $4

rm -rf log/*
mkdir report/$testgroup
mkdir logs/$testgroup

for ((varidx=0; varidx<${#testvar[@]}; ++varidx)); do
  cp -R logs/$testgroup/${testvar[$varidx]}/synthetic* log/
  for ((arrayidx=3; arrayidx<${#array[@]}; ++arrayidx)); do
    echo "$arrayidx" "${array[arrayidx]}"
    echo ${array[$arrayidx]}
    mkdir log/${array[$arrayidx]}

    bash run_fedavg.sh ${array[$arrayidx]} 0 0 ${testvar[$varidx]} | tee log/${array[$arrayidx]}/fedavg_drop0
#    mv -f log/GradientSolution.log log/${array[$arrayidx]}/fedavg_drop0.log
    bash run_fedavg.sh ${array[$arrayidx]} ${r1attackers[$arrayidx]} 0 ${testvar[$varidx]} | tee log/${array[$arrayidx]}/fedavg_drop0.5
#    mv -f log/GradientSolution.log log/${array[$arrayidx]}/fedavg_drop0.5.log
    bash run_fedavg.sh ${array[$arrayidx]} ${r2attackers[$arrayidx]} 0 ${testvar[$varidx]} | tee log/${array[$arrayidx]}/fedavg_drop0.9
#    mv -f log/GradientSolution.log log/${array[$arrayidx]}/fedavg_drop0.9.log

    bash run_fedprox.sh ${array[$arrayidx]} 0 0 ${testvar[$varidx]} | tee log/${array[$arrayidx]}/fedprox_drop0_mu0
#    mv -f log/GradientSolution.log log/${array[$arrayidx]}/fedavg_drop0_mu0.log
    bash run_fedprox.sh ${array[$arrayidx]} ${r1attackers[$arrayidx]} 0 ${testvar[$varidx]} | tee log/${array[$arrayidx]}/fedprox_drop0.5_mu0
#    mv -f log/GradientSolution.log log/${array[$arrayidx]}/fedavg_drop0.5_mu0.log
    bash run_fedprox.sh ${array[$arrayidx]} ${r2attackers[$arrayidx]} 0 ${testvar[$varidx]} | tee log/${array[$arrayidx]}/fedprox_drop0.9_mu0
#    mv -f log/GradientSolution.log log/${array[$arrayidx]}/fedavg_drop0.9_mu0.log

    bash run_fedprox.sh ${array[$arrayidx]} 0 ${mus[$arrayidx]} ${testvar[$varidx]} | tee log/${array[$arrayidx]}/fedprox_drop0_mu${mus[$arrayidx]}
#    mv -f log/GradientSolution.log log/${array[$arrayidx]}/fedprox_drop0_mu${mus[$arrayidx]}.log
    bash run_fedprox.sh ${array[$arrayidx]} ${r1attackers[$arrayidx]} ${mus[$arrayidx]} ${testvar[$varidx]} | tee log/${array[$arrayidx]}/fedprox_drop0.5_mu${mus[$arrayidx]}
#    mv -f log/GradientSolution.log log/${array[$arrayidx]}/fedprox_drop0.5_mu${mus[$arrayidx]}.log
    bash run_fedprox.sh ${array[$arrayidx]} ${r2attackers[$arrayidx]} ${mus[$arrayidx]} ${testvar[$varidx]} | tee log/${array[$arrayidx]}/fedprox_drop0.9_mu${mus[$arrayidx]}
#    mv -f log/GradientSolution.log log/${array[$arrayidx]}/fedprox_drop0.9_mu${mus[$arrayidx]}.log
  done

  python3 plot_final.py accuracy
  python3 plot_final.py loss
  python3 plot_final.py gradient

  mkdir report/$testgroup/${testvar[$varidx]}
  mkdir logs/$testgroup/${testvar[$varidx]}
  rm -rf report/$testgroup/${testvar[$varidx]}/*
  rm -rf logs/$testgroup/${testvar[$varidx]}/*
  mv -f *.pdf report/$testgroup/${testvar[$varidx]}/
  mv -f log/* logs/$testgroup/${testvar[$varidx]}/
done

for ((arrayidx=0; arrayidx<${#array[@]}; ++arrayidx)); do
  python3 plot_testvar.py accuracy $testgroup $arrayidx
  python3 plot_testvar.py loss $testgroup $arrayidx
  python3 plot_testvar.py gradient $testgroup $arrayidx
  python3 plot_alg.py alg clientsel $arrayidx
done

mv -f *.pdf report/$testgroup/
