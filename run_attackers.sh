#array=( "synthetic_1_1" "mnist" "nist" "shakespeare" "sent140" )
#array=( "synthetic_iid" "synthetic_0_0" "synthetic_1_1" "mnist" "nist" )
array=( "mnist" "nist" )
mus=( 1 1 1 1 1 1 )
r1attackers=( 15 15 15 300 60 )
r2attackers=( 20 20 20 600 120 )
activeusers=( 30 30 30 250 50 )
#testvar=( 0.001 0.01 0.05 0.1 0.5 ) #rate
#testvar=( 5 10 15 20 30 ) #cpr batch
#testvar=( 1 2 3 4 5 ) #epochs eval
#testvar=( "fedavg" "divfl_lazy" "divfl_stochastic" "powerchoice" "fedprox" ) #clientsel=
testvar=( "divfl_stochastic" ) #clientsel=
testgroup="clientsel_0720"   #$4


rm -rf log/*
mkdir log
mkdir report/$testgroup
mkdir logs/$testgroup

cp -f ${0##*/} logs/$testgroup
cp -f ${0##*/} report/$testgroup

for ((varidx=0; varidx<${#testvar[@]}; ++varidx)); do
#for ((varidx=0; varidx<1; ++varidx)); do
  cp -R logs/$testgroup/${testvar[$varidx]}/synthetic* log/
  for ((arrayidx=0; arrayidx<${#array[@]}; ++arrayidx)); do
    echo "$arrayidx" "${array[arrayidx]}"
    echo ${array[$arrayidx]}
    mkdir log/${array[$arrayidx]}

    if [ ${array[$arrayidx]} == "mnist" ]
    then
        echo "Attacker MNIST"
        attacker=( 0 10 100 200 300 400 500 600 700 800 900 1000 ) 
    elif [ ${array[$arrayidx]} == "nist" ]
    then
        echo "Attacker NIST"
        attacker=( 0 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 )
    else
        echo "Attacker Synthetic"
        attacker=( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 15 17 20 22 24 26 28 30 ) 
    fi

    for ((attackeridx=0; attackeridx<${#attacker[@]}; ++attackeridx)); do
      if [ "fedprox" == ${testvar[$varidx]} ]
      then
          bash run_fedprox.sh ${array[$arrayidx]} ${activeusers[$arrayidx]} ${attacker[$attackeridx]} 0 ${testvar[$varidx]} | tee log/${array[$arrayidx]}/fedavg_drop_${attacker[$attackeridx]}
          bash run_fedprox.sh ${array[$arrayidx]} ${activeusers[$arrayidx]} ${attacker[$attackeridx]} 1 ${testvar[$varidx]} | tee log/${array[$arrayidx]}/fedavg_drop_def_${attacker[$attackeridx]}
      else
          bash run_fedavg.sh ${array[$arrayidx]} ${activeusers[$arrayidx]} ${attacker[$attackeridx]} 0 ${testvar[$varidx]} | tee log/${array[$arrayidx]}/fedavg_drop_${attacker[$attackeridx]}
          bash run_fedavg.sh ${array[$arrayidx]} ${activeusers[$arrayidx]} ${attacker[$attackeridx]} 1 ${testvar[$varidx]} | tee log/${array[$arrayidx]}/fedavg_drop_def_${attacker[$attackeridx]}
      fi
    done
  done

  mkdir logs/$testgroup/${testvar[$varidx]}
  #rm -rf logs/$testgroup/${testvar[$varidx]}/*
  mv -f log/* logs/$testgroup/${testvar[$varidx]}/
done

