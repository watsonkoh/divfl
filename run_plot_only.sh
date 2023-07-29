array=( "synthetic_iid" "synthetic_0_0" "synthetic_1_1" "mnist" "nist" )
testgroup="clientsel_0718"   $4
reportgroup="clientsel_0729"   $4

for ((arrayidx=0; arrayidx<${#array[@]}; ++arrayidx)); do
  python3 plot_clients.py accuracy $testgroup $arrayidx Def
#  python3 plot_clients.py loss $testgroup $arrayidx Def
#  python3 plot_clients.py gradient $testgroup $arrayidx Def
  python3 plot_clients.py accuracy $testgroup $arrayidx Attack
#  python3 plot_clients.py loss $testgroup $arrayidx Attack
#  python3 plot_clients.py gradient $testgroup $arrayidx Attack

  python3 plot_clients.py accuracy $testgroup $arrayidx 1-1 0
#  python3 plot_clients.py loss $testgroup $arrayidx 1-1 0
#  python3 plot_clients.py gradient $testgroup $arrayidx 1-1 0
  python3 plot_clients.py accuracy $testgroup $arrayidx 1-1 3
#  python3 plot_clients.py loss $testgroup $arrayidx 1-1 3
#  python3 plot_clients.py gradient $testgroup $arrayidx 1-1 3
  python3 plot_clients.py accuracy $testgroup $arrayidx 1-1 6
#  python3 plot_clients.py loss $testgroup $arrayidx 1-1 6
#  python3 plot_clients.py gradient $testgroup $arrayidx 1-1 6
done

mkdir report/$reportgroup/
mv -f *.pdf report/$reportgroup/

#  python3 plot_testvar.py accuracy $testgroup $arrayidx Avg_Def
#  python3 plot_testvar.py loss $testgroup $arrayidx Avg_Def
#  python3 plot_testvar.py gradient $testgroup $arrayidx Avg_Def
#  python3 plot_alg.py alg $testgroup $arrayidx
