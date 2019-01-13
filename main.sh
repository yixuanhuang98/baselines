xterm -e 'python -m baselines.fcn.run_humanoid --seed=1 --reward_scale=0.1 --env=Walker2d-v2' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=2 --reward_scale=0.1 --env=Walker2d-v2' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=3 --reward_scale=0.1 --env=Walker2d-v2' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=4 --reward_scale=0.1 --env=Walker2d-v2' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=5 --reward_scale=0.1 --env=Walker2d-v2' &
wait
xterm -e 'python -m baselines.fcn.run_humanoid --seed=1 --reward_scale=0.1 --env=Swimmer-v2' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=2 --reward_scale=0.1 --env=Swimmer-v2' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=3 --reward_scale=0.1 --env=Swimmer-v2' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=4 --reward_scale=0.1 --env=Swimmer-v2' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=5 --reward_scale=0.1 --env=Swimmer-v2' &
