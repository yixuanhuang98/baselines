xterm -e 'python -m baselines.fcn.run_humanoid --seed=6 --reward_scale=0.1 --env=HalfCheetah-v2' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=7 --reward_scale=0.1 --env=HalfCheetah-v2' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=8 --reward_scale=0.1 --env=HalfCheetah-v2' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=9 --reward_scale=0.1 --env=HalfCheetah-v2' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=10 --reward_scale=0.1 --env=HalfCheetah-v2' &
wait
xterm -e 'python -m baselines.fcn.run_humanoid --seed=6 --reward_scale=0.1 --env=Hopper-v2' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=7 --reward_scale=0.1 --env=Hopper-v2' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=8 --reward_scale=0.1 --env=Hopper-v2' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=9 --reward_scale=0.1 --env=Hopper-v2' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=10 --reward_scale=0.1 --env=Hopper-v2' &