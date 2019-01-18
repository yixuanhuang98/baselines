# xterm -e 'python -m baselines.fcn.run_humanoid --seed=11 --reward_scale=0.1 --env=HalfCheetah-v2' &
# xterm -e 'python -m baselines.fcn.run_humanoid --seed=12 --reward_scale=0.1 --env=HalfCheetah-v2' &
# xterm -e 'python -m baselines.fcn.run_humanoid --seed=13 --reward_scale=0.1 --env=HalfCheetah-v2' &
# xterm -e 'python -m baselines.fcn.run_humanoid --seed=14 --reward_scale=0.1 --env=HalfCheetah-v2' &
# xterm -e 'python -m baselines.fcn.run_humanoid --seed=15 --reward_scale=0.1 --env=HalfCheetah-v2' &
# wait
#xterm -e 'python -m baselines.fcn.run_humanoid --seed=11 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/fcn_ho_11' &
#xterm -e 'python -m baselines.fcn.run_humanoid --seed=12 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/fcn_ho_12' &
#xterm -e 'python -m baselines.fcn.run_humanoid --seed=13 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/fcn_ho_13' &
#xterm -e 'python -m baselines.fcn.run_humanoid --seed=14 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/fcn_ho_14' &
#xterm -e 'python -m baselines.fcn.run_humanoid --seed=15 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/fcn_ho_15' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=11 --reward_scale=1 --env=Hopper-v2 --model-path=model/fcn_ho_11 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=12 --reward_scale=1 --env=Hopper-v2 --model-path=model/fcn_ho_12 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=13 --reward_scale=1 --env=Hopper-v2 --model-path=model/fcn_ho_13 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=14 --reward_scale=1 --env=Hopper-v2 --model-path=model/fcn_ho_14 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=15 --reward_scale=1 --env=Hopper-v2 --model-path=model/fcn_ho_15 --num_timesteps=10 --play' &
wait