xterm -e 'python -m baselines.fcn.run_humanoid --seed=6 --reward_scale=1.0 --env=Humanoid-v2 --model-path=model/fcn_hu_6 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=7 --reward_scale=1.0 --env=Humanoid-v2 --model-path=model/fcn_hu_7 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=8 --reward_scale=1.0 --env=Humanoid-v2 --model-path=model/fcn_hu_8 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=9 --reward_scale=1.0 --env=Humanoid-v2 --model-path=model/fcn_hu_9 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=10 --reward_scale=1.0 --env=Humanoid-v2 --model-path=model/fcn_hu_10 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.scn.run_humanoid --seed=6 --reward_scale=0.1 --env=Humanoid-v2 --model-path=model/scn_hu_6 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.scn.run_humanoid --seed=7 --reward_scale=0.1 --env=Humanoid-v2 --model-path=model/scn_hu_7 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.scn.run_humanoid --seed=8 --reward_scale=0.1 --env=Humanoid-v2 --model-path=model/scn_hu_8 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.scn.run_humanoid --seed=9 --reward_scale=0.1 --env=Humanoid-v2 --model-path=model/scn_hu_9 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.scn.run_humanoid --seed=10 --reward_scale=0.1 --env=Humanoid-v2 --model-path=model/scn_hu_10 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.fcn.run_humanoid --seed=6 --reward_scale=1.0 --env=Humanoid-v2 --model-path=model/fcn_hu_6' &
#xterm -e 'python -m baselines.fcn.run_humanoid --seed=7 --reward_scale=1.0 --env=Humanoid-v2 --model-path=model/fcn_hu_7' &
#xterm -e 'python -m baselines.fcn.run_humanoid --seed=8 --reward_scale=1.0 --env=Humanoid-v2 --model-path=model/fcn_hu_8' &
#xterm -e 'python -m baselines.fcn.run_humanoid --seed=9 --reward_scale=1.0 --env=Humanoid-v2 --model-path=model/fcn_hu_9' &
#xterm -e 'python -m baselines.fcn.run_humanoid --seed=10 --reward_scale=1.0 --env=Humanoid-v2 --model-path=model/fcn_hu_10' &