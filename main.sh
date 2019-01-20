xterm -e 'python -m baselines.fcn.run_humanoid --seed=1 --reward_scale=1.0 --env=Walker2d-v2 --model-path=model/fcn_wk_1 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=2 --reward_scale=1.0 --env=Walker2d-v2 --model-path=model/fcn_wk_2 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=3 --reward_scale=1.0 --env=Walker2d-v2 --model-path=model/fcn_wk_3 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=4 --reward_scale=1.0 --env=Walker2d-v2 --model-path=model/fcn_wk_4 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.fcn.run_humanoid --seed=5 --reward_scale=1.0 --env=Walker2d-v2 --model-path=model/fcn_wk_5 --num_timesteps=10 --play' &
wait
xterm -e 'python -m baselines.scn.run_humanoid --seed=1 --reward_scale=0.1 --env=Walker2d-v2 --model-path=model/scn_wk_1 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.scn.run_humanoid --seed=2 --reward_scale=0.1 --env=Walker2d-v2 --model-path=model/scn_wk_2 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.scn.run_humanoid --seed=3 --reward_scale=0.1 --env=Walker2d-v2 --model-path=model/scn_wk_3 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.scn.run_humanoid --seed=4 --reward_scale=0.1 --env=Walker2d-v2 --model-path=model/scn_wk_4 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.scn.run_humanoid --seed=5 --reward_scale=0.1 --env=Walker2d-v2 --model-path=model/scn_wk_5 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.fcn.run_humanoid --seed=6 --reward_scale=1.0 --env=Walker2d-v2 --model-path=model/fcn_wk_6' &
#xterm -e 'python -m baselines.fcn.run_humanoid --seed=7 --reward_scale=1.0 --env=Walker2d-v2 --model-path=model/fcn_wk_7' &
#xterm -e 'python -m baselines.fcn.run_humanoid --seed=8 --reward_scale=1.0 --env=Walker2d-v2 --model-path=model/fcn_wk_8' &
#xterm -e 'python -m baselines.fcn.run_humanoid --seed=9 --reward_scale=1.0 --env=Walker2d-v2 --model-path=model/fcn_wk_9' &
#xterm -e 'python -m baselines.fcn.run_humanoid --seed=10 --reward_scale=1.0 --env=Walker2d-v2 --model-path=model/fcn_wk_10' &