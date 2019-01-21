xterm -e 'python -m baselines.ppo1relu16.run_humanoid --seed=1 --reward_scale=0.1 --env=Walker2d-v2 --model-path=model/relu_wk_1' &
xterm -e 'python -m baselines.ppo1relu16.run_humanoid --seed=2 --reward_scale=0.1 --env=Walker2d-v2 --model-path=model/relu_wk_2' &
xterm -e 'python -m baselines.ppo1relu16.run_humanoid --seed=3 --reward_scale=0.1 --env=Walker2d-v2 --model-path=model/relu_wk_3' &
xterm -e 'python -m baselines.ppo1relu16.run_humanoid --seed=4 --reward_scale=0.1 --env=Walker2d-v2 --model-path=model/relu_wk_4' &
xterm -e 'python -m baselines.ppo1relu16.run_humanoid --seed=5 --reward_scale=0.1 --env=Walker2d-v2 --model-path=model/relu_wk_5' &
