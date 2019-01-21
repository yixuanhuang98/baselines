#xterm -e 'python -m baselines.ppo3relu64.run_humanoid --seed=11 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlpr_ho_11 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3relu64.run_humanoid --seed=12 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlpr_ho_12 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3relu64.run_humanoid --seed=13 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlpr_ho_13 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3relu64.run_humanoid --seed=14 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlpr_ho_14 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3relu64.run_humanoid --seed=15 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlpr_ho_15 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3relu64.run_humanoid --seed=11 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlpr_ho_11 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3relu64.run_humanoid --seed=12 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlpr_ho_12 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3relu64.run_humanoid --seed=13 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlpr_ho_13 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3relu64.run_humanoid --seed=14 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlpr_ho_14 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3relu64.run_humanoid --seed=15 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlpr_ho_15 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3.run_humanoid --seed=11 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_11 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3.run_humanoid --seed=12 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_12 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3.run_humanoid --seed=13 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_13 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3.run_humanoid --seed=14 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_14 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3.run_humanoid --seed=15 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_15 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3.run_humanoid --seed=1 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_1 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3.run_humanoid --seed=2 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_2 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3.run_humanoid --seed=3 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_3 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3.run_humanoid --seed=4 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_4 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3.run_humanoid --seed=5 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_5 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.ppo3.run_humanoid --seed=1 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_1' &
xterm -e 'python -m baselines.ppo3.run_humanoid --seed=2 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_2' &
xterm -e 'python -m baselines.ppo3.run_humanoid --seed=3 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_3' &
xterm -e 'python -m baselines.ppo3.run_humanoid --seed=4 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_4' &
xterm -e 'python -m baselines.ppo3.run_humanoid --seed=5 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_5' &
wait
xterm -e 'python -m baselines.relu3.run_humanoid --seed=1 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/relu3_ho_1' &
xterm -e 'python -m baselines.relu3.run_humanoid --seed=2 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/relu3_ho_2' &
xterm -e 'python -m baselines.relu3.run_humanoid --seed=3 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/relu3_ho_3' &
xterm -e 'python -m baselines.relu3.run_humanoid --seed=4 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/relu3_ho_4' &
xterm -e 'python -m baselines.relu3.run_humanoid --seed=5 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/relu3_ho_5' &
wait
xterm -e 'python -m baselines.linear.run_humanoid --seed=6 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/linear_ho_6' &
xterm -e 'python -m baselines.linear.run_humanoid --seed=7 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/linear_ho_7' &
xterm -e 'python -m baselines.linear.run_humanoid --seed=8 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/linear_ho_8' &
xterm -e 'python -m baselines.linear.run_humanoid --seed=9 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/linear_ho_9' &
xterm -e 'python -m baselines.linear.run_humanoid --seed=10 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/linear_ho_10' &
wait
xterm -e 'python -m baselines.linear.run_humanoid --seed=6 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/linear_ho_6 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.linear.run_humanoid --seed=7 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/linear_ho_7 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.linear.run_humanoid --seed=8 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/linear_ho_8 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.linear.run_humanoid --seed=9 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/linear_ho_9 --num_timesteps=10 --play' &
xterm -e 'python -m baselines.linear.run_humanoid --seed=10 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/linear_ho_10 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3.run_humanoid --seed=1 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_1 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3.run_humanoid --seed=2 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_2 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3.run_humanoid --seed=3 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_3 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3.run_humanoid --seed=4 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_4 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3.run_humanoid --seed=5 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_5 --num_timesteps=10 --play' &
#xterm -e 'python -m baselines.ppo3relu64.run_humanoid --seed=1 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_1' &
#xterm -e 'python -m baselines.ppo3relu64.run_humanoid --seed=2 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_2' &
#xterm -e 'python -m baselines.ppo3relu64.run_humanoid --seed=3 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_3' &
#xterm -e 'python -m baselines.ppo3relu64.run_humanoid --seed=4 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_4' &
#xterm -e 'python -m baselines.ppo3relu64.run_humanoid --seed=5 --reward_scale=0.1 --env=Hopper-v2 --model-path=model/mlp3_ho_5' &
wait 