#xterm -e 'python -m baselines.hyb.run_humanoid --seed=1 --reward_scale=1.0 --env=Walker2d-v2 --model-path=model/hyb_wk_1' &
xterm -e 'python -m baselines.hyb.run_humanoid --seed=2 --reward_scale=1.0 --env=Walker2d-v2 --model-path=model/hyb_wk_2' &
xterm -e 'python -m baselines.hyb.run_humanoid --seed=3 --reward_scale=1.0 --env=Walker2d-v2 --model-path=model/hyb_wk_3' &
xterm -e 'python -m baselines.hyb.run_humanoid --seed=4 --reward_scale=1.0 --env=Walker2d-v2 --model-path=model/hyb_wk_4' &
xterm -e 'python -m baselines.hyb.run_humanoid --seed=5 --reward_scale=1.0 --env=Walker2d-v2 --model-path=model/hyb_wk_5' &
