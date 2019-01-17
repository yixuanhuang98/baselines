import os
os.system('python -m baselines.fcn.run_humanoid --seed=1 --reward_scale=1 --env=Humanoid-v2 --model-path=model/fcn_hm')
os.system('python -m baselines.fcn.run_humanoid --seed=1 --reward_scale=1 --env=HalfCheetah-v2 --model-path=model/fcn_hc')
os.system('python -m baselines.fcn.run_humanoid --seed=1 --reward_scale=1 --env=Hopper-v2 --model-path=model/fcn_hp')
os.system('python -m baselines.fcn.run_humanoid --seed=1 --reward_scale=1 --env=InvertedDoublePendulum-v2 --model-path=model/fcn_idp')

os.system('python -m baselines.scn.run_humanoid --seed=1 --reward_scale=1 --env=Humanoid-v2 --model-path=model/scn_hm')
os.system('python -m baselines.scn.run_humanoid --seed=1 --reward_scale=1 --env=HalfCheetah-v2 --model-path=model/scn_hc')
os.system('python -m baselines.scn.run_humanoid --seed=1 --reward_scale=1 --env=Hopper-v2 --model-path=model/scn_hp')
os.system('python -m baselines.scn.run_humanoid --seed=1 --reward_scale=1 --env=InvertedDoublePendulum-v2 --model-path=model/scn_idp')
