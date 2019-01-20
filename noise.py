import os
env_id = "Hopper-v2"
seed_list = [11,12,13,14,15]
std_list=[0.1,0.5,1.0,1.5,2.0]

for seed in seed_list:
	f = open('noise.txt','a+')
	f.write("\n----------%d----------\n" % seed)
	f.close()
	os.system("python -m baselines.fcn.run_humanoid --seed="+str(seed)+" --reward_scale=1 --env="+env_id+ " --model-path=model/fcn_ho_"+str(seed)+ " --num_timesteps=10 --play")

	for std in std_list:
		os.system("python -m baselines.fcn.run_humanoid --seed="+str(seed)+" --reward_scale=1 --env="+env_id+ " --model-path=model/fcn_ho_"+str(seed)+ " --num_timesteps=10 --play --acstd="+str(std))
	for std in std_list:
		os.system("python -m baselines.fcn.run_humanoid --seed="+str(seed)+" --reward_scale=1 --env="+env_id+ " --model-path=model/fcn_ho_"+str(seed)+ " --num_timesteps=10 --play --obstd="+str(std))
