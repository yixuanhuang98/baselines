import os
env_id = "Humanoid-v2"
seed_list = [4,5]
std_list=[0.1,0.5,1.0,1.5,2.0]

for seed in seed_list:
	f = open('noise.txt','a+')
	f.write("\n----------%d----------\n" % seed)
	f.close()
	os.system("python -m baselines.scn.run_humanoid --seed="+str(seed)+" --reward_scale=1 --env="+env_id+ " --model-path=model/scn_hm"+str(seed)+ " --num_timesteps=10 --play")

	for std in std_list:
		os.system("python -m baselines.scn.run_humanoid --seed="+str(seed)+" --reward_scale=1 --env="+env_id+ " --model-path=model/scn_hm"+str(seed)+ " --num_timesteps=10 --play --acstd="+str(std))
	for std in std_list:
		os.system("python -m baselines.scn.run_humanoid --seed="+str(seed)+" --reward_scale=1 --env="+env_id+ " --model-path=model/scn_hm"+str(seed)+ " --num_timesteps=10 --play --obstd="+str(std))
