import numpy as np

file_sac="../sac.txt"
file_random="../random.txt"
file_greedy="../greedy.txt"
output_file="../plot.txt"

with open(file_random) as f:
    random_tmp=f.read().split("num_vehicles")[1:]
with open(file_greedy) as f:
    greedy_tmp=f.read().split("num_vehicles")[1:]
with open(file_sac) as f:
    sac_tmp=f.read().split("num_vehicles")[1:]

random_result=[]
greedy_result=[]
for i in range(10):
    random_item = np.array([0.0]*17)
    greedy_item = np.array([0.0]*17)
    for j in range(100):
        random = random_tmp[j*10+i].split('\n')[2:]
        tmp=np.array([0.0]*17)
        for k in random:
            tmp+=np.array([float(r) for r in k.split(' ')])
        random_item+=tmp/len(random)
        greedy = greedy_tmp[j*10+i].split('\n')[3]
        greedy_item+=np.array([float(g) for g in greedy.split(' ')])
    random_result.append(random_item)
    greedy_result.append(greedy_item)

sac_result=[]
for i in range(10):
    sac=sac_tmp[i].split('\n')[10:]
    utility=-999
    sac_result_tmp=""
    for s in sac:
        if float(s.split(' ')[0])>utility:
            utility = float(s.split(' ')[0])
            sac_result_tmp=s
    sac_result.append(np.array([float(s) for s in sac_result_tmp.split(' ')]))

output=open(output_file,'w+')
output.write("utility to density:\n")
output.write("completion ratio comparison:\n")
output.write("completion ratio to priority:\n")
output.write("average lantency comparison:\n")
output.write("average latency to priority:\n")
output.write("utility to number of tasks:\n")
output.close()