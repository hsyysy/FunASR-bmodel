import numpy as np

with open("record_sent.txt","r") as f:
    sent = f.readlines()
with open("record_recv.txt","r") as f:
    recv = f.readlines()

# sent[id] = time from first sent
sent = np.array([ float(item[5:14]) for item in sent ])

# recv[i][j] for each j
# j=0 -> 0 for online and 1 for offline
# j=1 -> start id
# j=2 -> end id
# j=3 -> time from first sent
recv = [[int(item[0]), int(item[2:6]), int(item[7:11]), float(item[12:21])] for item in recv]

online_time = []
offline_time = []
alltime = []

for item in recv:
    proc_time = item[3] - sent[item[2]]
    if item[0] == 1:
        # offline
        offline_time.append(proc_time)
    else:
        # online
        online_time.append(proc_time)
    alltime.append(proc_time)

print("online  time avg: "+str(np.mean(online_time))+", std = "+str(np.std(online_time)))
print("offline time avg: "+str(np.mean(offline_time))+", std = "+str(np.std(offline_time)))
print("allline time avg: "+str(np.mean(alltime))+", std = "+str(np.std(alltime)))
