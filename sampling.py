__author__ = 'gardenia'
import random
# This script will extract about m=1,000,000 samples from one-day dataset by reservoir sampling
date = 160915
target_hour = 23
time_gap = 1
m = 100
files = ["%d/sm_play_u9_%d%02d.csv" % (date,date,i) for i in range(target_hour-time_gap,target_hour)]
samples = []
i = 0

def reservoir_sample(line, i):
    if len(samples)<m:
        samples.append(line)
    else:
        j = random.randint(0, i-1)
        if j<m:
            samples[j] = line

if __name__=="__main__":
    for f_name in files:
        inf = open(f_name,"r")
        header = inf.readline()
        for line in inf:
            reservoir_sample(line, i)
            i += 1
        inf.close()

    with open("sampling_%d_%d-%d.csv" % (date, target_hour-time_gap, target_hour), "w") as out:
        out.write(header)
        for line in samples:
            out.write(line)


