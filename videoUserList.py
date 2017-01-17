__author__ = 'gardenia'
import helper
import json
from collections import defaultdict
if __name__=="__main__":
    video = defaultdict(list)
    user = defaultdict(list)
    filename = ["sm_play_u9_160914%02d.csv" % i for i in range(0,24)]
    for f in filename:
        print f
        inf = open(f, "r")
        inf.readline()
        for line in inf:
            row = line.strip().split('\t')
            try:
                u,v = row[12],row[16]
                user[u].append(v)
                video[v].append(u)
            except:
                pass
    print len(user), len(video)
f = open("video_160914_cn.json", "w")

json.dump(video, f)
f.close()
f = open("user_160914_cn.json", "w")

json.dump(user, f)
f.close()