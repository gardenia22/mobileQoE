__author__ = 'gardenia'
import gensim
import json

f = open('/Volumes/SQ/word2vec/user_160914_cn.json', "r")
data = json.load(f)
f.close()
f = open('user_count.csv', "w")
for k,v in data.iteritems():
    f.write("%s,%d\n" % (str(k), len(v)))
f.close()
# data = [v for k,v in data.iteritems()]
# model = gensim.models.Word2Vec(data,size=20, window=5, min_count=5, workers=4)
# model.save('/Volumes/SQ/word2vec/video_model_20')
