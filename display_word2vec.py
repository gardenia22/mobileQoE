__author__ = 'gardenia'
from gensim.models import Word2Vec
v_model = Word2Vec.load('/Volumes/SQ/word2vec/video_model')

keys = [u"红楼梦",u"天龙八部",u"吧啦啦小魔仙",u"火影忍者",u"寻龙诀"]
dup = set()
label = []
X = []
for key in keys:
    label.append(key)
    dup.add(key)
    X.append(v_model[key])
    for name, value in v_model.most_similar(key)[:5]:
        if name not in dup:
            label.append(name)
            X.append(v_model[name])
            dup.add(name)
        # for name2, value2 in v_model.most_similar(name):
        #     if name2 not in dup:
        #         label.append(name2)
        #         X.append(v_model[name2])
        #         dup.add(name2)