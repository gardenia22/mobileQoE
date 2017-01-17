import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
import urlparse
import json
from gensim.models import Word2Vec
all_features = ['_ipispid_v', '_ipgid_v', 'B_v', 'C_v', 'F_v', 'K_v', 'R_v', '_X_v', 'C1_v', 'D1_v', 'FT_v', 'DD_v', 'dp_v', 'Dim_ContentType_G_v', 'Dim_VideoLength_G_v', 'd_FN_v', 'PM_v', 'KL_v', 'PT_v', 'D3_v', 'F1_v', 'C2_v', 'PK_v', 'F2_v', 'KS_v', 'PP_v', 'AC_v', 'RL_v', 'F3_v', 'PM2_v', 'D7_v', 'D6_v', 'LB_v', 'BD_v', 'PH2_v', 'Tab_v', 'video_w2v_0', 'video_w2v_1', 'video_w2v_2', 'video_w2v_3', 'video_w2v_4', 'video_w2v_5', 'video_w2v_6', 'video_w2v_7', 'video_w2v_8', 'video_w2v_9', 'video_w2v_10', 'video_w2v_11', 'video_w2v_12', 'video_w2v_13', 'video_w2v_14', 'video_w2v_15', 'video_w2v_16', 'video_w2v_17', 'video_w2v_18', 'video_w2v_19', 'user_w2v_0', 'user_w2v_1', 'user_w2v_2', 'user_w2v_3', 'user_w2v_4', 'user_w2v_5', 'user_w2v_6', 'user_w2v_7', 'user_w2v_8', 'user_w2v_9', 'user_w2v_10', 'user_w2v_11', 'user_w2v_12', 'user_w2v_13', 'user_w2v_14', 'user_w2v_15', 'user_w2v_16', 'user_w2v_17', 'user_w2v_18', 'user_w2v_19',  'FN', 'FM', 'FM_class','video_count','user_count',"N","M","L"]
def raw_to_csv(filename):
    columns=["_tms", "_ipv", "_ipi", "_ipa", "_ipc", "_ipp", "_ipt", "_ipispid", "_ipgid", "_udef", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "_X", "C1", "D1", "VVID", "FT", "FN", "DD", "DO", "ut", "dp", "d_F", "d_G", "Dim_LiveOndemand_G", "Dim_Copyright_G", "Dim_SubCategoryId_G", "Dim_ContentType_G", "Dim_VideoLength_G", "d_I", "d_Y1", "d_FN", "PM", "np", "KL", "PT", "D3", "D2", "F1", "C2", "D5", "FM", "K1", "L1", "L2", "L3", "M1", "PW", "PK", "PD", "W1", "W2", "F2", "G1", "KS", "KW", "LH", "LC", "PP", "PV", "AC", "RL", "R1", "R2", "R3", "R4", "D_pure", "_cids", "_cidTV", "D_RTV", "PH", "F3", "S1", "d_S1", "L4", "RU", "PM2", "D7", "D6", "Y7", "C3", "G3", "LB", "W4", "W5", "ZT", "BC", "BD", "BE", "rdch", "S1", "F", "d_S1", "TC1", "TC2", "TC3", "TC4", "TC0", "DR", "PH2", "Tab", "_pdt", "_pho"]
    #columns = ["_tms", "_ipv", "_ipi", "_ipa", "_ipc", "_ipp", "_ipt", "_ipispid", "_ipgid", "_udef", "B", "C", "D", "E", "F", "G", "H", "_C", "_D", "_E", "_F", "_G", "_H", "_I", "_J", "_K", "_L", "_M", "_N", "_O", "_P", "_Q", "_R", "_S", "_T", "_U", "_V", "_W", "_X", "_Y", "_Z", "_A1", "_B1", "_C1", "_D1", "_E1", "_F1", "_G1", "_J1", "_K1", "_L1", "_M1", "_N1", "_O1", "_P1", "_Q1", "_R1", "_S1", "_T1", "_K2", "_L2", "_O2", "_pdt", "_pho"]
    out = open(filename.split('.')[0]+'.csv','w')
    for i in range(len(columns)):
        out.write("%s" % columns[i])
        if i+1==len(columns):
            out.write("%s" % '\n')
        else:
            out.write("%s" % '\t')

    f = open(filename,'r')
    for line in f:
        tokens = line.strip().split('\t')
        m = {}
        for t in tokens:
            filed = t.split('=')
            m[filed[0]] = '='.join(filed[1:])

        for i in range(len(columns)):
            if columns[i] in m:
                out.write("%s" % m[columns[i]])
            if i+1==len(columns):
                out.write("%s" % '\n')
            else:
                out.write("%s" % '\t')
    out.close()
    f.close()

def vvid(url):
    par = urlparse.parse_qs(urlparse.urlparse(url).query)
    if 'vvid' in par:
        return par['vvid'][0]
    else:
        return ""



def load_csv(filename):
    data = pd.read_csv(filename,sep='\t',error_bad_lines=False, warn_bad_lines=True)
    return data

def bin_class(x):
    if np.isnan(x) or np.isinf(x):
        return 11
    y = int(x*10)
    if y<0:
        return 1
    if y>10:
        return 10
    return y+1

def map_func(d):
    label = dict()
    count = 1
    for v in d:
        label[v] = count
        count += 1
    f = lambda x:label[x]
    return f
feature_names = []
def encode_str(train, test, key):
    d = Counter(train[key].fillna(-42))
    d += Counter(test[key].fillna(-42))
    train[key+'_v'] = train[key].fillna(-42).apply(map_func(d))
    test[key+'_v'] = test[key].fillna(-42).apply(map_func(d))
    return train, test

def addWord2vec(data, v_model, u_model, data_u, data_v):

    vd = data['H'].apply(lambda x: unicode(str(x), 'utf-8'))
    v_matrix = vd.apply(lambda x: v_model[x] if x in v_model.vocab else v_model.seeded_vector(x))
    data['video_count'] = vd.apply(lambda x: len(data_v[x]) if x in data_v else 0)
    for i in range(0,20):
        data['video_w2v_%d' % i] = v_matrix.apply(lambda x:x[i])


    u_matrix = data['D'].apply(lambda x: u_model[x] if x in u_model.vocab else u_model.seeded_vector(x))
    data['user_count'] = data['D'].apply(lambda x: len(data_u[x]) if x in data_u else 0)
    for i in range(0,20):
        data['user_w2v_%d' % i] = u_matrix.apply(lambda x:x[i])
    return data


def map_viewTime(x):
    x = x/60
    if x<2:
        return 0
    if x<5:
        return 1
    if x<15:
        return 2
    if x<45:
        return 3
    #30 300 900 3600
    return 4

def map_class(x):
    x = x/60
    if x==0:
        return 0
    if x<5:
        return 1
    elif x<15:
        return 2
    elif x<30:
        return 3
    elif x<40:
        return 4
    else:
        return 5

def encode_target(data):
    #data['completion'] = (data['I']-data['L']-data['M'])/data['FM']
    #data['completion_class'] = data['completion'].apply(bin_class)
    data['target'] = data['I'].apply(map_viewTime)
    return data

def get_trainset(train, test):
    v_model = Word2Vec.load('/Volumes/SQ/word2vec/video_model_20')
    u_model = Word2Vec.load('/Volumes/SQ/word2vec/user_model_20')
    f_u = open('/Volumes/SQ/word2vec/user_160914_cn.json', "r")
    data_u = json.load(f_u)
    f_u.close()
    f_v = open('/Volumes/SQ/word2vec/video_160914_cn.json', "r")
    data_v = json.load(f_v)
    f_v.close()

    train = addWord2vec(train, v_model, u_model, data_u, data_v)
    test = addWord2vec(test, v_model, u_model, data_u, data_v)
    train = encode_target(train)
    test = encode_target(test)
    train['FM_class'] = train['FM'].apply(map_viewTime)
    test['FM_class'] = test['FM'].apply(map_viewTime)
    #train['FM_bin'] = train['FM'].apply(lambda x:int(x/60))
    #test['FM_bin'] = test['FM'].apply(lambda x:int(x/60))
    #features = ["_ipispid", "_ipgid","B", "C",  "F", "K","R","Y3","_X","C1","D1","FT","DD","ut","dp","Dim_ContentType_G","Dim_VideoLength_G","d_FN","PM","np","KL","D3","F1","C2","KS","F3","PM2","D7","D6","LB","BD"]
    features = ["_ipispid", "_ipgid","B", "C",  "F", "K","R","_X","C1","D1","FT","DD","dp","Dim_ContentType_G","Dim_VideoLength_G","d_FN","PM","KL","PT","D3","F1","C2","PK","F2","KS","PP","AC","RL","F3","PM2","D7","D6","LB","BD","PH2","Tab","FM_class"]
    non_change_features = ["FN","FM","video_count", "user_count","N","M","L"]
    train[non_change_features] = train[non_change_features].fillna(-1)
    test[non_change_features] = test[non_change_features].fillna(-1)
    w2v_features = ['video_w2v_%d' % i for i in range(0,20)]+ ['user_w2v_%d' % i for i in range(0,20)]
    add_features = w2v_features+non_change_features
    for f in features:
        train, test = encode_str(train, test, f)
    # for f in non_change_features:
    #     train, test = bin(train, test, f)
    enc = OneHotEncoder(handle_unknown='ignore', n_values='auto', dtype=np.int)
    new_features = map(lambda x:x+'_v',features)


    for f in features:
        d = Counter(train[f].fillna(-42))
        for v in d:
            feature_names.append(f+'&'+str(v))

    enc.fit(train[new_features])
    train_X = pd.DataFrame(enc.transform(train[new_features]).toarray())
    for k in add_features:
        train_X[k] = list(train[k])
    #train_y = train['completion_class']
    #train_y = train['d_I'].apply(lambda x: min(max(1,int((x-100)/4)),5))
    train_y = train['target']
    test_X = pd.DataFrame(enc.transform(test[new_features]).toarray())
    for k in add_features:
        test_X[k] = list(test[k])
    #test_y = test['completion_class']
    #test_y = test['d_I'].apply(lambda x: min(max(1,int((x-100)/4)),5))
    test_y = test['target']
    return train_X, train_y, test_X, test_y

def plot_cdf(data,label="line_1"):
    n = len(data)
    return plt.plot(np.sort(data),np.arange(n)/float(n),label=label)
#plt.show()

def bin_value(data,key,bin=5):
    b = data.loc[:,key].apply(lambda x:int(x/bin)*bin)
    g = data.loc[:,[key,'I']]
    g.loc[:,'bin'] = b
    grouped = g[['bin','I']].groupby('bin',as_index=False).mean()
    xvals = np.array(grouped['bin']);
    yvals = np.array(grouped['I']);
    plt.plot(xvals, yvals);




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 1000
        cm = cm.astype('int')
        cm = cm.astype('float') / 10
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)



    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




if __name__=="__main__":
    #raw_to_csv("sample_21.log")
    train = load_csv('/Volumes/SQ/word2vec/sampling_160915_22-23.csv')
    test = load_csv('/Volumes/SQ/word2vec/sampling_160915_23-24.csv')
    train_X, train_y, test_X, test_y = get_trainset(train, test)
    cm = confusion_matrix(test_y, pred_1)
    plt.figure()
    plot_confusion_matrix(cm, classes=range(1,12), normalize=True, title='Normalized confusion matrix')
    for i in range(0,24):
        filename = "sm_play_u9_160914%02d.log" % i
        print filename
        raw_to_csv(filename)