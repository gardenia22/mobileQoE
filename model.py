__author__ = 'gardenia'

from helper import *
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from numpy import mean
from collections import Counter,defaultdict
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn import linear_model as lm
from feature_label import labels
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

def to_categorical(labels,num_class=12):
    '''
    transfrom label to one-hot encoding target
    :param labels:
    :param num_class:
    :return:
    '''
    y_binary = np.zeros((len(labels),num_class))
    labels = list(labels)
    for i in range(len(labels)):
        if labels[i]<num_class:
            y_binary[i][int(labels[i])] = 1.0
    return y_binary

def get_old_trainset():
    """
    :return: X features Y target
    """
    data = load_csv("record_G_1000.csv")
    features = ['A','C','F','K','R','E']
    data = data[data['F']>=0]
    data = data[data['K']>=0]
    data = data[data['R']>=0]
    data = data[data['I']>0] # target y greater than 0
    for f in features:
        data[f] = data[f].apply(lambda x:str(x))
    train, test = train_test_split(data, test_size=0.2, random_state=22)
    dv = DictVectorizer(sparse=False)
    df = train[features].convert_objects(convert_numeric=True)
    train_X = dv.fit_transform(df.to_dict(orient='records'))
    test_X = dv.transform(test[features].convert_objects(convert_numeric=True).to_dict(orient='records'))
    # train_X = pd.DataFrame(train_X)
    # test_X = pd.DataFrame(test_X)
    enc = OneHotEncoder(handle_unknown='ignore', n_values='auto', dtype=np.int)
    enc.fit(train_X)

    train_X = pd.DataFrame(enc.transform(train_X).toarray())
    test_X = pd.DataFrame(enc.transform(test_X).toarray())
    train_y = train['I']
    test_y = test['I']
    return dv, train_X, train_y, test_X, test_y

def dl(train,target):

    # expected input data shape: (batch_size, data_dim)
    model = Sequential()
    data_dim = len(train.columns)
    print len(train.columns)

    model.add(Dense(500,input_dim=data_dim,activation='relu'))  # return a single vector of dimension 32
    model.add(Dense(500,input_dim=data_dim,activation='relu'))

    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))

    nb_classes = 5
    model.add(Dense(nb_classes,activation='relu'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    y_binary = to_categorical(target.as_matrix(),nb_classes)
    class_count = Counter(target)
    model.fit(train.as_matrix(),y_binary,batch_size=1000, nb_epoch=20,)

    return model

def gbr(train, target):
    model = GradientBoostingRegressor(loss='lad',
                                n_estimators=10, max_depth=3,
                                learning_rate=0.1, min_samples_leaf=100,
                                min_samples_split=500, verbose=2)
    model.fit(train, target)
    return model

def svm(train, target):
    model = SVC(kernel='rbf', max_iter=20, verbose=True)
    model.fit(train, target)
    return model

def lr(train, target):

    model = lm.LogisticRegression()
    model.fit(train, target)
    return model

def dt(train, target):
    model = DecisionTreeClassifier(max_depth=10,min_samples_leaf=100,min_samples_split=100)
    model.fit(train, target)
    return model

def gbdt(train, target):
    model = GradientBoostingClassifier(max_depth=5,n_estimators=10, min_samples_leaf=50,min_samples_split=300)
    model.fit(train, target)
    return model

def get_feature_importance(weight, feature_names):
    assert(len(feature_names)==len(weight[0]))
    sum_w = map(abs,weight[0])
    for i in range(1,11):
        sum_w = [a+b for a,b in zip(sum_w, map(abs,weight[i]))]
    n = len(feature_names)
    fw = sorted([(sum_w[i],feature_names[i]) for i in range(n)], reverse=True)
    fw_dict = defaultdict(int)
    count = defaultdict(int)
    for x in fw:
        fw_dict[x[1].split('&')[0]] += x[0]
        count[x[1].split('&')[0]] += 1
    bfw = sorted([(fw_dict[x]/count[x],x) for x in fw_dict], reverse=True)
    return fw,bfw

def mape(y_truth, y_pred):
    '''
    calculate mean absolute percentage error
    :param y_truth:
    :param y_pred:
    :return: mape
    '''
    count = 0
    sum = 0
    for p,t in zip(list(y_pred), list(y_truth)):
        if np.isnan(p):
            p=1.0
        if t>0:
            sum += abs(p-t)/t
            count += 1
    return sum/count

def abs_error(y_truth, y_pred):
    ret = []
    for p,t in zip(list(y_pred), list(y_truth)):
        ret.append(abs(p-t)/t)
    return ret

def acc(y_truth, y_pred):
    '''
    calculate accuracy
    :param y_truth:
    :param y_pred:
    :return: accuracy
    '''
    count = 0
    tot = 0
    for p,t in zip(list(y_pred), list(y_truth)):
        if round(p)==t:
            count+=1
        tot+=1
    return count*1.0/tot

def predict_dl(pred):
    '''
    :param pred: predict result from deep learning model
    :return: predict class from deep learning model(highest predict socre)
    '''
    p = []
    for x in pred:
        ans = 1
        pp = 0
        for i in range(len(x)):
            if x[i]>pp:
                pp = x[i]
                ans = i
        p.append(ans)
    return p

def run_model(train_X, train_y, test_X, test_y):
    model_1 = dt(train_X, train_y) # decision tree
    pred_1 = model_1.predict(test_X)
    print "decision tree", acc(pred_1, test_y)

    model_2 = lr(train_X, train_y) # logistic regression
    pred_2 = model_2.predict(test_X)
    print "logistic regression", acc(pred_2, test_y)

    model_3 = svm(train_X, train_y) # support vector machine
    pred_3 = model_3.predict(test_X)
    print "support vector machine", acc(pred_3, test_y)
    #cm = confusion_matrix(test_y, pred_1)
    #plot_confusion_matrix(cm, classes=range(1,12), normalize=True, title='Normalized confusion matrix')

def split_qoe(data):
    '''
    split dataset by QoE buffer time and buffer event
    :param data:
    :return:
    '''
    good_qoe = data[data['M']==0][data['N']==0]
    poor_qoe = data[data['N']!=0]
    return good_qoe, poor_qoe

if __name__=="__main__":
    train_path = '/Volumes/SQ/word2vec/sampling_160915_22-23.csv'
    test_path = '/Volumes/SQ/word2vec/sampling_160915_23-24.csv'
    train = load_csv(train_path)
    test = load_csv(test_path)
    train_good, train_poor = split_qoe(train)
    test_good, test_poor = split_qoe(test)

    train_X, train_y, test_X, test_y = get_trainset(train_good, test_good)
    run_model(train_X, train_y, test_X, test_y)
    train_X, train_y, test_X, test_y = get_trainset(train_poor, test_poor)
    run_model(train_X, train_y, test_X, test_y)