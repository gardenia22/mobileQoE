import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kendalltau
def plot_cdf(data,label):
	n = len(data)
	return plt.plot(np.sort(data),np.arange(n)/float(n),label=label)

def bin_value(data,key,bin=5):
	b = data.loc[:,key].apply(lambda x:int(x/bin)*bin)
	g = data.loc[:,[key,'I']]
	g.loc[:,'bin'] = b
	grouped = g[['bin','I']].groupby('bin',as_index=False).mean()
	xvals = np.array(grouped['bin'])
	yvals = np.array(grouped['I'])
	plt.plot(xvals, yvals)
	return xvals, yvals

def save_data_metrics(data, name='data'):
    data['be'] = data['N']/data['I']
    data['br'] = data['M']/(data['M']+data['I'])
    data['viewratio'] = (data['I']-data['L']-data['M'])/data['FM']
    pd.DataFrame(np.sort(data['I'])).to_csv('%s/view_time_cdf.csv' % name, index=False, header=None)
    pd.DataFrame(np.sort(data['L'])).to_csv('%s/join_time_cdf.csv' % name, index=False, header=None)
    pd.DataFrame(np.sort(data['viewratio'])).to_csv('%s/view_ratio_cdf.csv' % name, index=False, header=None)
    pd.DataFrame(np.sort(data[data['be']<=1][data['be']>=0]['be'])).to_csv('%s/buffer_ratio_cdf.csv' % name, index=False, header=None)
    pd.DataFrame(np.sort(data[data['br']<=1][data['br']>=0]['br'])).to_csv('%s/buffer_event_cdf.csv' % name, index=False, header=None)
    pd.DataFrame(np.sort(data[data['FN']>0])).to_csv('%s/bitratio_cdf.csv' % name, index=False, header=None)
    x, y = bin_value(data[data['be']<=0.05][data['be']>=0],'be',bin=0.001)
    pd.DataFrame({'x':x, 'y':y}).to_csv('%s/buffer_event_view_time.csv' % name, index=False, header=None, sep=' ')
    x, y = bin_value(data[data['br']<=1][data['br']>=0],'br',bin=0.01)
    pd.DataFrame({'x':x, 'y':y}).to_csv('%s/buffer_ratio_view_time.csv' % name, index=False, header=None, sep=' ')
    x, y = bin_value(data,'L',bin=1)
    pd.DataFrame({'x':x, 'y':y}).to_csv('%s/join_time_view_time.csv' % name, index=False, header=None, sep=' ')
    x, y = bin_value(data[data['FN']>0], 'FN', bin=100)
    pd.DataFrame({'x':x, 'y':y}).to_csv('%s/bitratio_view_time.csv' % name, index=False, header=None, sep=' ')


def save_correlation(data, name='data'):
    cor_jt = []
    cor_br = []
    cor_be = []
    cor_fn = []
    count = 0
    for group, log in data.groupby('G'):
        if len(log)>10:
            tau, p_value = kendalltau(log['I'][log['M']>0], log['L'][log['M']>0])
            cor_jt.append(tau)
            tau, p_value = kendalltau(log['I'][log['M']>0][log['I']>0], log['br'][log['M']>0][log['I']>0])
            cor_br.append(tau)
            tau, p_value = kendalltau(log['I'][log['M']>0][log['I']>0], log['be'][log['I']>0][log['M']>0])
            cor_be.append(tau)
            tau, p_value = kendalltau(log['I'][log['M']>0][log['I']>0], log['FN'][log['I']>0][log['M']>0])
            cor_fn.append(tau)
            count+=1
        #print count
    pd.DataFrame(np.sort(cor_jt)).to_csv('%s/cor_join_time.csv' % name, index=False, header=None)
    pd.DataFrame(np.sort(cor_br)).to_csv('%s/cor_buffer_ratio.csv' % name, index=False, header=None)
    pd.DataFrame(np.sort(cor_be)).to_csv('%s/cor_buffer_event.csv' % name, index=False, header=None)
    pd.DataFrame(np.sort(cor_fn)).to_csv('%s/cor_bitratio.csv' % name, index=False, header=None)

def save_information_gain(data, name='data'):
    cor_jt = []
    cor_br = []
    cor_be = []
    cor_fn = []
    count = 0
    data['vt'] = data['I'].apply(lambda x:int(x/60))
    for group, log in data.groupby('G'):
        if len(log)>10:
            gain = information_gain(log['vt'], log['L'])
            cor_jt.append(gain)
            bin = 0.01
            gain = information_gain(log[log['br']<=1][log['br']>=0]['vt'], log[log['br']<=1][log['br']>=0]['br'].apply(lambda x:int(x/bin)*bin))
            cor_br.append(gain)
            bin = 0.001
            gain = information_gain(log[log['be']<=0.05][log['be']>=0]['vt'], log[log['be']<=0.05][log['be']>=0]['be'].apply(lambda x:int(x/bin)*bin))
            cor_be.append(gain)
            bin = 100
            gain = information_gain(log[log['FN']>=0][log['FN']<=2000]['vt'], log[log['FN']>=0][log['FN']<=2000]['FN'].apply(lambda x:int(x/bin)*bin))
            cor_fn.append(gain)

        count+=1
        #print count
    pd.DataFrame(np.sort(cor_jt)).to_csv('%s/ig_join_time.csv' % name, index=False, header=None)
    pd.DataFrame(np.sort(cor_br)).to_csv('%s/ig_buffer_ratio.csv' % name, index=False, header=None)
    pd.DataFrame(np.sort(cor_be)).to_csv('%s/ig_buffer_event.csv' % name, index=False, header=None)
    pd.DataFrame(np.sort(cor_fn)).to_csv('%s/ig_bitratio.csv' % name, index=False, header=None)


def information_gain(y, x):
    "Computes the information gain of x on y"

    def entropy(x):
        p_x = x.value_counts(normalize=True)
        return -p_x.map(lambda x: x * np.log2(x)).sum()

    prior_entropy = entropy(y)
    df = pd.concat([x, y], axis=1)
    df.columns = ["x","y"]

    p_x = x.value_counts(normalize=True)
    cond_entropy = pd.Series(dict([(x_i, entropy(df[df["x"] == x_i]["y"])) for x_i in p_x.index]))
    post_entropy = p_x.dot(cond_entropy)
    information_gain = prior_entropy - post_entropy
    return information_gain

def rank_information_gain(X, y):
    information_gains = pd.Series(dict([(x, information_gain(x,y)) for x in X]))
    information_gains.column = ["feature", "information_gain"]
    return information_gains

def bin_abandonment(data, key, bin):
    b = data.loc[:,key].apply(lambda x:int(x/bin)*bin)
    g = data.loc[:,[key,'W']]
    g.loc[:,'bin'] = b
    bin_value = []
    abd_value = []
    for name,log in g[['bin','W']].groupby('bin'):
        bin_value.append(name)
        abd_value.append(len(log[log['W']==1])*1.0/len(log))
    return bin_value, abd_value

def save_abandonment_rate(data,name='data'):
    x, y = bin_abandonment(data[data['be']<=0.05][data['be']>=0],'be',bin=0.001)
    pd.DataFrame({'x':x, 'y':y}).to_csv('%s/buffer_event_abandonment.csv' % name, index=False, header=None, sep=' ')
    x, y = bin_abandonment(data[data['br']<=1][data['br']>=0],'br',bin=0.01)
    pd.DataFrame({'x':x, 'y':y}).to_csv('%s/buffer_ratio_abandonment.csv' % name, index=False, header=None, sep=' ')
    x, y = bin_abandonment(data,'L',bin=1)
    pd.DataFrame({'x':x, 'y':y}).to_csv('%s/join_time_abandonment.csv' % name, index=False, header=None, sep=' ')

#l1, = plot_cdf(live['I'],'live')
#l2, = plot_cdf(vod['I'],'vod')
#plt.legend(handles=[l1,l2],loc=4)