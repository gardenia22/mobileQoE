__author__ = 'gardenia'
from collections import defaultdict
city = defaultdict(int)
country = defaultdict(int)
isp = defaultdict(int)
province = defaultdict(int)

def save_dict(d, filename):
    with open(filename,"w") as f:
        for k in d:
            f.write("%s %d\n" % (k,d[k]))

for i in range(0,24):
        filename = "sm_play_u9_160914%02d.csv" % i
        with open(filename,'r') as f:
            f.readline()

            for line in f:
                token = line.strip().split('\t')
                if len(token)==129:
                    isp[token[2]] += 1
                    country[token[4]] += 1
                    province[token[5]] += 1
                    city[token[6]] += 1



save_dict(city, "city_coverage.csv")
save_dict(country, "country_coverage.csv")
save_dict(isp, "isp_coverage.csv")
save_dict(province, "province_coverage.csv")