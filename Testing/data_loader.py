import pandas as pd
import numpy as np



def get_data(start_year, end_year, target='ERA', future=False):
    infos=[]

    for x in range(start_year, end_year):
        infos.append('../Data/'+str(x)+'clean.csv')
    ls = []
    # Read in data
    for x in range(start_year, end_year):
        if future:
            ls.append('../Labels/Players_Stats_'+str(x + 1) + '.csv')
        else:
            ls.append('../Labels/Players_Stats_' + str(x) + '.csv')
    out=[]
    out_labels=[]
    for x in range(len(ls)):
        info = infos[x]
        l = ls[x]
        tdf = pd.read_csv(info)
        labels = pd.read_csv(l)

        names = np.array(labels[['Name']].values.tolist())
        names = names[:,0]
        t_stat = np.array(labels[[target]].values.tolist())
        t_stat = t_stat[:, 0]

        label_map = {}
        for idx,name in enumerate(names):
            label_map[name] = t_stat[idx]
        data = tdf.values
        i=0
        for x in range(len(tdf)):
            name = tdf.iloc[x]['Name']
            try:
                test = label_map[name]
                out_labels.append((name,test))
                out.append(tdf.iloc[x])
            except KeyError as e:
                i+=1
    df = pd.DataFrame(out)
    df_l = pd.DataFrame(out_labels,columns=['Name',target])
    return(df,df_l)
