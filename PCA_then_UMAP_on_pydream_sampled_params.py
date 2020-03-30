import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
import umap

sampled_params = {}

sampled_params[0] = np.load('dreamzs_5chain_S_Sage_NM_mult_sampled_params_chain_0_40000.npy')[()]
sampled_params[1] = np.load('dreamzs_5chain_S_Sage_NM_mult_sampled_params_chain_1_40000.npy')[()]
sampled_params[2] = np.load('dreamzs_5chain_S_Sage_NM_mult_sampled_params_chain_2_40000.npy')[()]
sampled_params[3] = np.load('dreamzs_5chain_S_Sage_NM_mult_sampled_params_chain_3_40000.npy')[()]
sampled_params[4] = np.load('dreamzs_5chain_S_Sage_NM_mult_sampled_params_chain_4_40000.npy')[()]

df0 = pd.DataFrame(sampled_params[0])
df1 = pd.DataFrame(sampled_params[1])
df2 = pd.DataFrame(sampled_params[2])
df3 = pd.DataFrame(sampled_params[3])
df4 = pd.DataFrame(sampled_params[4])

df_tot = pd.concat([df0,df1,df2,df3,df4],ignore_index=True)

dups = np.unique(df_tot,axis=0,return_counts=True)[1]
df = copy.deepcopy(df_tot)
df = df_tot.drop_duplicates()
df['duplicates'] = dups
df_x = df.sort_values('duplicates',ascending=False) #[:5000]

Sage_params = pd.read_pickle('Sage_NEv2toNonNE_80Knm_top5K_paramsets.pickle')
Oliver_params = pd.read_pickle('Oliver_NEv2toNonNE_80Knm_top5K_paramsets.pickle')

pca = sklearnPCA(n_components=29)

Sage_params.drop('duplicates',axis=1,inplace=True)
Oliver_params.drop('duplicates',axis=1,inplace=True)

df_S = Sage_params
df_O = Oliver_params

df_S['dset'] = 'Sage'
df_O['dset'] = 'Oliver'
df_both = pd.concat([df_S,df_O])
num_std = StandardScaler().fit_transform(df_both.iloc[:,:-1])
df_std = pd.DataFrame(num_std,columns=df_both.columns[:-1],index=df_both.index)
df_std['dset'] = ''
df_std['dset'].iloc[:5000] = 'Sage'
df_std['dset'].iloc[5000:] = 'Oliver'


pca.fit(df_std[df_std.dset=='Sage'].iloc[:,:-1])#with standardized

Sage_inSagespace = pca.transform(df_std[df_std.dset=='Sage'].iloc[:,:-1])
Oliver_inSagespace = pca.transform(df_std[df_std.dset=='Oliver'].iloc[:,:-1])

x_scatter_S = []
y_scatter_S = []

for i in range(len(Sage_inSagespace)):
     x_scatter_S.append(Sage_inSagespace[i][0])
     y_scatter_S.append(Sage_inSagespace[i][1])

x_scatter_O = []
y_scatter_O = []

for i in range(len(Oliver_inSagespace)):
     x_scatter_O.append(Oliver_inSagespace[i][0])
     y_scatter_O.append(Oliver_inSagespace[i][1])

plt.scatter(x_scatter_S,y_scatter_S,color='blue')
plt.scatter(x_scatter_O,y_scatter_O,color='green')
plt.show()

Sage_df = pd.DataFrame(Sage_inSagespace)
Sage_df['dset'] = 'Sage'
Sage_df['dset'] = 0
Oliver_df = pd.DataFrame(Oliver_inSagespace)
Oliver_df['dset'] = 'Oliver'
Oliver_df['dset'] = 1


umap_df = pd.concat([Sage_df,Oliver_df],ignore_index=True)
reducer = umap.UMAP()
embedding = reducer.fit_transform(umap_df.iloc[:,:-1])


plt.scatter(embedding[:,0],embedding[:,1],c=[sns.color_palette()[x] for x in umap_df['dset']],s=0.5)
plt.gca().set_aspect('equal','datalim')
plt.title('UMAP projection of Sage (blue) and Oliver (orange)\nfitted parameters plotted on Sage parameter eigenspace',fontsize=14)
plt.show()

