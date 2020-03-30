import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from pysb.simulator import ScipyOdeSimulator
import numpy as np
from scipy.stats import norm,uniform,multinomial
from four_state_sclc import model
import seaborn as sns
from matplotlib import pyplot as plt
import copy
from itertools import chain
import sys
import pandas as pd

nchains = 5
total_iterations = 40000

# SHould be the same as what was used in the pydream run (including where the model is impored from)
tspan = np.linspace(0,60,1001)
solver = ScipyOdeSimulator(model,integrator='lsoda',compiler='cython')
param_values = np.array([p.value for p in model.parameters])
parameters_idxs = list(np.arange(1,30))
rates_mask = np.concatenate((np.zeros(1,dtype=bool),np.ones(len(param_values[1:]),dtype=bool)))

def normalize(trajectory, trajectories):
    """even though this is not really needed, if the data is already between 1 and 0!"""
    """Rescale a matrix of model trajectories to 0-1"""
    ymin = min([y for x in trajectories for y in x])
    ymax = max([y for x in trajectories for y in x])
    return (trajectory - ymin) / (ymax - ymin)

sampled_params = {}

sampled_params[0] = np.load('dreamzs_5chain_S_Sage_NM_mult_sampled_params_chain_0_40000.npy')[()]
sampled_params[1] = np.load('dreamzs_5chain_S_Sage_NM_mult_sampled_params_chain_1_40000.npy')[()]
sampled_params[2] = np.load('dreamzs_5chain_S_Sage_NM_mult_sampled_params_chain_2_40000.npy')[()]
sampled_params[3] = np.load('dreamzs_5chain_S_Sage_NM_mult_sampled_params_chain_3_40000.npy')[()]
sampled_params[4] = np.load('dreamzs_5chain_S_Sage_NM_mult_sampled_params_chain_4_40000.npy')[()]

logps = {}
logps[0] = np.load('dreamzs_5chain_S_Sage_NM_mult_logps_chain_0_40000.npy')[()]
logps[1] = np.load('dreamzs_5chain_S_Sage_NM_mult_logps_chain_1_40000.npy')[()]
logps[2] = np.load('dreamzs_5chain_S_Sage_NM_mult_logps_chain_2_40000.npy')[()]
logps[3] = np.load('dreamzs_5chain_S_Sage_NM_mult_logps_chain_3_40000.npy')[()]
logps[4] = np.load('dreamzs_5chain_S_Sage_NM_mult_logps_chain_4_40000.npy')[()]

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

# Now we have the dataframe with the parameter sets

all_vals = {}
ind = 0
tot_vals = []
tspan = np.linspace(0,60,1001)

for i in range(0,len(df_x)):
    if i % 1000 == 0:
        print (i)
    Y = np.copy(df_x.iloc[i][:-1])
    param_values[rates_mask] = 10 ** Y
    sim = solver.run(param_values=param_values,tspan=tspan).all
    sim_data_norm = {}
    tot_vals.append(np.zeros(1001))
    for obs in ['NE_obs','NEv1_obs','NEv2_obs','NonNE_obs']:
        toAdd = normalize(sim[obs], [list(sim[z]) for z in ['NE_obs','NEv1_obs','NEv2_obs','NonNE_obs']])
        tot_vals[ind] = np.sum([tot_vals[ind],toAdd],axis=0)
        if not obs in all_vals:
            all_vals[obs] = [toAdd]
        else:
            all_vals[obs].append(toAdd)
    ind += 1

np.save('all_vals_40K_sims_Sage_NM_NEv2_to_NonNE_mult.npy',all_vals,allow_pickle=True)
np.save('tot_vals_40K_sims_Sage_NM_NEv2_to_NonNE_mult.npy',tot_vals,allow_pickle=True)

## Plot log-p-values
iters = [i for i in range(20000,40000)] #number of iterations
fig = plt.figure(figsize=(18,10))
plt.rcParams.update({'font.size': 16})
columns = 3
rows = 2
for chain in range(0,5):
    fig.add_subplot(rows,columns,chain+1)
    plt.plot(iters,logps[chain][0:,0], color='b',label=('Chain '+str(chain)))
    #plt.ylim(-50,0)
    #plt.xlim(0,1000)
    plt.legend()

fig.text(0.5,0.001,'Iteration',ha='center', fontsize=20)
fig.text(0.001,0.5,'Log likelihood',va='center',rotation='vertical',fontsize=20)

plt.tight_layout(pad=2)
plt.show()

fig.savefig('SCLC_logpvalues.png')
plt.close(fig)

## Plot the proportions individually - this really only works wih 5K paramsets or less (and ideally less)
fig = plt.figure(figsize=(18,10))
columns = 2
rows = 2

NE_vals = all_vals['NE_obs']
NEv1_vals = all_vals['NEv1_obs']
NEv2_vals = all_vals['NEv2_obs']
NonNE_vals = all_vals['NonNE_obs']

fig.add_subplot(rows,columns,1)
plt.title('NE proportion over time,\n1000 sets of parameters with best p-vals')
plt.ylim(0,1)
#plt.xlim(0,20)
for i in np.true_divide(NE_vals,tot_vals):
    plt.plot(tspan,i,color='b',alpha=0.05)

fig.add_subplot(rows,columns,2)
plt.title('NEv1 proportion over time,\n1000 sets of parameters with best p-vals')
plt.ylim(0,1)
#plt.xlim(0,20)
for i in np.true_divide(NEv1_vals,tot_vals):
    plt.plot(tspan,i,color='orange',alpha=0.05)

fig.add_subplot(rows,columns,3)
plt.title('NEv2 proportion over time,\n1000 sets of parameters with best p-vals')
plt.ylim(0,1)
#plt.xlim(0,20)
for i in np.true_divide(NEv2_vals,tot_vals):
    plt.plot(tspan,i,color='g',alpha=0.05)

fig.add_subplot(rows,columns,4)
plt.title('NonNE proportion over time,\n1000 sets of parameters with best p-vals')
plt.ylim(0,1)
#plt.xlim(0,20)
for i in np.true_divide(NonNE_vals,tot_vals):
    plt.plot(tspan,i,color='r',alpha=0.05)

plt.tight_layout()
#plt.show()
fig.savefig('SCLC_proportions_over_time_per_subtype.png')
plt.close(fig)

## Plot trajectories overlaid
plot_vals = {}
plot_vals['NE'] = []
plot_vals['NEv1'] = []
plot_vals['NEv2'] = []
plot_vals['NonNE'] = []

plot_vals['NE'] = np.true_divide(NE_vals,tot_vals)
plot_sum_prev = copy.deepcopy(np.true_divide(NE_vals,tot_vals))

plot_vals['NEv1'] = np.sum([plot_sum_prev,np.true_divide(NEv1_vals,tot_vals)],axis=0)
plot_sum_prev = np.sum([plot_sum_prev,np.true_divide(NEv1_vals,tot_vals)],axis=0)

plot_vals['NEv2'] = np.sum([plot_sum_prev,np.true_divide(NEv2_vals,tot_vals)],axis=0)
plot_sum_prev = np.sum([plot_sum_prev,np.true_divide(NEv2_vals,tot_vals)],axis=0)

plot_vals['NonNE'] = np.sum([plot_sum_prev,np.true_divide(NonNE_vals,tot_vals)],axis=0)

# Plot trajectories by counts as well as percentages - every 25th paramset out of all of them
for i in np.arange(0,5000,25):
#for i in [0]:
	fig = plt.figure(figsize=(8,4))
	fig.add_subplot(1,2,1)
	plt.plot(tspan,NE_vals[i],label='NE')
	plt.plot(tspan,NEv1_vals[i],label='NEv1')
	plt.plot(tspan,NEv2_vals[i],label='NEv2')
	plt.plot(tspan,NonNE_vals[i],label='NonNE')
	plt.ylabel('Normalized cell count')
	plt.title('Normalized SCLC subtype cell count over time')
	plt.legend()
	plt.tight_layout()
	fig.add_subplot(1,2,2)
	plt.plot(tspan,plot_vals['NE'][i],color='darkblue')
	plt.fill_between(tspan, plot_vals['NE'][i], label='NE')
	plt.plot(tspan,plot_vals['NEv1'][i],color='darkorange')
	plt.fill_between(tspan, plot_vals['NE'][i], plot_vals['NEv1'][i], label='NEv1')
	plt.plot(tspan,plot_vals['NEv2'][i],color='darkgreen')
	plt.fill_between(tspan, plot_vals['NEv1'][i], plot_vals['NEv2'][i], label='NEv2')
	plt.plot(tspan,plot_vals['NonNE'][i],color='darkred')
	plt.fill_between(tspan, plot_vals['NEv2'][i], plot_vals['NonNE'][i], label='NonNE')
	plt.ylabel('Proportion of population')
	plt.title('Proportion of the SCLC population over time')
	plt.legend()
	plt.tight_layout()
	fig.text(0.5,0.005,'Time (days) ; parameter set '+str(i),ha='center', fontsize=12)
	plt.show()

# Plot all trajectories at once (this doesn't usually work well)
for i in plot_vals['NE']:
	plt.plot(tspan,i,color='darkblue')#,alpha=0.01)

for i in plot_vals['NEv1']:
	plt.plot(tspan,i,color='darkorange')#,alpha=0.01)

for i in plot_vals['NEv2']:
	plt.plot(tspan,i,color='darkgreen')#,alpha=0.01)

for i in plot_vals['NonNE']:
	plt.plot(tspan,i,color='darkred')#,alpha=0.01)

plt.ylim(0,1)
plt.show()

## Plot the average proportion over time
avg_vals = {}

avg_vals['NE'] = np.mean(np.true_divide(NE_vals,tot_vals),axis=0)

sum_prev = np.zeros(1001)
sum_prev = np.sum([sum_prev,np.mean(np.true_divide(NE_vals,tot_vals),axis=0)],axis=0)

avg_vals['NEv1'] = np.sum([sum_prev,np.mean(np.true_divide(NEv1_vals,tot_vals),axis=0)],axis=0)
sum_prev = np.sum([sum_prev,np.mean(np.true_divide(NEv1_vals,tot_vals),axis=0)],axis=0)

avg_vals['NEv2'] = np.sum([sum_prev,np.mean(np.true_divide(NEv2_vals,tot_vals),axis=0)],axis=0)
sum_prev = np.sum([sum_prev,np.mean(np.true_divide(NEv2_vals,tot_vals),axis=0)],axis=0)

avg_vals['NonNE'] = np.sum([sum_prev,np.mean(np.true_divide(NonNE_vals,tot_vals),axis=0)],axis=0)

# Plot all the different ones with averages on top
for i in plot_vals['NE']:
	plt.plot(tspan,i,color='darkblue')#,alpha=0.01)

for i in plot_vals['NEv1']:
	plt.plot(tspan,i,color='darkorange')#,alpha=0.01)

for i in plot_vals['NEv2']:
	plt.plot(tspan,i,color='darkgreen')#,alpha=0.01)

for i in plot_vals['NonNE']:
	plt.plot(tspan,i,color='darkred')#,alpha=0.01)

plt.plot(tspan,avg_vals['NE'],linewidth=5,color='navy',linestyle='--')
plt.fill_between(tspan, avg_vals['NE'], label='NE')
plt.plot(tspan,avg_vals['NEv1'],linewidth=5,color='orangered',linestyle='--')
plt.fill_between(tspan, avg_vals['NE'], avg_vals['NEv1'], label='NEv1')
plt.plot(tspan,avg_vals['NEv2'],linewidth=5,color='olive',linestyle='--')
plt.fill_between(tspan, avg_vals['NEv1'], avg_vals['NEv2'], label='NEv2')
plt.plot(tspan,avg_vals['NonNE'],linewidth=5,color='darkred',linestyle='--')
plt.fill_between(tspan, avg_vals['NEv2'], avg_vals['NonNE'], label='NonNE')

#plt.xlim(0,20)
#plt.ylim(0,1)
plt.xlabel('time (days)')
plt.ylabel('proportion of the population')
plt.title('Proportion of the SCLC population over time, average of trajectories\nfrom top 5K parameter sets')
plt.legend()
plt.tight_layout()
plt.show()

# Plot averages by themselves
plt.plot(tspan,avg_vals['NE'],linewidth=5,color='navy',linestyle='--')
plt.fill_between(tspan, avg_vals['NE'], label='NE')
plt.plot(tspan,avg_vals['NEv1'],linewidth=5,color='orangered',linestyle='--')
plt.fill_between(tspan, avg_vals['NE'], avg_vals['NEv1'], label='NEv1')
plt.plot(tspan,avg_vals['NEv2'],linewidth=5,color='olive',linestyle='--')
plt.fill_between(tspan, avg_vals['NEv1'], avg_vals['NEv2'], label='NEv2')
plt.plot(tspan,avg_vals['NonNE'],linewidth=5,color='darkred',linestyle='--')
plt.fill_between(tspan, avg_vals['NEv2'], avg_vals['NonNE'], label='NonNE')

#plt.xlim(0,20)
#plt.ylim(0,1)
plt.xlabel('time (days)')
plt.ylabel('proportion of the population')
plt.title('Proportion of the SCLC population over time, average of trajectories\nfrom top 5K parameter sets')
plt.legend()
plt.tight_layout()
plt.show()

## Plot the parameter distributions

# Use the sampled_params_list from pydream run to plot initial distributions
sampled_params_list = list()

sp_k_NE_div_0 = norm(loc=np.log10(.428),scale=.25)
sampled_params_list.append(sp_k_NE_div_0)
sp_k_NE_div_x = norm(loc=np.log10(1.05),scale=1)
sampled_params_list.append(sp_k_NE_div_x)
sp_KD_Kx_NE_div = norm(loc=np.log10(1000),scale=1)
sampled_params_list.append(sp_KD_Kx_NE_div)
sp_k_NE_die_0 = norm(loc=np.log10(0.365),scale=.5)
sampled_params_list.append(sp_k_NE_die_0)
sp_k_NE_die_x = norm(loc=np.log10(0.95),scale=1)
sampled_params_list.append(sp_k_NE_die_x)
sp_KD_Kx_NE_die = norm(loc=np.log10(1000),scale=1)
sampled_params_list.append(sp_KD_Kx_NE_die)
sp_k_NEv1_div_0 = norm(loc=np.log10(.428),scale=.25)
sampled_params_list.append(sp_k_NEv1_div_0)
sp_k_NEv1_div_x = norm(loc=np.log10(1.05),scale=1)
sampled_params_list.append(sp_k_NEv1_div_x)
sp_KD_Kx_NEv1_div = norm(loc=np.log10(1000),scale=1)
sampled_params_list.append(sp_KD_Kx_NEv1_div)
sp_k_NEv1_die_0 = norm(loc=np.log10(0.365),scale=.5)
sampled_params_list.append(sp_k_NEv1_die_0)
sp_k_NEv1_die_x = norm(loc=np.log10(0.95),scale=1)
sampled_params_list.append(sp_k_NEv1_die_x)
sp_KD_Kx_NEv1_die = norm(loc=np.log10(1000),scale=1)
sampled_params_list.append(sp_KD_Kx_NEv1_die)
sp_k_NEv2_div_0 = norm(loc=np.log10(.428),scale=.25)
sampled_params_list.append(sp_k_NEv2_div_0)
sp_k_NEv2_div_x = norm(loc=np.log10(1.05),scale=1)
sampled_params_list.append(sp_k_NEv2_div_x)
sp_KD_Kx_NEv2_div = norm(loc=np.log10(1000),scale=1)
sampled_params_list.append(sp_KD_Kx_NEv2_div)
sp_k_NEv2_die_0 = norm(loc=np.log10(0.365),scale=.5)
sampled_params_list.append(sp_k_NEv2_die_0)
sp_k_NEv2_die_x = norm(loc=np.log10(0.95),scale=1)
sampled_params_list.append(sp_k_NEv2_die_x)
sp_KD_Kx_NEv2_die = norm(loc=np.log10(1000),scale=1)
sampled_params_list.append(sp_KD_Kx_NEv2_die)
sp_k_nonNE_div_0 = norm(loc=np.log10(.428),scale=.5)
sampled_params_list.append(sp_k_nonNE_div_0)
sp_k_nonNE_div_x = norm(loc=np.log10(0.95),scale=1)
sampled_params_list.append(sp_k_nonNE_div_x)
sp_KD_Kx_nonNE_div = norm(loc=np.log10(1000),scale=1)
sampled_params_list.append(sp_KD_Kx_nonNE_div)
sp_k_nonNe_die = norm(loc=np.log10(0.365),scale=.5)
sampled_params_list.append(sp_k_nonNe_die)
sp_kf_diff_ne_nev1 = uniform(loc=np.log10(0.05),scale=2.5)
sampled_params_list.append(sp_kf_diff_ne_nev1)
sp_kr_diff_ne_nev1 = uniform(loc=np.log10(0.05),scale=2.5)
sampled_params_list.append(sp_kr_diff_ne_nev1)
sp_kf_diff_ne_nev2 = uniform(loc=np.log10(0.05),scale=2.5)
sampled_params_list.append(sp_kf_diff_ne_nev2)
sp_kr_diff_ne_nev2 = uniform(loc=np.log10(0.05),scale=2.5)
sampled_params_list.append(sp_kr_diff_ne_nev2)
sp_kf_diff_nev1_nev2 = uniform(loc=np.log10(0.05),scale=2.5)
sampled_params_list.append(sp_kf_diff_nev1_nev2)
sp_kr_diff_nev1_nev2 = uniform(loc=np.log10(0.05),scale=2.5)
sampled_params_list.append(sp_kr_diff_nev1_nev2)
sp_kf_diff_nev2_nonNe = uniform(loc=np.log10(0.05),scale=2.5)
sampled_params_list.append(sp_kf_diff_nev2_nonNe)


#burnin = int(total_iterations/2)
burnin = 0
samples = np.concatenate(tuple([sampled_params[i][burnin:, :] for i in range(nchains)]))
ndims = len(param_values)-1

paramcolorlist = [
'gray',
'deepskyblue',
'lightcoral',
'lawngreen',
'plum',
'peachpuff',
'darkseagreen',
'goldenrod',
'mediumpurple',
'gold',
'springgreen',
'khaki',
'slateblue',
'yellowgreen',
'mediumturquoise',
'darkslategray',
'sandybrown',
'olivedrab',
'darkcyan',
'gainsboro',
'steelblue',
'darksalmon',
'mediumorchid',
'sienna',
'hotpink',
'crimson',
'orangered',
'navajowhite',
'navy'
]

plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(18,10))
columns = 5
rows = 6
for dim in range(ndims):
    fig.add_subplot(rows, columns, dim+1)
    param_toprint = (model.parameters[int(parameters_idxs[dim])].name)
    if 'KD' in param_toprint:
        plt.xlim(-1,8)
    else:
        plt.xlim(-3.5,3.5)
    #plt.xlim(-3,3)
    #plt.ylim(0,0.8)
    sns.distplot(samples[:, dim], color=paramcolorlist[dim], norm_hist=True,label=param_toprint)
    #plt.plot(np.linspace(-6,10,100),sampled_params_list[dim].dist.pdf(np.linspace(-6,10,100)))
    currax = plt.gca()
    currax.axes.get_yaxis().set_visible(True)
    currax.tick_params(labelsize=10)
    plt.title(param_toprint)


fig.text(0.5,0.005,'Log parameter value',ha='center', fontsize=20)
fig.text(0.005,0.5,'Probability',va='center',rotation='vertical',fontsize=20)
plt.tight_layout(pad=3)
plt.show()


fig.savefig('SCLC_paramvalues.png')
plt.close(fig)
