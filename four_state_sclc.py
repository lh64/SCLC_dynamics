from pysb import *
from pysb.simulator import ScipyOdeSimulator
from pysb.simulator import BngSimulator
import numpy as np
import matplotlib.pyplot as plt

def k_fate(ename, k_fate_0, KD_Kx_fate, k_fate_x, effector_cell_obs):
    return Expression(ename, (k_fate_0*KD_Kx_fate + k_fate_x*effector_cell_obs) / (KD_Kx_fate + effector_cell_obs))

Model()

Monomer('NE')
Monomer('NEv1')
Monomer('NEv2')
Monomer('NonNE')

Parameter('NE_init', 100)
Initial(NE(), NE_init)

Observable('NE_obs', NE())
Observable('NEv1_obs', NEv1())
Observable('NEv2_obs', NEv2())
Observable('NonNE_obs', NonNE())
Observable('NE_all', NE()+NEv1()+NEv2())

Parameter('k_NE_div_0', 1) # TPCs divide approximately once per day in culture
Parameter('k_NE_div_x', 2)
Parameter('KD_Kx_NE_div', 1000)
k_fate('k_NE_div', k_NE_div_0, KD_Kx_NE_div, k_NE_div_x, NonNE_obs)
Parameter('k_NE_die_0', 0.9)
Parameter('k_NE_die_x', 0.1)
Parameter('KD_Kx_NE_die', 1000)
k_fate('k_NE_die', k_NE_die_0, KD_Kx_NE_die, k_NE_die_x, NonNE_obs)

# Parameter('k_ne_div', 1)
# Parameter('k_ne_die', 0.9)
Rule('NE_div', NE() >> NE() + NE(), k_NE_div)
Rule('NE_die', NE() >> None, k_NE_die)

# Parameter('k_nev1_div', 1)
# Parameter('k_nev1_die', 0.9)
Rule('NEv1_div', NEv1() >> NEv1() + NEv1(), k_NE_div)
Rule('NEv1_die', NEv1() >> None, k_NE_die)

# Parameter('k_nev2_div', 1)
# Parameter('k_nev2_die', 0.9)
Rule('NEv2_div', NEv2() >> NEv2() + NEv2(), k_NE_div)
Rule('NEv2_die', NEv2() >> None, k_NE_die)

# Parameter('k_nonNe_div', 0.9)
Parameter('k_nonNE_div_0', 1.1)
Parameter('k_nonNE_div_x', 0.9)
Parameter('KD_Kx_nonNE_div', 1000)
k_fate('k_nonNE_div', k_nonNE_div_0, KD_Kx_nonNE_div, k_nonNE_div_x, NE_all)
Parameter('k_nonNe_die', 0.1)
Rule('NonNE_div', NonNE() >> NonNE() + NonNE(), k_nonNE_div)
Rule('NonNE_die', NonNE() >> None, k_nonNe_die)

Parameter('kf_diff_ne_nev1', 0.1)
Parameter('kr_diff_ne_nev1', 0.1)
Rule('NE_diff_NEv1', NE() | NEv1(), kf_diff_ne_nev1, kr_diff_ne_nev1)

Parameter('kf_diff_ne_nev2', 0.1)
Parameter('kr_diff_ne_nev2', 0.075)
Rule('NE_diff_NEv2', NE() | NEv2(), kf_diff_ne_nev2, kr_diff_ne_nev2)

Parameter('kf_diff_nev1_nev2', 0.1)
Parameter('kr_diff_nev1_nev2', 0.1)
Rule('NEv1_diff_NEv2', NEv1() | NEv2(), kf_diff_nev1_nev2, kr_diff_nev1_nev2)

Parameter('kf_diff_nev1_nonNe', 5) 
Rule('NEv1_diff_NonNE', NEv1() >> NonNE(), kf_diff_nev1_nonNe)

tspan = np.linspace(0, 20, 101)

sim = ScipyOdeSimulator(model, verbose=True)
x = sim.run(tspan)

plt.figure()
for obs in model.observables[:4]:
    label = obs.name[:obs.name.find('_')]
    plt.plot(tspan, x.all[obs.name], lw=3, label=label)
plt.xlabel('time (d)', fontsize=16)
plt.ylabel('cell count', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(loc=0)
plt.tight_layout()

# cell_tot = np.array([sum(x.observables[i]) for i in range(len(x.observables))])
cell_tot = sum(x.all[obs.name] for obs in [NE_obs, NEv1_obs, NEv2_obs, NonNE_obs])

plt.figure()
label = [obs.name[:obs.name.find('_')] for obs in model.observables[:4]]
plt.fill_between(tspan, x.all[model.observables[0].name] / cell_tot, label=label[0])
sum_prev = x.all[model.observables[0].name]
for i in range(1,len(model.observables[:4])-1):
    plt.fill_between(tspan, (x.all[model.observables[i].name] + sum_prev) / cell_tot, sum_prev / cell_tot, label=label[i])
    sum_prev += x.all[model.observables[i].name]
plt.fill_between(tspan, [1]*len(tspan), sum_prev / cell_tot, label=label[-1])
plt.xlabel('time (d)', fontsize=16)
plt.ylabel('cell fraction', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc=(0.75,0.6), framealpha=1)
plt.tight_layout()

plt.show()
    







