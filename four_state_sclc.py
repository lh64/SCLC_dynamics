from pysb import *
from pysb.simulator import ScipyOdeSimulator
from pysb.simulator import BngSimulator
import numpy as np
import matplotlib.pyplot as plt

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

Parameter('k_ne_div', 1)
Parameter('k_ne_die', 0.9)
Rule('NE_div', NE() >> NE() + NE(), k_ne_div)
Rule('NE_die', NE() >> None, k_ne_die)

Parameter('k_nev1_div', 1)
Parameter('k_nev1_die', 0.9)
Rule('NEv1_div', NEv1() >> NEv1() + NEv1(), k_nev1_div)
Rule('NEv1_die', NEv1() >> None, k_nev1_die)

Parameter('k_nev2_div', 1)
Parameter('k_nev2_die', 0.9)
Rule('NEv2_div', NEv2() >> NEv2() + NEv2(), k_nev2_div)
Rule('NEv2_die', NEv2() >> None, k_nev2_die)

Parameter('k_nonNe_div', 0.9)
Parameter('k_nonNe_die', 1)
Rule('NonNE_div', NonNE() >> NonNE() + NonNE(), k_nonNe_div)
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

Parameter('kf_diff_nev1_nonNe', 0.1)
Rule('NEv1_diff_NonNE', NEv1() >> NonNE(), kf_diff_nev1_nonNe)

tspan = np.linspace(0, 100, 101)

sim = ScipyOdeSimulator(model, verbose=True)
x = sim.run(tspan)

plt.figure()
for obs in model.observables:
    label = obs.name[:obs.name.find('_')]
    plt.plot(tspan, x.all[obs.name], lw=3, label=label)
plt.xlabel('time')
plt.ylabel('cell count')
plt.legend(loc=0)
plt.tight_layout()

cell_tot = np.array([sum(x.observables[i]) for i in range(len(x.observables))])

plt.figure()
label = [obs.name[:obs.name.find('_')] for obs in model.observables]
plt.fill_between(tspan, x.all[model.observables[0].name] / cell_tot, label=label[0])
sum_prev = x.all[model.observables[0].name]
for i in range(1,len(model.observables)-1):
    plt.fill_between(tspan, (x.all[model.observables[i].name] + sum_prev) / cell_tot, sum_prev / cell_tot, label=label[i])
    sum_prev += x.all[model.observables[i].name]
plt.fill_between(tspan, [1]*len(tspan), sum_prev / cell_tot, label=label[-1])
plt.xlabel('time')
plt.ylabel('cell fraction')
plt.legend(loc=0, framealpha=1)
plt.tight_layout()

plt.show()
    







