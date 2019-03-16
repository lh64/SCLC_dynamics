from pysb import *
from pysb.integrate import odesolve
from pysb.bng import run_ssa
import numpy as np
import matplotlib.pyplot as plt

Model()

Monomer('S', ['state'], {'state' : ['_0', 'g']}) # tumor-propagating cells (TPC; stem-like; 'g' is a secreted factor from N cells)
Monomer('N', ['state'], {'state' : ['_0', 'f']}) # Hes1+/NOTCH-active cells ('f' is a secreted factor from S cells)
Monomer('C') # CD44+ cells

Initial(S(state='_0'), Parameter('S_0', 100))

Observable('S_tot', S())
Observable('N_tot', N())
Observable('C_tot', C())

Observable('Sg', S(state='g'))
Observable('Nf', N(state='f'))

#1.  TPC stem-like cells (S cells) proliferate fast
#2.  S cells have 20-30% apoptosis rate
#5.  N-cells have a slower proliferation rate than S Cells.

Parameter('k_S_div', 1) # /day

Rule('S0_div', S(state='_0') >> S(state='_0') + S(state='_0'), k_S_div)
Rule('Sg_div', S(state='g') >> S(state='g') + S(state='g'), k_S_div)

Parameter('k_S0_die', 0.2)
Parameter('k_Sg_die', 0.1*k_S0_die.value)

Rule('S0_die', S(state='_0') >> None, k_S0_die)
Rule('Sg_die', S(state='g') >> None, k_Sg_die)

Parameter('k_N0_div', 0.3)
Parameter('k_Nf_div', 0.1*k_N0_div.value)

Rule('N0_div', N(state='_0') >> N(state='_0') + N(state='_0'), k_N0_div)
Rule('Nf_div', N(state='f') >> N(state='f') + N(state='f'), k_Nf_div)

Parameter('k_N_die', 0.2)

Rule('N0_die', N() >> None, k_N_die)

Parameter('k_C_div', 0.3)
Parameter('k_C_die', 0.2)

Rule('C_div', C() >> C() + C(), k_C_div)
Rule('C_die', C() >> None, k_C_die)

#3.  S cells can differentiate into other cell types 
#    (e.g. Hes1+/NOTCH-active (N-Cells) and CD44+ cells (C cells?))

Parameter('k_S_diff_N', 0.5)
Parameter('k_S_diff_C', 0.1)

Rule('S_diff_N', S() >> N(state='_0'), k_S_diff_N)
Rule('S_diff_C', S() >> C(), k_S_diff_C)

#4.  N-cells support growth of the S-cell population (affects both 
#    proliferation and apoptosis)

Parameter('kf_N_binds_S', 100)
Parameter('kr_N_binds_S', 10000)
Parameter('kcat_N_binds_S', 1)
Parameter('Km_N_binds_S', (kr_N_binds_S.value + kcat_N_binds_S.value) / kf_N_binds_S.value)
Expression('keff_S_div_N', kcat_N_binds_S / (Km_N_binds_S + S_tot))

Rule('S0_div_N', S(state='_0') + N() >> S(state='_0') + S(state='_0') + N(), keff_S_div_N)
Rule('Sg_div_N', S(state='g') + N() >> S(state='g') + S(state='g') + N(), keff_S_div_N)

Parameter('k_S0_to_Sg', 1)
Parameter('k_Sg_to_S0', 1000)

Rule('S0_to_Sg', S(state='_0') + N() >> S(state='g') + N(), k_S0_to_Sg)
Rule('Sg_to_S0', S(state='g') >> S(state='_0'), k_Sg_to_S0)

# Rule('S_binds_N', S(n=None) + N(s=None) <> S(n=1) % N(s=1), kf_N_binds_S, kr_N_binds_S)
# Rule('S_div_N', S(n=1) % N(s=1) >> S(n=None) + S(n=None) + N(s=None), kcat_N_binds_S)

#6.  Could be some spatial aspects to the S and N interactions since 
#    NOTCH signaling involves cell-cell adhesion.  Some of this might be 
#    substituted by extracellular vesicles (EVs) since Notch ligands like 
#    delta-like ligand are carried on EVs.  Conversely, the EV-carried DLL 
#    might in fact have an inhibitory effect on NOTCH signaling if opposes 
#    the cell-cell interactions (need to check reviews again on this).

#7.  Other growth or survival factors may be carried on EVs or secreted in 
#    soluble fashion to affect S cell survival.

#8.  Role of CD44 unclear, need to dig into Anton Berns papers and also see 
#    how this relates to David's phenotypes.

#9.  In vivo tumors have on average 50% S cells, 20% N cells, 5% C cells, 
#    25% unknown.  Is this an endpoint of interest, what model rules, feedback 
#    loops, etc. get you to a steady state with those percentages?

#10. There is some data that the S cells can reduce the proliferation of the 
#    N cells in vitro (30-50% reduction as assessed by BrdU incorporation), 
#    maybe adding another level of regulation.

Parameter('k_N0_to_Nf', 1)
Parameter('k_Nf_to_N0', 1000)
 
Rule('N0_to_Nf', N(state='_0') + S() >> N(state='f') + S(), k_N0_to_Nf)
Rule('Nf_to_N0', N(state='f') >> N(state='_0'), k_Nf_to_N0)


###########################################################################

tspan = np.linspace(0, 7, 101)
x = odesolve(model, tspan, verbose=True)
# x = run_ssa(model, tspan[-1], len(tspan)-1, verbose=True)

cell_tot = x['S_tot'] + x['N_tot'] + x['C_tot']

plt.figure()
plt.plot(tspan, x['S_tot'], lw=3, label='TPC')
plt.plot(tspan, x['N_tot'], lw=3, label='Hes1+')
plt.plot(tspan, x['C_tot'], lw=3, label='CD44+')
plt.annotate("A" , (0.2, 0.8), xycoords='axes fraction', fontsize=24)
plt.xlabel('time (d)', fontsize=16)
plt.ylabel('cell count', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc=0)

plt.figure()
plt.fill_between(tspan, x['S_tot'] / cell_tot, color='b', label='TPC')
plt.fill_between(tspan, (x['S_tot'] + x['N_tot']) / cell_tot, x['S_tot'] / cell_tot, color='g', label='Hes1+')
plt.fill_between(tspan, [1]*len(tspan), (x['S_tot'] + x['N_tot']) / cell_tot, color='r', label='CD44+')
plt.xlabel('time (d)', fontsize=16)
plt.ylabel('cell fraction', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc=0)

plt.figure()
plt.plot(tspan, x['S_tot'] / cell_tot, lw=2, label='TPC')
plt.plot(tspan, x['N_tot'] / cell_tot, lw=2, label='Hes1+')
plt.plot(tspan, x['C_tot'] / cell_tot, lw=2, label='CD44+')
plt.legend(loc=0)

plt.figure()
# plt.plot(tspan, x['Sg'] / x['S_tot'], lw=2, label='S~g / S_tot')
plt.plot(tspan[1:], x['Nf'][1:] / x['N_tot'][1:], lw=2, label='N~f / N_tot')
plt.legend(loc=0)

plt.show()



