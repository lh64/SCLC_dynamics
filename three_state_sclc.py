from pysb import *
from pysb.integrate import odesolve
import numpy as np
import matplotlib.pyplot as plt

def k_fate(ename, k_fate_0, k_fate_x, KD_Kx_fate, effector_cell_obs):
    return Expression(ename, (k_fate_0*KD_Kx_fate + k_fate_x*effector_cell_obs) / (KD_Kx_fate + effector_cell_obs))

Model()

Monomer('TPC') # tumor-propagating cells (TPC; stem-like)
Monomer('Hes1') # Hes1+/NOTCH-active cells
Monomer('CD44') # CD44+ cells

Initial(TPC(), Parameter('TPC_0', 100))

Observable('TPC_tot', TPC())
Observable('Hes1_tot', Hes1())
Observable('CD44_tot', CD44())

# Hes1+ cells support the growth of the TPC population by both enhancing cell division and reducing apoptosis
Parameter('k_TPC_div_0', 1) # TPCs divide approximately once per day in culture
Parameter('k_TPC_div_x', 2)
Parameter('KD_Kx_TPC_div', 1000)
# Expression('k_TPC_div', (k_TPC_div_0*KD_Kx_TPC_div + k_TPC_div_x*Hes1_tot) / (KD_Kx_TPC_div + Hes1_tot))
k_fate('k_TPC_div', k_TPC_div_0, k_TPC_div_x, KD_Kx_TPC_div, Hes1_tot)
Parameter('k_TPC_die_0', 0.2)
Parameter('k_TPC_die_x', 0.1)
Parameter('KD_Kx_TPC_die', 1000)
# Expression('k_TPC_die', (k_TPC_die_0*KD_Kx_TPC_die + k_TPC_die_x*Hes1_tot) / (KD_Kx_TPC_die + Hes1_tot))
k_fate('k_TPC_die', k_TPC_die_0, k_TPC_die_x, KD_Kx_TPC_die, Hes1_tot)
Rule('TPC_div', TPC() >> TPC() + TPC(), k_TPC_div)
Rule('TPC_die', TPC() >> None, k_TPC_die)

# Evidence that TPCs inhibit proliferation of Hes1+ populations in vitro 
Parameter('k_Hes1_div_0', 0.5) # Hes1+ cells have a lower division rate than TPCs
Parameter('k_Hes1_div_x', 0.25)
Parameter('KD_Kx_Hes1_div', 1000)
# Expression('k_Hes1_div', (k_Hes1_div_0*KD_Kx_Hes1_div + k_Hes1_div_x*TPC_tot) / (KD_Kx_Hes1_div + TPC_tot))
k_fate('k_Hes1_div', k_Hes1_div_0, k_Hes1_div_x, KD_Kx_Hes1_div, TPC_tot)
Parameter('k_Hes1_die', 0.2)
Rule('Hes1_div', Hes1() >> Hes1() + Hes1(), k_Hes1_div)
Rule('Hes1_die', Hes1() >> None, k_Hes1_die)

Parameter('k_CD44_div', 0.3)
Parameter('k_CD44_die', 0.2)
Rule('CD44_div', CD44() >> CD44() + CD44(), k_CD44_div)
Rule('CD44_die', CD44() >> None, k_CD44_die)

Parameter('k_TPC_diff_Hes1', 0.5)
Parameter('k_TPC_diff_CD44', 0.1)
Rule('TPC_diff_Hes1', TPC() >> Hes1(), k_TPC_diff_Hes1)
Rule('TPC_diff_CD44', TPC() >> CD44(), k_TPC_diff_CD44)

###########################################################################

# In vivo tumors have on average 50% TPCs, 20% Hes1 cells, 5% CD44+ cells, and 25% unknown cell types.

tspan = np.linspace(0, 10, 101)
x = odesolve(model, tspan, verbose=True)
# x = run_ssa(model, tspan[-1], len(tspan)-1, verbose=True)

cell_tot = x['TPC_tot'] + x['Hes1_tot'] + x['CD44_tot']

print('TPC frac:   %g (%g)' % (x['TPC_tot'][-1]/cell_tot[-1], 50/80))
print('Hes1+ frac: %g (%g)' % (x['Hes1_tot'][-1]/cell_tot[-1], 25/80))
print('CD44+ frac: %g (%g)' % (x['CD44_tot'][-1]/cell_tot[-1], 5/80))

plt.figure()
plt.plot(tspan, x['TPC_tot'], lw=3, color='b', label='TPC')
plt.plot(tspan, x['Hes1_tot'], lw=3, color='g', label='Hes1+')
plt.plot(tspan, x['CD44_tot'], lw=3, color='r', label='CD44+')
# plt.annotate("A" , (0.2, 0.8), xycoords='axes fraction', fontsize=24)
plt.xlabel('time (d)', fontsize=16)
plt.ylabel('cell count', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(loc=0)
plt.tight_layout()

plt.figure()
plt.fill_between(tspan, x['TPC_tot'] / cell_tot, color='b', label='TPC')
plt.fill_between(tspan, (x['TPC_tot'] + x['Hes1_tot']) / cell_tot, x['TPC_tot'] / cell_tot, color='g', label='Hes1+')
plt.fill_between(tspan, [1]*len(tspan), (x['TPC_tot'] + x['Hes1_tot']) / cell_tot, color='r', label='CD44+')
plt.xlabel('time (d)', fontsize=16)
plt.ylabel('cell fraction', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc=(0.75,0.75), framealpha=1)
plt.tight_layout()
plt.show()

