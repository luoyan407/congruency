import pickle
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib import rc

# if you have installed Times New Roman on your machine, uncomment the following line
# rc('font', **{'family':'serif','serif':['Times New Roman'], 'size':12})
rc('text', usetex=True)


showLegend = False

print_range = 200

#-----------choose one block for visualization------------
input_file = 'figures_gd/z_results.npz'
output_svg = 'figures_gd/'
print_range = 50

# input_file = 'figures_rmsp/z_results.npz'
# output_svg = 'figures_rmsp/'
# print_range = 150

# input_file = 'figures_adam/z_results.npz'
# output_svg = 'figures_adam/'
# print_range = 200
#---------------------------------------------------------

data = np.load(input_file)

legends = data['names'].tolist()
colors = ['b', 'b', 'r', 'r']
linestyles = ['--', '-', '--', '-']


z = data['z']

x = np.linspace(1,z.shape[1],num=z.shape[1])

if not os.path.exists(os.path.dirname(output_svg)):
	os.makedirs(os.path.dirname(output_svg))


plt.figure(figsize=(12,7))
ax = plt.subplot()

for i in range(int(z.shape[0])):
	plt.plot(x[0:print_range],z[i,0:print_range], label=legends[i], linestyle=linestyles[i], linewidth=3, color=colors[i])

plt.xlabel('step', fontsize=50)
plt.ylabel(r'$z$', fontsize=60)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
ax.set_xlim([1,print_range])
plt.ticklabel_format(style='sci',axis='y', scilimits=(0,0))
plt.tick_params('both', which='major', labelsize=32)
plt.tick_params('both', which='minor', labelsize=32)
ax.yaxis.get_offset_text().set_fontsize(30)
plt.grid(True,linestyle='--')
if showLegend:
	plt.legend(loc=0,fontsize=29)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig('{}/{}.svg'.format(output_svg,'z_curve'), bbox_inches='tight', dpi=300)
plt.show()