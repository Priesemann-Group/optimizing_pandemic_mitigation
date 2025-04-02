import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

def CM_div(b,m):
    if whichcost  == "div":
        return	b * m/(1-m) # divergent cost 
    if whichcost == "divlog":
        return -b*np.log(1-m)
    if whichcost == "divsqrt":
        return b * (2/np.sqrt(1-m)-2)

def CI_cost(I,slope):
	a = 0.5*(5+np.sqrt(21)) # slope 5
	b = 0.001288575763893496
	return 1/a*I+a*(I-0.001)*(1-1/(1+np.exp(1000*(I-0.001))))+b
    
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

fig = plt.figure(figsize=(9/2.54,12/2.54))
gs = gridspec.GridSpec(1,2, width_ratios=[1.2,1], wspace = 0.3, figure = fig)
ax = gs[0].subgridspec(2,1, height_ratios=[1,1], hspace=0.5)
ax0 = fig.add_subplot(ax[0]) 
ax1 = fig.add_subplot(ax[1]) 


# Mitigation cost
labels = [r"$-\log(1-M)$",r"$\frac{2}{\sqrt{1-M}}-2$",r"$\frac{M}{1-M}$"]
lightness = [0.3,1,1.3]

for i,whichcost in enumerate(["divlog","divsqrt","div"]):
    b = 1
    m = np.linspace(0.01,0.99999,200)
    ax1.plot(m,CM_div(b,m),label=labels[i], color=adjust_lightness('cornflowerblue', lightness[i]))

ax1.set_xlabel(r"mitigation $M$")
ax1.set_ylabel(r"mitigation cost $C_M$")
ax1.set_xlim(0,1.05)
ax1.set_ylim(0,10)
ax1.set_xticks([0,0.5,1])
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '%g' % x))
ax1.set_yticks([0,5,10])
ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 1), frameon=False)
sns.despine(ax=ax1, top=True, right=True, left=False, bottom=False, offset=1, trim=True)

# Infection cost
I = np.arange(0,1,0.0001)
ax0.plot(I, CI_cost(I,5), label = 'slope: 5',color = 'indianred')
ax0.set_xlim(0,1)
ax0.set_xticks([0,0.5,1])
ax0.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '%g' % x))
ax0.set_ylim(0,5.01)
ax0.set_xlabel(r'infections $I$')
ax0.set_ylabel(r'infection cost $C_I$')
ax0.get_yaxis().set_ticks([0,1,2,3,4,5])
sns.despine(ax=ax0, top=True, right=True, left=False, bottom=False, offset=1, trim=True)
axins = ax0.inset_axes([1.4, 0.2, 0.6, 0.6],xlim=(0, 0.004), ylim=(0, 0.01))
axins.plot(I, CI_cost(I,5), color = 'indianred')
axins.get_yaxis().set_ticks([0,0.005,0.01])
axins.get_xaxis().set_ticks([0,0.002,0.004])
xlabels = ['0','','%g'%0.004]
ylabels = ['0','%g'%0.005,'%g'%0.01]
axins.set_xticklabels(xlabels, bbox = dict(ec="white", fc="white", pad = 0.1))
axins.set_yticklabels(ylabels, bbox = dict(ec="white", fc="white", pad=0.1))
ax0.indicate_inset_zoom(axins, edgecolor="grey")

fig.align_ylabels()

plt.savefig("costplots_for_SI.pdf", bbox_inches='tight')
