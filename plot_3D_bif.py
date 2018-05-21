from theoretical_prediction import *
import matplotlib as mat
from mpl_toolkits.mplot3d import Axes3D, proj3d


N = 500   #256   # 1000
m = 300    #128   # 500  # int(N/2)  # Donc beta = 1/2
sizes = [m, N-m]
n1 = sizes[0]
n2 = sizes[1]
f1 = n1/N
f2 = n2/N
f = f1/f2
beta = (n1*(n1-1) + n2*(n2-1)) / (N*(N-1))


###### Find saddle and Hopf bifurcation (theoretical) surface
alpha_array = np.linspace(1.2, np.pi/2-0.05, 100)     # Y
rho_array = np.linspace(0, 1, 100)                    # X
#delta_map_hopf = np.zeros((len(rho_array), len(alpha_array)))
#delta_map_saddle = np.zeros((len(rho_array), len(alpha_array)))
#i = 0
#for alpha in alpha_array:
#    print(alpha)
#    j = 0
#    for rho in rho_array:
#        solution = find_zero(alpha, f)
#        zeros_psi_det = solution[0]
#        zeros_psi_tr = solution[1]
#
#        delta_map_hopf[i, j] = find_delta(rho, zeros_psi_tr, alpha, f, beta)
#        delta_map_saddle[i, j] = find_delta(rho, zeros_psi_det, alpha, f, beta)
#        j += 1
#    i += 1
#print(np.shape(delta_map_hopf), np.shape(delta_map_saddle))
#
#import json
#import time
#timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
#with open('data/{}_delta_surface_hopf.json'.format(timestr, delta_map_hopf), 'w') as outfile:
#    json.dump(delta_map_hopf.tolist(), outfile)
#with open('data/{}_delta_surface_saddle.json'.format(timestr, delta_map_saddle), 'w') as outfile:
#    json.dump(delta_map_saddle.tolist(), outfile)
import json
with open('data/2018_05_21_04h47min30sec_delta_surface_hopf.json') as json_data:
    delta_map_hopf = np.array(json.load(json_data))
with open('data/2018_05_21_04h47min30sec_delta_surface_saddle.json') as json_data:
    delta_map_saddle = np.array(json.load(json_data))




####### Find homoclinic bifurcation suface (experimental)(data_calcul_qc3)

#import json
#delta_array_qsub = np.arange(0.1, 1+0.045, 0.045)            # Do not change
#rho_array_qsub = np.arange(0.5, 1+0.025, 0.025)              # Do not change
#alpha_array_qsub = np.arange(1.2, 1.56-0.024, 0.024)         # Do not change
#chimera_maps_cube = np.zeros((len(rho_array_qsub), len(delta_array_qsub), len(alpha_array_qsub)))
#
## Je fais quelque chose de mal pour remplir le cube de donnée
#for k in range(0, len(alpha_array_qsub)):
#    chimera_map = np.zeros((len(rho_array_qsub), len(delta_array_qsub)))
#    for i in range(0, len(delta_array_qsub)):
#        for j in range(0, len(rho_array_qsub)):
#            print(k, i, j)
#            with open('data_calcul_qc3/alpha_{}_delta_{}_rho_{}_chimeramap.json'.format(k, i, j)) as json_data:
#                R = json.load(json_data)
#            chimera_maps_cube[i, j, k] = R
#
## Ce n'est pas bon ici, il y a qqch que je fais de mal,
#delta_map_homo = np.zeros((len(rho_array_qsub), len(alpha_array_qsub)))
#for j in range(0, len(delta_array_qsub)):
#    for k in range(0, len(alpha_array_qsub)):
#        alpha = alpha_array_qsub[k]
#        iterator = 0
#        for i in range(0, len(rho_array_qsub)):
#            chimera_map = chimera_maps_cube[:, :, k]
#            # Pour visualiser les chimères pour chaque tranche
#            if iterator == 0:
#                plt.figure(figsize=(10, 10))
#                plt.title("$\\alpha={}$".format(alpha), fontsize=15)
#                plt.imshow(chimera_map)
#                plt.plot(np.linspace(0, 1, 200), find_delta_array(np.linspace(0, 1, 200), find_zero(alpha, f)[1], alpha, f, beta))
#                plt.plot(np.linspace(0, 1, 200), find_delta_array(np.linspace(0, 1, 200), find_zero(alpha, f)[0], alpha, f, beta))
#                plt.plot(rho_array_qsub, find_homoclinic_bifline(find_delta_array(rho_array_qsub, find_zero(alpha, f)[1], alpha, f, beta), rho_array_qsub, delta_array_qsub, chimera_map, beta))
#                plt.show()
#            iterator += 1
#
#            # Ce n'est pas bon ici,
#            delta_map_homo[i, k] = find_homoclinic_bifpoint(find_delta_array(rho_array_qsub, find_zero(alpha, f)[1], alpha, f, (n1**2 + n2**2)/(N**2)), rho_array_qsub, delta_array_qsub, rho_array_qsub[j], chimera_map)



#print(delta_map_homo, np.shape(delta_map_homo), np.shape(rho_array_qsub), np.shape(alpha_array_qsub))




fig = plt.figure(figsize=(15, 12))
plt.rc('axes', linewidth=2)
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'text.usetex': True,
    'font.family': 'serif',
    'xtick.labelsize': 15,
    'ytick.labelsize': 15
}

mat.rcParams.update(params)

cdict_homo = {
    'red': ((0, 132 / 255, 255 / 255), (0.01, 255 / 255, 255 / 255), (1.0, 255 / 255, 255 / 255)),
    'green': ((0, 35 / 255, 67 / 255), (0.01, 147 / 255, 147 / 255), (1.0, 102 / 255, 102 / 255)),
    'blue': ((0, 0 / 255, 0 / 255), (0.01, 76 / 255, 76 / 255), (1.0, 0, 0))
}

cdict_hopf = {
    'red': ((0, 26 / 255, 26 / 255), (0.01, 255 / 255, 255 / 255), (1.0, 255 / 255, 255 / 255)),
    'green': ((0, 63 / 255, 63 / 255), (0.01, 147 / 255, 147 / 255), (1.0, 102 / 255, 102 / 255)),
    'blue': ((0, 178 / 255, 178 / 255), (0.01, 76 / 255, 76 / 255), (1.0, 0, 0))
}
cdict_saddle = {
    'red': ((0, 40 / 255, 40 / 255), (0.001, 40 / 255, 40 / 255), (1.0, 40 / 255, 40 / 255)),
    'green': ((0, 205 / 255, 205 / 255), (0.001, 205 / 255, 205 / 255), (1.0, 205 / 255, 205 / 255)),
    'blue': ((0, 224 / 255, 224 / 255), (0.001, 224 / 255, 224 / 255), (1.0, 224, 224))
}

cm_homo = mat.colors.LinearSegmentedColormap('my_colormap', cdict_homo, 1024)
cm_hopf = mat.colors.LinearSegmentedColormap('my_colormap', cdict_hopf, 1024)
cm_saddle = mat.colors.LinearSegmentedColormap('my_colormap', cdict_saddle, 1024)

ax = fig.add_subplot(111, projection='3d')
plt.title("Community sizes ratio $f = {}$".format(np.round(f, 2)), fontsize=30)
#rho_array_qsub, alpha_array_qsub = np.meshgrid(rho_array_qsub, alpha_array_qsub)
#ax.scatter(rho_array_qsub, alpha_array_qsub, delta_map_homo, label="Homoclinic (experimental)")#, rstride=len(rho_array_qsub) // 10, cstride=len(rho_array_qsub) // 10, cmap=cm_homo, alpha=1, label="Homoclinic (experimental)")
rho_array, alpha_array = np.meshgrid(rho_array, alpha_array)
ax.plot_surface(rho_array, alpha_array, delta_map_hopf,   rstride=len(rho_array)//10, cstride=len(rho_array)//10, cmap=cm_hopf, alpha = 1, label="Hopf (theoretical)")
ax.plot_surface(rho_array, alpha_array, delta_map_saddle, rstride=len(rho_array)//10, cstride=len(rho_array)//10, cmap=cm_saddle, alpha = 0.4, label="Saddle (theoretical)")
ax.set_xlabel("$\\rho$", fontsize=35, labelpad=12)
ax.set_ylabel("$\\alpha$", fontsize=35, labelpad=12)
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel("$\\Delta$", fontsize=35, labelpad=12, rotation=0)
plt.tick_params(axis='both', which='major', labelsize=20)
#f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
#ax.legend(loc="lower left", bbox_to_anchor=f(0.6,1.55,0.4),
#          bbox_transform=ax.transData)
fake2Dline = mat.lines.Line2D([0],[0], linestyle="none", c='#dcabfc', marker='s')
fake2Dline2 = mat.lines.Line2D([0],[0], linestyle="none", c='#1a3fb2', marker='s')
fake2Dline3 = mat.lines.Line2D([0],[0], linestyle="none", c='#28cde0', marker='s')
lgnd = ax.legend([fake2Dline, fake2Dline2, fake2Dline3], ['Homoclinic (experimental)','Hopf (theoretical)', 'Saddle (theoretical)'], numpoints=1, fontsize=25, loc=2, bbox_to_anchor=(0.1, 0.8, 0, 0))
lgnd.legendHandles[0]._legmarker.set_markersize(15)
lgnd.legendHandles[1]._legmarker.set_markersize(15)
lgnd.legendHandles[2]._legmarker.set_markersize(15)
plt.tight_layout()
#plt.plot(alpha_array, zeros_psi_det)
#plt.plot(alpha_array, zeros_psi_tr)
plt.show()

















