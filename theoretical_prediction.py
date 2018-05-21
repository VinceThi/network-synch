# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
from scipy.optimize import newton
from geometrySBM import to_probability_space
from scipy.signal import argrelmin
from scipy.integrate import odeint, simps
plt.style.use('classic')
# from plot_r_map_good import give_levelset

# Note : phi = psi

def c(alpha): return np.cos(alpha)
def s(alpha): return np.sin(alpha)
def cm(psi, alpha): return np.cos(psi - alpha)
def cp(psi, alpha): return np.cos(psi + alpha)
def sm(psi, alpha): return np.sin(psi - alpha)
def sp(psi, alpha): return np.sin(psi + alpha)
def sm2(psi, alpha):return np.sin(2*alpha - psi)

#### Abrams 2008 results
def detAbrams(psi, beta): return np.sin(beta) + (np.sin(2*beta + psi)*(np.sin(beta - 2*psi) + 2*np.sin(beta + 2*psi)))/(np.sin(2*beta - psi)+2*np.sin(psi))
def psi_abrams_saddle(beta): return -2*beta -2*beta**2 + 2*beta**3 + 11/3*beta**4 - 12*beta**5 - 3271/180*beta**6
def delta_abrams_saddle(beta): return 2*beta -2*beta**2 - 7/3*beta**3 + 20/3*beta**4 + 181/60*beta**5
def psi_abrams_hopf(beta): return -0.5*np.arcsin(2*np.sin(2*beta))
def delta_abrams_hopf(beta): return 2 - np.sqrt(3) + (4*np.sqrt(3) - 6)*beta**2 + (26/np.sqrt(3) - 10)*beta**4
def trMJ_abrams(phi, beta): return 2*np.sin(2*beta) + np.sin(2*phi)
#print(psi_abrams_saddle(0.1), "\n")


### My results
def r(psi, alpha, f): return np.sqrt((2*f*s(alpha)*cm(psi, alpha) - np.sin(psi)) / (2*f**(-1)*sp(psi, alpha)*c(alpha) + np.sin(psi)))
def r2(psi, alpha, f): return (2*f*s(alpha)*cm(psi, alpha) - np.sin(psi)) / (2*f**(-1)*sp(psi, alpha)*c(alpha) + np.sin(psi))
def trMJ(psi, alpha, f): return f*cm(psi, alpha) + r2(psi, alpha, f)*cp(psi, alpha)
def detMJ(psi, alpha, f): return (-0.5*np.cos(2*alpha - 2*psi) - f**(-1)*np.cos(2*psi) + (s(alpha)*sm(psi, alpha)*cm(psi, alpha))/(c(alpha)) )*r2(psi, alpha, f) - 0.5
def find_delta_array(rho_array, psi, alpha, f, beta):
    delta_arr = np.zeros(len(rho_array))
    i = 0
    #print(((r(psi, alpha, f) * c(alpha) + f * cm(psi, alpha)) / (
    #    (beta - 1) * r(psi, alpha, f) * c(alpha) + f * beta * cm(psi, alpha))))
    for rho in rho_array:
        delta = ((r(psi,alpha,f)*c(alpha) + f*cm(psi,alpha))/((beta-1)*r(psi,alpha,f)*c(alpha) + f*beta*cm(psi,alpha)))*rho
        probability_space_tuple = to_probability_space(rho, delta, beta)  # IMPORTANT : return (q, p)
        p = probability_space_tuple[1]
        q = probability_space_tuple[0]

        if p > 1 or p < 0 or q > 1 or q < 0:
            delta_arr[i] = np.nan
            i += 1
        else:
            delta_arr[i] = delta
            i += 1
    return delta_arr
def find_delta(rho, psi, alpha, f, beta):
    delta = ((r(psi, alpha, f) * c(alpha) + f * cm(psi, alpha)) / (
            (beta - 1) * r(psi, alpha, f) * c(alpha) + f * beta * cm(psi, alpha))) * rho
    #print(delta, (2*f*s(alpha)*cm(psi, alpha) - np.sin(psi)) / (2*f**(-1)*sp(psi, alpha)*c(alpha) + np.sin(psi)))
    probability_space_tuple = to_probability_space(rho, delta, beta)  # IMPORTANT : return (q, p)
    p = probability_space_tuple[1]
    q = probability_space_tuple[0]

    if p > 1 or p < 0 or q > 1 or q < 0:
        delta_value = np.nan
    else:
        delta_value = delta
    return delta_value
def find_zero(alpha, f):
    psiarr = np.linspace(-2, 0, 10000)
    trace_array = trMJ(psiarr, alpha, f)
    min_trindex_array = argrelmin(trace_array)[0]
    # print(min_trindex_array)
    min_trace_array = trace_array[min_trindex_array]
    # print(min_trace_array)
    minmin_trindex = min_trindex_array[np.where(min_trace_array == min(min_trace_array))]
    guess_trace = psiarr[minmin_trindex] - 0.005

    det_array = detMJ(psiarr, alpha, f)
    min_detindex_array = argrelmin(det_array)[0]
    min_det_array = det_array[min_detindex_array]
    # print(min_det_array)
    minmin_detindex = min_detindex_array[np.where(min_det_array == min(min_det_array))]
    guess_det = psiarr[minmin_detindex]-0.005

    zeros_psi_det = newton(detMJ, guess_det, args=(alpha, f,))
    zeros_psi_tr = newton(trMJ, guess_trace, args=(alpha, f,))
    #print(zeros_psi_det, zeros_psi_tr)
    return zeros_psi_det, zeros_psi_tr
def Takens_Bogdanov_function(psi, alpha, f):
    return (-f/2*np.cos(2*alpha - 2*psi) - np.cos(2*psi) + f*(s(alpha)*sm(psi, alpha)*cm(psi, alpha))/(c(alpha)) )*(cm(psi, alpha)/cp(psi, alpha)) + 0.5
def find_Takens_Bogdanov():
    return "À faire"
def theta_bifurcation(psi, alpha, f, beta):
    return np.arctan(((r(psi, alpha, f) * c(alpha) + f * cm(psi, alpha)) / ((beta - 1) * r(psi, alpha, f) * c(alpha) + f * beta * cm(psi, alpha))))
def find_homoclinic_bifline(hopf_bifline, rho_array, delta_array, chimera_map, beta):
    for k in hopf_bifline[::-1]:
        if np.isnan(k):
            pass
        else:
            hopf_last_point = k
            break

    delta_index = 0
    i = 0
    for delta in delta_array:
        if np.abs(delta - hopf_last_point) < 0.1:
            delta_index = i
            break
        else:
            i += 1

    rho_index = 0
    j = 0
    for rho in rho_array:
        if np.abs(chimera_map[delta_index, j] - 0.97) < 0.01:
            rho_index = j
        else:
            j += 1

    slope = (delta_array[delta_index] - 0) / (rho_array[rho_index] - 0)

    delta_homobif = np.zeros(len(rho_array))
    m = 0
    for rho in rho_array:
        delta = rho * (delta_array[delta_index] - 0) / (rho_array[rho_index] - 0)
        probability_space_tuple = to_probability_space(rho, delta, beta)  # IMPORTANT : return (q, p)
        p = probability_space_tuple[1]
        q = probability_space_tuple[0]

        if p > 1 or p < 0 or q > 1 or q < 0:
            delta_homobif[m] = np.nan
            m += 1
        else:
            delta_homobif[m] = delta
            m += 1

    return delta_homobif
def find_homoclinic_bifpoint(hopf_bifline, rho_array, delta_array, rho, chimera_map):
    for k in hopf_bifline[::-1]:
        if np.isnan(k):
            pass
        else:
            hopf_last_point = k
            break

    delta_index = 0
    i = 0
    for delta in delta_array:
        if np.abs(delta - hopf_last_point) < 0.1:
            delta_index = i
            break
        else:
            i += 1

    rho_index = 0
    j = 0
    for rho in rho_array:
        if np.abs(chimera_map[delta_index, j] - 0.97) < 0.01:
            rho_index = j
        else:
            j += 1

    slope = (delta_array[delta_index] - 0) / (rho_array[rho_index] - 0)


    return slope*rho


"""  # Old
#def r2old(psi, alpha): return sm2(psi, alpha)/(2*sp(psi, alpha)*c(alpha) + np.sin(psi))
#def oldtr(psi, alpha, f):
#    #return (((2*f-1)*s(alpha)*cm(psi, alpha) - f**2*sm(psi, alpha)*c(alpha))/(2*f*sp(psi,alpha)*c(alpha) + s(alpha)*cm(psi,alpha) + f**2*sm(psi,alpha)*c(alpha))) \
#    #       - ((cm(psi, alpha) + f**2)/(3*cm(psi, alpha) - 2*f**2*cm(psi, alpha)-2*f*cp(psi, alpha) - f))
#    #return r2(psi, alpha, f) - ((cm(psi, alpha) + f**2)/(3*cm(psi, alpha) - 2*f**2*cm(psi, alpha)-2*f*cp(psi, alpha) - f))
#    #return (1-3*r2(psi, alpha, f))*f**(-1)*cm(psi, alpha) + 2*f*r2(psi, alpha, f)*cm(psi, alpha) + (1+r2(psi,alpha,f))*f*cm(psi,alpha) + 2*r2(psi, alpha, f)*cp(psi,alpha)
#    #return (1-3*r2(psi, alpha, f))*f**(-1)*cm(psi, alpha) + 2*f*r2(psi, alpha, f)*cm(psi, alpha) + (1+r2(psi,alpha,f))*f*cm(psi,alpha) + 2*r2(psi, alpha, f)*cp(psi,alpha)
#    return f/(-np.sqrt(r2(psi, alpha, f)))*cm(psi, alpha)  -np.sqrt(r2(psi, alpha, f))*np.cos(psi + alpha)
#
#def detMJ(psi, alpha, q, bool):
#    return -1/(4*r(psi, alpha, bool))*(3*r(psi, alpha, bool)**2-1)*(r(psi, alpha, bool)**2+1)*q**2*cm(psi, alpha)**2\
#           - 0.5*r(psi, alpha, bool)*(3*r(psi, alpha, bool)**2 - 1)*q**2*cm(psi,alpha)*cp(psi,alpha) \
#           + 0.5*(r(psi, alpha, bool)**2-1)*q**2*s(alpha)*sm(psi, alpha)*cm(psi,alpha)/c(alpha)\
#           + (r(psi, alpha, bool)**2-1)**2/(4*r(psi, alpha, bool)**2)*q**2*sm(psi, alpha)**2\
#           + 0.5*(r(psi, alpha, bool)**2-1)*q**2*sm(psi, alpha)*sp(psi, alpha)
#def trMJ2(psi, alpha, q, bool):
#    return (3*r(psi,alpha,bool)**3 - r(psi,alpha,bool)**2 - r(psi,alpha,bool) - 1)/(2*r(psi,alpha,bool)) - r(psi,alpha,bool)*q*cp(psi, alpha)
#def deltafun(rho, psi, alpha):
#    return 2*rho*(r(psi, alpha, 1)*c(alpha) + cm(psi,alpha))/(cm(psi, alpha)-r(psi, alpha, 1)*c(alpha))



#import sympy as sym
#
#def dot_rhosym(rho, phi):
#    return (1-rho**2)/2*(np.cos(phi + np.pi/2 - 0.1) * np.dot(np.array([[0.64, 0.36],[0.36, 0.64]]), rho*np.cos(phi)) + np.sin(phi+np.pi/2 - 0.1) * np.dot(np.array([[0.64, 0.36],[0.36, 0.64]]), rho*np.sin(phi)))
#
#def dot_phisym(rho, phi):
#    return (1+rho**2)/(2*rho)*(np.cos(phi + np.pi/2 - 0.1) * np.dot(np.array([[0.64, 0.36],[0.36, 0.64]]), rho*np.sin(phi)) - np.sin(phi+np.pi/2 - 0.1) * np.dot(np.array([[0.64, 0.36],[0.36, 0.64]]), rho*np.cos(phi)))
#
#rho, phi =sym.symbols('rho, phi')
#F = sym.Matrix([dot_rhosym, dot_phisym])
#print(F.jacobian([rho, phi]))
"""



##################################### Plot 3D delta, rho, alpha ########################################################
"""
from mpl_toolkits.mplot3d import Axes3D, proj3d
#print(newton(trMJ, -0.18, args=(1.43, 1,))   )

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
#for k in range(0, len(alpha_array_qsub)):
#    chimera_map = np.zeros((len(rho_array_qsub), len(delta_array_qsub)))
#    for i in range(0, len(delta_array_qsub)):
#        for j in range(0, len(rho_array_qsub)):
#            print(k, i, j)
#            with open('data_calcul_qc3/alpha_{}_delta_{}_rho_{}_chimeramap.json'.format(k, i, j)) as json_data:
#                R = json.load(json_data)
#            chimera_maps_cube[i, j, k] = R
#
#
#delta_map_homo = np.zeros((len(rho_array_qsub), len(alpha_array_qsub)))
#for j in range(0, len(delta_array_qsub)):
#    for k in range(0, len(alpha_array_qsub)):
#        alpha = alpha_array_qsub[k]
#        iterator = 0
#        for i in range(0, len(rho_array_qsub)):
#            chimera_map = chimera_maps_cube[:, :, k]
#            if iterator == 0:
#                plt.figure(figsize=(10, 10))
#                plt.title("$\\alpha={}$".format(alpha), fontsize=15)
#                plt.imshow(chimera_map)
#                plt.plot(np.linspace(0, 1, 200), find_delta_array(np.linspace(0, 1, 200), find_zero(alpha, f)[1], alpha, f, beta))
#                plt.plot(np.linspace(0, 1, 200), find_delta_array(np.linspace(0, 1, 200), find_zero(alpha, f)[0], alpha, f, beta))
#                plt.plot(rho_array_qsub, find_homoclinic_bifline(find_delta_array(rho_array_qsub, find_zero(alpha, f)[1], alpha, f, beta), rho_array_qsub, delta_array_qsub, chimera_map, beta))
#                plt.show()
#            iterator += 1
#            delta_map_homo[i, k] = find_homoclinic_bifpoint(find_delta_array(rho_array_qsub, find_zero(alpha, f)[1], alpha, f, (n1**2 + n2**2)/(N**2)), rho_array_qsub, delta_array_qsub, rho_array_qsub[j], chimera_map)
#


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
"""



############################### Plot volume between bifurcations vs f ##################################################
"""
n1array = np.arange(500, 670, 1)
alpha_array = np.linspace(1.4, 1.5, 100)
rho_array = np.linspace(0, 1, 100)

volume_hopf_list = []
volume_saddle_list = []

N = 1000

for m in n1array:
    sizes = [m, N - m]
    n1 = sizes[0]
    n2 = sizes[1]
    f1 = n1 / N
    f2 = n2 / N
    f = f1 / f2
    betainf = f1 ** 2 + f2 ** 2  # beta not inf =(n1*(n1-1) + n2*(n2-1)) / (N*(N-1))

    volume_hop = 0
    volume_sad = 0
    for alpha in alpha_array:
        delta_array_hopf = np.zeros(len(rho_array))
        delta_array_saddle = np.zeros(len(rho_array))
        i = 0
        for rho in rho_array:
            solution = find_zero(alpha, f)
            zeros_psi_det = solution[0]
            zeros_psi_tr = solution[1]

            delta_array_hopf[i] = find_delta(rho, zeros_psi_tr, alpha, f, betainf)
            delta_array_saddle[i] = find_delta(rho, zeros_psi_det, alpha, f, betainf)
            
            i += 1
        volume_hop += simps(delta_array_hopf, rho_array)
        volume_sad += simps(delta_array_hopf, rho_array)

    volume_hopf_list += volume_hop
    volume_saddle_list += volume_sad

farray = n1array / (N * np.ones(len(n1array)) - n1array)

fig = plt.figure(figsize=(6, 5))
plt.rc('axes', linewidth=2)
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'text.usetex': True,
    'font.family': 'serif',
    'xtick.labelsize': 15,
    'ytick.labelsize': 15
}

mat.rcParams.update(params)

ax = fig.add_subplot(111)
colorlist = ["#ffbf00", "#ff6600", "#b72114", "#661d05"]
ax.plot(farray, volume_hopf_list, linewidth=2, color=colorlist[0])
ax.plot(farray, volume_saddle_list, linewidth=2, color=colorlist[1])
ax.set_xlabel("$f$", fontsize=35, labelpad=12)
ax.set_ylabel("$Volume$", fontsize=35, labelpad=12)
ax.legend(loc=4, fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlim([1, 2])
plt.tight_layout()
plt.show()
"""



############################### Plot angle between bifurcations vs f ###################################################
"""
n1array = np.arange(500, 670, 1)
alpha_array = np.linspace(1.4, 1.5, 4)

theta_hopf_matrix = np.zeros((len(alpha_array), len(n1array)))
theta_saddle_matrix = np.zeros((len(alpha_array), len(n1array)))
N = 1000

i = 0
for alpha in alpha_array:
    j = 0
    for m in n1array:
        print(alpha, m)
        sizes = [m, N-m]
        n1 = sizes[0]
        n2 = sizes[1]
        f1 = n1/N
        f2 = n2/N
        f = f1/f2
        betainf = f1**2 + f2**2  #not inf (n1*(n1-1) + n2*(n2-1)) / (N*(N-1))
        solution = find_zero(alpha, f)
        zeros_psi_det = solution[0]
        zeros_psi_tr = solution[1]

        theta_saddle_matrix[i, j] = theta_bifurcation(zeros_psi_det, alpha, f, betainf)
        theta_hopf_matrix[i, j] = theta_bifurcation(zeros_psi_tr, alpha, f, betainf)

        j += 1
    i += 1


theta_stable_chimera_matrix = theta_hopf_matrix - theta_saddle_matrix
farray = n1array/(N*np.ones(len(n1array))-n1array)

fig = plt.figure(figsize=(8, 5))
plt.rc('axes', linewidth=2)
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'text.usetex': True,
    'font.family': 'serif',
    'xtick.labelsize': 15,
    'ytick.labelsize': 15
}

mat.rcParams.update(params)

ax = fig.add_subplot(111)
colorlist = ["#ffbf00", "#ff6600", "#b72114", "#661d05"]
k = 0
for angles in theta_stable_chimera_matrix:
    print(angles)
    ax.plot(farray, angles, label="$\\alpha = {}$".format( np.round(alpha_array[k], 3) ), linewidth=6, color=colorlist[k])
    k += 1
ax.set_xlabel("$f$", fontsize=35, labelpad=12)
ax.set_ylabel("$\\theta$", fontsize=35, labelpad=20, rotation=0)
ax.legend(loc=4, fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_xlim([1, 2])
plt.tight_layout()
plt.show()
"""



##################################### Plot T-B vs psi ##################################################################
"""
Beta = 0.2239
alpha = np.pi/2 - Beta
f = 1

plt.figure(figsize=(10, 6))
plt.rc('axes', linewidth=2)
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'text.usetex': True,
    'font.family': 'serif',
    'xtick.labelsize': 15,
    'ytick.labelsize': 15
}
mat.rcParams.update(params)
plt.title("$\\alpha = {}, f = {}$".format(np.round(alpha, 2), np.round(f, 2)), fontsize=25, y=1.01)
psiarr = np.linspace(-2*np.pi, 2*np.pi, 1000000)
plt.plot(psiarr, Takens_Bogdanov_function(psiarr, alpha, f), linewidth=2, label='With sizes')
plt.plot(psiarr, psiarr*0, "k")
#plt.scatter(find_zero(alpha, f)[0], 0, s=50, color='r')
legend = plt.legend(loc=1, fontsize=20)
legend.get_frame().set_linewidth(2)
plt.ylabel('Takens-Bogdanov', fontsize=25)
plt.xlabel('$\\phi$', fontsize=25)
plt.xlim([-np.pi, np.pi])
plt.ylim([-12, 12])
plt.show()


"""



##################################### Plot det vs psi ##################################################################
"""
#Beta = 0.22
alpha = 1.2# np.pi/2 - Beta
f = 2#7/3

plt.figure(figsize=(10, 6))
plt.rc('axes', linewidth=2)
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'text.usetex': True,
    'font.family': 'serif',
    'xtick.labelsize': 15,
    'ytick.labelsize': 15
}
mat.rcParams.update(params)
plt.title("$\\alpha = {}, f = {}$".format(np.round(alpha, 2), np.round(f, 2)), fontsize=25, y=1.01)
psiarr = np.linspace(-2*np.pi, 2*np.pi, 1000000)
plt.plot(psiarr, detMJ(psiarr, alpha, f), linewidth=2, label='With sizes')
#plt.plot(psiarr, detAbrams(psiarr, Beta), linewidth=2, label='No sizes')
plt.plot(psiarr, psiarr*0, "k")
plt.scatter(find_zero(alpha, f)[0], 0, s=50, color='r')
#plt.plot(psiarr, r2old(psiarr, np.pi/2-0.1))
#plt.plot(psiarr, r2(psiarr, np.pi/2-0.1, 1))
legend = plt.legend(loc=1, fontsize=20)
legend.get_frame().set_linewidth(2)
plt.ylabel('Determinant', fontsize=25)
plt.xlabel('$\\phi$', fontsize=25)
plt.xlim([-np.pi, np.pi])
plt.ylim([-12, 12])
plt.show()


"""



###################################### Plot trace vs psi ###############################################################
"""
#Beta = 0.15
alpha = 1.45 #np.pi/2 - Beta
f = (130/200)/(70/200)
print(newton(trMJ, -0.15, args=(alpha, 1.5,))   )

plt.figure(figsize=(10, 6))
plt.rc('axes', linewidth=2)
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'text.usetex': True,
    'font.family': 'serif',
    'xtick.labelsize': 15,
    'ytick.labelsize': 15
}
mat.rcParams.update(params)
#plt.title("$\\alpha = {}, f = {}$".format(np.round(alpha, 2), np.round(f, 2)), fontsize=25, y=1.01)
psiarr = np.linspace(-2*np.pi, 2*np.pi, 100000)


plt.plot(psiarr, trMJ(psiarr, alpha, f), linewidth=2, label='$f = 1.5, \\alpha = {}$'.format(np.round(alpha, 3)))
#plt.plot(psiarr, trMJ(psiarr, np.pi/2 - 0.1, f), linewidth=2, label='$f = 1, \\alpha =  {}$'.format(np.round(np.pi/2 - 0.1, 3) ))
#plt.plot(psiarr, trMJ(psiarr, np.pi/2 - 0.01, f), linewidth=2, label='$f = 1, \\alpha = {}$'.format(np.round(np.pi/2 - 0.01, 3)))
#plt.plot(psiarr, trMJ_abrams(psiarr, Beta), label='No sizes')
plt.plot(psiarr, psiarr*0, "k")
#plt.plot(psiarr, r2old(psiarr, np.pi/2-0.1))
#plt.plot(psiarr, r2(psiarr, np.pi/2-0.1, 1))
legend = plt.legend(loc=1, fontsize=20)
legend.get_frame().set_linewidth(2)
plt.ylabel('Trace', fontsize=25)
plt.xlabel('$\\phi$', fontsize=25)
plt.xlim([-np.pi-0.5, np.pi+0.5])
plt.ylim([-1, 1])
plt.show()

#print(  newton(trMJ, 0.277732335434, args=(np.pi/2-0.1, 1,))  )
#print(  newton(trMJ2, 0.45, args=(np.pi//2-0.1, 1, 1))  )
"""



############################################ Plot r vs psi #############################################################
"""
psi_array = np.linspace(0, np.pi, 1000)                                           
                                                                                  
alpha = np.pi/2 - 0.1                                                             
q = 1   # Because it doesn't affect the position of the zeros of detMJ or trMJ    
bool = 1 # r<0 for bool=0 , so we take bool = 1                                   

plt.figure(figsize=(10, 8))
plt.title("$\\alpha = {}$".format(round(alpha, 3)), fontsize=35, y=1.01)
plt.plot(psi_array, r(psi_array, alpha, 1), label="$Root\:R_{+}$")
plt.plot(psi_array, r(psi_array, alpha, 0), label="$Root\:R_{-}$")
plt.xlabel("$\\psi$", fontsize=35)
plt.ylabel("$R$", fontsize=35)
#plt.ylim([0,1])
plt.legend(loc="best", fontsize=25)
plt.show()
"""



######################################## Plot detMJ vs psi #############################################################
"""
psi_array = np.linspace(-np.pi, np.pi, 100000)

beta = 0.1
alpha = np.pi / 2 - beta
q = 1  # Because it doesn't affect the position of the zeros of detMJ or trMJ
bool = 1  # r<0 for bool=0 , so we take bool = 1


psi = psi_abrams_saddle(0.1)
print("psi = ", psi, "résultat de l'équation 16 de l'article pour beta=0.1")
print("psi = ", newton(detAbrams, -0.15, args=(beta,)), "on exclut cette solution")
print("psi = ", newton(detAbrams, -0.21, args=(beta,)), "ce qui concorde avec le résultat de l'équation 16 de l'article pour beta=0.1")


plt.figure(figsize=(10, 8))
plt.title("$\\alpha = {}$".format(round(alpha, 3)), fontsize=35, y=1.01)
#plt.plot(psi_array, detMJ(psi_array, alpha, q, bool))
plt.plot(psi_array, detAbrams(psi_array, beta))
plt.plot(psi_array, np.zeros(len(psi_array)), color='k')
plt.xlabel("$\\psi$", fontsize=35)
plt.ylabel("$R$", fontsize=35)
plt.ylim([-0.5, 0.5])
plt.xlim([-0.23, -0.15])
plt.show()
"""



########################################### Theoric chimera map ########################################################
"""
rho_array = np.linspace(0, 1, 1000)
delta_array = np.linspace(-1, 1, 1000)
psi = psi_abrams_saddle(0.1)
#psi = psi_abrams_hopf(0.1)
#print(psi)

#psi = -2.94549628646
#psi = -1.85462832071
#psi = -3.24159265359
#psi = 0.196096367127
#psi = 1.28696433288
#psi = 3.04159265359

#psi = -0.217764255024  # Abrams 2008 (que j'ai trouvé avec Newton-Raphson sur eq 15 de l'article)

alpha = np.pi/2 - 0.1
q = 1     # Because it doesn't affect the position of the zeros of detMJ or trMJ
bool = 1  # r<0 for bool=0 , so we take bool = 1


#for x0 in np.linspace(0, 1, 10):
#    print(newton(detMJ, x0, args=(alpha, q, bool)))


def R(rho, delta, psi, alpha):
   return (0.5*delta - rho)*np.cos(psi - alpha)/((rho +0.5*delta)*np.cos(alpha))

chimera_map = np.zeros((len(rho_array),len(delta_array)))
i=0
for delta in delta_array:

    j = 0  # On fixe une colonne j
    for rho in rho_array:
        probability_space_tuple = to_probability_space(rho, delta, 0.5)  # IMPORTANT : return (q, p)
        p = probability_space_tuple[1]
        q = probability_space_tuple[0]
        # print(p,q)
        if p > 1 or p < 0 or q > 1 or q < 0:
            j += 1
        else:
            chimera_map[i,j] = R(rho, delta, psi, alpha)


            j += 1
    i += 1

#levelset_map = give_levelset(chimera_map, )


plt.figure(figsize=(10, 8))

# For saddle
cdict = {
    'red': ((0.0, 1.0, 1.0), (0.999, 0.3, 0.3), (1.0, 0.1, 0.1)),
    'green': ((0.0, 1.0, 1.0), (0.999, .59, .59), (1.0, 0.25, .25)),
    'blue': ((0.0, 1.0, 1.0), (0.999, .75, .75), (1.0, 0.7, 0.7))
}

# The one I normally use
#cdict = {
#    'red': ((0.0, 1.0, 1.0), (0.6, 0.3, 0.3), (1.0, 0.1, 0.1)),
#    'green': ((0.0, 1.0, 1.0), (0.6, .59, .59), (1.0, 0.25, .25)),
#    'blue': ((0.0, 1.0, 1.0), (0.6, .75, .75), (1.0, 0.7, 0.7))
#}

# For Hopf
#cdict = {
#    'red': ((0.0, 1.0, 1.0), (0.01, 0.3, 0.3), (1.0, 0.1, 0.1)),
#    'green': ((0.0, 1.0, 1.0), (0.01, .59, .59), (1.0, 0.25, .25)),
#    'blue': ((0.0, 1.0, 1.0), (0.01, .75, .75), (1.0, 0.7, 0.7))
#}

import matplotlib as mat
cm = mat.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

# plt.title("$\\sigma = {}$".format(sigma), fontsize=50)
plt.imshow(chimera_map, cmap=cm, vmin=0, vmax=1,
           extent=[rho_array.min(), rho_array.max(), delta_array.min(), delta_array.max()],
           interpolation='nearest', origin='lower', aspect=0.5)
plt.xlabel("$\\rho$", fontsize=40)
plt.ylabel("$\\Delta$", fontsize=40)
cbar = plt.colorbar()
cbar.set_label("$R_r \\neq 1$", rotation=360, fontsize=40, labelpad=70)
fig = plt.gcf()
plt.show()

"""


############################## Plot bifurcation map Abrams 2008 ########################################################
"""
beta_array = np.linspace(0, 0.25, 1000)

plt.figure(figsize=(10, 8))
plt.plot(beta_array, delta_abrams_saddle(beta_array), 'r--')
plt.plot(beta_array, delta_abrams_hopf(beta_array), 'b')
plt.show()
"""

"""
alpha_array = np.linspace(np.pi/2 - 0.25, np.pi/2, 1000)
delta_array = np.linspace(0, 0.5, 1000)

r_map_saddle = np.zeros((len(alpha_array), len(delta_array)))
r_map_hopf = np.zeros((len(alpha_array), len(delta_array)))
i = 0  # On fixe une ligne pour itérer sur les colonnes j
for delta in delta_array:
    j = 0  # On fixe une colonne j
    for alpha in delta_array:
        r_map_saddle[i, j] = r(psi_abrams_saddle(np.pi/2 - alpha), alpha, 1)
        r_map_hopf[i, j] = r(psi_abrams_hopf(np.pi/2 - alpha), alpha, 1)
        j += 1
    i += 1



cdict = {
    'red': ((0.0, 1.0, 1.0), (0.6, 0.3, 0.3), (1.0, 0.1, 0.1)),
    'green': ((0.0, 1.0, 1.0), (0.6, .59, .59), (1.0, 0.25, .25)),
    'blue': ((0.0, 1.0, 1.0), (0.6, .75, .75), (1.0, 0.7, 0.7))
 }

import matplotlib as mat

cm = mat.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

# plt.title("$\\sigma = {}$".format(sigma), fontsize=50)
plt.imshow(r_map_saddle, cmap=cm, vmin=0, vmax=1,
           extent=[alpha_array.min(), alpha_array.max(), delta_array.min(), delta_array.max()],
           interpolation='nearest', origin='lower', aspect=0.5)
plt.xlabel("$\\alpha$", fontsize=40)
plt.ylabel("$\\Delta$", fontsize=40)
cbar = plt.colorbar()
cbar.set_label("$R_r \\neq 1$", rotation=360, fontsize=40, labelpad=70)
fig = plt.gcf()
plt.show()


# plt.title("$\\sigma = {}$".format(sigma), fontsize=50)
plt.imshow(r_map_hopf, cmap=cm, vmin=0, vmax=1,
           extent=[alpha_array.min(), alpha_array.max(), delta_array.min(), delta_array.max()],
           interpolation='nearest', origin='lower', aspect=0.5)
plt.xlabel("$\\alpha$", fontsize=40)
plt.ylabel("$\\Delta$", fontsize=40)
cbar = plt.colorbar()
cbar.set_label("$R_r \\neq 1$", rotation=360, fontsize=40, labelpad=70)
fig = plt.gcf()
plt.show()


"""





