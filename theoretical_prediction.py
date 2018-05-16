import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
from plot_r_map_good import give_levelset
from scipy.optimize import newton
from geometrySBM import to_probability_space
from scipy.signal import argrelmin
#plt.style.use('classic')

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

    return zeros_psi_det, zeros_psi_tr
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
#"""
from mpl_toolkits.mplot3d import Axes3D, proj3d

#print(newton(trMJ, -0.18, args=(1.43, 1,))   )

N = 1000   #256   # 1000
m = 600    #128   # 500  # int(N/2)  # Donc beta = 1/2
sizes = [m, N-m]
n1 = sizes[0]
n2 = sizes[1]
f1 = n1/N
f2 = n2/N
f = f1/f2
beta = (n1*(n1-1) + n2*(n2-1)) / (N*(N-1))

alpha_array = np.linspace(1.2, np.pi/2-0.05, 100)    # Y
rho_array = np.linspace(0, 1, 100)                    # X
delta_map_hopf = np.zeros((len(rho_array), len(alpha_array)))
delta_map_saddle = np.zeros((len(rho_array), len(alpha_array)))
i = 0
for alpha in alpha_array:
    print(alpha)
    j = 0
    for rho in rho_array:
        solution = find_zero(alpha, f)
        zeros_psi_det = solution[0]
        zeros_psi_tr = solution[1]

        delta_map_hopf[i, j] = find_delta(rho, zeros_psi_tr, alpha, f, beta)
        delta_map_saddle[i, j] = find_delta(rho, zeros_psi_det, alpha, f, beta)
        j += 1
    i += 1

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

cdict_hopf = {
    'red': ((0, 255 / 255, 255 / 255), (0.01, 255 / 255, 255 / 255), (1.0, 255 / 255, 255 / 255)),
    'green': ((0, 67 / 255, 67 / 255), (0.01, 147 / 255, 147 / 255), (1.0, 102 / 255, 102 / 255)),
    'blue': ((0, 0 / 255, 0 / 255), (0.01, 76 / 255, 76 / 255), (1.0, 0, 0))
}
cdict_saddle = {
    'red': ((0, 255 / 255, 255 / 255), (0.001, 255 / 255, 255 / 255), (1.0, 255 / 255, 255 / 255)),
    'green': ((0, 220 / 255, 220 / 255), (0.001, 147 / 255, 147 / 255), (1.0, 102 / 255, 102 / 255)),
    'blue': ((0, 94 / 255, 94 / 255), (0.001, 76 / 255, 76 / 255), (1.0, 0, 0))
}

cm_hopf = mat.colors.LinearSegmentedColormap('my_colormap', cdict_hopf, 1024)
cm_saddle = mat.colors.LinearSegmentedColormap('my_colormap', cdict_saddle, 1024)

ax = fig.add_subplot(111, projection='3d')
plt.title("Community sizes ratio $f = {}$".format(np.round(f, 2)), fontsize=25)
rho_array, alpha_array = np.meshgrid(rho_array, alpha_array)
ax.plot_surface(rho_array, alpha_array, delta_map_hopf,   rstride=len(rho_array)//10, cstride=len(rho_array)//10, cmap=cm_hopf, alpha = 0.9, label="Hopf")
ax.plot_surface(rho_array, alpha_array, delta_map_saddle, rstride=len(rho_array)//10, cstride=len(rho_array)//10, cmap=cm_saddle, alpha = 0.5, label="Saddle")
ax.set_xlabel("$\\rho$", fontsize=35, labelpad=12)
ax.set_ylabel("$\\alpha$", fontsize=35, labelpad=12)
ax.zaxis.set_rotate_label(False)  # disable automatic rotation
ax.set_zlabel("$\\Delta$", fontsize=35, labelpad=12, rotation=0)
#f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
#ax.legend(loc="lower left", bbox_to_anchor=f(0.6,1.55,0.4),
#          bbox_transform=ax.transData)
fake2Dline = mat.lines.Line2D([0],[0], linestyle="none", c='#ff4300', marker='s')
fake2Dline2 = mat.lines.Line2D([0],[0], linestyle="none", c='#ffdc5e', marker='s')
lgnd = ax.legend([fake2Dline, fake2Dline2], ['Hopf', 'Saddle'], numpoints=1, fontsize=25, loc=2, bbox_to_anchor=(0.1, 0.8, 0, 0))
lgnd.legendHandles[0]._legmarker.set_markersize(15)
lgnd.legendHandles[1]._legmarker.set_markersize(15)
plt.tight_layout()
#plt.plot(alpha_array, zeros_psi_det)
#plt.plot(alpha_array, zeros_psi_tr)
plt.show()
#"""





##################################### Plot det vs psi ##################################################################
"""
Beta = 0.22
alpha = np.pi/2 - Beta
f = 7/3

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
Beta = 0.15
alpha = np.pi/2 - Beta
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


plt.plot(psiarr, trMJ(psiarr, np.pi/2 - 0.15, f), linewidth=2, label='$f = 1, \\alpha = {}$'.format(np.round(np.pi/2 - 0.15, 3)))
plt.plot(psiarr, trMJ(psiarr, np.pi/2 - 0.1, f), linewidth=2, label='$f = 1, \\alpha =  {}$'.format(np.round(np.pi/2 - 0.1, 3) ))
plt.plot(psiarr, trMJ(psiarr, np.pi/2 - 0.01, f), linewidth=2, label='$f = 1, \\alpha = {}$'.format(np.round(np.pi/2 - 0.01, 3)))
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





