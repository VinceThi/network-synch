import tkinter.simpledialog
from tkinter import messagebox
import time
import json
from synchro_integration_parameters import *
from synchro_integration import integrate_sync_dynamics_SBM, generate_chimera_map
from synchro_dynamics import kuramoto_odeint
import matplotlib.pyplot as plt
#from theoretical_prediction import *
plt.style.use('classic')
import matplotlib as mat

############################################ Espace Chaos ##############################################################


plist = np.linspace(0.6203, 0.64659, 50)
i = 0
while i < len(plist):
    print(i)
    #thetaszero = give_initial_conditions('Uniform', N)
    pq = [[plist[i], 1-plist[i]],
      [1-plist[i], plist[i]]]

    with open('data/2018_05_08_16h12min10sec_chimera_repeat_thetazeros.json') as json_data:
        thetaszero = np.array(json.load(json_data))
    solutions = integrate_sync_dynamics_SBM(kuramoto_odeint, thetaszero, sigma_array, alpha, freq_distr, adjacency_mat, sizes, pq, nbfreq, nbadjmat, timelist, r12t=True)
    rt1list = solutions[4]
    rt2list = solutions[5]
    tosavelist = ["numberoftimepoints = {}\n".format(numberoftimepoints),
                 "timelist = np.linspace({},{},{})\n".format(timelist[0], timelist[-1], len(timelist)),
                 "deltat = {}".format(deltat), "\n",
                 "number of nodes : N = {}\n".format(N),
                 "size of the first community: m = {}\n".format(m),
                 "sizes = {}\n".format(sizes),
                 "adjacency_matrix_type = {}\n".format(adjacency_mat),
                 "# affinity/density matrix: pq = {}\n".format(pq),
                 "nb_SBM = {}\n".format(nbadjmat), "sigma = {}\n".format(sig), "\n",
                 "alpha = {}\n".format(alpha),
                 "freq_distr = {}\n".format(freq_distr),
                 "nbfreq = {}\n".format(nbfreq),
                 "init_cond = {}\n".format(init_cond),
                 "nbCI = {}\n".format(nbCI),
                 "rho_array = np.linspace({}, {}, {})\n".format(rho_array[0], rho_array[-1], len(rho_array)),
                 "delta_array = np.linspace({}, {}, {})\n".format(delta_array[0], delta_array[-1], len(delta_array))]

    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
    if rt2list[-1] < 0.97 or rt1list[-1] < 0.97:
        if rt2list[-1] > 0.97 or rt1list[-1] > 0.97:
            #plt.figure(figsize=(12, 8))
            #plt.plot(timelist, rt1list, label="$r_1(t)$")
            #plt.plot(timelist, rt2list, label="$r_2(t)$")
            #plt.legend(loc='best', fontsize=20)
            #plt.ylabel('$Order\:parameters$', fontsize=25)
            #plt.xlabel('$Time\:t$', fontsize=25)
            #fig = plt.gcf()
            #plt.show()
            #if messagebox.askyesno("Python", "Would you like to save the parameters, the data and the plot?"):

                #window = tkinter.Tk()
                #window.withdraw()    # hides the window
                #file = tkinter.simpledialog.askstring("File: ", "Enter your file name")


                #fig.savefig("data/{}_{}_figure.jpg".format(timestr, file))

                #f = open('data/{}_{}_parameters.txt'.format(timestr, file), 'w')
                #f.writelines(tosavelist)#[line1, line2, line3, "\n", line4, line5, line6, line7, "\n", line8, line9, "\n", line10, line11, "\n", line12, line13, "\n", line14])

                #f.close()

                #with open('data/{}_{}_rt1list.json'.format(timestr, file), 'w') as outfile:
                #    json.dump(rt1list.tolist(), outfile)
            file = ''
            with open('data/rt2list_' +str(i)+'.json', 'w') as outfile:
                json.dump(rt2list.tolist(), outfile)

                #with open('data/{}_{}_thetazeros.json'.format(timestr, file), 'w') as outfile:
                #    json.dump(thetaszero.tolist(), outfile)

    i += 1