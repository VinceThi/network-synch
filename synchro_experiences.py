# @author: Vincent Thibeault

import tkinter.simpledialog
from tkinter import messagebox
import time
import json
from synchro_parameters import *
from synchro_integration import integrate_sync_dynamics_SBM, generate_chimera_map, integrate_reduced_sync_dynamics_SBM
from synchro_dynamics import kuramoto_odeint
import matplotlib.pyplot as plt
import json
from theoretical_prediction import *
import matplotlib as mat
import seaborn as sns



############################################ Plot r_1 and r_2  #########################################################
"""
### Operation LFC (Looking For Chimeras)
for i in range(0, nbCI):
    thetas0 = np.random.uniform(-np.pi, np.pi, size=(1, N))[0]
    solutions = integrate_sync_dynamics_SBM(kuramoto_odeint, thetas0, coupling, alpha, freq_distr, adjacency_mat, sizes, pq, nbfreq, nbadjmat, timelist, r12t=True)
    rtlist = solutions[3]
    rt1list = solutions[4]
    rt2list = solutions[5]
    #dot_theta = solutions[6]

    line1 = "numberoftimepoints = {}\n".format(numberoftimepoints)
    line2 = "timelist = np.linspace({},{},{})\n".format(timelist[0], timelist[-1], len(timelist))
    line3 = "deltat = {}".format(deltat)
    line4 = "Number of nodes : N = {}\n".format(N)
    line5 = "Size of the first community: m = {}\n".format(m)
    line6 = "sizes = {}\n".format(sizes)
    line7 = "adjacency_matrix_type = {}\n".format(adjacency_mat)
    line8 = "Affinity/density matrix: pq = {}\n".format(pq)
    line9 = "nb_SBM = {}\n".format(nbadjmat)
    line10 = "sigma = {}\n".format(sig)
    line11 = "alpha = {}\n".format(alpha)
    line12 = "freq_distr = {}\n".format(freq_distr)
    line13 = "nbfreq = {}\n".format(nbfreq)
    line14 = "thetas0 = {}\n".format(thetas0)
    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
    plt.figure(figsize=(12, 8))
    plt.plot(timelist, rt1list, label="$r_1(t)$")
    plt.plot(timelist, rt2list, label="$r_2(t)$")
    plt.legend(loc='best', fontsize=20)
    plt.ylabel('$Order\:parameters$', fontsize=25)
    plt.xlabel('$Time\:t$', fontsize=25)
    fig = plt.gcf()
    plt.show()


    if rt2list[-1] < 0.97 or rt1list[-1] < 0.97:
        if rt2list[-1] > 0.97 or rt1list[-1] > 0.97:
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
            plt.plot(timelist, rt1list, label="$R_1$", color="#ff9900", linewidth=2)
            plt.plot(timelist, rt2list, label="$R_2$", color="#ff3700", linewidth=2)
            legend = plt.legend(loc='best', fontsize=20)
            legend.get_frame().set_linewidth(2)

            plt.ylabel('$Order\:parameters$', fontsize=25)
            plt.xlabel('$Time\:t$', fontsize=25)
            fig = plt.gcf()
            plt.show()
            if messagebox.askyesno("Python", "Would you like to save the parameters, the data and the plot?"):

                window = tkinter.Tk()
                window.withdraw()  # hides the window
                file = tkinter.simpledialog.askstring("File: ", "Enter your file name")


                fig.savefig("data/{}_{}_r1_r2.jpg".format(timestr, file))

                f = open('data/{}_{}.txt'.format(timestr, file), 'w')
                f.writelines([line1, line2, line3, "\n", line4, line5, line6, line7, "\n", line8, line9, "\n", line10, line11, "\n", line12, line13, "\n", line14])

                f.close()

                with open('data/{}_{}_rt1list.json'.format(timestr, file), 'w') as outfile:
                    json.dump(rt1list.tolist(), outfile)
                with open('data/{}_{}_rt2list.json'.format(timestr, file), 'w') as outfile:
                    json.dump(rt2list.tolist(), outfile)
                #with open('data/{}_{}_dot_theta.json'.format(file, timestr), 'w') as outfile:
                #    json.dump(dot_theta.tolist(), outfile)
            break
    print(i)
"""



############################################ Plot chimera SBM ##########################################################
"""

thetas0 = [0.04461508,  2.01459067,  4.70343455,  0.80495653,  1.65641329,  5.47445849,
  1.21340442,  5.92437955,  3.62638888,  1.3661011 ,  1.83424522,  4.82157024,
  4.44077113,  1.20586385,  1.80156938,  0.46500221,  6.11100881,  2.27097027,
  0.29094057,  0.15458182,  5.43926913,  6.16756314,  2.36648424,  3.89722125,
  5.05847002,  3.5176789 ,  0.4587184 ,  1.54330464,  2.47519421,  5.44555294,
  5.54672234,  3.55412502,  6.11100881,  1.22597205,  0.20610909,  0.32738669,
  5.07229441,  6.24045537,  5.97904872,  3.16138667,  0.2394133 ,  3.09477824,
  4.51743366,  4.76690106,  0.03581774,  5.36323498,  5.25703853,  3.76588954,
  2.68381683,  3.9562891 ,  3.7740585 ,  1.27372903,  0.49202261,  2.01710419,
  4.78763765,  1.96054987,  3.65529442,  2.2766257 ,  1.38495254,  3.30591438,
  0.99032904,  1.75067049,  2.51101195,  3.63455784,  4.09893167,  2.98669665,
  1.90399555,  4.16679686,  2.85725008,  1.15119467,  4.55639331,  6.2159485 ,
  1.52822349,  0.04335831,  3.84695074,  0.33429889,  1.83675874,  1.61054145,
  3.48123278,  0.58879334,  5.61144562,  4.5633055 ,  5.8753658 ,  4.93467889,
  4.10333034,  0.37263015,  1.3290266 ,  5.85714274,  3.31722525,  5.46126248,
  4.61671792,  4.22712147,  4.54194053,  3.59936848,  0.07540576,  5.89861591,
  2.39601816,  0.78421995,  4.07945185,  0.56302971,  0.98027494,  4.59723809,
  2.94019642,  5.3626066 ,  0.46500221,  3.60125362,  4.50989308,  2.41738313,
  5.38020128,  1.67714987,  2.51792415,  1.10909312,  6.18515781,  2.39224787,
  4.05243145,  0.85397028,  1.4704124 ,  4.61169087,  2.3206124 ,  0.8432878 ,
  3.93303899,  4.33771659,  0.94257205,  1.55650065,  1.06133613,  0.29910953,
  3.22988024,  1.69285941,  1.28001285,  3.96068777,  6.26684739,  1.58100752,
  0.82443636,  5.11816625,  5.43298532,  6.03246114,  4.06060041,  0.52344168,
  2.12455741,  5.19671392,  4.91959774,  2.36648424,  4.46025096,  0.94822749,
  1.81288025,  6.07393431,  1.18198535,  4.75056315,  6.15311036,  0.62146917,
  0.77479423,  5.74968953,  3.62010507,  4.45585229,  1.43522305,  2.98983855,
  2.89872325,  3.3894891 ,  0.36571796,  2.57259332,  2.29233523,  1.62122393,
  1.50057471,  6.13740083,  5.64789174,  3.77280174,  1.41574322,  0.70441551,
  2.20499022,  2.06674632,  4.56016359,  0.66985454,  0.43106962,  2.56882304,
  3.57737513,  1.90588069,  4.56204874,  1.46161506,  3.95063367,  2.94585186,
  4.08008023,  2.03595564,  2.13272637,  6.19521192,  5.93694717,  4.13600617,
  2.05292193,  3.57988866,  4.00718799,  0.54669179,  4.54194053,  4.42443322,
  5.06789574,  4.7995769 ,  1.59169001,  5.28028864,  1.89394145,  6.27375959,
  2.2125308 ,  4.59346781,  0.2733459 ,  4.19821593,  5.42732988,  2.6241206 ,
  4.42191969,  4.62991393,  4.42066293,  2.51792415,  1.39374988,  5.32301858,
  4.18250639,  4.957929  ,  5.66422966,  1.78208956,  2.43372104,  2.61029621,
  3.52082081,  1.00478181,  2.37402481,  1.67212282,  6.27941502,  5.95077156,
  5.70884474,  5.0792066 ,  5.48639773,  1.38055387,  0.69624656,  4.55765007,
  2.94270995,  1.30074943,  0.31984612,  3.62701726,  4.26670949,  5.37454585,
  4.4250616 ,  4.78952279,  4.38735872,  2.76739155,  4.59723809,  6.01046779,
  5.05344297,  2.63605984,  4.01095828,  3.14379199,  1.31645897,  0.53600931,
  2.25588911,  4.33017601,  4.86744208,  5.7672842 ,  2.084341  ,  6.01737999,
  4.85298931,  0.44929268,  2.50347137,  4.01410018]
solutions = integrate_sync_dynamics_SBM(kuramoto_odeint, thetas0, coupling, alpha, freq_distr, adjacency_mat, sizes, pq, nbfreq, nbadjmat, timelist, r12t=True)
rtlist = solutions[3]
rt1list = solutions[4]
rt2list = solutions[5]
#dot_theta = solutions[6]

line1 = "numberoftimepoints = {}\n".format(numberoftimepoints)
line2 = "timelist = np.linspace({},{},{})\n".format(timelist[0], timelist[-1], len(timelist))
line3 = "deltat = {}".format(deltat)
line4 = "Number of nodes : N = {}\n".format(N)
line5 = "Size of the first community: m = {}\n".format(m)
line6 = "sizes = {}\n".format(sizes)
line7 = "adjacency_matrix_type = {}\n".format(adjacency_mat)
line8 = "Affinity/density matrix: pq = {}\n".format(pq)
line9 = "nb_SBM = {}\n".format(nbadjmat)
line10 = "sigma = {}\n".format(sig)
line11 = "alpha = {}\n".format(alpha)
line12 = "freq_distr = {}\n".format(freq_distr)
line13 = "nbfreq = {}\n".format(nbfreq)
line14 = "thetas0 = {}\n".format(thetas0)
timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")
#plt.figure(figsize=(12, 8))
#plt.plot(timelist, rt1list, label="$r_1(t)$")
#plt.plot(timelist, rt2list, label="$r_2(t)$")
#plt.legend(loc='best', fontsize=20)
#plt.ylabel('$Order\:parameters$', fontsize=25)
#plt.xlabel('$Time\:t$', fontsize=25)
#fig = plt.gcf()
#plt.show()


if rt2list[-1] < 0.9 or rt1list[-1] < 0.9:
    if rt2list[-1] > 0.9 or rt1list[-1] > 0.9:
        plt.figure(figsize=(12, 8))
        plt.plot(timelist, rt1list, label="$r_1(t)$")
        plt.plot(timelist, rt2list, label="$r_2(t)$")
        plt.legend(loc='best', fontsize=20)
        plt.ylabel('$Order\:parameters$', fontsize=25)
        plt.xlabel('$Time\:t$', fontsize=25)
        fig = plt.gcf()
        plt.show()
        if messagebox.askyesno("Python", "Would you like to save the parameters, the data and the plot?"):

            window = tkinter.Tk()
            window.withdraw()  # hides the window
            file = tkinter.simpledialog.askstring("File: ", "Enter your file name")


            fig.savefig("data/{}_{}_r1_r2.jpg".format(timestr, file))

            f = open('data/{}_{}.txt'.format(timestr, file), 'w')
            f.writelines([line1, line2, line3, "\n", line4, line5, line6, line7, "\n", line8, line9, "\n", line10, line11, "\n", line12, line13, "\n", line14])

            f.close()

            with open('data/{}_{}_rt1list.json'.format(timestr, file), 'w') as outfile:
                json.dump(rt1list.tolist(), outfile)
            with open('data/{}_{}_rt2list.json'.format(timestr, file), 'w') as outfile:
                json.dump(rt2list.tolist(), outfile)
            #with open('data/{}_{}_dot_theta.json'.format(file, timestr), 'w') as outfile:
            #    json.dump(dot_theta.tolist(), outfile)



"""



### Reproduce Abrams 2008 results (approximatively)
"""

thetas01 = [5.57160746,  5.53466196,  3.0534701,   1.89842059,  5.45235141,  2.61961181,
  3.90604404,  4.02467176,  4.69471734, 1.25721512,  5.40397039,  3.91289278,
  2.67616105,  3.63259708,  0.25723618, 2.77977181,  2.9670126 ,  1.68610964,
  5.37544445,  5.81074788,  5.13058627, 4.86914031,  0.19723116,  2.85780975,
  4.50936152,  2.33749397,  5.51468123, 4.87140228,  3.96397559,  2.2394753 ,
  0.68682185,  0.61110871,  3.2896574 , 2.10268899,  5.99019745,  1.40926973,
  0.50592714,  0.80023448,  2.56664403, 4.63188486,  6.11014465,  6.11077298,
  4.22114893,  1.75321473,  2.10092968, 3.87016669,  1.65607571,  4.7223008 ,
  6.18862242,  2.4681227 ,  0.93469599, 2.85491946,  3.35984128,  2.87546568,
  3.12899474,  5.03294459,  5.3393786 , 0.56762864,  6.08344085,  1.98462676,
  5.61722384,  4.94070651,  5.80861158, 2.00605263,  2.14327877,  4.09736894,
  4.30025503,  0.80746022,  5.96261399, 3.57749299,  5.46303293,  5.2931339 ,
  3.83485484,  0.05139697,  2.53032686, 4.12872235,  5.22797661,  2.1731242 ,
  4.97696085,  1.85267855,  2.17111356, 5.37971706,  2.91643246,  3.16242162,
  3.27552009,  3.14501702,  2.31437162, 4.83753558,  1.37157024,  0.08406986,
  1.28253661,  4.64734165,  3.95599586, 3.79363673,  5.49589432,  5.51254492,
  2.6429855 ,  2.47503427,  0.85873152, 3.23442764,  1.09454183,  6.22487676,
  1.70571337,  3.64698572,  4.85638532, 2.22401851,  3.31523022,  3.2130646 ,
  3.564738  ,  3.552737  ,  0.71006987, 2.83927417,  5.0641095 ,  5.41930152,
  2.42508245,  4.19268582,  4.04817111, 4.05703049,  3.29317601,  2.98001893,
  2.38386434,  4.37735048,  3.23562146, 1.93718823,  2.75143436,  1.45143032,
  3.10587239,  1.09466749,  4.5847605 , 2.57939903,  3.07590129,  5.15672458,
  2.50890098,  3.45465549,  2.83965116, 4.29189831,  2.50978064,  2.82909531,
  5.16885125,  1.31106256,  6.0179694 , 6.18378432,  0.9882921 ,  6.16883019,
  6.11322345,  0.90485056,  2.96996573, 5.86911925,  3.19905296,  6.08966126,
  0.43090516,  4.2557068 ,  2.30400426, 4.97708652,  5.68451742,  0.84189242,
  4.78664127,  4.60593505,  1.73920309, 5.54220185,  3.82882292,  3.47469905,
  0.81456029,  0.86243864,  0.02136304, 1.64470303,  4.82101063,  0.90497623,
  5.4678082 ,  4.23252161,  1.89741527, 5.72158859,  5.27629479,  1.35806125,
  0.81889573,  4.92738602,  0.22054201, 0.23028104,  1.37640834,  4.88761306,
  1.8293677 ,  3.39961424,  2.30507241, 1.51244066,  5.89538323,  5.29702951,
  2.78373026,  2.66629635,  1.69069641, 4.45677074,  5.38097371,  1.36817728,
  3.0604445 ,  5.41440059,  1.23453259, 4.2232224 ,  2.44456052,  2.64141469,
  1.33091762,  1.60945401,  0.89071326, 5.09508592,  2.83091745,  4.88817856,
  2.92133339,  3.27413777,  0.10895152,  0.09783017,  1.60197695,  1.12671206,
  2.2346372 ,  5.80779475,  1.40311214,  0.77547849,  4.29183547,  4.29862138,
  1.76735204,  0.11272147,  1.40443162,  4.50848187,  1.15197071,  5.47968354,
  1.78557346,  2.72290841,  4.62088918,  6.23078302,  4.21379753,  0.56574366,
  2.15465145,  5.58084383,  0.89674517,  0.89976113,  4.86166325,  1.46142068,
  0.91276746,  5.43720878,  5.33309536,  1.56710492,  0.70391229,  4.19306281,
  4.52098553,  1.6966655 ,  0.69122013,  0.3245926 ,  5.24223959,  1.76804319,
  2.31305214,  5.01007357,  3.73715033,  1.53204439,  0.00955054,  1.95170254,
  5.74866939,  4.31866494,  1.31206788,  1.33921151]
solutions_almoststable = integrate_sync_dynamics_SBM(kuramoto_odeint, thetas01, coupling, alpha, freq_distr, adjacency_mat, sizes, pq1, nbfreq, nbSBM, timelist)
rt1list_almoststable = solutions_almoststable[4]

thetas02 = [ 5.15798123,  5.2832692,   2.55897847,  5.37977989,  6.10241626,  3.44032969,
  1.57012088,  1.24213532,  4.38143459,  1.17100896,  5.12537117,  3.32421526,
  4.80040158,  4.78268282,  1.11822967,  0.41714484,  4.09994508,  0.33935823,
  1.02517477,  5.98944346,  1.15636899,  6.26559221,  5.70211052,  5.37337098,
  5.33359802,  3.39176018,  3.75977002,  2.0996102 ,  3.49367446,  5.10758958,
  4.51419963,  3.07445615,  3.35073057,  5.08679203,  4.63628314,  4.79524932,
  6.07602662,  2.14755138,  5.42809807,  6.17618159,  5.46686571,  3.16562608,
  5.5631879 ,  3.67972144,  4.27424238,  4.01053446,  1.38683853,  1.14003254,
  4.36673179,  3.40432667,  0.37800021,  5.21377647,  1.17396208,  2.446257  ,
  0.59257313,  2.65863079,  0.90384524,  5.06875911,  1.01870302,  4.77954119,
  3.71723243,  6.09537902,  6.04341656,  4.17509272,  3.10405025,  4.08706442,
  4.05256939,  5.76645098,  3.96893935,  5.12518268,  6.25465936,  4.05690483,
  1.95258219,  6.00043914,  0.55518781,  0.14482887,  0.56417285,  1.96785048,
  0.66294551,  2.73007132,  5.57738804,  5.851212  ,  1.91796149,  2.17789947,
  6.27024182,  0.8016168 ,  1.69886463,  2.26749859,  2.75859726,  5.61357955,
  3.09676168,  2.74923522,  5.23840681,  2.23526553,  4.67115516,  4.86637568,
  1.77552026,  4.53568833,  1.73499331,  1.2068863 ,  1.85060507,  6.16047347,
  4.3561131 ,  6.03342619,  0.13798013,  6.09041525,  5.98272038,  2.30431842,
  1.36403034,  5.72303374,  3.50423032,  2.34321173,  0.39144636,  2.56878034,
  5.06015106,  0.13245087,  3.90227409,  5.01830462,  3.8796544 ,  5.33761929,
  2.11814578,  5.76890145,  0.60840692,  4.03629577,  3.73143257,  1.67297765,
  4.73976823,  4.78563595,  5.20102148,  2.02213775,  0.67884213,  0.71616462,
  5.85045801,  2.3182044 ,  4.1402207 ,  1.45670825,  4.8406772 ,  2.22659464,
  1.13538294,  3.25151808,  2.24173727,  2.25612591,  2.26291182,  4.07795371,
  5.23658466,  1.04936527,  1.52821161,  2.69815242,  0.64824271,  0.08048841,
  0.07759811,  2.12153873,  2.1707994 ,  6.25271155,  3.76159216,  0.80708322,
  5.88878582,  3.14143557,  6.04373072,  0.12528797,  4.75491086,  2.28546868,
  3.48676289,  1.77156181,  4.00670167,  2.16954275,  2.01975011,  2.46089697,
  3.38616809,  3.35217571,  5.39705882,  1.930151  ,  5.31675891,  4.3365722 ,
  5.26994871,  0.73614535,  0.99953911,  5.95620507,  5.75696328,  3.65446278,
  5.02534186,  0.86658558,  3.32471792,  2.44053924,  1.03906074,  5.67295625,
  5.97122204,  5.22169337,  5.25128747,  1.18068516,  2.6867169 ,  1.10283571,
  0.8385623 ,  6.27300645,  2.94326193,  1.90438968,  5.42200332,  0.96906536,
  1.82540925,  2.47076167,  3.93758594,  5.89959301,  0.3961588 ,  0.30134458,
  0.70485478,  5.63770723,  5.84675089,  5.87552817,  3.68606752,  4.24640759,
  5.21792342,  1.82138797,  3.70648808,  6.18233917,  5.38392683,  4.39280727,
  5.27415849,  5.1577299 ,  4.56214081,  5.41370943,  2.99183143,  2.5938505 ,
  4.69270671,  3.71767226,  5.2954587 ,  2.8372007 ,  1.04553249,  1.03899791,
  0.84698185,  3.28375114,  5.00341333,  0.20973482,  0.55625596,  0.1797009 ,
  1.34235313,  5.6490799 ,  2.41974169,  3.98496164,  3.30888414,  4.29472577,
  0.00628325,  4.21970379,  0.63573905,  0.55223468,  0.37240812,  5.9492935 ,
  4.70835199,  3.02613797,  4.53933262,  5.36513992,  4.32972346,  3.56530349,
  2.52184447,  0.21086581,  4.06501022,  0.51510068]
solutions_breathing = integrate_sync_dynamics_SBM(kuramoto_odeint, thetas02, coupling, alpha, freq_distr, adjacency_mat, sizes, pq2, nbfreq, nbSBM, timelist)
rt2list_breathing = solutions_breathing[5]

thetas03 = [4.73530713,  2.77223191,  0.65314364,  0.10505591,  2.0332591,   0.23034388,
  2.15741608,  0.26201145,  3.87418797,  5.84335794,  2.34390289,  3.16330128,
  2.46441559,  2.24399924,  2.48936008,  6.03782447,  4.50678539,  3.3507934 ,
  0.74688971,  5.75840842,  0.08702299,  1.51595928,  1.18828789,  5.30563756,
  3.44755542,  4.89603262,  5.63952937,  0.72923378,  1.96973546,  2.63720491,
  5.07925213,  3.26552972,  6.01463928,  1.09523298,  0.65955256,  0.70133616,
  6.09198607,  2.63400045,  0.02500733,  3.24001974,  5.56733485,  1.20971376,
  0.98659562,  2.12304671,  5.49250136,  1.78544779,  5.48653228,  4.62591578,
  2.9203909 ,  5.52121581,  0.32515809,  5.96820608,  0.97170432,  5.64066035,
  1.26689132,  5.74584193,  5.91417014,  5.86779977,  0.37699489,  2.83292809,
  5.5016749 ,  0.40363586,  3.04624436,  2.04720791,  5.09458326,  4.32494819,
  2.16225418,  1.70313724,  6.12057485,  0.74274276,  1.98594624,  4.62384231,
  3.92307164,  6.21695987,  1.11276325,  2.38040856,  5.70167069,  3.17938639,
  4.24804123,  4.26192721,  5.84662523,  5.00372749,  6.11202963,  5.04884121,
  6.24008222,  2.62865969,  4.21574534,  5.42992021,  6.03336336,  1.89427365,
  0.8259958 ,  2.56589004,  1.13525727,  6.10851101,  1.80304089,  5.32103152,
  2.04136449,  1.08901257,  1.7063417 ,  0.00879655,  5.42445378,  5.71631066,
  0.33558828,  3.54262097,  4.13167548,  2.05286283,  3.57818415,  4.75497369,
  4.57514713,  5.93534469,  2.73346427,  4.71287593,  2.49325569,  4.240627  ,
  5.35125394,  4.08228915,  4.2878142 ,  2.17614016,  6.02984474,  4.80737598,
  1.16849566,  0.63793818,  1.70728418,  3.7173581 ,  0.76957223,  3.69970217,
  6.22405994,  0.47815518,  5.75300483,  1.91356322,  2.5398774 ,  0.80582657,
  0.04165794,  4.44627771,  0.91798255,  6.27947819,  5.91448431,  1.41969992,
  3.4513882 ,  5.03445257,  1.10604017,  2.05418231,  0.34539015,  5.31449694,
  6.23084585,  2.19737754,  3.25063843,  1.20217387,  1.28668355,  2.20152448,
  2.54408717,  5.81916743,  4.15272436,  3.19968128,  1.8463953 ,  6.03455718,
  2.48414498,  2.52624275,  1.72160999,  1.11414556,  1.35592495,  1.90979327,
  4.20399567,  0.90830635,  2.85862657,  5.56934549,  2.09294996,  3.46854147,
  4.64702749,  5.14315276,  1.01461891,  1.30729261,  3.40552049,  5.47541093,
  1.42811947,  2.21534763,  3.31120894,  0.87613612,  5.82846664,  6.06352295,
  4.978029  ,  4.30905157,  5.82557635,  1.07468676,  1.18332412,  3.88763412,
  0.22079334,  2.77952048,  3.72376701,  0.56480118,  4.5037066 ,  4.56433995,
  1.55070564,  1.979286  ,  1.58388119,  6.09374538,  3.18698912,  1.13707942,
  2.14899653,  5.44864429,  5.26108933,  5.07868664,  1.32406888,  0.5704561 ,
  1.27889233,  3.87475346,  4.10270971,  0.84867833,  5.02006393,  2.68985853,
  6.10738002,  6.01419945,  4.72418578,  0.85615539,  0.54808774,  1.06067512,
  4.01637788,  0.41965814,  3.67412935,  2.38034573,  5.98818681,  1.97821784,
  4.49742335,  1.37571718,  1.29868456,  4.02090181,  3.32220462,  3.98540146,
  5.46077096,  3.40369835,  0.65509145,  1.46431098,  4.4868675 ,  4.12916218,
  1.65148894,  1.14656712,  5.77688117,  5.16338482,  2.56746085,  5.47591359,
  5.01070189,  2.1808526 ,  4.56785857,  5.26674426,  1.83841557,  1.20098005,
  0.4300255 ,  3.54764756,  1.82000566,  1.2293175 ,  0.20495955,  1.3204246 ,
  0.27658858,  3.03637966,  0.41959531,  6.07659211]
solutions_longbreathing = integrate_sync_dynamics_SBM(kuramoto_odeint, thetas03, coupling, alpha, freq_distr, adjacency_mat, sizes, pq3, nbfreq, nbSBM, timelist)
rt1list_longbreathing = solutions_longbreathing[4]

plt.figure(figsize=(12, 8))

plt.subplot(311)
plt.plot(timelist, rt1list_almoststable)
plt.ylim([0,1])
#plt.xlim([10, 20])

plt.subplot(312)
plt.plot(timelist, rt2list_breathing)
plt.ylabel('$R_{r}$', fontsize=35)
plt.ylim([0,1])
#plt.xlim([10, 20])

plt.subplot(313)
plt.plot(timelist, rt1list_longbreathing)
plt.xlabel('$t$', fontsize=35)
plt.ylim([0,1])
#plt.xlim([10, 20])

plt.show()
"""



##################################################### Chimera map ######################################################
"""
density_map_bool = False
solutions = generate_chimera_map(rho_array, delta_array, beta, sizes, kuramoto_odeint, coupling, alpha, init_cond, freq_distr, adjacency_mat,  nbCI, nbfreq, nbadjmat, timelist, density_map_bool=density_map_bool)
chimera_map = solutions[0]
density_map = solutions[1]
tosavelist = ["numberoftimepoints = {}\n".format(numberoftimepoints),
             "timelist = np.linspace({},{},{})\n".format(timelist[0], timelist[-1], len(timelist)),
             "deltat = {}".format(deltat), "\n",
             "number of nodes : N = {}\n".format(N),
             "size of the first community: m = {}\n".format(m),
             "sizes = {}\n".format(sizes),
             "adjacency_matrix_type = {}\n".format(adjacency_mat),
             "# affinity/density matrix: pq = there are multiple matrix (we're trying to have all combinations)\n",
             "nb_SBM = {}\n".format(nbadjmat),
             "sigma = {}\n".format(sig), "\n",
             "alpha = {}\n".format(alpha),
             "freq_distr = {}\n".format(freq_distr),
             "nbfreq = {}\n".format(nbfreq),
             "init_cond = {}\n".format(init_cond),
             "nbCI = {}\n".format(nbCI),
             "rho_array = np.linspace({}, {}, {})\n".format(rho_array[0], rho_array[-1], len(rho_array)),
             "delta_array = np.linspace({}, {}, {})\n".format(delta_array[0], delta_array[-1], len(delta_array)),
             "Total simulation time = {} minutes".format(solutions[2])]
#line1 = "numberoftimepoints = {}\n".format(numberoftimepoints)
#line2 = "timelist = np.linspace({},{},{})\n".format(timelist[0], timelist[-1], len(timelist))
#line3 = "deltat = {}".format(deltat)
#line4 = "Number of nodes : N = {}\n".format(N)
#line5 = "Size of the first community: m = {}\n".format(m)
#line6 = "sizes = {}\n".format(sizes)
#line7 = "adjacency_matrix_type = {}\n".format(adjacency_mat)
#line8 = "Affinity/density matrix: pq = there are multiple matrix (trying to have all combinations)\n"
#line9 = "nb_SBM = {}\n".format(nbadjmat)
#line10 = "sigma = {}\n".format(sig)
#line11 = "alpha = {}\n".format(alpha)
#line12 = "freq_distr = {}\n".format(freq_distr)
#line13 = "nbfreq = {}\n".format(nbfreq)
#line14 = "nbCI = {}".format(nbCI)
#line15 = "rho_array = np.linspace({}, {}, {})\n".format(rho_array[0], rho_array[-1], len(rho_array))
#line16 = "delta_array = np.linspace({}, {}, {})\n".format(delta_array[0], delta_array[-1], len(delta_array))
timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

plt.figure(figsize=(10, 8))
plt.rc('axes', linewidth=2)
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'text.usetex': True,
    'font.family': 'serif',
    'xtick.labelsize': 15,
    'ytick.labelsize': 15
}
mat.rcParams.update(params)
#cdict = {
#    'red': ((0.0, 1.0, 1.0), (0.6, 0.3, 0.3), (1.0, 0.1, 0.1)),
#    'green': ((0.0, 1.0, 1.0), (0.6, .59, .59), (1.0, 0.25, .25)),
#    'blue': ((0.0, 1.0, 1.0), (0.6, .75, .75), (1.0, 0.7, 0.7))
#}
cdict = {                                                                                           
    'red': ((0, 255 / 255, 255 / 255), (0.9, 255 / 255, 255 / 255), (1.0, 255 / 255, 255 / 255)),   
    'green': ((0, 255 / 255, 255 / 255), (0.9, 147 / 255, 147 / 255), (1.0, 102 / 255, 102 / 255)), 
    'blue': ((0, 255 / 255, 255 / 255), (0.9, 76 / 255, 76 / 255), (1.0, 0, 0))                     
}                                                                                                  

cm = mat.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

#plt.title("$\\sigma = {}$".format(sigma), fontsize=50)
plt.imshow(chimera_map, cmap=cm, vmin=0, vmax=1,
           extent=[rho_array.min(), rho_array.max(), delta_array.min(), delta_array.max()],
           interpolation='nearest', origin='lower', aspect=0.5)
plt.xlabel("$\\rho$", fontsize=40)
ylab = plt.ylabel("$\\Delta$", fontsize=40, labelpad=20)
ylab.set_rotation(0)
cbar = plt.colorbar(pad=0.05)
cbar.set_label("$R_r$", rotation=360, fontsize=40, labelpad=30)
fig = plt.gcf()
plt.show()
if messagebox.askyesno("Python", "Would you like to save the parameters, the data and the plot?"):

    window = tkinter.Tk()
    window.withdraw()    # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")


    fig.savefig("data/{}_{}_figure.jpg".format(timestr, file))

    f = open('data/{}_{}_parameters.txt'.format(timestr, file), 'w')
    f.writelines(tosavelist)#[line1, line2, line3, "\n", line4, line5, line6, line7, "\n", line8, line9, "\n", line10, line11, "\n", line12, line13, "\n", line14])

    f.close()

    with open('data/{}_{}_chimeramatrix.json'.format(timestr, file), 'w') as outfile:
        json.dump(chimera_map.tolist(), outfile)
    if density_map_bool == True:
        with open('data/{}_{}_densitymatrix.json'.format(timestr, file), 'w') as outfile:
            json.dump(density_map.tolist(), outfile)
"""



################################################# Plot filtered (old)chimera map #######################################
"""
import json
with open('data/2018_04_27_20h35min15sec_chimera_map_100x100_alpha1_47_matrix.json') as json_data:
    chimera_map = np.array(json.load(json_data))

beta = 0.1
alpha = np.pi/2 - beta

filtered_chimera_map = np.zeros((len(chimera_map[0]), len(chimera_map[0])))
for i in range(0, len(chimera_map[0])):
    for j in range(0, len(chimera_map[0])):
        if chimera_map[i,j] < 0.85:
            filtered_chimera_map[i,j] = chimera_map[i,j]

rho_array = np.linspace(0, 1, 50)
delta_array = np.linspace(0, 1, 50)
plt.figure(figsize=(10, 8))
cdict = {
    'red': ((0.0, 1.0, 1.0), (0.6, 0.3, 0.3), (1.0, 0.1, 0.1)),
    'green': ((0.0, 1.0, 1.0), (0.6, .59, .59), (1.0, 0.25, .25)),
    'blue': ((0.0, 1.0, 1.0), (0.6, .75, .75), (1.0, 0.7, 0.7))
}

cm = mat.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

#plt.title("$\\sigma = {}$".format(sigma), fontsize=50)
plt.imshow(filtered_chimera_map, cmap=cm, vmin=0, vmax=0.8,
           extent=[rho_array.min(), rho_array.max(), delta_array.min(), delta_array.max()],
           interpolation='nearest', origin='lower', aspect=0.5)
plt.plot(rho_array, delta(rho_array, psi_abrams_saddle(beta), alpha), color="k", label='$Saddle$')
plt.plot(rho_array, delta(rho_array, psi_abrams_hopf(beta), alpha), color="r", label='$Hopf$')
plt.plot(np.linspace(0, 1, 10000), 0.72*np.linspace(0, 1, 10000), color="y", label='$Homoclinic$') ## La pente est approximative ici mais tiré de Abrams 2008 , expérimentalement

#plt.plot(np.linspace(0, 1, 10000), 0.356727*np.linspace(0, 1, 10000), color="k", label='$Saddle$')
#plt.plot(np.linspace(0, 1, 10000), 0.5388*np.linspace(0, 1, 10000), color="r", label='$Hopf$')
#plt.plot(np.linspace(0, 1, 10000), 0.891*np.linspace(0, 1, 10000), color="g", label='Hopf')
#plt.plot(np.linspace(0, 1, 10000), 0.2653*np.linspace(0, 1, 10000), color="y", label='$Hopf$')

plt.xlabel("$\\rho$", fontsize=40)
plt.ylabel("$\\Delta$", fontsize=40)
cbar = plt.colorbar()
cbar.set_label("$R_r \\neq 1$", rotation=360, fontsize=40, labelpad=70)
plt.legend(loc=4)
plt.show()

"""



###################################################### Plot density map ################################################
"""

with open('data/2018_05_06_20h38min34sec_chimera_map_densitymatrix.json') as json_data:
    density_map = np.array(json.load(json_data))

Beta = 0.1
alpha = np.pi / 2 - Beta

rho_array = np.linspace(0, 1, 50)
delta_array = np.linspace(0, 0.55, 50)
i = 0
for delta in delta_array:
    j = 0  # On fixe une colonne j
    for rho in rho_array:
        probability_space_tuple = to_probability_space(rho, delta, beta)  # IMPORTANT : return (q, p)
        p = probability_space_tuple[1]
        q = probability_space_tuple[0]
        # print(p,q)
        if p > 1 or p < 0 or q > 1 or q < 0:
            density_map[i, j] = np.nan
            j += 1
        else:
            j += 1
    i += 1


plt.figure(figsize=(14*rho_array[-1], 14*delta_array[-1]/2))
plt.rc('axes', linewidth=2)

labelfontsize = 30
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'text.usetex': True,
    'font.family': 'serif',
    'xtick.labelsize': 15,
    'ytick.labelsize': 15
}
mat.rcParams.update(params)

cdict = {
    'red':   ((0, 255/255, 255/255),(0.05, 255/255,  255/255), (1.0, 255/255, 255/255)),
    'green': ((0, 102/255, 102/255), (0.05, 147/255, 147/255), (1.0, 255/255, 255/255)),
    'blue':  ((0, 0, 0 ),    (0.05, 76/255, 76/255),   (1.0, 255/255, 255/255))
}
cm = mat.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
plt.imshow(density_map, cmap=cm, vmin=0, vmax=1,
           extent=[rho_array.min(), rho_array.max(), delta_array.min(), delta_array.max()],                                                                                               
           interpolation='nearest', origin='lower', aspect=0.5)
plt.plot(np.linspace(0, 0.73, 10000), 0.72 * np.linspace(0, 0.73, 10000), color="#ad2727", label='$Homoclinic$', linewidth=5)  ## La pente est approximative ici mais tirée de Abrams 2008 , expérimentalement
plt.plot(rho_array[0:39], deltafun(rho_array[0:39], psi_abrams_hopf(beta), alpha), color="#ff4300", label='$Hopf$', linewidth=5)
plt.plot(rho_array[0:42], deltafun(rho_array[0:42], psi_abrams_saddle(beta), alpha), color="#ffdc5e", label='$Saddle$', linewidth=5)

# Ici je triche sur les ordonnées à l'origine pour cacher les trous blancs
plt.plot(np.linspace(0, 0.5, 10000), 2*np.linspace(0, 0.5, 10000) - 0.03, color="k", linewidth=18)
plt.plot(np.linspace(0.5, 1.0, 10000), -2*np.linspace(0, 0.5, 10000) + 0.97, color="k", linewidth=18)



#plt.plot(rho_array, 2*np.sqrt(rho_array/256), color="g", label='$KS-bound$')
plt.xlabel("\\rho", fontsize=labelfontsize)
ylab = plt.ylabel("\\Delta", fontsize=labelfontsize, labelpad=20, position=(-0.05, 0.45))
plt.rc('axes', linewidth=2)
plt.xlim([0, 1])
plt.ylim([0, 0.55])
ylab.set_rotation(0)
cbar = plt.colorbar(pad=0.02)
cbar.set_label("$\\frac{n_{chim}}{n_{CI}}$", rotation=360, fontsize=labelfontsize, labelpad=40)
plt.legend(loc=2, fontsize=15)
plt.show()
"""

#### colormap old
#cdict = {
#    'red': ((0.0, 1.0, 1.0), (0.6, 0.3, 0.3), (1.0, 0.1, 0.1)),
#    'green': ((0.0, 1.0, 1.0), (0.6, .59, .59), (1.0, 0.25, .25)),
#    'blue': ((0.0, 1.0, 1.0), (0.6, .75, .75), (1.0, 0.7, 0.7))
#}
#params = {
#    'text.latex.preamble': ['\\usepackage{gensymb}'],
#    'image.origin': 'lower',
#    'image.interpolation': 'nearest',
#    'image.extent': [rho_array.min(), rho_array.max(), delta_array.min(), delta_array.max()],
#    'axes.grid': False,
#    'axes.labelsize': 30, # fontsize for x and y labels (was 10)
#    'axes.titlesize': 30,
#    'font.size': 30, # was 10
#    'legend.fontsize': 20, # was 10
#    'xtick.labelsize': 20,
#    'ytick.labelsize': 20,
#    'text.usetex': True,
#    'figure.figsize': [15*rho_array[-1], 15*delta_array[-1]/2],
#    'font.family': 'serif',
#}
#mat.rcParams.update(params)
#fig, ax = plt.subplots()
#img1 = ax.imshow(density_map, cmap=cm)
#fig.colorbar(img1, ax=ax)
#plt.show()


#cdict = {
#    'red': ((0.0, 1.0, 1.0), (0.6, 0.3, 0.3), (1.0, 0.1, 0.1)),
#    'green': ((0.0, 1.0, 1.0), (0.6, .59, .59), (1.0, 0.25, .25)),
#    'blue': ((0.0, 1.0, 1.0), (0.6, .75, .75), (1.0, 0.7, 0.7))
#}
#cm = mat.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
#sns.set(font_scale=1.5, rc={'text.usetex' : True})
#sns.set_style("white")
#ax = sns.heatmap(density_map, cmap=cm, vmin=0, vmax=1)#, cbar_kws={'label': '$\\frac{n_{chim}}{n_{CI}}$', 'orientation':'vertical'})
#ax.invert_yaxis()
#ax.set_xlabel("\\rho", fontsize=40)
#ax.set_ylabel("\\Delta", fontsize=40)
#plt.yticks(rotation=0)
#ax.collections[0].colorbar.set_label("$\\frac{n_{chim}}{n_{CI}}$", size=40)
#ax.collections[0].colorbar.set_rotation(90)

#plt.imshow(density_map, cmap=cm, vmin=0, vmax=1,
#           extent=[rho_array.min(), rho_array.max(), delta_array.min(), delta_array.max()],
#           interpolation='nearest', origin='lower', aspect=0.5)
#plt.plot(rho_array, delta(rho_array, psi_abrams_saddle(beta), alpha), color="k", label='$Saddle$')
#plt.plot(rho_array, delta(rho_array, psi_abrams_hopf(beta), alpha), color="r", label='$Hopf$')
#plt.plot(np.linspace(0, 1, 10000), 0.72 * np.linspace(0, 1, 10000), color="y", label='$Homoclinic$')  ## La pente est approximative ici mais tirée de Abrams 2008 , expérimentalement
# plt.plot(rho_array, 2*np.sqrt(rho_array/256), color="g", label='$KS-bound$')


#cbar = plt.colorbar()
#cbar.set_label("$\\frac{n_{chim}}{n_{CI}}$", rotation=360, fontsize=40, labelpad=70)
#plt.legend(loc=4)
#plt.show()


#xticklist = np.linspace(0, len(rho_array)-1, 5)
#yticklist = np.linspace(0, len(delta_array)-1, 5)
#rholist = []
#deltalist = []
#for i in xticklist:
#    rholist.append(np.round(rho_array[int(i)], 2))
#for i in yticklist:


############################################ Integrate reduced dynamics ################################################
"""
solutions = integrate_reduced_sync_dynamics_SBM(w0, coupling, alpha, sizes, pq, timelist)
rho1 = solutions[0][:, 0]
rho2 = solutions[0][:, 1]
phi1 = solutions[0][:, 2]
phi2 = solutions[0][:, 3]
Rt = solutions[1]
Rmoy = solutions[2]

plt.plot(timelist, rho1)
plt.plot(timelist, rho2)
plt.plot(timelist, np.imag(rho1*np.exp(1j*phi1)))
plt.plot(timelist, np.imag(rho2*np.exp(1j*phi1)))
plt.plot(timelist, Rt)
plt.plot(timelist, Rmoy*np.ones(len(timelist)))
#plt.plot(timelist, phi1)
#plt.plot(timelist, phi2)
#plt.plot(solutions[:, 0]*np.cos(solutions[:, 2]-solutions[:, 3]), solutions[:, 0]*np.sin(solutions[:, 2]-solutions[:, 3]))
#plt.plot(solutions[:, 1]*np.cos(solutions[:, 2]-solutions[:, 3]), solutions[:, 1]*np.sin(solutions[:, 2]-solutions[:, 3]))
plt.show()

"""


#################################### Replot chimera map with theoretical predictions ###################################
"""
with open('data/2018_05_11_12h27min13sec_heteroblocksizes_chimeramatrix.json') as json_data:
    chimera_map = np.array(json.load(json_data))

rho_array = np.linspace(0.6, 0.9, 20)
delta_array = np.linspace(0.1, 0.4, 20)

plt.figure(figsize=(10, 8))
plt.rc('axes', linewidth=2)
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'text.usetex': True,
    'font.family': 'serif',
    'xtick.labelsize': 15,
    'ytick.labelsize': 15
}
mat.rcParams.update(params)
# cdict = {
#    'red': ((0.0, 1.0, 1.0), (0.6, 0.3, 0.3), (1.0, 0.1, 0.1)),
#    'green': ((0.0, 1.0, 1.0), (0.6, .59, .59), (1.0, 0.25, .25)),
#    'blue': ((0.0, 1.0, 1.0), (0.6, .75, .75), (1.0, 0.7, 0.7))
# }
cdict = {
    'red': ((0, 255 / 255, 255 / 255), (0.9, 255 / 255, 255 / 255), (1.0, 255 / 255, 255 / 255)),
    'green': ((0, 255 / 255, 255 / 255), (0.9, 147 / 255, 147 / 255), (1.0, 102 / 255, 102 / 255)),
    'blue': ((0, 255 / 255, 255 / 255), (0.9, 76 / 255, 76 / 255), (1.0, 0, 0))
}
cm = mat.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

#plt.title("$\\sigma = {}$".format(sigma), fontsize=50)
plt.imshow(chimera_map, cmap=cm, vmin=0, vmax=1,
           extent=[rho_array.min(), rho_array.max(), delta_array.min(), delta_array.max()],
           interpolation='nearest', origin='lower', aspect=0.5)
plt.plot(np.linspace(0, 1, 200), find_delta_array(np.linspace(0, 1, 200), -0.141267, alpha, f, betainf), color="k", label='$Hopf$', linewidth=5)    #  Normally Hopf is this color : #ff4300
plt.plot(np.linspace(0, 1, 200), find_delta_array(np.linspace(0, 1, 200), -0.1544717, alpha, f, betainf), color="r", label='$Saddle$', linewidth=5)
plt.xlabel("$\\rho$", fontsize=40)
ylab = plt.ylabel("$\\Delta$", fontsize=40, labelpad=20)
ylab.set_rotation(0)
cbar = plt.colorbar(pad=0.05)
cbar.set_label("$R_r$", rotation=360, fontsize=40, labelpad=30)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.show()
"""