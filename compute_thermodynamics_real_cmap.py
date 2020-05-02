import numpy as np
import csv
import os

out_file = open("thermodynamic-real-cmap.dat", "a+")
for temp in os.listdir("dataset"):
    print("Current file ",temp, "...")
    dataset_path = "dataset/"
    train_file = temp # Name of the .dat train file in the dataset dir

    print("Loading data...")
    train_patterns_list = []
    with open(dataset_path+train_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=' ')
        for i,row in enumerate(readCSV):
            if (i%2==0):
                row=list(map(int,row[:-1])) #exclude last space character
                train_patterns_list.append(row)

    samples = np.asarray(train_patterns_list)

    # Compute mean energy and variance of the energy for nblocks
    nblocks = 100
    nfpxblocks = int(np.size(samples, 0)/nblocks)
    av_E = np.zeros(nblocks)
    av_E2 = np.zeros(nblocks)
    hc = np.zeros(nblocks)
    hc2 = np.zeros(nblocks)
    av_E_over_blocks = 0
    av_E2_over_blocks = 0
    av_hc_over_blocks = 0
    av_hc2_over_blocks = 0

    cmap_energy = samples.sum(1)/np.size(samples, 1)
    for i in range(nblocks):
        sum = 0
        for j in range(nfpxblocks):
            k = j + i*nfpxblocks
            sum += cmap_energy[k]
        av_E[i] = sum/nfpxblocks
        av_E2[i] = (av_E[i])**2

        sum2 = 0
        for j in range(nfpxblocks):
            k = j + i*nfpxblocks
            sum2 += (cmap_energy[k] - av_E[i])**2
        hc[i] = (sum2)/(nfpxblocks-1)
        hc2[i] = (hc[i])**2

    for i in range(nblocks):
        av_E_over_blocks += av_E[i]
        av_E2_over_blocks += av_E2[i]
        av_hc_over_blocks += hc[i]
        av_hc2_over_blocks += hc2[i]

    av_E_over_blocks /= nblocks
    av_E2_over_blocks /= nblocks
    std_err_E = np.sqrt( (av_E2_over_blocks - av_E_over_blocks**2)/(nblocks-1) )

    av_hc_over_blocks /= nblocks
    av_hc2_over_blocks /= nblocks
    std_err_C_v = np.sqrt( (av_hc2_over_blocks - av_hc_over_blocks**2)/(nblocks-1) )

    '''
    nener = np.size(cmap_energy)
    print("Computing variance...")
    sum = 0
    for i in range(nener):
        sum = sum + (cmap_energy[i] - ave_energy)**2
    alt_variance = sum/(nener-1)
    '''
    print("cmap av_energy: ", av_E_over_blocks, " std_err: ", std_err_E)
    print("av_hc: ", av_hc_over_blocks, " std_err: ", std_err_C_v)#," alternative variance: ", alt_variance)
    out_file.write(temp[4:-5]+" "+ str(av_E_over_blocks) + " " + str(std_err_E)+" "+ str(av_hc_over_blocks) + " " + str(std_err_C_v)+"\n")
