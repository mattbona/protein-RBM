#!/home/matteob/anaconda3/envs/paysage-rbm/bin/python
### Author: Matteo Bonamassa
### E-mail: matteo.bonamassa1@gmail.com
### Date: February 2020
###
### First attempt constructing an RBM that
### is trained with protein-G contact maps
### at various temperature [119-122 GROMACS_REF_T].

import os
from paysage import preprocess as pre
from paysage import layers
from paysage.models import BoltzmannMachine
from paysage import fit
from paysage import optimizers
from paysage import backends as be
from paysage import schedules, batch
from paysage import penalties as pen
import numpy as np
import matplotlib.pyplot as plt
import csv

# import the Gprotein_util module
from importlib import util
filename = os.path.join(os.path.dirname(__file__), "util/Gprotein_util.py")
spec = util.spec_from_file_location("Gprotein_util", location=filename)
Gprotein_util = util.module_from_spec(spec)
spec.loader.exec_module(Gprotein_util)

be.set_seed(137) # for determinism

for temperature in os.listdir("../dataset"):

#    out_file1 = open("results/KL-div/KL&ReverseKL-div-"+temperature, "a+")

    dataset_path = "../dataset/"
    train_file = temperature # Name of the .dat train file in the dataset dir
    print("\nRead data from: ", train_file)

    print("Load data...")
    train_patterns_list = []
    with open(dataset_path+train_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=' ')
        for i,row in enumerate(readCSV):
            if (i%2==0):
                row=list(map(int,row[:-1])) #exclude last space character
                train_patterns_list.append(row)

    samples = np.asarray(train_patterns_list)

    def run(num_epochs=1, show_plot=False):
        num_hidden_units = 1
        batch_size = 100
        mc_steps = 10
        beta_std = 0.6

        # set up the reader to get minibatches
        with batch.in_memory_batch(samples, batch_size, train_fraction=0.95) as data:

            # set up the model and initialize the parameters
            vis_layer = layers.BernoulliLayer(data.ncols)
            hid_layer = layers.BernoulliLayer(num_hidden_units, center=False)
            rbm = BoltzmannMachine([vis_layer, hid_layer])

            rbm.connections[0].weights.add_penalty({'matrix': pen.l2_penalty(0.001)}) # Add regularization term

            rbm.initialize(data, method='hinton') # Initialize weights

            cd = fit.SGD(rbm, data)
            learning_rate = schedules.PowerLawDecay(initial=0.01, coefficient=0.1)
            opt = optimizers.ADAM(stepsize=learning_rate)

            print("Train the model...")
            cd.train(opt, num_epochs, mcsteps=mc_steps, method=fit.pcd, verbose=False)
            '''
            # write on file KL divergences
            reverse_KL_div = [ cd.monitor.memory[i]['ReverseKLDivergence'] for i in range(0,len(cd.monitor.memory)) ]
            KL_div = [ cd.monitor.memory[i]['KLDivergence'] for i in range(0,len(cd.monitor.memory)) ]
            for i in range(0,len(cd.monitor.memory)):
            	out_file1.write(str(KL_div[i])+" "+str(reverse_KL_div[i])+"\n")
            out_file1.close()

            # save weights on file
            filename = "results/weights/weights-"+temperature[:-4]+".jpg"
            Gprotein_util.show_weights(rbm, show_plot=False, n_weights=8, Filename=filename, random=False)
            '''
        return rbm

    if __name__ == "__main__":
        rbm = run(show_plot = False)
        print("Train done!\n")

        n_fantasy = 10000
        fantasy_steps = 10
        print("Create fantasy particles...")
        fantasy_particles = Gprotein_util.compute_fantasy_particles(rbm, n_fantasy, fantasy_steps,run_mean_field=False)

        # Compute mean energy and variance of the energy for nblocks
        nblocks = 100
        nfpxbloc = int(n_fantasy/nblocks)
        av_E = np.zeros(nblocks)
        av_E2 = np.zeros(nblocks)
        hc = np.zeros(nblocks)
        hc2 = np.zeros(nblocks)
        av_E_over_blocks = 0
        av_E2_over_blocks = 0
        av_hc_over_blocks = 0
        av_hc2_over_blocks = 0

        cmap_energy = fantasy_particles.sum(1)/np.size(fantasy_particles, 1)
        for i in range(nblocks):
            sum = 0
            for j in range(nfpxbloc):
                k = j + i*nfpxbloc
                sum += cmap_energy[k]
            av_E[i] = sum/nfpxbloc
            av_E2[i] = (av_E[i])**2

            sum2 = 0
            for j in range(nfpxbloc):
                k = j + i*nfpxbloc
                sum2 += (cmap_energy[k] - av_E[i])**2
            hc[i] = (sum2)/(nfpxbloc-1)
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

#        print("<E> = ",av_E_over_blocks ,"\tvariance_E = ",std_err_E, "<C_v> = ", av_hc_over_blocks ,"\tvariance_C_v = ",std_err_C_v )
        out_file = open("thermodynamic-fantasy-cmap.dat", "a+")
        out_file.write(temperature[4:-5]+" "+ str(av_E_over_blocks) + " " + str(std_err_E)+" "+ str(av_hc_over_blocks) + " " + str(std_err_C_v)+"\n")

        # Save mean fantasy cmap and variance of single contacts
"""
        # Mean
        print("Compute the mean of single contacts...")
        mean_cmap = fantasy_particles.sum(0)/np.size(fantasy_particles, 0)
        #reshape mean_cmap as a contact map for better visualizing it
        triu_i = np.triu_indices(56,1)
        Z1 = np.zeros((56,56))
        Z1[triu_i] = mean_cmap
        plt.matshow(Z1)
        colorbar = plt.colorbar()
        colorbar.set_label("% precence of contact on average")

        plt.savefig("results/mean_fantasy_cmap/mean_cmap-"+ temperature[:-4]+".png")

        # Variance
        print("Compute the variance of single contacts...")
        ncontact = np.size(fantasy_particles,1)
        nfpart = np.size(fantasy_particles,0)
        var = np.zeros(ncontact)

        for i in range(ncontact):
            sum = 0
            print("#Contact: ", i, '\r', end='')
            for j in range(nfpart):
                sum = sum + ( fantasy_particles[j,i] - mean_cmap[i])**2
            var[i] = sum/(nfpart-1)

        # reshape var as a contact map for better visualizing it
        Z2 = np.zeros((56,56))
        Z2[triu_i] = var
        plt.matshow(Z2)
        colorbar = plt.colorbar()
        colorbar.set_label("Variance")

        plt.savefig("results/variance_fantasy_cmap/variance_cmap-"+ temperature[:-4]+".png")
"""
