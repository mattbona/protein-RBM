#!/home/matteob/.local/envs/miniconda3/envs/rbm-paysage/bin/python
### Author: Matteo Bonamassa
### E-mail: matteo.bonamassa1@gmail.com
### Date: February 2020
###
### First attempt constructing an RBM that
### is trained with protein-G contact maps
### at various temperature [119-122 GROMACS_REF_T].
###

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
import csv

# import the Gprotein_util module
from importlib import util
filename = os.path.join(os.path.dirname(__file__), "util/Gprotein_util.py")
spec = util.spec_from_file_location("Gprotein_util", location=filename)
Gprotein_util = util.module_from_spec(spec)
spec.loader.exec_module(Gprotein_util)

be.set_seed(137) # for determinism

out_file = open("results/energy-vs-var-fanatasy-cmap.dat", "a+")
out_file1 = open("results/KL&ReverseKL-div.dat", "a+")
for temperature in os.listdir("../dataset"):

    dataset_path = "../dataset/"
    train_file = temperature # Name of the .dat train file in the dataset dir

    train_patterns_list = []
    with open(dataset_path+train_file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=' ')
        for i,row in enumerate(readCSV):
            if (i%2==0):
                row=list(map(int,row[:-1])) #exclude last space character
                train_patterns_list.append(row)

    samples = np.asarray(train_patterns_list)

    def run(num_epochs=10, show_plot=False):
        num_hidden_units = 10
        batch_size = 100
        mc_steps = 1
        beta_std = 0.6

        # set up the reader to get minibatches
        with batch.in_memory_batch(samples, batch_size, train_fraction=0.95) as data:

            # set up the model and initialize the parameters
            vis_layer = layers.BernoulliLayer(data.ncols)
            hid_layer = layers.BernoulliLayer(num_hidden_units, center=False)

            rbm = BoltzmannMachine([vis_layer, hid_layer])
            rbm.connections[0].weights.add_penalty({'matrix': pen.l2_penalty(0.001)})
            rbm.initialize(data, method='hinton')

#            print('training with persistent contrastive divergence')
            cd = fit.SGD(rbm, data)

            learning_rate = schedules.PowerLawDecay(initial=0.01, coefficient=0.1)
            opt = optimizers.ADAM(stepsize=learning_rate)

            cd.train(opt, num_epochs, mcsteps=mc_steps, method=fit.pcd, verbose=False)
            reverse_KL_div = [ cd.monitor.memory[i]['ReverseKLDivergence'] for i in range(0,len(cd.monitor.memory)) ]
            KL_div = [ cd.monitor.memory[i]['KLDivergence'] for i in range(0,len(cd.monitor.memory)) ]
            for i in range(0,len(cd.monitor.memory)):
            	out_file1.write(str(KL_div[i])+" "+str(reverse_KL_div[i])+"\n")
    #        Gprotein_util.show_metrics(rbm, cd.monitor)

        return rbm

    if __name__ == "__main__":
        print("Computing file: ", temperature)
        rbm = run(show_plot = False)
        print("Train done!")

        n_fantasy = 100000
        fantasy_steps = 10
        print("Creating fantasy particles...")
        fantasy_particles = Gprotein_util.compute_fantasy_particles(rbm, n_fantasy, fantasy_steps,run_mean_field=False)

        cmap_energy = fantasy_particles.sum(1)/np.size(fantasy_particles, 1)
        av_E = cmap_energy.sum(0)/np.size(cmap_energy, 0)
        av_E2 = (np.square(cmap_energy)).sum(0)/np.size(cmap_energy, 0)
        var = av_E2 - av_E**2
        print("Mean E:\t",av_E,"\t",var)
        out_file.write(temperature+" "+ str(av_E) + " " + str(var)+"\n")
