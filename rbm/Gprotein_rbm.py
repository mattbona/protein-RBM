### Author: Matteo Bonamassa
### E-mail: matteo.bonamassa1@gmail.com
### Date: February 2020
###
### First attempt constructing an RBM that
### is trained with protein-G contact maps
### at various temperature [119-122 GROMACS_REF_T].
###

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
import mnist_util as util

be.set_seed(137) # for determinism

### Load Data
dataset_path="./data/rbm-dataset/"
train_file="sim-120T" # Name of the .dat train file in the dataset dir

train_patterns_list = []
with open(dataset_path+train_file+'.dat') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=' ')
    for i,row in enumerate(readCSV):
        if (i%2==0):
            row=list(map(int,row[:-1])) #exclude last space character
            train_patterns_list.append(row)

samples = np.asarray(train_patterns_list)
###

def run(num_epochs=10, show_plot=False):
    num_hidden_units = 1540
    batch_size = 100
    mc_steps = 10
    beta_std = 0.6
    train_fraction = 0.5

    # set up the reader to get minibatches
    with batch.in_memory_batch(samples, batch_size, train_fraction) as data:

        # set up the model and initialize the parameters
        vis_layer = layers.BernoulliLayer(data.ncols)
        hid_layer = layers.BernoulliLayer(num_hidden_units, center=False)

        rbm = BoltzmannMachine([vis_layer, hid_layer])
        rbm.connections[0].weights.add_penalty({'matrix': pen.l2_penalty(0.001)})
        rbm.initialize(data, method='hinton')

        print('training with persistent contrastive divergence')
        cd = fit.SGD(rbm, data)

        learning_rate = schedules.PowerLawDecay(initial=0.01, coefficient=0.1)
        opt = optimizers.ADAM(stepsize=learning_rate)

        cd.train(opt, num_epochs, mcsteps=mc_steps, method=fit.pcd)
#        util.show_metrics(rbm, cd.monitor)

    return rbm

if __name__ == "__main__":

    rbm = run(show_plot = True)

    n_fantasy = 100
    fantasy_steps = 100
    beta_std = 0.6
    run_mean_field = True

    fantasy_particles = util.compute_fantasy_particles(rbm, n_fantasy, fantasy_steps,beta_std=beta_std,run_mean_field=run_mean_field)

    FP = fantasy_particles.sum(2)/1.540
    av_E = FP[:,fantasy_steps-1].sum()/n_fantasy
    av_E2 = (np.square(FP[:,fantasy_steps-1])).sum()/n_fantasy
    var = av_E2 - av_E**2

    print("Mean E:\t",av_E,"\t",var)
