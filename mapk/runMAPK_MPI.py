import getopt
import sys
import pdb
sys.path.append('../models/mapk')
import numpy as np
sys.path.append('../../emcee')
import emcee
from emcee.utils import MPIPool
import warnings
import matplotlib.pyplot as plt
import time
import multiprocessing
import scipy.misc
import pickle

from mapk import Model

class Object(object):
    pass

def likelihood_sim(args):
    model, param_values, tspan = args
    #print param_values
    y, yobs = model.simulate(tspan, param_values=param_values,view=False,sample_init=True)
    #print yobs['ERKpp']
    return y[0,1],yobs

def likelihood(x,opts):
    param_values = 10**x
    
    nl = opts.num_likelihood_eval # number of likelihood evaluations
    nc = len(opts.exp_data)
    nt = len(opts.tspan)-1
    time_start  = time.time()
    if opts.use_parallel:
        inputs1 = (opts.model,param_values,opts.tspan)
        inputs = []
        for i in range(nl):
            inputs.append(inputs1)
        pool = multiprocessing.Pool(4)
        result = pool.map_async(likelihood_sim,inputs)
        while not result.ready():
            try:
                outputs = result.get()
            except KeyboardInterrupt as e:
                pool.terminate()
                raise
        pool.close()
        pool.join()
    else:
        outputs = [None] * nl
        MEK0 = [None] * nl
        for k in range(nl):
            MEK0[k], outputs[k] = likelihood_sim((opts.model,param_values,opts.tspan))
            #print outputs[k]['ERKpp']
    time_stop = time.time()

    if opts.likelihood_method=='mc3':
        lh = np.zeros((nc,nl))
        lhsum = np.zeros(nc)
        for k in range(nl):
            #pdb.set_trace()
            #yobs = outputs[k].astype('float')[1:]
            if opts.use_parallel:
                yobs = outputs[k][1]['ERKpp'][1:]
            else:
                yobs = outputs[k]['ERKpp'][1:]
            #yobs = yobs / mcmc.options.opts.model.initial_conditions[0].value
            for i in range(nc):
                likelihoods = (opts.exp_data[i][1:] - yobs) ** 2 / (2 * opts.exp_data_var[1:] ** 2)
                lh[i,k] = np.sum(likelihoods)
        for i in range(nc):
            lhsum[i] = -scipy.misc.logsumexp(-lh[i,:])  # stable version of log(sum(exp(lh)))
        #print np.sum(lhsum)
        #np.savetxt('lhmat.txt',lh)
        #np.savetxt('MEK0.txt',MEK0)
        #pdb.set_trace()
        return -np.sum(lhsum)-np.log10(nl)
    elif opts.likelihood_method=='multinom':
        L = 10
        nexp = 1   
        lmax = np.max(opts.exp_data)
        # Prepare observations
        data_counts = np.ndarray(shape=([nt]+[L]*nexp))
        data_counts[:] = 0
        discretize = lambda x,L: int(min(np.floor(L*(x/lmax)),L-1))
        for i in range(nc):
            for t in range(nt):
                exp_data_d = discretize(opts.exp_data[i][t],L)
                data_counts[(t,exp_data_d)]+=1
        
        # Simulate the opts.model with current parameters
        sim_counts = np.ndarray(shape=[nt]+[L]*nexp)
        sim_counts_norm = np.ndarray(shape=[nt]+[L]*nexp)
        sim_counts[:] = 0
        for k in range(nl):
            yobs = outputs[k].astype('float')[1:]
            #yobs = yobs/mcmc.options.opts.model.initial_conditions[0].value
            for t in range(nt):
                normcdfs = scipy.stats.norm.cdf([lmax*(l+1)/float(L) for l in range(L)],loc=yobs[t],scale=opts.exp_data_var[t+1])
                prev = 0
                for l in range(L):
                    sim_counts[t,l] += normcdfs[l]-prev
                    prev = normcdfs[l]
                #pdb.set_trace()
        for t in range(nt):
            sim_counts_norm[t,:] = sim_counts[t,:]/np.sum(sim_counts[t,:])
        # Where data_counts=0 we should use 0 irrespective of sim_counts_norm
        lh = 0
        #pdb.set_trace()
        for s,d in zip(sim_counts_norm.reshape(-1),data_counts.reshape(-1)):
            if d>0:
                if s==0:
                    return np.inf
                else:
                    lh += d * np.log(s)
        return -lh

def prior(x,opts):
    return np.sum(-(x - opts.prior_mean)**2 / (2 * opts.prior_var**2)) 

#def likelihood(x,opts):
#     with warnings.catch_warnings(record=True) as w:
#         tstart = time.time()
#         _, yobs = opts.model.simulate(opts.tspan, 10**x,view=False,sample_init=True)
#         tend = time.time()
#         if tend-tstart>1:
#            print x
#            #plt.plot(opts.tspan,yobs['ERKpp'])
#            #plt.show()
#         if w!=[]:
#             print 'returned...'
#             return -np.inf
#     llh = -np.sum((opts.exp_data[0][1:] - yobs['ERKpp'][1:]) ** 2 / (2 * opts.exp_data_var[1:]**2))
#     return llh

def lnprob(x,opts):
    lnpri = prior(x,opts)
    lnlhood = likelihood(x,opts)
    #print lnpri, lnlhood
    #print x
    return lnpri + lnlhood

nwalkers = 40
ndim = 12
nsteps = 10
var_scale = 0.0

pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

opts, extraparams = getopt.getopt(sys.argv[1:],'w:n:c:v:',['nwalkers=','nsteps=','ncells=','var_scale='])


for o,p in opts:
    if o in ['-w','--nwalkers']:
        nwalkers = int(p)
    elif o in ['-n','--nsteps']:
        nsteps = int(p)
    elif o in ['-c','--ncells']:
        ncells = int(p)
    elif o in ['-v','--var_scale']:
		var_scale = float(p)
print ncells, nwalkers, nsteps, var_scale


opts = Object()
opts.model = Model(var_scale)
opts.tspan = np.linspace(0,2400,24)
opts.exp_data, opts.exp_data_var,exp0 = opts.model.generate_data(opts.tspan,ncells)
opts.p0 = np.zeros((nwalkers,ndim))
prior_var = 1.0
for i in range(nwalkers):
    for k,p in enumerate(opts.model.parameters):
        opts.p0[i,k] = np.log10(p.value) + 0.1*prior_var*np.random.randn()
opts.prior_mean = np.array([np.log10(p.value) for p in opts.model.parameters])
opts.prior_var = np.array([prior_var for p in opts.model.parameters])
#opts.model.integrator.set_integrator('vode',nsteps=1e5)
opts.num_likelihood_eval = 1000
opts.use_parallel = False
opts.likelihood_method = 'mc3'

#opts.num_likelihood_eval = 1
#x1 = np.array([np.log10(p.value) for p in opts.model.parameters])
#x2 = np.array([np.log10(p.value)-0.1 for p in opts.model.parameters])
#p11 = lnprob(x1,opts)
#p12 = lnprob(x2,opts)

#opts.num_likelihood_eval = 1000
#p21 = lnprob(x1,opts)
#p22 = lnprob(x2,opts)

#print p11, p12
#print 'diff1:', p11-p12
#print p21, p22
#print 'diff2:', p21-p22

#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[opts], a=1.3)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[opts], a=1.25, pool=pool)
time_start = time.time()
i = 1
time_one_start = time_start
for results in sampler.sample(opts.p0,iterations=nsteps):
    if i % 100 == 0:
        time_one_end = time.time()
        print i, np.mean(sampler.acceptance_fraction), np.mean(sampler.acor)
        print np.var(sampler.flatchain)
        print 'Time ', time_one_end-time_one_start
        time_one_start = time_one_end
    i+=1

time_stop = time.time()
print str(time_stop-time_start)+'s'

pool.close()

fname = '/home/bmg16/mcmccell/src/models/mapk/output/mapk_emcee_v%.2f_c%d_w%d_n%d' % (var_scale,ncells,nwalkers,nsteps)
print fname
np.savetxt(fname+'.txt',sampler.flatchain,fmt='%.5f')

sampler.pool = []
pickle.dump(sampler,open(fname+'.pkl','wb'))

#sampler.pool = []
#pickle.dump(sampler,open(fname+'.pkl','wb'))


#labels = ['log10('+p.name+')' for p in opts.model.parameters]
#truths = [np.log10(p.value) for p in opts.model.parameters]
#figure = triangle.corner(sampler.flatchain, labels=labels,truths=truths,quantiles=[0.16, 0.5, 0.84],show_titles=True, title_args={'fontsize': 12})

#figure.savefig('sample_distrib.png')


#cov = 0.001*np.identity(ndim)
#samplerMC = emcee.MHSampler(cov,ndim,lnprob,args=[opts])
#samplerMC.run_mcmc(opts.p0[0], nwalkers*nsteps)
