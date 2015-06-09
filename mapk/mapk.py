"""
MAPK model
"""

import numpy as np
import scipy.weave, scipy.integrate
import collections
import itertools
import distutils.errors
import pdb

#_use_inline = False
# try to inline a C statement to see if inline is functional
#try:
#    scipy.weave.inline('int i;', force=1)
#    _use_inline = True
#except distutils.errors.CompileError:
#    pass

_use_inline = True

Parameter = collections.namedtuple('Parameter', 'name value')
Observable = collections.namedtuple('Observable', 'name species coefficients')
Initial = collections.namedtuple('Initial', 'species_index value ll ul')


class Model(object):
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['integrator']
        return state
    def __setstate__(self,state):
        self.__dict__.update(state)
        self.init_solver()
    def init_solver(self):
        self.integrator = scipy.integrate.ode(self.ode_rhs)
        self.integrator.set_integrator('vode', method='bdf', with_jacobian=True, rtol=1e-3, atol=1e-6,nsteps=500,order=5)
 
    def __init__(self, var_scale=0.0):
        global rng
        self.y = None
        self.yobs = None
        self.y0 = np.empty(9)
        self.ydot = np.empty(9)
        self.init_solver()
        self.sim_param_values = np.empty(12)
        self.parameters = [None] * 12
        self.observables = [None] * 4
        self.initial_conditions = [None] * 4

        self.parameters[0] = Parameter('kf1', 0.01)
        self.parameters[1] = Parameter('kr1', 0.1)
        self.parameters[2] = Parameter('kc1', 1.0)
        self.parameters[3] = Parameter('kf2', 0.01)
        self.parameters[4] = Parameter('kr2', 0.1)
        self.parameters[5] = Parameter('kc2', 10.0)
        self.parameters[6] = Parameter('hf1', 0.01)
        self.parameters[7] = Parameter('hr1', 0.1)
        self.parameters[8] = Parameter('hc1', 1.0)
        self.parameters[9] = Parameter('hf2', 0.001)
        self.parameters[10] = Parameter('hr2', 0.1)
        self.parameters[11] = Parameter('hc2', 10.0)

        self.observables[0] = Observable('ERKpp', [5, 7], [1, 1])
        self.observables[1] = Observable('ERKtot', [0,2,3,4,5,7,8], [1,1,1,1,1,1,1])
        self.observables[2] = Observable('MEKtot', [1,2,4], [1,1,1])
        self.observables[3] = Observable('MKPtot', [6,7,8], [1,1,1])

        mek0 = 500
        mkp0 = 1500
        var_scale = 0.5
        self.initial_conditions[0] = Initial(0, 1300, 1300, 1300) # ERK
        self.initial_conditions[1] = Initial(1, mek0, mek0*(1-var_scale), mek0*(1+var_scale)) # MEK
        self.initial_conditions[2] = Initial(6, mkp0, mkp0*(1-var_scale), mkp0*(1+var_scale)) # MKP
        self.initial_conditions[3] = Initial(5, 200, 200, 200) # ERKpp
    
        rng = np.random.RandomState(None)
    if _use_inline:    
        def ode_rhs(self, t, y, p):
            ydot = self.ydot
            scipy.weave.inline(r'''
                ydot[0] = -p[0]*y[0]*y[1]+p[1]*y[2]+p[11]*y[8];	// ERK
                ydot[1] = -p[0]*y[0]*y[1]+(p[1]+p[2])*y[2]-p[3]*y[1]*y[3]+(p[4]+p[5])*y[4]; // MEK
                ydot[2] = p[0]*y[0]*y[1]-(p[1]+p[2])*y[2];		// ERK-MEK
                ydot[3] = p[2]*y[2]+p[4]*y[4]+p[8]*y[7]+p[10]*y[8]-p[3]*y[1]*y[3]-p[9]*y[3]*y[6];	// ERKp
                ydot[4] = p[3]*y[1]*y[3]-(p[4]+p[5])*y[4];		// ERKp-MEK
                ydot[5] = p[5]*y[4]-p[6]*y[5]*y[6]+p[7]*y[7];	// ERKpp
                ydot[6] = -p[6]*y[5]*y[6]+(p[7]+p[8])*y[7]-p[9]*y[3]*y[6]+(p[10]+p[11])*y[8];	// MKP
                ydot[7] = p[6]*y[5]*y[6]-(p[7]+p[8])*y[7];	// ERKpp-MKP
                ydot[8] = p[9]*y[3]*y[6]-(p[10]+p[11])*y[8];	// ERKp-MKP
                ''', ['ydot', 't', 'y', 'p'])
            return ydot
        
    else:
        def ode_rhs(self, t, y, p):
            ydot = self.ydot
            ydot[0] = -p[0]*y[0]*y[1]+p[1]*y[2]+p[11]*y[8];	# ERK
            ydot[1] = -p[0]*y[0]*y[1]+(p[1]+p[2])*y[2]-p[3]*y[1]*y[3]+(p[4]+p[5])*y[4]; # MEK
            ydot[2] = p[0]*y[0]*y[1]-(p[1]+p[2])*y[2];		# ERK-MEK
            ydot[3] = p[2]*y[2]+p[4]*y[4]+p[8]*y[7]+p[10]*y[8]-p[3]*y[1]*y[3]-p[9]*y[3]*y[6];	# ERKp
            ydot[4] = p[3]*y[1]*y[3]-(p[4]+p[5])*y[4];		# ERKp-MEK
            ydot[5] = p[5]*y[4]-p[6]*y[5]*y[6]+p[7]*y[7];	# ERKpp
            ydot[6] = -p[6]*y[5]*y[6]+(p[7]+p[8])*y[7]-p[9]*y[3]*y[6]+(p[10]+p[11])*y[8];	# MKP
            ydot[7] = p[6]*y[5]*y[6]-(p[7]+p[8])*y[7];	# ERKpp-MKP
            ydot[8] = p[9]*y[3]*y[6]-(p[10]+p[11])*y[8];	# ERKp-MKP
            return ydot
#    def get_initial_conditions(self):
#        y0 = self.y0
#        for ic in self.initial_conditions:
#            self.y0[ic.species_index] = self.sim_param_values[ic.param_index]
#        y0[0] = rng.rand()*((self.parameters[16].value-self.parameters[15].value))+self.parameters[15].value
    def generate_data(self, tspan, N=1000):
        sigma = 2e-2
        psim = [p.value for p in self.parameters]
        #print psim
        yall = []
        y0all = []
        for i in range(N):
            y, yobs = self.simulate(tspan,psim,view=False,sample_init=True)
            y0all.append(self.y0.copy())
            print y[0][1]
            noise_var = np.maximum(sigma*yobs.astype('float'),2)
            #noise_var = 0
            yobs = np.maximum(0,yobs.astype('float') + noise_var*rng.randn(yobs.size))
            #pdb.set_trace()
            #yobs = yobs/self.initial_conditions[0].value
            yall.append(yobs)
        return yall, noise_var, y0all
								
    def simulate(self, tspan, param_values=None, view=False, sample_init=False):
        if param_values is not None:
            # accept vector of parameter values as an argument
            if len(param_values) != len(self.parameters):
                raise Exception("param_values must have length %d" % len(self.parameters))
            self.sim_param_values[:] = param_values
        else:
            # create parameter vector from the values in the model
            self.sim_param_values[:] = [p.value for p in self.parameters]

        self.y0.fill(0)
        for ic in self.initial_conditions:
            if sample_init:
                self.y0[ic.species_index] = ic.ll+(ic.ul-ic.ll)*np.random.rand()
            else:
                self.y0[ic.species_index] = ic.value
        #print self.y0[1]
        if self.y is None or len(tspan) != len(self.y):
            self.y = np.empty((len(tspan), len(self.y0)))
            if len(self.observables):
                self.yobs = np.ndarray(len(tspan), zip((obs.name for obs in self.observables),
                                                          itertools.repeat(float)))
            else:
                self.yobs = np.ndarray((len(tspan), 0))
            self.yobs_view = self.yobs.view(float).reshape(len(self.yobs), -1)
        # perform the actual integration
        self.integrator.set_initial_value(self.y0, tspan[0])
        #print self.sim_param_values
        self.integrator.set_f_params(self.sim_param_values)
        #print '-----'
        #print self.integrator.f_params
        #print '-----'
        self.y[0] = self.y0
        t = 1
        while self.integrator.successful() and self.integrator.t < tspan[-1]:
            self.y[t] = self.integrator.integrate(tspan[t])
            #print t,self.integrator.f_params[0][0],self.y[t][5]+self.y[t][7]
            t += 1
        for i, obs in enumerate(self.observables):
            #self.yobs_view[:, i] = \
            self.yobs[obs.name] = \
                (self.y[:, obs.species] * obs.coefficients).sum(1)
            #print "yobs_view:", self.yobs_view[:,i]
            #print "y:", (self.y[:, obs.species] * obs.coefficients).sum(1)
            #print "yobs:", self.yobs
        if view:
            y_out = self.y.view()
            yobs_out = self.yobs.view()
            for a in y_out, yobs_out:
                a.flags.writeable = False
        else:
            y_out = self.y.copy()
            yobs_out = self.yobs.copy()
        #print yobs_out
        return (y_out, yobs_out)
    

