import sys
sys.path.append('../../../tools/pysb')
from pysb import *

Model()

Monomer('ERK',['b','p1','p2'],{'p1':['u','p'],'p2':['u','p']})
Monomer('MEK',['b'])
Monomer('MKP',['b'])

Parameter('kf1', 0.001)
Parameter('kr1', 0.01)
Parameter('kc1', 0.01)
Parameter('kf2', 0.001)
Parameter('kr2', 0.01)
Parameter('kc2', 1)
Parameter('hf1', 0.001)
Parameter('hr1', 0.01)
Parameter('hc1', 0.1)
Parameter('hf2', 0.001)
Parameter('hr2', 0.01)
Parameter('hc2', 1)

Parameter('ERK0',1500)
Parameter('MEK0',2500)
Parameter('MKP0',800)

Rule('ERK_binds_MEK', ERK(b=None,p1='u',p2='u') + MEK(b=None) <> ERK(b=1,p1='u',p2='u')%MEK(b=1),kf1, kr1)
Rule('ERKp_binds_MEK', ERK(b=None,p1='p',p2='u') + MEK(b=None) <> ERK(b=1,p1='p',p2='u')%MEK(b=1),kf2, kr2)
Rule('ERK_phosph1',ERK(b=1,p1='u',p2='u')%MEK(b=1) >> ERK(b=None,p1='p',p2='u') + MEK(b=None), kc1)
Rule('ERK_phosph2',ERK(b=1,p1='p',p2='u')%MEK(b=1) >> ERK(b=None,p1='p',p2='p') + MEK(b=None), kc2)
Rule('ERKpp_binds_MKP', ERK(b=None,p1='p',p2='p') + MKP(b=None) <> ERK(b=1,p1='p',p2='p')%MKP(b=1),hf1,hr1)
Rule('ERKp_binds_MKP', ERK(b=None,p1='p',p2='u') + MKP(b=None) <> ERK(b=1,p1='p',p2='u')%MKP(b=1),hf2,hr2)
Rule('ERK_dephosph1',ERK(b=1,p1='p',p2='p')%MKP(b=1) >> ERK(b=None,p1='p',p2='u') + MKP(b=None), hc1)
Rule('ERK_dephosph2',ERK(b=1,p1='p',p2='u')%MEK(b=1) >> ERK(b=None,p1='u',p2='u') + MEK(b=None), hc2)

Initial(ERK(b=None,p1='u',p2='u'),ERK0)
Initial(MEK(b=None),MEK0)
Initial(MKP(b=None),MKP0)

Observable('ERKpp',ERK(p1='p',p2='p'))
Observable('ERKp1',ERK(p1='p'))
Observable('ERKp2',ERK(p2='p'))
#Observable('ERKtot',ERK())
#Observable('MEKtot',MEK())
#Observable('MKPtot',MKP())

if __name__ == '__main__':
    from pysb.integrate import Solver
    import numpy
    import matplotlib.pyplot as plt
    ts = numpy.linspace(0,100,100)
    solver = Solver(model,ts)
    solver.run()
    plt.ion()
    plt.plot(ts,solver.yobs)
    plt.show()
