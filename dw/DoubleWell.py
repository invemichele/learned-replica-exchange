#!/usr/bin/env python

# useful things, in no particular order

import numpy as np
import matplotlib.pyplot as plt

import openmm
from openmm import unit
kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)

import torch
import bgflow as bg
from bgflow.utils.types import assert_numpy

#define the model in bgflow
class ModifiedWolfeQuapp(bg.Energy):
    def __init__(self, dim=2, theta=-0.3*np.pi/2, scale1=2, scale2=15, beta=1.):
        super().__init__(dim)
        assert dim >= 2
        self._scale1 = scale1
        self._scale2 = scale2
        self._beta = beta
        self._c = torch.cos(torch.as_tensor(theta))
        self._s = torch.sin(torch.as_tensor(theta))

    def _energy(self, x):
        xx = self._c * x[..., [0]] - self._s * x[..., [1]]
        yy = self._s * x[..., [0]] + self._c * x[..., [1]]
        e4 = xx.pow(4) + yy.pow(4)
        e2 = -2 * xx.pow(2) - 4 * yy.pow(2) + 2 * xx * yy
        e1 = 0.8 * xx + 0.1 * yy + 9.28
        v = x[..., 2:]
        ev = self._scale2 * 0.5 * v.pow(2).sum(dim=-1, keepdim=True)
        return self._beta * (self._scale1 * (e4 + e2 + e1) + ev)

    @property
    def potential_str(self):
      x_str = f'({self._c:g}*x-{self._s:g}*y)'
      y_str = f'({self._s:g}*x+{self._c:g}*y)'
      pot_str = f'{self._scale1:g}*({x_str}^4+{y_str}^4-2*{x_str}^2-4*{y_str}^2+2*{x_str}*{y_str}+0.8*{x_str}+0.1*{y_str}+9.28)'
      if self.dim >= 3:
        pot_str += f'+{self._scale2:g}*0.5*z^2'
      return pot_str

    
#define the openmm system
class DoubleWellOpenMM():
    def __init__(self, model=ModifiedWolfeQuapp(2)):
        '''
        Simulate a multidimensional double-well system using OpenMM.
        The given model provides the potential for the first 3 dof
        and the scale for the remaining Gaussian ones.
        Since all degrees of freedom but the firts two are uncoupled, 
        we use N particles in 3D instead of one particle in dim dimensions.
        '''
        dim = model.dim
        n_particles = 1 + (dim - 1) // 3
        partial_dim = dim % 3
        
        system = openmm.System()
        
        #first particle feels nontrivial potential
        model_potential = openmm.CustomExternalForce(model.potential_str)
        model_potential.addParticle(0)
        system.addParticle(1.0)
        system.addForce(model_potential)
        #other particles feel an harmonic potential
        if dim > 3:
            normal_potential = openmm.CustomExternalForce(f'{model._scale2}*0.5*(x^2+y^2+z^2)')
            for i in range(1, n_particles-bool(partial_dim)):
                normal_potential.addParticle(i)
                system.addParticle(1.0)
            system.addForce(normal_potential)
            #do not add a potential to the left over MD dimensions
            if partial_dim > 0:
                if partial_dim == 1:
                    dof = 'x^2'
                elif partial_dim == 2:
                    dof = '(x^2+y^2)'
                else:
                    assert False
                partial_potential = openmm.CustomExternalForce(f'{model._scale2}*0.5*{dof}')
                partial_potential.addParticle(n_particles-1)
                system.addParticle(1.0)
                system.addForce(partial_potential)
        
        #some arbitrary initial positions
        init_posA = np.array([[-1.5, 0.7, 0]]) #basin A
        init_posB = np.array([[2, -1.0, 0]]) #basin B
        if dim > 3:
            init_posA = np.append(init_posA, [0.0]*((n_particles-1)*3)).reshape(-1, 3)
            init_posB = np.append(init_posB, [0.0]*((n_particles-1)*3)).reshape(-1, 3)
        
        #properties:
        self.dim = dim #intrinsic dimension
        self.MD_dim = 3 * n_particles #MD dimension (multiple of 3), MD_dim >= dim
        
        self.system = system
        self.topology = openmm.app.topology.Topology()
        
        self.positions = init_posA
        self.pos_basinA = init_posA
        self.pos_basinB = init_posB
   
  
#some plotting functions
model2D = ModifiedWolfeQuapp(2)
nbins = 101
x_bins = np.linspace(-3, 3, nbins)
y_bins = x_bins
XY = np.array(np.meshgrid(x_bins, y_bins))
DoubleWellXY = model2D.energy(torch.as_tensor(XY.T).reshape(nbins**2,2)).view(nbins, nbins).cpu().numpy().T

FES = np.zeros(len(x_bins))
for i in range(len(x_bins)):
    FES[i] = -np.logaddexp.reduce(-DoubleWellXY[:,i])
FES -= min(FES)


def plot_doublewell(traj=None, traj2=None, grid=False):
    levels = np.linspace(0,20,9)
    
    plt.contourf(XY[0], XY[1], DoubleWellXY, levels)
    plt.colorbar()
    plt.gca().set_box_aspect(1)
    plt.xlabel('x')
    plt.ylabel('y')
    if grid:
        plt.grid(linestyle='dashed')
    if traj is not None:
        xy_samples = assert_numpy(traj).reshape(len(traj),-1)[:,:2]
        plt.plot(xy_samples[:,0], xy_samples[:,1], '.r')
    if traj2 is not None:
        xy_samples = assert_numpy(traj2).reshape(len(traj2),-1)[:,:2]
        plt.plot(xy_samples[:,0], xy_samples[:,1], '+m')
    plt.show()
  
  
def plot_fes(data, temp=1/kB, bins='auto', w=None, show=True):
    x_samples = assert_numpy(data)
    beta = 1/(temp*kB)
    FES_t = [-np.logaddexp.reduce(-beta*DoubleWellXY[:,i]) for i in range(len(x_bins))]
    FES_t -= min(FES_t)
    plt.plot(x_bins, FES_t, '--', label='reference')
    if w is not None and bins == 'auto':
        bins = 50
    hist, edges = np.histogram(x_samples, bins=bins, weights=w)
    fes_estimate = -np.log(np.where(hist!=0, hist/hist.max(), np.nan))
    xrange = edges[:-1]+(edges[1]-edges[0])/2
    plt.plot(xrange, fes_estimate, label='estimate')
    plt.xlabel('x')
    plt.ylabel('FES')
    plt.ylim(0, FES_t[-1])
    plt.xlim(x_bins[0], x_bins[-1])
    plt.legend()
    if show:
        plt.show()
    fes_estimate[np.isnan(fes_estimate)] = 1000
    
    print(f'ref DeltaF: {np.logaddexp.reduce(-FES_t[x_bins<0])-np.logaddexp.reduce(-FES_t[x_bins>0]):g} [kBT]')
    print(f'    DeltaF: {np.logaddexp.reduce(-fes_estimate[xrange<0])-np.logaddexp.reduce(-fes_estimate[xrange>0]):g} [kBT]')

    
def get_xy_grid(dim, prior, ctx, vmax=5, grid=np.linspace(-3.5, 3.5, 20)):
    myX, myY = np.meshgrid(grid,grid)
    if dim > 2:
        xy_grid = torch.as_tensor(np.c_[myX.flatten(), myY.flatten(), np.zeros((len(myX.flatten()), dim-2))], **ctx)
    else:
        xy_grid = torch.as_tensor(np.c_[myX.flatten(), myY.flatten()], **ctx)
    priorMask = assert_numpy(prior.energy(xy_grid).view(-1)) < vmax
    return xy_grid, priorMask

  
def plot_training(reporter, n_epochs):
    fig, ax = plt.subplots()
    report = reporter._raw[0]
    epochs = np.linspace(0, n_epochs, len(report))
    line1, = ax.plot(epochs, report, label='KLD')
    ax.set_ylim(min(report), np.percentile(report, 98))
    ax.set_ylabel('KLD')
    ax.set_xlim(epochs[0], epochs[-1])
    ax.set_xlabel('epochs')
    print(f'KLD from {report[0]:g} to {report[-1]:g}')
    if len(reporter._raw) == 2:
        report = reporter._raw[1]
        ax2 = ax.twinx()
        ax2.plot([], []) #fixes color
        line2, = ax2.plot(epochs, report, label='NLL')
        ax2.set_ylabel('NLL')
        print(f'NLL from {report[0]:g} to {report[-1]:g}')
        plt.legend([line1, line2], ['KLD', 'NLL'])
    plt.show()