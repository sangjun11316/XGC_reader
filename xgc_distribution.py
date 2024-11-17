from typing import *
"""
This module provides classes and functions for handling velocity grids and XGC distribution data.
Classes:
    VelocityGrid: Represents a velocity grid with parallel and perpendicular velocity components.
    XGCDistribution: Represents the XGC distribution data and provides methods for initialization, updating, and removing Maxwellian components.
Functions:
    flux_surface_average(var, x): Computes the flux surface average using xgc_reader data and scatters back to the mesh.
    adios2_write_array(file, name, data): Writes an array to an ADIOS2 file.
Classes:
    VelocityGrid:
        __init__(self, nvpara: int, nvperp: int, vpara_max: float, vperp_max: float):
            Initializes the VelocityGrid with the given parameters.
    XGCDistribution:
        __init__(self, vgrid: 'VelocityGrid', nnodes: int):
            Initializes the XGCDistribution with the given velocity grid and number of nodes.
        __init__(self, vgrid: 'VelocityGrid', nnodes: int, f: np.ndarray, den: np.ndarray, temp_ev: np.ndarray, flow: np.ndarray, fg_temp_ev: np.ndarray, mass: float, has_maxwellian=True):
            Initializes the XGCDistribution with the given parameters and data arrays.
        __init__(self, filename, dir='./', var_string='i_f', time_step=0, mass_au=2.0):
            Initializes the XGCDistribution from an ADIOS2 file.
        update_maxwellian(self, x):
            Updates the Maxwellian distribution function.
        remove_maxwellian(self, get=False, remove=True):
            Removes or retrieves the Maxwellian component from the distribution function.
        fix_axis_value(self, var, sz=10):
            Fixes the axis value by averaging the first few elements.
        save_f(filename, array_dict, engine_type="BP4"):
            Saves the given arrays to a file using ADIOS2.
Functions:
    flux_surface_average(var, x):
        Computes the flux surface average using xgc_reader data and scatters back to the mesh.
    adios2_write_array(file, name, data):
        Writes an array to an ADIOS2 file.
"""
import xgc_reader
import numpy as np
import adios2


class VelocityGrid:

    def __init__(self, nvperp: int, nvpara: int,
                 vperp_max: float, vpara_max: float):
        self.nvperp = nvperp
        self.nvpara = nvpara
        self.nvpdata = nvpara*2 +1

        self.vpara=np.linspace(-vpara_max, vpara_max, self.nvpdata)
        self.vperp=np.linspace(0, vperp_max, nvperp)

        self.dvperp = self.vperp[1] - self.vperp[0]
        self.dvpara = self.vpara[1] - self.vpara[0]

        self.vpara_max = vpara_max
        self.vperp_max = vperp_max
        
class XGCDistribution:
    # Class constants
    PROTON_MASS: ClassVar[float] = 1.6726219e-27
    E_CHARGE: ClassVar[float] = 1.602176634e-19
    EV_TO_JOULE: ClassVar[float] = 1.602176634e-19
    MU0_FACTOR: ClassVar[float] = 1/3
    
    #initialize from data passing
    def __init__(self, vgrid: 'VelocityGrid', nnodes: int, f: np.ndarray, den: np.ndarray, temp_ev: np.ndarray, flow: np.ndarray, fg_temp_ev: np.ndarray, mass: float, has_maxwellian=True) -> None:
        self.vgrid = vgrid
        self.nnodes = nnodes
        self.den = den
        self.temp_ev = temp_ev
        self.flow = flow
        self.fg_temp_ev = fg_temp_ev
        self.mass = mass
        self.has_maxwellian = has_maxwellian
        if(has_maxwellian):
            self.f = f
        else:
            self.f_g = f

    #initialize from adios2 file like xgc.f0.00000.bp
    @classmethod
    def from_xgc_output(cls, filename, var_string='i_f', dir = './', time_step = 0, mass_au=2.0):
        with adios2.open(filename,"rra") as file:
            it=time_step
            ftmp=file.read(var_string,start=[],count=[],step_start=it, step_count=1)
            step=file.read('step',start=[],count=[],step_start=it, step_count=1)
            ftmp = np.squeeze(ftmp) # remove time step dimension
            ftmp = np.mean(ftmp,axis=0) # toroidal average
            ftmp = np.swapaxes(ftmp, 0, 1)
            print('Loading '+var_string + ' from '+ filename+ '. The toroidal averaged distribution has dim of', np.shape(ftmp))
            nvperp = ftmp.shape[1]
            nvpara = int((ftmp.shape[2]-1)/2)
            nnodes=ftmp.shape[0]

        with adios2.open(dir + 'xgc.f0.mesh.bp',"rra") as file:
            vp_max = file.read('f0_vp_max')
            smu_max = file.read('f0_smu_max')
            vgrid = VelocityGrid(nvperp, nvpara, smu_max, vp_max)
            
            # read the specific species.
            fg_temp_ev = file.read('f0_fg_T_ev')[0,:]  # need species index handling

        mass = mass_au * cls.PROTON_MASS

        # Initialize other arrays with zeros or appropriate values
        den = np.zeros(nnodes)
        temp_ev = np.zeros(nnodes)
        flow = np.zeros(nnodes)

        return cls(vgrid, nnodes, ftmp, den, temp_ev, flow, fg_temp_ev, mass, has_maxwellian=True)

    #initialize from distribution input file -- same format as that of save()
    @classmethod
    def from_xgc_input(cls, filename):
        with adios2.open(filename, "rra") as fr:
            f_g = fr.read("f_g")
            den = fr.read("density")
            temp_ev = fr.read("temperature")
            flow = fr.read("flow")
            fg_temp_ev = fr.read("fg_temp_ev")
            mass = fr.read("mass")
            vgrid = VelocityGrid(fr.read("nvperp"), fr.read("nvpara"), fr.read("vperp_max"), fr.read("vpara_max"))
            nnodes = fr.read("nnodes")

        return cls(vgrid, nnodes, f_g, den, temp_ev, flow, fg_temp_ev, mass, has_maxwellian=False)


    # update maxwellian distribution function
    def update_maxwellian_moments(self, x): # x required for flux surface average

        # update_maxwellian_moments only for full-f distribution including maxwellian component.
        if(not self.has_maxwellian):
            self.add_maxwellian()

        # apply mu0_factor at zero vperp
        self.f[:,0,:] = self.f[:,0,:] * self.MU0_FACTOR

        #get vspace volume
        vspace_vol = self.fg_temp_ev * np.sqrt(1/(np.pi*2)) * self.vgrid.dvperp * self.vgrid.dvpara

        # get moments

        # particle density
        ptls = np.sum(self.f, axis=(1, 2))
        print(np.shape(ptls))
        den = ptls * vspace_vol

        # flow
        flow = np.sum(self.f * self.vgrid.vpara[np.newaxis, np.newaxis, :], axis=(1, 2))
        print(np.shape(flow))
        flow = flow/ ptls * np.sqrt(self.fg_temp_ev * self.EV_TO_JOULE/ self.mass) 

        # temperature
        en1 = np.add.outer(self.vgrid.vperp**2/2, self.vgrid.vpara**2/2) # check factoer 1/2
        
        temp = np.sum(self.f * en1[np.newaxis,:, :], axis=(1, 2)) /ptls * self.fg_temp_ev * 2/3

        # undo multiplication by MU0_FACTOR
        self.f[:,0,:] = self.f[:,0,:] / self.MU0_FACTOR

        # flox surface average
        self.den = flux_surface_average(den, x)
        self.flow = flux_surface_average(flow/x.mesh.r, x)*x.mesh.r # u/R (parallel rotation freq) flux average
        self.temp_ev = flux_surface_average(temp, x) - 0.5 * self.flow**2 * self.mass/self.EV_TO_JOULE



    def remove_maxwellian(self, get=False, add=False):        
        if(get):
            fm = np.zeros((self.nnodes, self.vgrid.nvperp, self.vgrid.nvpdata))
        else: # check invalid cases and set parameters
            #check invalid cases
            if(not add and not self.has_maxwellian):
                print('No maxwellian component to remove')
                return
            if(add and self.has_maxwellian):
                print('Maxwellian component already exists')
                return
            
            #set parameters
            if not add: # remove maxwellian component
                sign = -1
                f_array = self.f
            else: # add maxwellian component
                sign = 1
                f_array = self.f_g


        #actual calculation
        for i in range(self.vgrid.nvperp):
            for j in range(self.vgrid.nvpdata):
                v_n = np.sqrt(self.fg_temp_ev*self.E_CHARGE/self.mass)           
                en =0.5* self.mass * ((self.vgrid.vpara[j]*v_n-self.flow)**2 + (self.vgrid.vperp[i]*v_n)**2) / (self.temp_ev*self.EV_TO_JOULE) # normalized energy by T
                tmp = self.den * np.exp(-en) / (self.temp_ev)**1.5 * self.vgrid.vperp[i]  * np.sqrt(self.fg_temp_ev)
                if(get):
                    fm[:,i,j] = tmp
                else:
                    f_array[:,i,j]=f_array[:,i,j] + sign * tmp

        # when get=True, return the maxwellian component        
        if(get):
            return fm
        
        
        if(sign==-1): # remove maxwellian component
            self.has_maxwellian = False
            self.f_g = self.f
            delattr(self,'f')
        if(sign==1): # add maxwellian component
            self.has_maxwellian = True
            self.f = self.f_g
            delattr(self,'f_g')

    def add_maxwellian(self):
        self.remove_maxwellian(add=True)

    def get_maxwellian(self):
        return self.remove_maxwellian(get=True)

    def fix_axis_value(self,var, sz=10):
        sz2 = np.min([sz, var.shape[0]])
        var[0]=np.mean(var[1:sz2])



    def save(self, filename="xgc.f_init.bp"):
        #when writing to file, remove maxwellian component
        if(self.has_maxwellian):
            self.remove_maxwellian()

        with adios2.open(filename, mode='w') as fw:
            adios2_write_array(fw, "f_g", self.f_g) # maxwellian component is removed.
            adios2_write_array(fw, "density", self.den)
            adios2_write_array(fw, "temperature", self.temp_ev)
            adios2_write_array(fw, "flow", self.flow)
            adios2_write_array(fw, "fg_temp_ev", self.fg_temp_ev)
            fw.write("mass", np.array([self.mass]))

            fw.write("nvpara", np.array([self.vgrid.nvpara], dtype=np.int32))
            fw.write("nvperp", np.array([self.vgrid.nvperp], dtype=np.int32))
            fw.write("vpara_max", np.array([self.vgrid.vpara_max], dtype=np.float64))
            fw.write("vperp_max", np.array([self.vgrid.vperp_max], dtype=np.float64))
            fw.write("nnodes", np.array([self.nnodes], dtype=np.int32))

    def zero_out_fg(self):
        if(self.has_maxwellian):
            delattr(self,'f')
        self.f_g = np.zeros((self.nnodes, self.vgrid.nvperp, self.vgrid.nvpdata))
        self.has_maxwellian = False

# interpolate flux surface moments to mesh
def interp_flux_surface_moments(psi_surf, moments_surf, x):
    var_mesh = np.interp(x.mesh.psi, psi_surf, moments_surf)  # region 3 need to ba handled separately
    return var_mesh

#flux surface average using xgc_reader data and scatter back to the mesh
# private region need to be implemented.
def flux_surface_average(var, x):
    fsa_surf = x.fsa_simple(var)    
    var_favg = interp_flux_surface_moments(x.mesh.psi_surf, fsa_surf, x)
    return (var_favg)

        
# write an array to an adios2 file - upto 4D array.        
def adios2_write_array(file, name, data):
    if (data.ndim == 1):
        start=[0]
        count=[data.shape[0]]
    elif (data.ndim == 2):
        start=[0,0]
        count=[data.shape[0], data.shape[1]]
    elif (data.ndim == 3):
        start=[0,0,0]
        count=[data.shape[0], data.shape[1], data.shape[2]]
    elif (data.ndim == 4):
        start=[0,0,0,0]
        count= [data.shape[0], data.shape[1], data.shape[2], data.shape[3]]
    else:
        print("Data dimension not supported")
        
    file.write(name, np.ascontiguousarray(data), data.shape, start=start, count=count)

