# Electron-diffraction

This program is a numerical solution to one of physics most popular experiment: the diffraction of a single electron through two slits. More precisely it is a python script that displays an animation of the propagation of a gaussian wave packet and its interaction with an arbitrary number of slits. The program solves the two-dimensional time-dependant Schrödinger equation using Crank-Nicolson algorithm and perfectly reflecting boundary conditions.

## Running

To run this code simply clone this repository and run the animate_wave_function.py script with python (the numpy and matplotlib modules are required):
 
```
$ git clone https://github.com/FelixDesrochers/Electron-diffraction/
$ cd Electron-diffraction
$ python animate_wave_function.py 
```

The parameters of the simulation (shape and size of the potential, shape and speed of the wave packet, etc.) can be modified at the beginning of the animate_wave_function.py script. 

## Examples

Here are some examples of animations produced by the program. We can see that in the large gaussian wave packet limit, a diffraction pattern similar to the one obtain in the Fraunhoffer approximation of the diffraction of a plane wave. For a more elaborate dicusssion on the expected shape of the diffraction pattern of a gaussian wave packet see for example

Zecca, A. (2013). Gaussian wave packets passing through two slits: contribution of confinement and tunneling to the diffraction pattern. Adv. Studies Theor. Phys, 7, 287.

Zecca, A. (2011). Diffraction of Gaussian wave packets by a single slit. The European Physical Journal Plus, 126(2), 18.

### Large gaussian wave packet through one slit
![Alt text](https://github.com/FelixDesrochers/Electron-diffraction/blob/master/animation/one_slit_thick.gif?raw=true "Title")

### Small gaussian wave packet through one slit
![Alt text](https://github.com/FelixDesrochers/Electron-diffraction/blob/master/animation/2D_oneslit_dx008_dt0005_yf1.gif?raw=true "Title")

### Large gaussian wave packet through two slits
![Alt text](https://github.com/FelixDesrochers/Electron-diffraction/blob/master/animation/2D_2slits_dx008_dt0005_yf10.gif?raw=true "Title")
