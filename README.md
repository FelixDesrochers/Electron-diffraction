# Electron-diffraction

A python script that displays an animation of the propagation of a gaussian wave packet and its interaction with an arbitrary number of slits. The program solves the two-dimensional time-dependant Schrödinger equation using Crank-Nicolson algorithm.

## Running

To run this code simply clone this repository and run the animate_wave_function.py script with python (the numpy and matplotlib modules are required):
 
```
$ git clone https://github.com/FelixDesrochers/Electron-diffraction/
$ cd Electron-diffraction
$ python animate_wave_function.py 
```

The parameters of the simulation can be modified at the beginning of the animate_wave_function.py script. 

## Examples

### Large gaussian wave packet through one slit
![Alt text](https://github.com/FelixDesrochers/Electron-diffraction/blob/master/animation/one_slit_thick.gif?raw=true "Title")

### Small gaussian wave packet through one slit
![Alt text](https://github.com/FelixDesrochers/Electron-diffraction/blob/master/animation/2D_oneslit_dx008_dt0005_yf1.gif?raw=true "Title")

### Large gaussian wave packet through two slits
![Alt text](https://github.com/FelixDesrochers/Electron-diffraction/blob/master/animation/2D_2slits_dx008_dt0005_yf10.gif?raw=true "Title")
