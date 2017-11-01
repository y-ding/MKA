# Multiresolution Kernel Approximation for Gaussian Process Regression

This is the MATLAB implementation of the Multiresolution Kernel Approximation for Gaussian Process Regression as described in:

Y. Ding, R. Kondor and J. Eskreis-Winkler, [Multiresolution kernel approximation for Gaussian process regression](https://arxiv.org/abs/1708.02183) (2017)

## Requirements
* C++11
* [Eigen](http://eigen.tuxfamily.org/index.php)
* [pMMF](http://people.cs.uchicago.edu/~risi/MMF/index.html)
* [Matio](https://sourceforge.net/projects/matio/)

## Installation/Setup
Install pMMF to the subdirectory of MKA folder. Make sure to change the variables in Makefile.options to your environment settings. Run the following command to create the pMMF executable in the MKA directory.
```bash
make all
```

## Run the demo
After compiling, you can run the script "MGP/gprExperiments/demo_1D" to run the toy experiment for Snelson's 1D data in the paper. The folder "gprExperiments" also includes all the experiment scripts on real data in the paper.

## Contact
If you have any questions/comments/concerns, please contact dingy[at]uchicago.edu. In particular,
it would be fantastic if you could report any bugs!