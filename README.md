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
This program package was written by Y. Ding, R. Kondor and J. Eskreis-Winkler. If you have any questions/comments/concerns, please contact dingy[at]uchicago.edu. 

===MKA Copyright===
Copyright (c) 2017 Y. Ding, R. Kondor and J. Eskreis-Winkler.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither name of copyright holders nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.