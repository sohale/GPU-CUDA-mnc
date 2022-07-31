Parallel numerical integration using Monte Carlo on GPU

Last commmit: on 8 May 2015

Based on a method by Allen Genz 1999.


Changes log:
* Wrote MC kernel for CUDA using Matlab Parallel toolbox as host (2014)
* Pushed code to github
* Set up docker-based Nvidia compiler
* Rewrote practice cuda code
* Attempted to use AWS GPU instances (The AWS instance didn't actually have the GPU)
* Attempted GCP instances (failed)
* Wrote pure cuda based host (instead of Matlab host) code
* Change of default branch
