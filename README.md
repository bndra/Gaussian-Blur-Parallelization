# Gaussian-Blur-Parallelization
Several NVIDIA CUDA C++ implementations of a Gaussian Blur operation. Sequential, partially parallel, fully parallel implementations were made, each as an upgrade to the last. Created as an exercise in learning CUDA C++.

Performs the blur on a randomized 16*16 greyscale pixel matrix; aka a 2D array of 8 bit unsigned integers.

Images included are outputs from nvprof, a cuda diagnostic/performance measuring tool:
1: seq_nvprof.png
2: pconvolutions_nvprof.png
3: pblur_nvprof.png

My sequential Gaussian Blur takes 653 microseconds which was improved to around 350 and then all the way down to 5.67 microseconds. The speedup is about 110 times as fast, which I'm pretty happy with. More formal analysis of theoretical speedup factors/asymptotic analysis could be performed for potentially even faster speeds.

Accuracy of calculations were checked by hand.
