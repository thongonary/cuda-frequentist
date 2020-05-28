## GPU-Accelerated Toy Monte-Carlo Generator for Frequentist Hypothesis Testing


This projects aims to parallelize the Monte Carlo simulation of the test statistics used in frequentist hypothesis testing for binned histograms using CUDA.

The following 2 scenarios are implemented: 

### 1. Neyman-Pearson hypothesis testing
The alternative hypothesis is explicitly required, ie, signal templated needs to be provided.

Usage: 
```
    ./neyman-pearson <number of bins> \
                     <background template file> \
                     <signal template file> \
                     <observed data file> \
                     <number of toys> \
                     [--out <output file> (optional)] \
                     [--GPUonly <0 or 1> (optional)] 
```

### 2. Improved chisquare goodness-of-fit testing
The saturated model is used as the alternative hypothesis, therefore no signal template is required.

Usage:
```
    ./goodness-of-fit <number of bins> \
                      <background template file> \
                      <observed data file> \
                      <number of toys> \
                      <output file> \
                     [--out <output file> (optional)] \
                     [--GPUonly <0 or 1> (optional)] 
```

### Parameters:
<ul>
<li> <code>number of bins</code>: Number of bins used in the histograms for the test. The provided examples in `resources` directory use 20 bins. </li>
<li> <code>background template file</code>: A text file containing the count of each bin in the background template histogram. Example can be found in `resources/background_template.txt`. </li>
<li> <code>signal template file</code>: A text file containing the count of each bin in the signal template histogram. Only required for Neyman-Pearson hypothesis test. Example can be found in `resources/signal_template.txt`. </li>
<li> <code>observed data file</code>: A text file containing the count of each bin in the observed data histogram. Example can be found in `resources/observed_data.txt`. </li>
<li> <code>number of toys</code>: Number of toy Monte Carlo simulation to obtain the test statistics distribution. For a large number of toys (above 1e7), depending on the available device memory, the generation on GPU may be done by batches. Generating more than 1 billion toys, however, might cause segfault due to the host being out of memory. </li>
<li> <code>--out [string]</code> (optional): Destination to save the generated test statistics. Note that for a large number of toys (above 1e7), saving the output to disk may take a long time depending on the disk IO. </li>
<li> <code>--GPUonly [integer]</code> (optional): Whether to run only the generation on GPU. For larger number of toys (above 1e8), generating on CPU might take hours, while on GPU should take less than 1 minute. </li>
</ul>

### Example running commands and outputs

Neyman-Pearson test with 1e6 Monte Carlo toys, running on both CPU and GPU
```
$ ./neyman-pearson 20 resources/background_template.txt resources/signal_template.txt resources/observed_data.txt 1e6 --out np-test.out

Reading 20 bins from background file resources/background_template.txt
Reading 20 bins from data file resources/observed_data.txt
Reading 20 bins from signal file resources/signal_template.txt
Generating 1e+06 toy experiments to obtain the test statistics distribution on CPU
  ████████████████████████████████████████▏ 100.0% [1000000/1000000 | 60.0 kHz | 17s<0s]
Free device memory: 12096/12212 MB
+  Using 977 blocks with 1024 threads per block
Toy-generation run time:
+ On CPU: 16693.4 ms
+ On GPU: 32.1331 ms
Gained a 520-time speedup with GPU
Saving the toy experiments' test statistics to np-test.out.-37.530582.cpu and np-test.out.-37.530582.gpu
  ████████████████████████████████████████▏ 100.0% [1000000/1000000 | 806.1 kHz | 1s<0s]
p-value from Neyman-Pearson hypothesis test: less than 1e-06 (CPU), less than 1e-06 (GPU)
```

Goodness of fit test with 1e6 Monte Carlo toys, running only on both CPU and GPU
```
$ ./goodness-of-fit 20 resources/background_template.txt resources/observed_data.txt 1e6 --out gof-test --GPUonly 0

Reading 20 bins from background file resources/background_template.txt
Reading 20 bins from data file resources/observed_data.txt
Generating 1e+06 toy experiments to obtain the test statistics distribution on CPU
  ████████████████████████████████████████▏ 100.0% [1000000/1000000 | 69.1 kHz | 14s<0s]
Free device memory: 12096/12212 MB
+  Using 977 blocks with 1024 threads per block
Toy-generation run time:
+ On CPU: 14496.6 ms
+ On GPU: 30.9741 ms
Gained a 468-time speedup with GPU
Saving the toy experiments' test statistics to gof-test.1681.881348.cpu and gof-test.1681.881348.gpu
  ████████████████████████████████████████▏ 100.0% [1000000/1000000 | 781.2 kHz | 1s<0s]
p-value from Goodness-of-fit test: 0.003805 (CPU), 0.003793 (GPU)
```

Neyman-Pearson test with 1 billion Monte Carlo toys, running only on GPU and not writing the output to disk
```
$ ./neyman-pearson 20 resources/background_template.txt resources/signal_template.txt resources/observed_data.txt 1e9 --GPUonly 1
Generating 1e+09 toy experiments to obtain the test statistics distribution on GPU only.
Reading 20 bins from background file resources/background_template.txt
Reading 20 bins from data file resources/observed_data.txt
Reading 20 bins from signal file resources/signal_template.txt
Generating in 4 batches
+ Batch 1 of 4: Generating 317207348 toys
    -- Using 220711 blocks with 1024 threads per block
+ Batch 2 of 4: Generating 317207348 toys
    -- Using 220711 blocks with 1024 threads per block
+ Batch 3 of 4: Generating 317207348 toys
    -- Using 220711 blocks with 1024 threads per block
+ Batch 4 of 4: Generating 48377984 toys
    -- Using 47244 blocks with 1024 threads per block
Toy-generation run time on GPU: 29261.3 ms
  ████████████████████████████████████████▏ 100.0% [1000000000/1000000000 | 121.2 MHz | 8s<0s]
p-value from Neyman-Pearson hypothesis test: 2.4e-08 (GPU)
```

