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
<li> <code>number of bins</code>: Number of bins used in the histograms for the test. The provided examples in <code>resources</code> directory use 20 bins. </li>
<li> <code>background template file</code>: A text file containing the count of each bin in the background template histogram. Example can be found in <code>resources/background_template.txt</code>. </li>
<li> <code>signal template file</code>: A text file containing the count of each bin in the signal template histogram. Only required for Neyman-Pearson hypothesis test. Example can be found in <code>resources/signal_template.txt</code>. </li>
<li> <code>observed data file</code>: A text file containing the count of each bin in the observed data histogram. Example can be found in <code>resources/observed_data.txt</code>. </li>
<li> <code>number of toys</code>: Number of toy Monte Carlo simulation to obtain the test statistics distribution. For a large number of toys (above 1e7), depending on the available device memory, the generation on GPU may be done by batches if the output is kept with the <code>--out</code> option. </li>
<li> <code>--out [string]</code> (optional): Destination to save the generated test statistics. Note that for a large number of toys (above 1e7), saving the output to disk may take a long time depending on the disk IO. If this option is not specified, the generated test statistics will not be kept and only the p-value will be computed.</li>
<li> <code>--GPUonly [integer]</code> (optional): Whether to run the generation only on GPU. </li>
</ul>

### Example running commands and outputs

Neyman-Pearson test with 1e6 Monte Carlo toys, running on both CPU and GPU
```
$ ./neyman-pearson 20 resources/background_template.txt resources/signal_template.txt resources/observed_data.txt 1e6 --out np-test.out

[INPUT] Will save output to disk
[INPUT] Reading 20 bins from background file resources/background_template.txt
[INPUT] Reading 20 bins from data file resources/observed_data.txt
[INPUT] Reading 20 bins from signal file resources/signal_template.txt

Generating 1000000 toy experiments to obtain the test statistics distribution on CPU
  ████████████████████████████████████████▏ 100.0% [1000000/1000000 | 62.4 kHz | 16s<0s]

Generating 1000000 toy experiments to obtain the test statistics distribution on GPU
[INFO] Free device memory: 11790/12209 MB
+  Using 976 blocks with 1024 threads per block

Toy-generation run time:
+ On CPU: 16054.7 ms
+ On GPU: 33.7967 ms
Gained a 475-time speedup with GPU

Saving the toy experiments' test statistics to np-test.out.-37.530582.cpu and np-test.out.-37.530582.gpu
  ████████████████████████████████████████▏ 100.0% [1000000/1000000 | 961.5 kHz | 1s<0s]

p-value from Neyman-Pearson hypothesis test: less than 1e-06 (CPU), less than 1e-06 (GPU)
```

Goodness of fit test with 1e6 Monte Carlo toys, running only on both CPU and GPU
```
$ ./goodness-of-fit 20 resources/background_template.txt resources/observed_data.txt 1e6 --out gof-test --GPUonly 0

[INPUT] Will save output to disk
[INPUT] Reading 20 bins from background file resources/background_template.txt
[INPUT] Reading 20 bins from data file resources/observed_data.txt

Generating 1000000 toy experiments to obtain the test statistics distribution on CPU
  ████████████████████████████████████████▏ 100.0% [1000000/1000000 | 70.8 kHz | 14s<0s]

Generating 1000000 toy experiments to obtain the test statistics distribution on GPU
[INFO] Free device memory: 11790/12209 MB
+  Using 976 blocks with 1024 threads per block

Toy-generation run time:
+ On CPU: 14130.6 ms
+ On GPU: 32.747 ms
Gained a 432-time speedup with GPU

Saving the toy experiments' test statistics to gof-test.1681.881348.cpu and gof-test.1681.881348.gpu
  ████████████████████████████████████████▏ 100.0% [1000000/1000000 | 862.1 kHz | 1s<0s]

p-value from Goodness-of-fit test: 0.003805 (CPU), 0.003912 (GPU)
```

Neyman-Pearson test with 1 billion Monte Carlo toys, running only on GPU and not writing the output to disk
```
$ ./neyman-pearson 20 resources/background_template.txt resources/signal_template.txt resources/observed_data.txt 1e9 --GPUonly 1

[INPUT] Use GPU only
[INPUT] Reading 20 bins from background file resources/background_template.txt
[INPUT] Reading 20 bins from data file resources/observed_data.txt
[INPUT] Reading 20 bins from signal file resources/signal_template.txt

Generating 1000000000 toy experiments to obtain the test statistics distribution on GPU
[INFO] Free device memory: 11794/12209 MB
+  Using 239027 blocks with 1024 threads per block
Toy-generation run time on GPU: 26842.7 ms

p-value from Neyman-Pearson hypothesis test: 1.5e-08 (GPU)
```

