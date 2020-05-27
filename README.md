## GPU-Accelerated Toy Monte-Carlo Generator for Frequentist Hypothesis Testing


This projects aims to parallelize the Monte Carlo simulation of the test statistics used in frequentist hypothesis testing for binned histograms using CUDA.

The following 2 scenarios are implemented: 

### 1. Neyman-Pearson hypothesis testing
The alternative hypothesis is explicitly required, ie, signal templated needs to be provided.

How to run: 
```
    ./neyman-pearson <number of bins> \
                     <background template file> \
                     <signal template file> \
                     <observed data file> \
                     <number of toys> \
                     <output file> 
```

### 2. Improved chisquare goodness-of-fit testing
The saturated model is used as the alternative hypothesis, therefore no signal template is required.

How to run:
```
    ./goodness-of-fit <number of bins> \
                      <background template file> \
                      <observed data file> \
                      <number of toys> \
                      <output file>
```

### Example running commands and outputs
```
$ ./neyman-pearson 20 resources/background_template.txt resources/signal_template.txt resources/observed_data.txt 1e7 test.out

Reading 20 bins from background file resources/background_template.txt
Reading 20 bins from data file resources/background_template.txt
Reading 20 bins from signal file resources/background_template.txt
Generating 1e+07 toy experiments to obtain the test statistics distribution
  ████████████████████████████████████████▏ 100.0% [10000000/10000000 | 63.7 kHz | 157s<0s]
p-value from Neyman-Pearson hypothesis test: less than 1e-07
Saving the toy experiments' test statistics to test.out.cpu
  ████████████████████████████████████████▏ 100.0% [10000000/10000000 | 1.9 MHz | 5s<0s]
```
```
$ ./goodness-of-fit 20 resources/background_template.txt resources/observed_data.txt 1e6 test.out

Reading 20 bins from background file resources/background_template.txt
Reading 20 bins from data file resources/background_template.txt
Generating 1e+06 toy experiments to obtain the test statistics distribution
  ████████████████████████████████████████▏ 100.0% [1000000/1000000 | 71.2 kHz | 14s<0s]
p-value from Goodness-of-fit test: 0.003805
Saving the toy experiments' test statistics to test.out.cpu
  ████████████████████████████████████████▏ 100.0% [1000000/1000000 | 1.9 MHz | 1s<0s]
```
