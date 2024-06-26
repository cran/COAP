# COAP
High-Dimensional Covariate-Augmented Overdispersed Poisson Factor Model

=========================================================================
<!-- badges: start -->

[![](https://www.r-pkg.org/badges/version-ago/COAP)](https://cran.r-project.org/package=COAP)
[![](https://cranlogs.r-pkg.org/badges/COAP?color=orange)](https://cran.r-project.org/package=COAP)
[![](https://cranlogs.r-pkg.org/badges/grand-total/COAP?color=orange)](https://cran.r-project.org/package=COAP)
<!-- badges: end -->


The current Poisson factor models often assume that the factors are unknown, which overlooks the explanatory potential of certain observable covariates. This study focuses on high dimensional settings, where the number of the count response variables and/or covariates can diverge as the sample size increases. A covariate-augmented overdispersed Poisson factor model is proposed to jointly perform a high-dimensional Poisson factor analysis and estimate a large coefficient matrix for overdispersed count data. 



Check out  our [Biometric paper](https://doi.org/10.1093/biomtc/ujae031) and  [Package Website](https://feiyoung.github.io/COAP/index.html) for a more complete description of the methods and analyses. 

# Installation
"COAP" depends on the 'Rcpp' and 'RcppArmadillo' package, which requires appropriate setup of computer. For the users that have set up system properly for compiling C++ files, the following installation command will work.
```{Rmd}
## Method 1：
if (!require("remotes", quietly = TRUE))
    install.packages("remotes")
remotes::install_github("feiyoung/COAP")

## Method 2: install from CRAN
install.packages("COAP")

```



## Usage
For usage examples and guided walkthroughs, check the `vignettes` directory of the repo. 

* [Simulated data](https://feiyoung.github.io/COAP/articles/COAPsimu.html)

## Simulated codes
For the codes in simulation study, check the `simu_code` directory of the repo.


## News

COAP version 1.1 released! (2023-07-29) 


