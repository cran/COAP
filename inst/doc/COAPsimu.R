## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----eval = FALSE-------------------------------------------------------------
#  library(COAP)
#  library(GFM)

## ----eval = FALSE-------------------------------------------------------------
#  n <- 200; p <- 200;
#  d= 50
#  rank0 <- 6;
#  q = 5;
#  datList <- gendata_simu(seed = 1, n=n, p=p, d= d, rank0 = rank0, q= q, rho=c(2, 2),
#                          sigma2_eps = 1)
#  X_count <- datList$X; Z <- datList$Z
#  H0 <- datList$H0; B0 <- datList$B0
#  bbeta0 <- cbind( datList$mu0, datList$bbeta0)
#  

## ----eval = FALSE-------------------------------------------------------------
#  hq <- 5; hr <- 6
#  system.time({
#    tic <- proc.time()
#    reslist <- RR_COAP(X_count, Z= Z, q=hq, rank_use= hr, epsELBO = 1e-6)
#    toc <- proc.time()
#    time_coap <- toc[3] - tic[3]
#  })

## ----eval = FALSE-------------------------------------------------------------
#  library(ggplot2)
#  dat_iter <- data.frame(iter=1:length(reslist$ELBO_seq), ELBO=reslist$ELBO_seq)
#  ggplot(data=dat_iter, aes(x=iter, y=ELBO)) + geom_line() + geom_point() + theme_bw(base_size = 20)
#  

## ----eval = FALSE-------------------------------------------------------------
#  library(GFM)
#  metricList <- list()
#  metricList$COAP <- list()
#  metricList$COAP$Tr_H <- measurefun(reslist$H, H0)
#  metricList$COAP$Tr_B <- measurefun(reslist$B, B0)
#  
#  norm_vec <- function(x) sqrt(sum(x^2/ length(x)))
#  metricList$COAP$err_bb <- norm_vec(reslist$bbeta-bbeta0)
#  metricList$COAP$err_bb1 <- norm_vec(reslist$bbeta[,1]-bbeta0[,1])
#  metricList$COAP$Time <- time_coap

## ----eval = FALSE-------------------------------------------------------------
#  metricList$LFM <- list()
#  tic <- proc.time()
#  fit_lfm <- Factorm(X_count, q=q)
#  toc <- proc.time()
#  time_lfm <- toc[3] - tic[3]
#  
#  hbb1 <- colMeans(X_count)
#  metricList$LFM$Tr_H <- measurefun(fit_lfm$hH, H0)
#  metricList$LFM$Tr_B <- measurefun(fit_lfm$hB, B0)
#  metricList$LFM$err_bb1 <- norm_vec(hbb1- bbeta0[,1])
#  metricList$LFM$err_bb <- NA
#  metricList$LFM$Time <- time_lfm

## ----eval = FALSE-------------------------------------------------------------
#  metricList$PoissonPCA <- list()
#  library(PoissonPCA)
#  tic <- proc.time()
#  fit_poispca <- Poisson_Corrected_PCA(X_count, k= hq)
#  toc <- proc.time()
#  time_ppca <- toc[3] - tic[3]
#  
#  hbb1 <- colMeans(X_count)
#  metricList$PoissonPCA$Tr_H <- measurefun(fit_poispca$scores, H0)
#  metricList$PoissonPCA$Tr_B <- measurefun(fit_poispca$loadings, B0)
#  metricList$PoissonPCA$err_bb1 <- norm_vec(log(1+fit_poispca$center)- bbeta0[,1])
#  metricList$PoissonPCA$err_bb <- NA
#  metricList$PoissonPCA$Time <- time_ppca

## ----eval =FALSE--------------------------------------------------------------
#  ## ZIPFA runs very slowly, so we do not run it here.
#  library(ZIPFA)
#  metricList$ZIPFA <- list()
#  system.time(
#    tic <- proc.time()
#    fit_zipfa <- ZIPFA(X_count, k=hq, display = FALSE)
#    toc <- proc.time()
#    time_zipfa <- toc[3] - tic[3]
#  )
#  
#  
#  
#  idx_max_like <- which.max(fit_zipfa$Likelihood)
#  hbb1 <- colMeans(X_count)
#  metricList$ZIPFA$Tr_H <- measurefun(fit_zipfa$Ufit[[idx_max_like]], H0)
#  metricList$ZIPFA$Tr_B <- measurefun(fit_zipfa$Vfit[[idx_max_like]], B0)
#  metricList$PoissonPCA$Time <- time_zipfa
#  

## ----eval = FALSE-------------------------------------------------------------
#  metricList$GFM <- list()
#  tic <- proc.time()
#  fit_gfm <- gfm(list(X_count),  type='poisson', q= q, verbose = F)
#  toc <- proc.time()
#  time_gfm <- toc[3] - tic[3]
#  metricList$GFM$Tr_H <- measurefun(fit_gfm$hH, H0)
#  metricList$GFM$Tr_B <- measurefun(fit_gfm$hB, B0)
#  metricList$GFM$err_bb1 <- norm_vec(fit_gfm$hmu- bbeta0[,1])
#  metricList$GFM$err_bb <- NA
#  metricList$GFM$Time <- time_gfm
#  

## ----eval = FALSE-------------------------------------------------------------
#  PLNPCA_run <- function(X_count, covariates, q,  Offset=rep(1, nrow(X_count))){
#    require(PLNmodels)
#  
#    if(!is.character(Offset)){
#      dat_plnpca <- prepare_data(X_count, covariates)
#      dat_plnpca$Offset <- Offset
#    }else{
#      dat_plnpca <- prepare_data(X_count, covariates, offset = Offset)
#    }
#  
#    d <- ncol(covariates)
#    #  offset(log(Offset))+
#    formu <- paste0("Abundance ~ 1 + offset(log(Offset))+",paste(paste0("V",1:d), collapse = '+'))
#  
#  
#    myPCA <- PLNPCA(as.formula(formu), data = dat_plnpca, ranks = q)
#  
#    myPCA1 <- getBestModel(myPCA)
#    myPCA1$scores
#  
#    res_plnpca <- list(PCs= myPCA1$scores, bbeta= myPCA1$model_par$B,
#                       loadings=myPCA1$model_par$C)
#  
#    return(res_plnpca)
#  }
#  
#  
#  
#    tic <- proc.time()
#    fit_plnpca <- PLNPCA_run(X_count,  covariates = Z[,-1], q= q)
#    toc <- proc.time()
#    time_plnpca <- toc[3] - tic[3]
#  message(time_plnpca, " seconds")
#  
#  metricList$PLNPCA$Tr_H <- measurefun(fit_plnpca$PCs, H0)
#  metricList$PLNPCA$Tr_B <- measurefun(fit_plnpca$loadings, B0)
#  metricList$PLNPCA$err_bb1 <- norm_vec(fit_plnpca$bbeta[,1]- bbeta0[,1])
#  metricList$PLNPCA$err_bb <- norm_vec(as.vector(fit_plnpca$bbeta) - as.vector(bbeta0))
#  metricList$PLNPCA$Time <- time_plnpca

## ----eval =FALSE--------------------------------------------------------------
#  ## GLLVM runs very slowly, so we do not run it here.
#  
#  library(gllvm)
#  colnames(Z) <- c(paste0("V",1: ncol(Z)))
#  tic <- proc.time()
#  fit <- gllvm(y=X_count, X=Z, family=poisson(), num.lv= q, control = list(trace=T))
#  toc <- proc.time()
#  time_gllvm <- toc[3] - tic[3]
#  
#  metricList$GLLVM <- list()
#  metricList$GLLVM$Tr_H <- measurefun(fit$lvs, H0)
#  metricList$GLLVM$Tr_B <- measurefun(fit$params$theta, B0)
#  metricList$GLLVM$err_bb1 <- norm_vec(fit$params$beta0- bbeta0[,1])
#  metricList$GLLVM$err_bb <- norm_vec(as.vector(cbind(fit$params$beta0,fit$params$Xcoef)) - as.vector(bbeta0))
#  metricList$GLLVM$Time <- time_gllvm
#  }
#  

## ----eval = FALSE-------------------------------------------------------------
#  PoisReg <- function(X_count, covariates){
#       library(stats)
#       hbbeta <- apply(X_count, 2, function(x){
#         glm1 <- glm(x~covariates+0, family = "poisson")
#         coef(glm1)
#       } )
#       return(t(hbbeta))
#  }
#  tic <- proc.time()
#  hbbeta_poisreg <- PoisReg(X_count, Z)
#  toc <- proc.time()
#  time_poisreg <- toc[3] - tic[3]
#  metricList$GLM <- list()
#  metricList$GLM$Tr_H <- NA
#  metricList$GLM$Tr_B <- NA
#  metricList$GLM$err_bb1 <- norm_vec(hbbeta_poisreg[,1]- bbeta0[,1])
#  metricList$GLM$err_bb <- norm_vec(as.vector(hbbeta_poisreg) - as.vector(bbeta0))
#  metricList$GLM$Time <- time_poisreg
#  

## ----eval = FALSE-------------------------------------------------------------
#  mrrr_run <- function(Y, X, rank0, q=NULL, family=list(poisson()), familygroup=rep(1,ncol(Y))){
#  
#  
#    require(rrpack)
#  
#    n <- nrow(Y); p <- ncol(Y)
#  
#    if(!is.null(q)){
#      rank0 <- rank0+q
#      X <- cbind(X, diag(n))
#    }
#  
#    svdX0d1 <- svd(X)$d[1]
#    init1 = list(kappaC0 = svdX0d1 * 5) ## this setting follows the example that authors provided.
#  
#    fit.mrrr <- mrrr(Y=Y, X=X[,-1], family = family, familygroup = familygroup,
#                     penstr = list(penaltySVD = "rankCon", lambdaSVD = 0.1),
#                     init = init1, maxrank = rank0)
#    hbbeta_mrrr <-t(fit.mrrr$coef[1:ncol(Z), ])
#    if(!is.null(q)){
#      Theta_hb <- (fit.mrrr$coef[(ncol(Z)+1): (nrow(Z)+ncol(Z)), ])
#      svdTheta <- svd(Theta_hb, nu=q, nv=q)
#      return(list(hbbeta=hbbeta_mrrr, factor=svdTheta$u, loading=svdTheta$v))
#    }else{
#      return(list(hbbeta=hbbeta_mrrr))
#    }
#  
#  
#  }
#  tic <- proc.time()
#  
#  res_mrrrz <- mrrr_run(X_count, Z, rank0)
#  toc <- proc.time()
#  time_mrrrz <- toc[3] - tic[3]
#  
#  metricList$MRRR_Z <- list()
#  metricList$MRRR_Z$Tr_H <- NA
#  metricList$MRRR_Z$Tr_B <-NA
#  metricList$MRRR_Z$err_bb1 <- norm_vec(res_mrrrz$hbbeta[,1]- bbeta0[,1])
#  metricList$MRRR_Z$err_bb <- norm_vec(as.vector(res_mrrrz$hbbeta) - as.vector(bbeta0))
#  metricList$MRRR_Z$Time <- time_mrrrz
#  

## ----eval = FALSE-------------------------------------------------------------
#  tic <- proc.time()
#  res_mrrrf <- mrrr_run(X_count, Z, rank0, q=q)
#  toc <- proc.time()
#  time_mrrrf <- toc[3] - tic[3]
#  metricList$MRRR_F <- list()
#  metricList$MRRR_F$Tr_H <- measurefun(res_mrrrf$factor, H0)
#  metricList$MRRR_F$Tr_B <- measurefun(res_mrrrf$loading, B0)
#  metricList$MRRR_F$err_bb1 <- norm_vec(res_mrrrf$hbbeta[,1]- bbeta0[,1])
#  metricList$MRRR_F$err_bb <- norm_vec(as.vector(res_mrrrf$hbbeta) - as.vector(bbeta0))
#  metricList$MRRR_F$Time <- time_mrrrf
#  

## ----eval = FALSE-------------------------------------------------------------
#  list2vec <- function(xlist){
#    nn <- length(xlist)
#    me <- rep(NA, nn)
#    idx_noNA <- which(sapply(xlist, function(x) !is.null(x)))
#    for(r in idx_noNA) me[r] <- xlist[[r]]
#    return(me)
#  }
#  
#  dat_metric <- data.frame(Tr_H = sapply(metricList, function(x) x$Tr_H),
#                           Tr_B = sapply(metricList, function(x) x$Tr_B),
#                           err_bb1 =sapply(metricList, function(x) x$err_bb1),
#                           err_bb = list2vec(lapply(metricList, function(x) x[['err_bb']])),
#                           Method = names(metricList))
#  dat_metric

## ----eval = FALSE, fig.width=9, fig.height=6----------------------------------
#  library(cowplot)
#  p1 <- ggplot(data=subset(dat_metric, !is.na(Tr_B)), aes(x= Method, y=Tr_B, fill=Method)) + geom_bar(stat="identity") + xlab(NULL) + scale_x_discrete(breaks=NULL) + theme_bw(base_size = 16)
#  p2 <- ggplot(data=subset(dat_metric, !is.na(Tr_H)), aes(x= Method, y=Tr_H, fill=Method)) + geom_bar(stat="identity") + xlab(NULL) + scale_x_discrete(breaks=NULL)+ theme_bw(base_size = 16)
#  p3 <- ggplot(data=subset(dat_metric, !is.na(err_bb1)), aes(x= Method, y=err_bb1, fill=Method)) + geom_bar(stat="identity") + xlab(NULL) + scale_x_discrete(breaks=NULL)+ theme_bw(base_size = 16)
#  p4 <- ggplot(data=subset(dat_metric, !is.na(err_bb)), aes(x= Method, y=err_bb, fill=Method)) + geom_bar(stat="identity") + xlab(NULL) + scale_x_discrete(breaks=NULL)+ theme_bw(base_size = 16)
#  plot_grid(p1,p2,p3, p4, nrow=2, ncol=2)

## ----eval = FALSE-------------------------------------------------------------
#  datList <- gendata_simu(seed = 1, n=n, p=p, d= d, rank0 = rank0, q= q, rho=c(3, 6),
#                          sigma2_eps = 1)
#  X_count <- datList$X; Z <- datList$Z
#  res1 <- selectParams(X_count=datList$X, Z=datList$Z, verbose=F)
#  
#  print(c(q_true=q, q_est=res1['hq']))
#  print(c(r_true=rank0, r_est=res1['hr']))

## -----------------------------------------------------------------------------
sessionInfo()

