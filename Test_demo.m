


clc;clear;close all;
addpath(genpath('.'));

% --------------------------------------- load data
data_id = '3sources3vbigRnSp';
[X,gt]  = data_load(data_id);


% --------------------------------------- paras setting

paras.lambda = 1e-1;
paras.beta   = 1e-2;
paras.gama   = 1e-2;
paras.Ns     = 30;
paras.Nc     = 80;

[nmi,ACC,AR,f,p,r,RI] = DSS_MVC_Learning(X,gt,paras);


        
        
% --------------------------------------- clustering 
[nmi,ACC,AR,f,p,r,RI] = DSS_MVC_Learning(X,gt,paras);
disp(['ACC=',num2str(ACC),' NMI=',num2str(nmi),' AR=',num2str(AR),' f-score=',num2str(f),' precison=',num2str(p),' recall=',num2str(r)])

 

