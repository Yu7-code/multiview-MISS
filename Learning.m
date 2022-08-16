
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------

function [nmi,ACC,AR,f,p,r,RI] = DSS_MVC_Learning(X,gt,paras)

% ---------------------------------- data pre-processing
V = size(X,2);      % number of views，返回x的列数
N = size(X{1},2);   % number of samples，
cls_num  = size(unique(gt),1);

lambda = paras.lambda;
beta   = paras.beta;
gama   = paras.gama;
Ns     = paras.Ns;
Nc     = paras.Nc;


for i = 1:V
    X{i} = X{i}./repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1);
end

% ---------------------------------------------- initialize variables

for i=1:V
    r=size(X{i},2);
    totalNum=r;
    randomIndex=1+floor(rand(1,floor(totalNum*0.1))*totalNum);
    X{i}(:,randomIndex)=0;
end

for i = 1:V
   
    E1{i} = zeros(Ns+Nc,N);
    E2{i} = zeros(Ns+Nc,N);
    EH{i} = zeros(Ns+Nc,N);
    Y1{i} = zeros(Ns+Nc,N);
    Y2{i} = zeros(Ns+Nc,N);
    YQ{i} = zeros(N,N);
    YH{i} = zeros(Ns+Nc,N);
    YM{i} = zeros(Ns+Nc,N);
    Hv{i} = rand(Ns+Nc,N);
    Tv{i} = zeros(Nc,N);
    Zv{i} = zeros(N,N);  
    Mv{i} = zeros(N,N);
    Q=X{i};
    X{i}=Q;
    E=Mv{i};
    option=any(Q,1);
    for j = 1:size(X{i},2)
        if option(j)==0
            for m =1:N
             E(m,j)=rand(1)*2;
            end
            Mv{i}=E;
        else if option(j)==1
                
                  E(j,j)=1;
                  Mv{i}=E;
        end
    end
    end
end
T   = rand(Ns,N);
Z    = zeros(N,N);
Yz   = zeros(N,N);


% ----------------------------------------------
IsConverge = 0;
mu         = 1e-4;
pho        = 1.5;  % 1.5
max_mu     = 1e6;
max_iter   = 200;
iter       = 1;
thresh     = 1e-6;

% -------------------------------------------------------------------------
while (IsConverge == 0&&iter<max_iter+1)
    
    sum_DtD = zeros(N,N);
    sum_DtC =zeros(N,N);
    
    for i = 1:V
        
        % ---------------------------------- update Pv
        P{i} = updatePP(Y1{i},mu,Hv{i} + E1{i}, X{i});
        
        % ---------------------------------- update Hv
        % ZZ    = (eye(N)-Z-Zv{i});
        tepM=Mv{i};
        Hv{i} = (P{i}*X{i}-E1{i}-EH{i}+[T;Tv{i}]+[T;Tv{i}]*tepM'+Y1{i}/mu+YH{i}/mu-YM{i}*tepM'/mu)/(2*eye(N)+Mv{i}*tepM');
        
        % ---------------------------------- update Zv
        tepB  = [T;Tv{i}];
        tepA  = [T;Tv{i}] - [T;Tv{i}]*Z - E2{i};
        Qv{i} = ComputeSaprseValues(Zv{i} + YQ{i}/mu, gama/mu);
        Zv{i} = (tepB'*tepB + eye(N) ) \ (tepB'*tepA + Qv{i} - diag(diag(Qv{i})) + (tepB'*Y2{i} - YQ{i})/mu);
        clear tepB tepA;
         % ----------------------------------update Mv
        tepH=Hv{i};
        Mv{i}=(tepH'*Hv{i})\(tepH'*[T;Tv{i}]-tepH'*YM{i}/mu);
        % ----------------------------------update Tv
        ZZ    = (eye(N)-Z-Zv{i});
        Tv{i}=(Hv{i}(Ns+1:end,:)*Mv{i}+YM{i}(Ns+1:end,:)/mu+Hv{i}(Ns+1:end,:)+EH{i}(Ns+1:end,:)-YH{i}(Ns+1:end,:)/mu+E2{i}(Ns+1:end,:)*ZZ'-Y2{i}(Ns+1:end,:)*ZZ'/mu) / (ZZ*ZZ'+2*eye(N)+eye(N)*1e-8);
        % ---------------------------------- update Ev
        G = [P{i}*X{i} - Hv{i}+ Y1{i}/mu; [T;Tv{i}] - [T;Tv{i}]*Z - [T;Tv{i}]*Zv{i} + Y2{i}/mu];
        Ev = solve_l1l2(G,lambda/mu);
        
        E1{i} = Ev(1:Ns+Nc,:);
        E2{i} = Ev(Ns+Nc+1:end,:);
        % ----------------------------------update EH(这里不知道对不对）
        G2=([T;Tv{i}]-Hv{i}+YH{i}/mu);
        EH{i}=solve_l1l2(G2,beta/mu);
        
        % ----------------------------------
        tepC    = [T;Tv{i}] - [T;Tv{i}]*Zv{i} - E2{i};
        tepD    = [T;Tv{i}];
        sum_DtD = sum_DtD + tepD'*tepD;
        sum_DtC = sum_DtC + (tepD'*(tepC + Y2{i}/mu));
        
    end
    
    % ---------------------------------- update J
    J = softth(Z+Yz/mu,1/mu);  %  + eye(N)*1e-8
    
    % ---------------------------------- update Z
    Z  =  inv(eye(size(sum_DtD,2)) + sum_DtD + eye(N)*1e-8)* (J - Yz/mu + sum_DtC);
    
    % ---------------------------------- update T
    tepTs = zeros(Ns,N);
    tepZ  = zeros(N,N);
    for i = 1:V
        Zvv   = eye(N) - Z - Zv{i};
        tepTs = (Hv{i}(1:Ns,:)*Mv{i}+YM{i}(1:Ns,:)/mu+Hv{i}(1:Ns,:)+EH{i}(1:Ns,:)-YH{i}(1:Ns,:)/mu+E2{i}(1:Ns,:));
        tepZ  = tepZ + Zvv*Zvv'+2*eye(N);
    end
    T  = tepTs * inv(tepZ + eye(N)*1e-8);   % revise 0926 tepHs / (tepZ + eye(N)*1e-8);
    
    % ---------------------------------- updata multipliers
    for i = 1:V
        Y1{i}= Y1{i} + mu*(P{i}*X{i}-Hv{i}-E1{i});
        Y2{i}=Y2{i}+mu*([T;Tv{i}]-[T;Tv{i}]*Z-[T;Tv{i}]*Zv{i}-E2{i});
        YQ{i}= YQ{i} + mu*(Zv{i} - Qv{i} + diag(diag(Qv{i})));
        YH{i}=YH{i}+mu*([T;Tv{i}]-Hv{i}-EH{i});
        YM{i}=YM{i}+mu*(Hv{i}*Mv{i}-[T;Tv{i}]);
    end
    Yz = Yz + mu*(Z-J);
    mu = min(pho*mu, max_mu);
    
    % ----------------------------------- convergence conditions
    min_err = 0;
    for i = 1:V
        err1(i) = norm(P{i}*X{i}-Hv{i}-E1{i},inf);
        err2(i) = norm([T;Tv{i}]-[T;Tv{i}]*Z-[T;Tv{i}]*Zv{i}-E2{i},inf);
        errH(i) = norm([T;Tv{i}]-Hv{i}-EH{i},inf);
        errM(i) = norm(Hv{i}*Mv{i}-[T;Tv{i}],inf);
        errQ(i) = norm(Zv{i}-Qv{i}+diag(diag(Qv{i})),inf);
        
    end
    max_err0 = max([err1(:);err2(:);errH(:);errM(:);errQ(:)]);
    max_err  = max([max_err0,norm(Z-J,inf)]);
    
    % -----------------------------------
    if max_err < thresh
        IsConverge = 1;
    end
    cov_val(iter) = max_err; % norm(Z-J,inf);
    iter          = iter + 1;
    
end

% -------------------------------------------------------------------------
Z_all = zeros(size(Z));

for j = 1:V
    Zi = Zv{j};
   
    Zi    = Z + Zi;
    Z_all = Z_all + (abs(Zi) + abs(Zi'));
    
end

Z_all = Z_all / V;
[nmi,ACC,AR,f,p,r,RI] = clustering(Z_all, cls_num, gt);
