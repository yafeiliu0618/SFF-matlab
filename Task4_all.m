% Task 4
% True distribution parameters
ks = [3 6];%degree of freedom
mus = [-3 -2 -1 0];%asy parameter
loc=1; scale=2;
T=[250 500 2000]; rep=100; B = 100;% bootsrapping parameters
alpha=0.05;  %alpha quantile for VaR / ES
alpha2=0.1;%nomial coverage probability

output_all = zeros(length(T),4,length(ks)*length(mus));

for ki = 1:length(ks)
for mui = 1: length(mus)

    k = ks(ki); mu = mus(mui);
    disp(['k = ', num2str(k), '; mu = ', num2str(mu)]);
    act_cov=zeros(length(T),2);ci_len=zeros(length(T),2);
    
    initvec=[k mu loc scale]; % guessed df mu=loc c=scale
    TrueES = loc+scale*nctES(alpha,k,mu); %theoretical ES
    
    for i=1:length(T)
    
        %generate data
        x_normal = normrnd(mu,1,T(i),rep);
        y_chi = chi2rnd(k,T(i),rep);
        transasyt = loc+scale*x_normal./((y_chi./k).^(1/2));
        
        [act_cov(i,:), ci_len(i,:)] = BSFitOnES(B, transasyt, ...
        TrueES, alpha, alpha2, initvec, @BS_nctES);
    end
    
    output_all(:,[1 2],((length(mus)*(ki-1))+mui))=act_cov;
    output_all(:,[3 4],length(mus)*(ki-1)+mui)=ci_len;
    disp([act_cov ci_len])

end
end

function [act_cov, ci_len] = BSFitOnES(B, data_TXrep, TrueES, alpha, alpha2, initvec, myfunc)

[T, rep]=size(data_TXrep); 
ES_wLS_analytic = TrueES;
bootsESvec1= zeros(rep,B); bootsESvec2 = zeros(rep,B);
% ES_para=zeros(rep,1);
for i=1:rep    
    [bootsESvec1(i,:), bootsESvec2(i,:)] = myfunc(data_TXrep(:,i), ...
        T, B, alpha, initvec);

end
bootsci1 = quantile(bootsESvec1,[alpha2/2,1-alpha2/2],2);
bootsci2 = quantile(bootsESvec2,[alpha2/2,1-alpha2/2],2);

actual_coverage_prob1=( (ES_wLS_analytic>=bootsci1(:,1) ) & ...
    (ES_wLS_analytic<=bootsci1(:,2) ) );
actual_coverage_prob2=( (ES_wLS_analytic>=bootsci2(:,1) ) & ...
    (ES_wLS_analytic<=bootsci2(:,2) ) );


act_cov=[mean(actual_coverage_prob1), mean(actual_coverage_prob2)];
ci_len=[mean(bootsci1(:,2)-bootsci1(:,1)), ...
    mean(bootsci2(:,2)-bootsci2(:,1))];
end

function [bootsESvec1, bootsESvec2] = BS_nctES(data, T, B, alpha, initvec)

bootsESvec1=zeros(B,1); bootsESvec2=zeros(B,1);

% non-parametric bootstrapping
ind = unidrnd(T,[T,B]);
bootstrap1 = data(ind);
bootsVaR1=quantile (bootstrap1, alpha, 1);

% parametric bootstrapping
[param]=nctmle(data,initvec);
paramle = param'; %df mu loc scale

x_normal = normrnd(paramle(2),1,T,B);
y_chi = chi2rnd(paramle(1),T,B);
bootstrap2 = paramle(3)+paramle(4)*x_normal./((y_chi./paramle(1)).^(1/2));

% bootstrap2 = paramle(2)+paramle(3)*trnd(paramle(1),T,B);
% bootsVaR2 = quantile (bootstrap2, alpha, 1);
  for b = 1:B
    % non-parametric bootstrapping
    bootstrap1b = bootstrap1(:,b);
    bootsESvec1(b)=mean(bootstrap1b(bootstrap1b<=bootsVaR1(b))); 
    
    % parametric boot
    bootstrap2b = bootstrap2(:,b);
    [param_bs]=nctmle(bootstrap2b,initvec);
%     paramle22_bs = param_bs';% df asy loc scale
    bootsESvec2(b) = param_bs(3)+param_bs(4)*nctES(alpha,param_bs(1),param_bs(2));
%     bootsESvec2(b)=mean(bootstrap2b(bootstrap2b<=bootsVaR2(b))); 
  end

end 

function [ES, VaR] = nctES(xi,v,theta)
howfar = nctinv(1e-7,v,theta); % how far into the left tail to integrate 
VaR = nctinv(xi,v,theta); % matlab routine for the quantile
I = quadl(@int,howfar,VaR,1e-6,[],v,theta); ES = I/xi;
end

function I = int(tvec,v,theta), pdf = nctpdf(tvec,v,theta); I = tvec.* pdf;    
end

function MLE = nctmle(x,initvec)
tol=1e-5; 
opts=optimset('Disp','none','LargeScale','Off', ...
'TolFun ' ,tol , 'TolX ' ,tol , 'Maxiter ' ,200);
A = [];b = [];
Aeq = [];
beq = [];
lb = [1,-Inf,-Inf,0.01];
ub = [10,Inf,Inf,100];
nonlcon = [];
MLE = fmincon(@(param) ncttloglik (param,x),initvec,A,b,Aeq,beq,lb,ub,nonlcon,opts);
% MLE = fminunc(@(param) ncttloglik (param,x) , initvec , opts) ;
end

function ll=ncttloglik(param,x)% df asy loc scale
% param = param';
df=param(1) ; mu=param(2) ; loc=param(3); scale=param(4);
x_trans = (x-loc)/scale;
ll = log(nctpdf(x_trans,df,mu)/scale); ll = -mean(ll);
end


