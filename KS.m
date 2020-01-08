%% Question 1.2%%

global kgrid psi0 t savgrid psi k z1 kss alph beta tau lambd rho zeta zeta_sim rho_sim

% parameters
 beta = (0.99)^40;
 alph = 0.3;
 lambd = 0.5;
 tau = 0;
 t=5000;

 % shocks 
  % eta
  n=11;
  mu=1;
  var=0.9;
  [eta,w]=qnwnorm(n,mu,var);
  % zeta
  sdz=0.13;
  zeta=normrnd(mu,sdz,[1,2]);
   a= repmat(zeta,1,t/2) ;
  zeta_sim = a(randperm(numel(a)));
  % rho
  sdr=0.5;
  rho=normrnd(mu,sdr,[1,2]);
  b = repmat(rho,1,t/2) ;
  rho_sim = b(randperm(numel(b)));

phis=zeros(2,11);
for i=1:2
    for j=1:11
        phis(i,j)= 1/(1+((1-alph)/(alph*(1+lambd)*rho(i)))*(lambd*eta(j)+tau*(1+lambd*(1-eta(j)))));
    end
end

probability_eta=exp(eta);
phi = sum(phis(1,:)*probability_eta(1,:)*0.5)+ sum(phis(2,:)*probability_eta(2,:)*0.5)  ; 
saving = beta*phi / (1 + beta * phi);
 
% Simulating capital in logs
 k=zeros(1,t);
 k(1)=exp(log(saving)+log(1-tau)+log(1-alph));

 for i=2:t-1
      k(i)=exp(log(saving)+log(1-tau)+log(1-alph)+alph*log(k(i-1)));
 end


%% Simple Krussel Smith Algorithm

% Guess for each good state and bad state, i.e. recession and boom.
 b = [zeta(1),rho(1)];
 g = [zeta(2),rho(2)];
 z1=vertcat(b,g);

% Guess for psi
psi0 = [length(z1),length(z1)];
 for i=1:2
     psi0(i,1) = log(saving)+log(1-tau)+log(1-alph)+log(zeta(1));
     psi0(i,2) = alph ;

 end
 
% State space
kss=exp(k(1));
y=1;
kgrid=linspace(0.5*kss,1.5*kss,5);
 
eps=0.0000001;
psi_guess= psi0;
psi_iter= ones(2,2);
iterations = 0;
delta = 0.95;

kmin=log(0.5)+k(i);
kmax=log(1.5)+k(i);

n=5;

savgrid=kmin:(kmax-kmin)/(n-1):kmax;
while abs(psi_iter-psi_guess)>eps
      savgrid=kmin:(kmax-kmin)/(n-1):kmax;
    [k_sim, c1,c2]=func_sim(savgrid);
     c1l = c1(1,:);
     c1l = c1l(500:end);
     c1h = c1(2,:);
     c1h = c1h(500:end);
     c2l = c2(1,:);
     c2l = c2l(500:end);
     c2h = c2(2,:);
     c2h = c2h(500:end);
     utill = 1/(t-500)*sum(((1-beta)/beta)*log(c1l)+((1-beta)/beta)*log(c2l));
     utilh = 1/(t-500)*sum(((1-beta)/beta)*log(c1h)+((1-beta)/beta)*log(c2h));
     psi_int = func_reg(k_sim(1,:),k_sim(2,:));
     psi_guess=psi_iter;
     psi_iter = delta*psi_int+(1-delta)*psi_guess;
               
    end
   

disp(['iteration # ', num2str(iterations)]);
disp(['saving', num2str(savgrid)]);
 disp(['utill ', num2str(utill)]);
disp(['utilh ', num2str(utilh)]);
%% Household problem
function [savgrid]=func_hh(psi)
global kgrid k z1 alph beta tau lambd
n=5;

savgrid= ([length(psi),length(kgrid)]);

for i=1:2
      for j=1:n
          k1(j,i)= exp(psi(j,1)+psi(j,2)*log(kgrid));
     end
 end

 for j=1:n
     for i=1:2
        variable = @(a) (1-beta*((1-tau)*(1-alph)*k.^alph*z1(i,1)-a)*(1+alph*k(i,j).^(alph-1)*z1(i,1)*z1(i,2)))/((a*(1+alph*k1(i,j).^(alph-1)*z1(i,1)*z1(i,2)))+lambd*(1-alph)*k1(i,j).^alph*z1(i,1)*(1-tau)+(1-lambd)*tau*(1-alph)*k1(i,j)^alph*z1(i,1)*(1+lambd))/(1-lambd);
        x0 = [0.01,2]; % initial interval
        variable(x0(1));
        savgrid(i,j) = fzero(variable,x0);
             
     end
     
 end
end

%% Simulation


function [k_sim, c1,c2]=func_sim(savgrid)

global  t z1 kss alph beta tau zeta_sim rho_sim

c = repmat(savgrid(1),1,t) ;
s1 = c(randperm(numel(c)));
d = repmat(savgrid(2),1,t) ;
s2 = d(randperm(numel(d)));
sav_sim = [length(s1),length(s2)];
k_sim = zeros(length(z1),t);
k_sim(1) = kss;
c1 = zeros(length(z1),t);
c2 = zeros(length(z1),t);
u = zeros(length(z1));

for i=1:range(z1)
    for j=2:t
     k_sim(i,j) = exp(log(sav_sim(i,j)+log(1-tau)+log(1-alph)+log(zeta_sim(j)+alph*log(k_sim(i,j-1)))));
     c1(i,j) = (1-sav_sim(i,j))*(1-tau)*(1-alph)*zeta_sim(j)*k_sim(i,j).^(alph);
     c2(i,j) = beta*c1(i,j)*(1-alph*k_sim(i,j).^(alph-1)*zeta_sim(j)*rho_sim(j));
        
    end
end
end

%% Regression 
function [psi]=func_reg(kl, kh)
global  t

    kl=zeros(1,t-1);
    kh=zeros(1,t-1);
    k1h=zeros(2,t);
    k1l=zeros(2,t);
    lnkl = log(kl);
    lnkh = log(kh);
    lnk1h = log(k1h);
    lnk1l = log(k1l); 
    Xl=lnkl(500:end);
    Xh=lnkh(500:end);
    Yh=lnk1h(500:end);
    Yl=lnk1l(500:end);

    %run the regression and calculate psi
    X = Xh;
    Yh = Xh + ones(1,length(Xh));
    mdl = fitlm(X,Yh);
    psi_h = mdl.Coefficients.Estimate;
    X1 = Xl;
    Yl = X1 + ones(1,length(Xl));
    mdl = fitlm(X1,Yl);
    psi_l = mdl.Coefficients.Estimate;%
    psi = [(psi_l),(psi_h)];
    
end

