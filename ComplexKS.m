%Complex Variant of Krussel-Smith Algorithm

global nj ny

tic

opt_det=false;          
opt_nosr=false;        
opt_ny = 5;             % 1=Markov chain with 5 number of states

% transtion matrix aggregate state

piZ = [ 95/100  5/100;...
        5/100  95/100];
    
gridz=[1.03 0.97];
    
    % parameters
 beta = (0.99)^40;
 alph = 0.3;
 lambd = 0.5;
 tau = 0;
 t=100;
 
 global maxit tol df r nj ny replrate L R delta gridx tau kgrid nk alpha z

tol = 1e-4;     
maxit = 100;    %iterations on r
df = 0.1;   

% Calibration:

% inital guess for psi
psi0 = [0.3;0.3];
psi1 = [1;1];

function func_calibr(opt_det,opt_nosr,opt_ny)

global betta tetta nj jr nx ny nk pi gridy netw pens sr epsi curv pini frac pop totpop grdfac delta alpha L R replrate piz gridz gridk ETA Z nshocks 

close all

rho = 0.04;
betta = 1/(1+rho);
tetta = 2;
delta = 0.05;
alpha = 0.33;

nj=80;
jr=45;

nx=30;         
curv=3.0;       % grid curvature
grdfac=60;      % saving grid scaling factor

% deterministic income component:

netw=1.0;
pens=0.4;%0;%
replrate=0.6;%0;%
epsi=ones(nj,1);
if (jr<nj),
    epsi(jr+1:nj)=0.0;
end
