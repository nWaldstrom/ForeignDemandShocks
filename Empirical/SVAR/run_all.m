%==========================================================================
%% housekeeping
%==========================================================================
%clear variables;
close all;
userpath('clear');
tic;

rng('default'); % reinitialize the random number generator to its startup configuration
rng(0);         % set seed

disp(strcat(svar, '-ALL'))

currdir=pwd;
addpath([currdir,'/helpfunctions']); % set path to helper functions



%==========================================================================
%% write data in Rubio, Waggoner, and Zha (RES 2010)'s notation
%==========================================================================
% Panel data...
Y = importdata(strcat('../data/MATLAB/Y_', svar, '.csv'));
X = importdata(strcat('../data/MATLAB/X_', svar, '.csv'));
Y = 100*Y.data;
X = 100*X.data;



%==========================================================================
%% model setup
%==========================================================================
nlag      = 4;               % number of lags
nvar      = 4;               % number of endogenous variables
nex       = size(X,2)-nlag*nvar; % set equal to 1 if a constant is included; 0 otherwise
m         = nvar*nlag + nex; % number of exogenous variables
nd        = 2e5;             % number of orthogonal-reduced-form (B,Sigma,Q) draws
iter_show = 2e5;             % display iteration every iter_show draws
horizon   = 40;              % maximum horizon for IRFs
index     = 40;              % define  horizons for the FEVD
NS        = 1;               % number of objects in F(THETA) to which we impose sign and zero restrictios: F(THETA)=[L_{0}]
e         = eye(nvar);       % create identity matrix
maxdraws  = 1e4;             % max number of importance sampling draws
conjugate = 'structural';    % structural or irfs or empty
nshocks = 3;                 % number of shocks
shock   = 2;                 % shock number

% Transform data
q = nvar*nlag+1;
X(1:end,q:end) = X(1:end,q:end)/100;
T = length(Y);


%==========================================================================
%% identification: declare Ss and Zs matrices
%==========================================================================
horizons = 0;

if nvar == 8 && nshocks == 3
    % sign restrictions
    S = cell(nvar,1);
    for ii=1:nvar
        S{ii}=zeros(0,nvar);
    end
    S{1} = [1 0 0 0 0 0  0  0];
    S{2} = [0 1 0 0 0 0  0  0];
    S{3} = [0 0 1 0 0 0  0  0];
    S{4} = [0 0 0 1 0 0  0  0];
    S{5} = [0 0 0 0 1 0  0  0];
    S{6} = [0 0 0 0 0 1  0  0
            0 0 0 0 0 0  1  0
            0 0 0 0 0 0  0  1];
    S{7} = [0 0 0 0 0 1  0  0
            0 0 0 0 0 0 -1  0
            0 0 0 0 0 0  0 -1];
    S{8} = [0 0 0 0 0 1  0  0
            0 0 0 0 0 0  1  0
            0 0 0 0 0 0  0 -1];
    
    % zero restrictions
    Z=cell(nvar,1);
    for i=1:nvar
        Z{i}=zeros(0,nvar);
    end
    Z{1} = [0 0 0 0 0 1 0 0;
            0 0 0 0 0 0 1 0
            0 0 0 0 0 0 0 1];
    Z{2} = Z{1};
    Z{3} = Z{1};
    Z{4} = Z{1};
    Z{5} = Z{1};
elseif nvar == 7 && nshocks == 3
    % sign restrictions
    S = cell(nvar,1);
    for ii=1:nvar
        S{ii}=zeros(0,nvar);
    end
    S{1} = [1 0 0 0 0 0 0];
    S{2} = [0 1 0 0 0 0 0];
    S{3} = [0 0 1 0 0 0 0];
    S{4} = [0 0 0 1 0 0 0];
    S{5} = [0 0 0 0 1 0 0
            0 0 0 0 0 1 0
            0 0 0 0 0 0 1];
    S{6} = [0 0 0 0 1 0 0
            0 0 0 0 0 -1 0
            0 0 0 0 0 0 -1];
    S{7} = [0 0 0 0 1 0 0
            0 0 0 0 0 1 0
            0 0 0 0 0 0 -1];
    
    % zero restrictions
    Z=cell(nvar,1);
    for i=1:nvar
        Z{i}=zeros(0,nvar);
    end
    Z{1} = [0 0 0 0 1 0 0;
            0 0 0 0 0 1 0
            0 0 0 0 0 0 1];
    Z{2} = Z{1};
    Z{3} = Z{1};
    Z{4} = Z{1};
elseif nvar == 6 && nshocks == 2
    % sign restrictions
    S = cell(nvar,1);
    for ii=1:nvar
        S{ii}=zeros(0,nvar);
    end
    S{1} = [1 0 0 0 0 0];
    S{2} = [0 1 0 0 0 0];
    S{3} = [0 0 1 0 0 0];
    S{4} = [0 0 0 1 0 0];
    S{5} = [0 0 0 0 1 0
            0 0 0 0 0 1];
    S{6} = [0 0 0 0 1 0
            0 0 0 0 0 -1];
    
    % zero restrictions
    Z=cell(nvar,1);
    for i=1:nvar
        Z{i}=zeros(0,nvar);
    end
    Z{1} = [0 0 0 0 1 0;
            0 0 0 0 0 1];
    Z{2} = Z{1};
    Z{3} = Z{1};
    Z{4} = Z{1};
elseif nvar == 3 && nshocks == 2
    % sign restrictions
    S = cell(nvar,1);
    for ii=1:nvar
        S{ii}=zeros(0,nvar);
    end
    S{1} = [1 0 0];
    S{2} = [0 1 0
            0 0 1];
    S{3} = [0 1 0
            0 0 -1];
    
    % zero restrictions
    Z=cell(nvar,1);
    for i=1:nvar
        Z{i}=zeros(0,nvar);
    end
    Z{1} = [0 1 0;
            0 0 1];
elseif nvar == 4 && nshocks == 3
    % sign restrictions
    S = cell(nvar,1);
    for ii=1:nvar
        S{ii}=zeros(0,nvar);
    end
    S{1} = [1 0 0 0];
    S{2} = [0 1 0 0
            0 0 1 0
            0 0 0 1];
    S{3} = [0 1 0 0
            0 0 -1 0
            0 0 0 -1];
    S{4} = [0 1 0 0
            0 0 1 0
            0 0 0 -1];

    % zero restrictions
    Z=cell(nvar,1);
    for i=1:nvar
        Z{i}=zeros(0,nvar);
    end
    Z{1} = [0 1 0 0;
            0 0 1 0
            0 0 0 1];
end


%==========================================================================
%% Setup info
%==========================================================================
info=SetupInfo(nvar,m,Z,@(x)chol(x));

% ZIRF()
info.nlag     = nlag;
info.horizons = horizons;
info.ZF       = @(x,y)ZIRF(x,y);

% functions useful to compute the importance sampler weights
fs      = @(x)ff_h(x,info);
r       = @(x)ZeroRestrictions(x,info);

if strcmp(conjugate,'irfs')==1
    fo              = @(x)f_h(x,info);
    fo_str2irfs     = @(x)StructuralToIRF(x,info);
    fo_str2irfs_inv = @(x)IRFToStructural(x,info);
    r_irfs          = @(x)IRFRestrictions(x,info); 
end


% function useful to check the sign restrictions
fh_S_restrictions  = @(y)StructuralRestrictions(y,S);

%% prior for reduced-form parameters
nnuBar              = 0;
OomegaBarInverse    = zeros(m);
PpsiBar             = zeros(m,nvar);
PphiBar             = zeros(nvar);

%% posterior for reduced-form parameters
nnuTilde            = T +nnuBar;
OomegaTilde         = (X'*X  + OomegaBarInverse)\eye(m);
OomegaTildeInverse  =  X'*X  + OomegaBarInverse;
PpsiTilde           = OomegaTilde*(X'*Y + OomegaBarInverse*PpsiBar);
PphiTilde           = Y'*Y + PphiBar + PpsiBar'*OomegaBarInverse*PpsiBar - PpsiTilde'*OomegaTildeInverse*PpsiTilde;
PphiTilde           = (PphiTilde'+PphiTilde)*0.5;

%% useful definitions
% definitios used to store orthogonal-reduced-form draws, volume elements, and unnormalized weights
Bdraws         = cell([nd,1]); % reduced-form lag parameters
Sigmadraws     = cell([nd,1]); % reduced-form covariance matrices
Qdraws         = cell([nd,1]); % orthogonal matrices
storevefh      = zeros(nd,1);  % volume element f_{h}
storevegfhZ    = zeros(nd,1);  % volume element g o f_{h}|Z
uw             = zeros(nd,1);  % unnormalized importance sampler weights

if strcmp(conjugate,'irfs')==1
    storevephi      = zeros(nd,1);  % volume element f_{h}
    storevegphiZ    = zeros(nd,1);  % volume element g o f_{h}|Z
end

% definitions related to IRFs; based on page 12 of Rubio, Waggoner, and Zha (RES 2010)
J      = [e;repmat(zeros(nvar),nlag-1,1)];
A      = cell(nlag,1);
extraF = repmat(zeros(nvar),1,nlag-1);
F      = zeros(nlag*nvar,nlag*nvar);
for l=1:nlag-1
    F((l-1)*nvar+1:l*nvar,nvar+1:nlag*nvar)=[repmat(zeros(nvar),1,l-1) e repmat(zeros(nvar),1,nlag-(l+1))];
end

% definition to facilitate the draws from B|Sigma
hh              = info.h;
cholOomegaTilde = hh(OomegaTilde)'; % this matrix is used to draw B|Sigma below

%% initialize counters to track the state of the computations
counter = 1;
record  = 1;
count   = 0;

while record<=nd
    
    
    %% step 1 in Algorithm 2
    Sigmadraw     = iwishrnd(PphiTilde,nnuTilde);
    cholSigmadraw = hh(Sigmadraw)';
    Bdraw         = kron(cholSigmadraw,cholOomegaTilde)*randn(m*nvar,1) + reshape(PpsiTilde,nvar*m,1);
    Bdraw         = reshape(Bdraw,nvar*nlag+nex,nvar);
    % store reduced-form draws
    Bdraws{record,1}     = Bdraw;
    Sigmadraws{record,1} = Sigmadraw;
    
   
    %% steps 2:4 of Algorithm 2
    w           = DrawW(info);   
    x           = [vec(Bdraw); vec(Sigmadraw); w];
    structpara  = ff_h_inv(x,info);
    
    % store the matrix Q associated with step 3
    Qdraw            = SpheresToQ(w,info,Bdraw,Sigmadraw);
    Qdraws{record,1} = reshape(Qdraw,nvar,nvar);


    %% check if sign restrictions hold
    signs      = fh_S_restrictions(structpara);
    
    
    if (sum(signs>0))==size(signs,1)
        
        count=count+1;
  
        %% compute importance sampling weights
        
        switch conjugate
            
            case 'structural'
                
                
                storevefh(record,1)   = (nvar*(nvar+1)/2)*log(2)-(2*nvar+m+1)*LogAbsDet(reshape(structpara(1:nvar*nvar),nvar,nvar));
                storevegfhZ(record,1) = LogVolumeElement(fs,structpara,r); 
                uw(record,1)          = exp(storevefh(record,1) - storevegfhZ(record,1));
                
            case 'irfs'
                
                irfpara                = fo_str2irfs(structpara);
                storevephi(record,1)   = LogVolumeElement(fo,structpara)   + LogVolumeElement(fo_str2irfs_inv,irfpara);%log(2)*nvar*(nvar+1)/2 - LogAbsDet(inv(reshape(structpara(1:nvar*nvar),nvar,nvar)*reshape(structpara(1:nvar*nvar),nvar,nvar)'))*(2*nvar*nlag-m-1)/2;
                storevegphiZ(record,1) = LogVolumeElement(fs,structpara,r) + LogVolumeElement(fo_str2irfs_inv,irfpara,r_irfs); 
                uw(record,1)           = exp(storevephi(record,1) - storevegphiZ(record,1));
                
            otherwise
                
                uw(record,1) = 1;
                
        end
        
    else
        
        uw(record,1) = 0;
        
    end
    
    if counter==iter_show
        
        display(['Number of draws = ',num2str(record)])
        display(['Remaining draws = ',num2str(nd-(record))])
        counter =0;
        
    end
    counter = counter + 1;
    
    record = record+1;
    
end
toc


% compute the normalized weights and estimate the effective sample size of the importance sampler
imp_w  = uw/sum(uw);
ne     = floor(1/sum(imp_w.^2));


%% useful definitions to store relevant objects
A0tilde       = zeros(nvar,nvar,ne);               % define array to store A0
Aplustilde    = zeros(m,nvar,ne);                  % define array to store Aplus
Ltilde        = zeros(horizon+1,nvar,nvar,ne);     % define array to store IRF
FEVD          = zeros(nvar,horizon,nvar,ne);       % define array to store FEVD

% initialize counter to track the state of the importance sampler
count_IRF     = 0;
for s=1:min(ne,maxdraws)
    
    %% draw: B,Sigma,Q
    is_draw     = randsample(1:size(imp_w,1),1,true,imp_w);
    Bdraw       = Bdraws{is_draw,1};
    Sigmadraw   = Sigmadraws{is_draw,1};
    Qdraw       = Qdraws{is_draw,1};
    
    x          = [reshape(Bdraw,m*nvar,1); reshape(Sigmadraw,nvar*nvar,1); Qdraw(:)];
    structpara = f_h_inv(x,info);
    

    LIRF = IRF_horizons(structpara, nvar, nlag, m, 0:horizon);
    
    for h=0:horizon
        Ltilde(h+1,:,:,s) =  LIRF(1+h*nvar:(h+1)*nvar,:);
    end
    

    % FEVD
    % compute matrix F: useful for FEVD
    hSigmadraw = hh(Sigmadraw);
    A0         = hSigmadraw\e;
    Aplus      = Bdraw*A0;
    % Page 8 ARRW
    for l=1:nlag-1
        A{l} = Aplus((l-1)*nvar+1:l*nvar,1:end);
        F((l-1)*nvar+1:l*nvar,1:nvar)=A{l}/A0;
    end
    A{nlag} = Aplus((nlag-1)*nvar+1:nlag*nvar,1:end);
    F((nlag-1)*nvar+1:nlag*nvar,:)=[A{nlag}/A0 extraF];
    
    % NOTE: Was Qdraw(:,1) before
    for i=1:horizon
        for j=1:nvar % Loop over shocks
            FEVD(j,i,:,s) = variancedecomposition(F',J,Sigmadraw, hh(Sigmadraw)'*Qdraw(:,j),nvar,i);
        end
    end
    
    % store weighted independent draws
    A0tilde(:,:,s)    = reshape(structpara(1:nvar*nvar),nvar,nvar);
    Aplustilde(:,:,s) = reshape(structpara(nvar*nvar+1:end),m,nvar);
    
end
A0tilde    = A0tilde(:,:,1:s);
Aplustilde = Aplustilde(:,:,1:s);
Ltilde     = Ltilde(:,:,:,1:s);
FEVD       = FEVD(:,:,:,1:s);

%addpath('plothelpfunctions');
%figname = 'fig_SVAR_SD.pdf';
save(strcat('../Data/MATLAB/Ltilde_', svar, '.mat'), 'Ltilde');
%store_results_and_plot_IRFs;

disp(['Effective obs.: ', num2str(ne)]);

disp(' ')

%%
mFEVD = mean(FEVD,4);
smFEVD = mFEVD(shock,:,:);
iFEVD = smFEVD(1,1:40,1);