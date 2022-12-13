% Performs optimal reward attack for the TD(0) learning algorithm.
% 
% Written by:
% -- 
% Xingang Guo               2022/12/13
% 
% email: xingang2@illinois.edu
% 
% Please send comments and especially bug reports to the
% above email address.

clear;
clc;
nStates = 3;
nActions = 2;

Max_iteration = 5000;
alpha = 0.01;
gamma = 0.7; % the discount factor 
st = 2;
Q_qlearn = zeros(nStates,nActions);
Q_dagger = zeros(nStates*nActions,1);
Q_att = zeros(nStates,nActions);
stateseen = [st]; 
% pick an initial action using an epsilon greedy policy derived from Q: 
epsilon = 0.1;
[dum,at_qlearn] = max(Q_qlearn(st,:));  % at \in [1,2,3,4]=[up,down,right,left]
if( rand<epsilon )         % explore ... with a random action 
    tmp=randperm(nActions); 
    at_qlearn=tmp(1); 
end
per = zeros(9,1);
% begin an episode
R_qlearn=0; 
for ite = 1:Max_iteration
    % propagate to state stp1 and collect a reward rew
    if at_qlearn == 1
        stp1=st-1;
        if stp1 == 0
            stp1 = 1;
        end
    else
        stp1=st+1;
        if stp1 == 4
            stp1 = 3;
        end
    end
    % compute the perturbations
    if [st,stp1] == [1,1]
        rew = 0;
        rew_att = per(1);
    elseif [st,stp1] == [1,2]
        rew = 0;
        rew_att = per(2);
    elseif [st,stp1] == [2,1]
        rew = 0;
        rew_att = per(4);
    elseif [st,stp1] == [2,3]
        rew = 1;
        rew_att = 1+per(6);
    elseif [st,stp1] == [3,2]
        rew = 0;
        rew_att = per(8);
    elseif [st,stp1] == [3,3]
        rew = 1;
        rew_att = 1+per(9);
    end
    [dum,atp1_qlearn] = max(Q_qlearn(stp1,:)); 
      if( rand<epsilon )         % explore ... with a random action 
        tmp=randperm(nActions); 
        atp1_qlearn=tmp(1); 
      end
    Q_qlearn(st,at_qlearn) = Q_qlearn(st,at_qlearn) + alpha*(rew + gamma*max(Q_qlearn(stp1,:)) - Q_qlearn(st,at_qlearn) ); 
    Q_att(st,at_qlearn) = Q_att(st,at_qlearn) + alpha*(rew_att + gamma*max(Q_att(stp1,:)) - Q_att(st,at_qlearn) ); 
    % update (st,at) pair: 
    st = stp1; 
    at_qlearn = atp1_qlearn; 
    options = optimoptions('fmincon','Display','none');
    problem.options = options;
    A = -[1 -1 0 0 0 0;0 0 1 -1 0 0;0 0 0 0 1 -1];
    b = -0.01*ones(3,1);
    fun = @(x)norm(x-reshape(Q_qlearn,6,1),2);
    x0 = reshape(Q_dagger,nStates*nActions,1);
    Q_dagger = fmincon(fun,x0,A,b,[],[],[],[],[],options);
    per(1) = Q_dagger(1) - (0+gamma*Q_dagger(1));
    per(2) = Q_dagger(2) - (0+gamma*Q_dagger(3));
    per(4) = Q_dagger(3) - (0+gamma*Q_dagger(1));
    per(6) = Q_dagger(4) - (1+gamma*Q_dagger(5));
    per(8) = Q_dagger(5) - (0+gamma*Q_dagger(3));
    per(9) = Q_dagger(6) - (1+gamma*Q_dagger(5));
end 




