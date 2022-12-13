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
% 
clear;
clc;
% Compute the true vlaue function 
gamma = 0.7;
nstate = 3;
P = [0.5 0.5 0;...
     0.5 0 0.5;...
     0 0.5 0.5];
R =[0 0.5 0.5]';
true = inv(eye(nstate)-gamma*P)*R;
% define the target value function
V_hat = [2;0.5;1];
per = zeros(9,1);
%%
Max_iteration = 5000;
alpha = 0.01;
gamma = 0.7; % the discount factor 
nStates=1+2;
st = 2;
Vtd = zeros(1,nStates); 
Vtd_att = zeros(1,nStates); 
VVtd = Vtd;
VVtd_att = Vtd_att;

stateseen = [st]; 
Rew = [];
Rew_att = [];
P11 = 0;
P12 = 0;
P21 = 0;
P23 = 0;
P32 = 0;
P33 = 0;
Num = 3;

for ei=1:Max_iteration  
    if (rand<0.5) 
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
    if [st,stp1] == [1,1]
        P11 = P11+1;
    elseif [st,stp1] == [1,2]
        P12 = P12+1;
    elseif [st,stp1] == [2,1]
        P21 = P21+1;
    elseif [st,stp1] == [2,3]
        P23 = P23+1;
    elseif [st,stp1] == [3,2]
        P32 = P32+1;
    elseif [st,stp1] == [3,3]
        P33 = P33+1;
    end
    % estimate the transition probability matrix
    P_hat = [P11/(P11+P12) P12/(P11+P12) 0; P21/(P21+P23) 0 P23/(P21+P23); 0 P32/(P32+P33) P33/(P32+P33)];
    % compute the estimated reward attack
    if sum(sum(isnan(P_hat))) == 0
        DeltaR = (eye(nstate)-gamma*P_hat)*(V_hat-true);
        C = [P_hat(1,1) P_hat(1,2) 0 zeros(1,6);zeros(1,3) P_hat(2,1) 0 P_hat(2,3) zeros(1,3);zeros(1,6) 0 P_hat(3,2) P_hat(3,3)];
        per = C'*inv(C*C')*DeltaR;
    end
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
    % update Vtd:
    Vtd(st) = Vtd(st) + alpha*(rew+gamma*Vtd(stp1) - Vtd(st));
    Vtd_att(st) = Vtd_att(st) + alpha*(rew_att+gamma*Vtd_att(stp1) - Vtd_att(st));
    Rew = [Rew rew];
    Rew_att = [Rew_att rew_att];
    stateseen = [stateseen;stp1]; 
    st = stp1; 
    VVtd = [VVtd;Vtd];
    VVtd_att = [VVtd_att;Vtd_att];
end

%% Plot the result 
figure();
plot(VVtd(:,1),VVtd(:,2),'LineWidth',2)
hold on;
plot(true(1),true(2),'p','markersize',15,'MarkerEdgeColor','r',...
    'MarkerFaceColor','r')
plot(VVtd_att(:,1),VVtd_att(:,2),'LineWidth',2)
plot(V_hat(1),V_hat(2),'o','markersize',10,'MarkerEdgeColor','b',...
    'MarkerFaceColor','b')
hold off;
grid on;
xlabel('$V(1)$','FontSize',25,'Interpreter','latex')
ylabel('$V(2)$','FontSize',25,'Interpreter','latex')
set(gca,'FontSize',20)
legend('Trajectory of clean TD(0)','True value function','Trajectory of poisoned TD(0)','Target value function')


%%
range = 0:Max_iteration;
figure();
Error = (VVtd_att(:,1)-V_hat(1)).^2 + (VVtd_att(:,2)-V_hat(2)).^2 + (VVtd_att(:,3)-V_hat(3)).^2;
plot(range,sqrt(Error),'LineWidth',2);
grid on;
xlabel('Iterations','FontSize',25,'Interpreter','latex')
ylabel(['$\|V_k - V^\dagger\|$'],'FontSize',25,'Interpreter','latex')
set(gca,'FontSize',25)

%%
range = 0:Max_iteration;
figure();
Error = (VVtd(:,1)-true(1)).^2 + (VVtd(:,2)-true(2)).^2 + (VVtd(:,3)-true(3)).^2;
plot(range,sqrt(Error),'LineWidth',2);
grid on;
xlabel('Iterations','FontSize',25,'Interpreter','latex')
ylabel('$\|V_k - V\|$','FontSize',25,'Interpreter','latex')
set(gca,'FontSize',25)
