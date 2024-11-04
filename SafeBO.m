% MIT License
% 
% Copyright (c) 2024 Oleksii Molodchyk, Johannes Teutsch, Timm Faulwasser
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.


clear
close all
addpath(fullfile(pwd,'helpers'))

%random seed:
rng(683639528); % rng('shuffle');
s = rng;

%%%%%%%%%% USER BEGIN %%%%%%%%%% 
%flag for plotting the learning progress step by step:
flag_plot = false;

%flag for storing the results as a .mat-file:
flag_store = true;

%number of learning steps t:
num_steps = 100;

%number of repeated (Monte Carlo) runs over noise realizations:
num_mc = 100;
%%%%%%%%%%% USER END %%%%%%%%%%% 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SYSTEM & KERNEL SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%standard deviation of measurement noise:
sigma_M = 1;

%system function: y = f(x) + noise
f = @(x) 0.01*x.^3 - 0.2*x.^2 + 0.2*x;

%dimension of input data:
nx = 1;

%bounds of input domain:
x_min = -5; 
x_max = 5; 

%lower bound for output:
f_min = f(-4.5);

%known safe input:
xsafe = x_max;

%kernel function:
params.sigmaSE = 4.21;
params.lSE = 3.59;
kernel_init = objKernel(nx,sigma_M,"se",params);

%initial data:
xdata_init = xsafe;
wdata_init = sigma_M*randn(size(xdata_init));
ydata_init = f(xdata_init) + wdata_init;

%initial kernel object:
kernel_init = kernel_init.addData(xdata_init,ydata_init);

if(flag_plot)
    figure(1)
    plot_progress(f, kernel_init, x_max, x_min, f_min)
    pause(0.1)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BAYESIAN OPTIMIZATION SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%variables to store evolution:
xsafe_min = xsafe*ones(num_steps,num_mc,3);
xsafe_max = xsafe*ones(num_steps,num_mc,3);
xdata_seq = zeros(num_steps,num_mc,3);
ydata_seq = zeros(num_steps,num_mc,3);
olgamma = zeros(num_steps,num_mc);
kernel_storage = cell(2,1);

%confidence for constraint satisfaction:
delta = 1e-3;

%upper bound to RKHS norm of unknown function:
B = 2.5;

%regularization for log-barrier function:
tau = 1e-6;

%solver options:
options = optimoptions('fmincon','Display','off');
options0 = optimset('Display','off');

%create noise sequence for Monte Carlo runs:
noise = sigma_M*randn(num_mc,num_steps); 

%variable for printing progress update
nbytes = [0;0;0];

%flag to indicate feasible point in safe region exists
flag_safe = true;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% BAYESIAN OPTIMIZATION ALGORITHM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%loop over comparison methods:
% j = 1: (P) proposed bound 
% j = 2: (A) bound from Abbasi-Yadkori (2013), Theorem 3.11
% j = 3: (F) bound from Fiedler et al. (2021), Proposition 2 
for j = 1:3
    fprintf(repmat('\b',1,sum(nbytes)))
    nbytes(1) = fprintf('Method %d/%d; Monte Carlo iteration ', j, 3);
    nbytes(2:3) = 0;
        
    %loop over noise realizations:
    for k = 1:num_mc
        fprintf(repmat('\b',1,sum(nbytes(2:3))))
        nbytes(2) = fprintf('%d/%d; Learning step: ', k, num_mc); 
        nbytes(3) = 0;
    
        %learning loop:
        for i = 1:num_steps
            if(i==1)
                kernel = kernel_init;
            end
            
            fprintf(repmat('\b',1,nbytes(3)))
            nbytes(3) = fprintf('%d/%d', i, num_steps);
    
           
            %choose error bound from comparison methods:
            if(j==1) %Proposed approach:
                %parameter:
                beta_WK = sqrt(2*log(2/delta));

                %error bound:
                eta = @(x) B*sqrt(kernel.var_ker(x) - kernel.var_wie(x)) + beta_WK*sqrt(kernel.var_wie(x));
            
            elseif(j==2) %Abbasi-Yadkori (2013) Theorem 3.11:
                %parameter:
                olgamma(i,k) = det( (1/sigma_M^2)*kernel.Gram + eye(size(kernel.Gram)) );
                beta_1 = sqrt(log(olgamma(i,k)) + 2*log(1/delta));

                %error bound:
                eta = @(x) (B + beta_1)*sqrt(kernel.var_ker(x));
                
            elseif(j==3) %Fiedler (2021), Proposition 2:
                %parameter:
                D = size(xdata_init,2)+i;
                beta_2 = sqrt(D + 2*sqrt(D)*sqrt(log(1/delta)) + 2*log(1/delta));
                
                %error bound:
                eta = @(x) B*sqrt(kernel.var_ker(x)) + beta_2*sqrt(kernel.var_wie(x));

            end

            %acquisition function: upper confidence bound
            acquisitionFun = @(x) kernel.mean_ker(x) + eta(x);
                
            %constraint function: lower confidence bound - f_min >= 0
            constraintFun = @(x) kernel.mean_ker(x) - eta(x) - f_min;
            
        
            %find good/feasible initial condition for optimization:
            gran = 0.05;
            x0grid = x_min+gran:gran:x_max-gran;
            objgrid = acquisitionFun(x0grid);
            if(tau>0) %if constraint is being considered
                barrgrid = constraintFun(x0grid);
                mask = (barrgrid > 0);
                if(all(~mask))
                    % disp("No feasible point exists! Go to safe fallback strategy...")
                    flag_safe = false;
                end
                objgrid(~mask) = NaN;
            end
            [~,idx0] = max(objgrid);
            xopt0 = x0grid(idx0(1));

            
            if(flag_safe)
                %if safe initial state exists, solve safe BO problem:
                %(negative objective for maximization)
                [xopt, ~, ef] = fmincon(@(x) -acquisitionFun(x) - tau*log(constraintFun(x)), xopt0,[],[],[],[],x_min,x_max,[],options);    
            else
                %safe fallback strategy:
                xopt = xsafe;
            end

        
            %apply action:
            xdata_seq(i,k,j) = xopt;
            ydata_seq(i,k,j) = f(xopt) + noise(k,i);
            
            %update kernel object:
            kernel = kernel.addData(xdata_seq(i,k,j),ydata_seq(i,k,j));
        
            %find safe region:
            if(tau>0)
                idxs = find(diff(mask)); %find changes in the safety indicator mask
                if(length(idxs)<2) %less than 2 changes found
                    if(mask(1) && ~mask(end)) %x_min is part of the safe set
                        idxs = [1,idxs];
                    elseif(~mask(1) && mask(end)) %x_max is part of the safe set
                        idxs = [idxs,length(mask)];
                    else %the full domain is safe
                        idxs = [1,length(mask)];
                    end
                end
                %determine boundary of safe region:
                xsafe_min(i,k,j) = max( [min([fzero(@(x) constraintFun(x),x0grid(idxs(1)),options0), xsafe]), x_min]);
                xsafe_max(i,k,j) = min( [max([fzero(@(x) constraintFun(x),x0grid(idxs(2)),options0), xsafe]), x_max]);
            end
    
            %plot learning progress
            if(flag_plot)
                plot_progress(f, kernel, x_max, x_min, f_min)
                pause(0.1)
            end

            %reset flag that indicates existence of safe action:
            flag_safe = true;

            %save kernel & constraint data for one MC run (k==1) for (P) 
            %and (A) (for learning progress plot, Figure 3 of paper):
            if((j==1 || j==2) && k==1)
                if(i==1)
                    numgrid = 100;
                    xgrid = linspace(x_min,x_max,numgrid);
                    kernel_storage{j}.xgrid = xgrid';
                end
                kernel_storage{j}.vargrid_ker(:,i) = kernel.var_ker(xgrid)';
                kernel_storage{j}.vargrid_wie(:,i) = kernel.var_wie(xgrid)';
                kernel_storage{j}.meangrid_ker(:,i) = kernel.mean_ker(xgrid)';
                kernel_storage{j}.barrgrid(:,i) = constraintFun(xgrid)';
                if(i==num_steps)
                    kernel_storage{j}.xdata = kernel.xdata';
                    kernel_storage{j}.ydata = kernel.ydata';
                end
            end
        
        end
    end
end
fprintf('\n')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STORE DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data used for figures in the paper

%true optimum:
xopttrue = fmincon(@(x) -f(x), 0,[],[],[],[],x_min,x_max,[],options);
fopttrue = f(xopttrue);

%parameter beta for confidence bound:
Dsteps = size(xdata_init,2)+(1:num_steps);
beta_res.p = sqrt(2*log(2/delta))*ones(num_steps,1);
beta_res.a = sqrt(log(olgamma)+2*log(1/delta));
beta_res.f = sqrt(Dsteps + 2*sqrt(Dsteps)*sqrt(log(1/delta)) + 2*log(1/delta))';

%params:
results.x_min = x_min;
results.x_max = x_max;
results.f = f;
results.y_min = f_min;
results.xopttrue = xopttrue;
results.fopttrue = fopttrue;
results.beta = beta_res;

%results:
results.xsafe_min = xsafe_min;
results.xsafe_max = xsafe_max;
results.xdata = xdata_seq;
results.ydata = ydata_seq;
results.wdata = noise;
results.kernel = kernel_storage; 

if(flag_store)
    save("data_BO.mat", "results")
    disp('Data has been stored!')
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLOT RESULTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%labels for legend:
leglab = cell(3,1);
leglab{1} = '(P): Proposed method';
leglab{2} = '(A): Abbasi-Yadkori (2013) Thm. 3.11';
leglab{3} = '(F): Fiedler et al. (2021) Prop. 2';

%regret:
feval = results.f(results.xdata);
cumregret = cumsum(results.fopttrue - feval);
cumregret_mean = squeeze(mean(cumregret,2));

%confidence interval (75%):
cumregret_conf_min = cumregret_mean - 1.15035*squeeze(std(cumregret,0,2));
cumregret_conf_max = cumregret_mean + 1.15035*squeeze(std(cumregret,0,2));

%size of safe region:
xsafe_size = results.xsafe_max - results.xsafe_min;
xsafe_size_mean = squeeze(mean(xsafe_size,2));

%confidence interval (75%):
xsafe_size_conf_min = xsafe_size_mean - 1.15035*squeeze(std(xsafe_size,0,2));
xsafe_size_conf_max = xsafe_size_mean + 1.15035*squeeze(std(xsafe_size,0,2));

figure
tend = size(results.xdata,1);
t = (1:tend)';

%plot regret:
subplot(2,1,1)
plot_results(t,cumregret_mean,cumregret_conf_max,cumregret_conf_min,leglab)
ylabel('Regret $\sum_{i=1}^{t} ( f(x_{\mathrm{opt}}) - f(x_i) )$', 'Interpreter','latex')
xlabel('Learning iterations t', 'Interpreter','latex')

%plot size of safe region:
subplot(2,1,2)
plot_results(t,xsafe_size_mean,xsafe_size_conf_max,xsafe_size_conf_min,leglab)
ylim([0,results.x_max-  results.x_min])
ylabel('Size of safe region', 'Interpreter','latex')
xlabel('Learning iterations t', 'Interpreter','latex')

figpos = [0.1 0.1 0.8 0.8];
set(gcf,'units','normalized','outerposition',figpos)


%Print cumulative regret relative to proposed method:
cumregret_rel_final = (cumregret_mean(end,:)/cumregret_mean(end,1)-1)*100;
disp(['Relative increase in mean cumulative regret wrt proposed method (P) at final time t = ', num2str(tend), ':'])
disp(['(A):  ', num2str(cumregret_rel_final(2)), '%'])
disp(['(F): ', num2str(cumregret_rel_final(3)), '%'])

