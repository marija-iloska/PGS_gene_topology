clear all
clc




%% SETTINGS for generating data
dim_y = 20; var_u =1;
p_s = 0.7; p_ns = 0.3;
T = 1e3;

% Generating data and matrices
[A, C, y, dim_x] = generate_mat(T, dim_y, p_s, p_ns, var_u);



%% Bayesian Ridge Regression

%Consider a Gaussian prior
mu_0 = zeros(dim_x, 1);
var_0 = 1; sig_0 = var_0*eye(dim_x);

% Obtain the posterior of the vectorized matrix (and likelihood
% counterparts)
[mu_c, sig_c, mu_x, sig_x] = mn_conjugate_var(y, var_u, mu_0, sig_0);

% Convert to an estimate of the coefficient matrix
C_est = reshape(mu_c, dim_y, dim_y);

% Compute MSE in the coefficients
MSE = sum(sum((C-C_est).^2))/dim_x;



%% Gibbs Sampler

% Settings for Gibbs Bernoulli
I = 3000;                       % Gibbs iterations
I0 = 1500;                      % Gibbs burn-in 
K = 2;                          % Thinning parameter
A_init = ones(dim_y, dim_y);    % Initial adjacency matrix
gamma = 0.1:0.1:0.8
R=100;

parpool(34)


tic
parfor run=1:R
    
      [f_score] = bernoulli_f(A, I, I0, K, A_init, C, mu_x, sig_x, gamma);
      f_bernoulli(run,:) = f_score;
         
end  
toc

% Find average of R runs
avg_f_bernoulli = mean(f_bernoulli,1);

save('gb20.mat','avg_f_bernoulli')





