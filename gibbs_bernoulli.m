function [A_s] = gibbs_bernoulli(iter_num, burn_in, thin_rate, A_init, C, mu, sig, gamma)

% Obtain dimensions
% testing file
dim_y = length(C(:, 1));

% Initialize arrays to store samples
A_s = zeros(dim_y, dim_y, iter_num+1);

% Set the initial states of the Markov chains
A_s(:, :, 1) = A_init;
A_old = A_s(:, :, 1);

% For loop for the Gibbs
for i = 2 :iter_num+1
    
    for j = 1:dim_y
        for k = 1:dim_y
            
            % Conside that A(j, k)=1
            A_old(j,k) = 1;
            C_temp = C.*A_old;
            log_pa1 = log(gamma) + logmvnpdf(C_temp(:)', mu', sig);

            % Conside that A(j, k)=0
            A_old(j,k) = 0;
            C_temp = C.*A_old;
            log_pa0 = log(1 - gamma) + logmvnpdf(C_temp(:)', mu', sig);
            
            % Sample the topology
            pa0 = exp(log_pa0 - max([log_pa0, log_pa1]));
            pa1 = exp(log_pa1 - max([log_pa0, log_pa1]));
            prob_1 = pa1/(pa1 + pa0);
            
            A_old(j,k) = rand <prob_1;
        end
    end
    
    A_s(:, :, i) = A_old;
    
end

% Apply burn-in
A_s = A_s(:, :, burn_in+1:thin_rate:iter_num+1);

end

