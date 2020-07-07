function [fs_3] = bernoulli_f(A, I, I0, K, A_init, C, mu_x, sig_x, gamma)


parfor prior = 1:length(gamma)       
    
    tic
    % Gibbs loop
    [A_s] = gibbs_bernoulli(I, I0, K, A_init, C, mu_x, sig_x, gamma(prior));
    toc
    
    % Compute fscore
    [~,~, fs_3(prior)] = adj_eval(A, mode(A_s, 3));

end   


end

