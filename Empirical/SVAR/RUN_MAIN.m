%% Settings
dep_vars = ["Y","YT","YNT","C","CT","CNT","E","RR","P","EX","IM","NX"];

%% Run "full" model
for svar = dep_vars
    run_all;
end

%% Run model for each country and var
v = 1;
for svar = ["Y","C","EX","IM","NX"]
    countries = readlines(strcat('..\Data\MATLAB\countries_', svar, '.txt'));
    N = size(countries, 1);
    FEVDs = nan(N, 40);

    for n = 1:N
        country = countries(n, 1);
        run_single;
        FEVDs(n,:) = iFEVD;
    end
    v = v+1;

    save(strcat('../Data/MATLAB/FEVDs_', svar, '.mat'), 'FEVDs');
end