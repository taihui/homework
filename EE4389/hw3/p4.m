clear;
clc;

tmp = load('synth.tr', '-ascii');
trn_data.X = tmp(:,1:2)';
trn_data.y = tmp(:,3);

tst_data = trn_data;
tmp = load('synth.te', '-ascii');
tst_data.X = tmp(:,1:2)';
tst_data.y = tmp(:,3);

[ypred, nrms, rms0, nmax] = xtal(trn_data, trn_data, 'RBF1', 25);
MSE = nrms* rms0


% draw figures


