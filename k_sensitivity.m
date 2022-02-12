data = readmatrix("VAM-TPL-Extended.csv");

x = data(:,1:3); 

y = data(:,4:5);



avg = mean(y);
rng('default')
n = length(y);
n_k = 50;
pt = n_k-1;

perf = zeros(pt,1);
ovrperf = zeros(pt,1);
minperf = zeros(pt,1);
maxperf = zeros(pt,1);

for j=1:pt
    
    k = j+1;
    fold = cvpartition(n,'KFold', k);

    for i=1:k
        idxTrain = fold.training(i);
        idxTest = fold.test(i);
        xTrain = x(idxTrain,:);
        yTrain = y(idxTrain,:);
        xTest = x(idxTest,:);
        yTest = y(idxTest,:);
        net = cascadeforwardnet(8);
        net = train(net,xTrain',yTrain');
        pred = net(xTest');
        perf(i,1) = perform(net,pred,yTest');
    end 

    ovrperf(j,1) = sum(perf)/j;
    minperf(j,1) = min(perf(perf>0)) - ovrperf(j,1);
    maxperf(j,1) = max(perf) - ovrperf(j,1);
end


folds = linspace(2,n_k,pt);


scatter(folds, ovrperf, "filled")
hold on 
polyfit(ovrperf,folds,5);
coef = polyfit(folds,ovrperf,5);
ypol = polyval(coef,folds);
plot(folds,ypol);


