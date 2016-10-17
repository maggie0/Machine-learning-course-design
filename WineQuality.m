clear all; close all; clc
file = fopen('winequality-red.csv','r');
data_str = textscan(file,'%s %s %s %s %s %s %s %s %s %s %s %s','Delimiter',';') ;
N = 1599;
D = 12;
data = zeros(N,D);
for i = 1:12
    data_col = data_str{i};
    data_col = data_col(2:end);
    data(:,i) = str2double(data_col);
end
for i = 1:N
    if data(i,D)<5.5
        data(i,D) = -1;
    else
        data(i,D) = 1;
    end
end
randomsort = randperm(N);
% 70% training data
trainloc = randomsort(1:floor(0.7*N));
trainresp = data(trainloc,D);
traindata = data(trainloc,1:D-1);
% 30% training data
testloc = randomsort(ceil(0.7*N):N);
testresp = data(testloc,D);
testdata = data(testloc,1:D-1);

% 5-fold cross-validation to tune SVM
k_fold = 5;
indices = crossvalind('Kfold',length(trainresp),k_fold);

%% ==== SVM with cross-validation ====
Cparams = [1e4,1e5,1e6,1e7,1e8,1e9,1e10];
gammas = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0];
stds = (0.5./gammas).^0.5;
misclassrateLin0 = zeros(length(Cparams),1);
misclassrateGau0 = zeros(length(Cparams),length(stds));
for k = 1:k_fold
    test = (indices == k); 
    train = ~test;
    Xtest = traindata(test,:);
    Ytest = trainresp(test);
    Xtraining = traindata(train,:);            
    Ytraining = trainresp(train);
    for i = 1:length(Cparams)
        Kn = Xtraining*Xtraining';
        [v,b,SVindex] = ksvm(Ytraining,Kn,Cparams(i));
        Kn_test = Xtraining(SVindex,:)*Xtest';
        O = sign(v'*Kn_test+b); 
        err = 0;
        for n = 1:length(O)
            if O(n) ~= Ytest(n)
                err = err+1;
            end
        end
        misclassrateLin0(i) = misclassrateLin0(i)+err;
        for j = 1:length(stds)
            Kn = grbf(Xtraining,Xtraining,stds(j)*ones(1,size(Xtraining,2)));
            [v,b,SVindex] = ksvm(Ytraining,Kn,Cparams(i));
            Kn_test = grbf(Xtraining(SVindex,:),Xtest,stds(j)*ones(1,size(Xtest,2)));
            O = sign(v'*Kn_test+b); 
            err = 0;
            for n = 1:length(O)
                if O(n) ~= Ytest(n)
                    err = err+1;
                end
            end
            misclassrateGau0(i,j) = misclassrateGau0(i,j)+err;
        end
    end
end
misclassrateLin0 = misclassrateLin0/length(trainresp);
misclassrateGau0 = misclassrateGau0/length(trainresp);
minLin = min(misclassrateLin0);
minGau = min(min(misclassrateGau0));
if minLin <= minGau
    InLin0 = find(misclassrateLin0 == minLin,1,'last');
    Kn = traindata*traindata';
    [v,b,SVindex] = ksvm(trainresp,Kn,Cparams(InLin0));
    Kn_test = traindata(SVindex,:)*testdata';
else
    [InGaurow0,InGaucol0] = find(misclassrateGau0 == minGau,1,'last');
    Kn = grbf(traindata,traindata,stds(InGaucol0)*ones(1,size(traindata,2)));
    [v,b,SVindex] = ksvm(trainresp,Kn,Cparams(InGaurow0));
    Kn_test = grbf(traindata(SVindex,:),testdata,stds(InGaucol0)*ones(1,size(testdata,2)));
end
O = sign(v'*Kn_test+b);
svmtestcm0 = zeros(2,2);
for n = 1:length(O)
    if testresp(n) == 1 
        if O(n) == 1
            svmtestcm0(2,2) = svmtestcm0(2,2)+1;
        else
            svmtestcm0(2,1) = svmtestcm0(2,1)+1;
        end
    else
        if O(n) == 1
            svmtestcm0(1,2) = svmtestcm0(1,2)+1;
        else
            svmtestcm0(1,1) = svmtestcm0(1,1)+1;
        end
    end
end
misclassratesvm0 = (svmtestcm0(1,2)+svmtestcm0(2,1))/length(testresp);
       
%% ==== SVM with cross-validation by using Libsvm ==== 
Cparams = [1e4,1e5,1e6,1e7,1e8,1e9,1e10];
gammas = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0];
misclassrateLin = zeros(length(Cparams),1);
misclassrateGau = zeros(length(Cparams),length(gammas));
for k = 1:k_fold
    test = (indices == k); 
    train = ~test;
    Xtest = traindata(test,:);
    Ytest = trainresp(test);
    Xtraining = traindata(train,:);            
    Ytraining = trainresp(train);
    for i = 1:length(Cparams)
        mysvm = svmtrain(Ytraining, Xtraining, ['-t 0 -c ' num2str(Cparams(i)/length(Ytraining))]);   % linear function
        [~, accuracy, ~] = svmpredict(Ytest, Xtest, mysvm);
        misclassrateLin(i) = misclassrateLin(i)+(1-accuracy(1)/100)*length(Ytest)/length(trainresp);
        for j = 1:length(gammas)
            mysvm = svmtrain(Ytraining, Xtraining, ['-t 2 -c ' num2str(Cparams(i)/length(Ytraining)) ' -g ' num2str(gammas(j))]);   % radial basis function            
            [~, accuracy, ~] = svmpredict(Ytest, Xtest, mysvm);
            misclassrateGau(i,j) = misclassrateGau(i,j)+(1-accuracy(1)/100)*length(Ytest)/length(trainresp);
        end
    end
end
minLin = min(misclassrateLin);
minGau = min(min(misclassrateGau));
if minLin <= minGau
    InLin = find(misclassrateLin == minLin,1,'last');
    mysvm = svmtrain(trainresp, traindata, ['-t 0 -c ' num2str(Cparams(InLin)/length(trainresp))]);
else
    [InGaurow,InGaucol] = find(misclassrateGau == minGau);
    [~,Ind] = min((length(Cparams)-InGaurow+InGaucol));
    mysvm = svmtrain(trainresp, traindata, ['-t 2 -c ' num2str(Cparams(InGaurow(Ind))/length(trainresp)) ' -g ' num2str(gammas(InGaucol(Ind)))]);
end
[predict_label, accuracy, dec_values] = svmpredict(testresp, testdata, mysvm);
svmtestcm = zeros(2,2);
for n = 1:length(predict_label)
    if testresp(n) == 1 
        if predict_label(n) == 1
            svmtestcm(2,2) = svmtestcm(2,2)+1;
        else
            svmtestcm(2,1) = svmtestcm(2,1)+1;
        end
    else
        if predict_label(n) == 1
            svmtestcm(1,2) = svmtestcm(1,2)+1;
        else
            svmtestcm(1,1) = svmtestcm(1,1)+1;
        end
    end
end
misclassratesvm = 1-accuracy(1)/100;

%% ==== SVM with cross-validation by using svmtrain & svmclassify ==== 
Cparams = [1e4,1e5,1e6,1e7,1e8,1e9,1e10];
gammas = [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0];
stds = (0.5./gammas).^0.5;
misclassrateLinMat = zeros(length(Cparams),1);
misclassrateGauMat = zeros(length(Cparams),length(stds));
options = optimset('Largescale','off','Algorithm','active-set','MaxIter',5000,'Display','off');
for k = 1:k_fold
    test = (indices == k); 
    train = ~test;
    Xtest = traindata(test,:);
    Ytest = trainresp(test);
    Xtraining = traindata(train,:);            
    Ytraining = trainresp(train);
    for i = 1:length(Cparams)
        mysvm = svmtrain(Xtraining,Ytraining,'kernel_function','linear','boxconstraint',Cparams(i)/length(Ytraining)*ones(size(Ytraining)),'method','QP','options',options);
        C = svmclassify(mysvm,Xtest);
        errRate = sum(Ytest~= C)/length(trainresp);  % mis-classification rate
        misclassrateLinMat(i) = misclassrateLinMat(i)+errRate;
        for j = 1:length(stds)
            mysvm = svmtrain(Xtraining,Ytraining,'kernel_function','rbf','boxconstraint',Cparams(i)/length(Ytraining)*ones(size(Ytraining)),'rbf_sigma',stds(j),'method','QP','options',options);
            C = svmclassify(mysvm,Xtest);
            errRate = sum(Ytest~= C)/length(trainresp);  % mis-classification rate
            misclassrateGauMat(i,j) = misclassrateGauMat(i,j)+errRate;
        end
    end
end
minLin = min(misclassrateLinMat);
minGau = min(min(misclassrateGauMat));
if minLin <= minGau
    InLinMat = find(misclassrateLinMat == minLin,1,'last');
    mysvm = svmtrain(traindata,trainresp,'kernel_function','linear','boxconstraint',Cparams(InLinMat)/length(trainresp)*ones(size(trainresp)),'method','QP','options',options);
else
    [InGaurowMat,InGaucolMat] = find(misclassrateGauMat == minGau);
    [~,Ind] = min((length(Cparams)-InGaurowMat+InGaucolMat));
    mysvm = svmtrain(traindata,trainresp,'kernel_function','rbf','boxconstraint',Cparams(InGaurowMat(Ind))/length(trainresp)*ones(size(trainresp)),'rbf_sigma',stds(InGaucolMat(Ind)),'method','QP','options',options);
end
C = svmclassify(mysvm,testdata);
misclassratesvmMat = sum(testresp~= C)/length(testresp);  % mis-classification rate
svmtestcmMat = confusionmat(testresp,C); % the confusion matrix

%% ==== Decision tree with cross-validation ====
minparents = [10,20,30,40,50];
misclassratesplit = zeros(length(minparents),1);
for k = 1:k_fold
    test = (indices == k); 
    train = ~test;
    Xtest = traindata(test,:);
    Ytest = trainresp(test);
    Xtraining = traindata(train,:);            
    Ytraining = trainresp(train);
    for i = 1:length(minparents)
        mytree = ClassificationTree.fit(Xtraining,Ytraining,'MinParent',minparents(i));
%         view(mytree,'mode','graph')
        [label,~,~,~] = predict(mytree,Xtest);
        err = 0;
        for n = 1:length(label)
            if label(n) ~= Ytest(n)
                err = err+1;
            end
        end
        misclassratesplit(i) = misclassratesplit(i)+err;
    end
end
misclassratesplit = misclassratesplit/length(trainresp);
[minTree,minInd] = min(misclassratesplit);
mytree = ClassificationTree.fit(traindata,trainresp,'MinParent',minparents(minInd));
[label,~,~,~] = predict(mytree,testdata);
treetestcm = zeros(2,2);
for n = 1:length(label)
    if testresp(n) == 1
        if label(n) == 1
            treetestcm(2,2) = treetestcm(2,2)+1;
        else
            treetestcm(2,1) = treetestcm(2,1)+1;
        end
    else
        if label(n) == 1
            treetestcm(1,2) = treetestcm(1,2)+1;
        else
            treetestcm(1,1) = treetestcm(1,1)+1;
        end
    end
end
misclassrateTree = (treetestcm(1,2)+treetestcm(2,1))/length(testresp);