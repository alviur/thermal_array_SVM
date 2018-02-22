
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors: Alexander Gomez Villa - Jefferson Cunalata - Fabio Arnez
% -------------------------------------------------------------------------
% Pure Matlab implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Paths
load('positive.mat')
load('negative.mat')
addpath('/home/lex/Desktop/embebbed/libsvm-master/matlab');

data=[positive;negative];
positiveLabels = ones(1022,1);
negativesLabels = zeros(1020,1);
labels = [positiveLabels;negativesLabels];


data2 = [data,labels];
shuffledData = data2(randperm(size(data2,1)),:);


dataS = shuffledData(:,1:end-1);
labelsS = shuffledData(:,end);

train = dataS(1:1021,:);
trainLabel = labelsS(1:1021,:);

test = dataS(1021:end,:);
testLabel = labelsS(1021:end,:);

model = svmtrain(trainLabel, train, ['-t',' 2',' -g' ,' 0.001', ' -c', ' 1']);

[predicted_label, accuracy, decision_values] = svmpredict(testLabel, test, model);

error = sum(abs(predicted_label-testLabel))/size(test,1)
