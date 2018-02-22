%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Authors: Alexander Gomez Villa - Jefferson Cunalata - Fabio Arnez
% -------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('positive.mat')
load('negative.mat')

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


SVMStruct = svmtrain(train,trainLabel,'kernel_function','rbf');
prediction = svmclassify(svmStruct,test);


error = sum(abs(prediction-testLabel))/size(test,1);