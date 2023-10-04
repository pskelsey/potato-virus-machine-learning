% pvyMachineLearning.m
% Author: Peter Skelsey
% Organisation: The James Hutton Institute, Dundee, UK
% Date modified: 29/09/2023

% This script develops a suite of machine learning algorithms for
% predicting potato virus Y incidence at the landscape-scale using a nested
% cross-validation procedure for model training, tuning and testing. The
% best performing algorithm is then finalized and interpreted using Shapley
% values.

%% Nested cross-validation

% Load data
T = readtable('yourData.csv');

% Outer loop data partition
k = 10;
rng(0,'twister');
cvpOuter = cvpartition(T.cases,'KFold',k,'Stratify',true);

% Storage
S = struct();

% Model loop
for iModel = 1:7

    % Storage
    trainResults = NaN(k,14);
    testResults = NaN(k,14);

    % Nested CV outer loop: model performance
    for iOuter = 1:k
    
        % Data partitioning
        trainOut = T(cvpOuter.training(iOuter),:);
        testOut = T(cvpOuter.test(iOuter),:);
        cvpInner = cvpartition(trainOut.cases,'KFold',k,'Stratify',true);
        
        % Set up Bayesian optimisation
        hypopts = struct('CVPartition',cvpInner,...
            'AcquisitionFunctionName','expected-improvement-plus',...
            'useparallel',true,'Verbose',1,'ShowPlots',0);

        % Nested CV inner loop: model optimisation
        switch iModel
            case 1
                mdl = LinearDiscriminant(trainOut,hypopts);
            case 2
                mdl = KNearestNeighbor(trainOut,hypopts);
            case 3
                mdl = DecisionTree(trainOut,hypopts);
            case 4
                mdl = SupportVectorMachine(trainOut,hypopts);
            case 5
                mdl = NeuralNetwork(trainOut,hypopts);
            case 6
                mdl = RandomForest(trainOut,hypopts);
            otherwise
                mdl = RandomUndersamplingBoosting(trainOut,hypopts);
        end % Inner loop
                           
        % Performance
        [metricsR,metricsT] = Performance(mdl,trainOut,testOut);

        % Storage
        trainResults(iOuter,:) = metricsR;
        testResults(iOuter,:) = metricsT;
    
    end % Outer loop

    % Storage
    S(iModel).trainResults = trainResults;
    S(iModel).testResults = testResults;   

end % Model loop

% Save
save pvyResults.mat S

%% Finalized model - Random Forest

% Model optimisation
rng(0,'twister');
hypopts = struct('CVPartition',cvpOuter,...
    'AcquisitionFunctionName','expected-improvement-plus',...
    'useparallel',true,'Verbose',1,'ShowPlots',0);
mdl = RandomForest(T,hypopts);
save pvyRF.mat mdl

% Model interpretation
load pvyRF.mat mdl
shap = ShapleyValues(T,mdl);
save shapleyValuesRF.mat shap

%% Local functions

% Discriminant analysis classifier
function mdl = LinearDiscriminant(T,hypopts)
    T.weights = [];
    mdl = fitcdiscr(T,'cases','Cost', [0 1;2 0],'ClassNames', [0; 1],...
        'OptimizeHyperparameters','all',...
        'HyperparameterOptimizationOptions',hypopts);
end

% k-nearest neighbor classifier
function mdl = KNearestNeighbor(T,hypopts)
    T.weights = [];
    mdl = fitcknn(T,'cases','ClassNames',[0,1],'Cost',[0 1;2 0],...
        'Standardize',true,'OptimizeHyperparameters',...
        {'NumNeighbors','Distance'},...
        'HyperparameterOptimizationOptions',hypopts);
end

% Decision tree classifier
function mdl = DecisionTree(T,hypopts)
    T.weights = [];
    mdl = fitctree(T,'cases','ClassNames',[0,1],'Cost',[0 1;2 0],...
        'OptimizeHyperparameters','auto',...
        'HyperparameterOptimizationOptions',hypopts);
end

% Support vector machine classifier
function mdl = SupportVectorMachine(T,hypopts)
    T.weights = [];
    mdl = fitcsvm(T,'cases','ClassNames',[0,1],'Cost',[0 1;2 0],...
        'Standardize',true, 'OptimizeHyperparameters',...
        {'BoxConstraint','KernelScale'},...
        'HyperparameterOptimizationOptions',hypopts);
end

% Neural network classifier
function mdl = NeuralNetwork(T,hypopts)
    mdl = fitcnet(T,'cases','ClassNames',[0,1],'Standardize',true,...
        'Weights','weights','OptimizeHyperparameters',...
        {'Activations','Lambda','LayerWeightsInitializer',...
        'LayerBiasesInitializer','LayerSizes'},...
        'HyperparameterOptimizationOptions',hypopts);
end

% Ensemble of classifiers - Random forest
function mdl = RandomForest(T,hypopts)
    t = templateTree('PredictorSelection','interaction-curvature',...
        'MaxNumSplits',20,'Reproducible',true);
    mdl = fitcensemble(T,'cases','Learners',t,'ClassNames',[0,1],...
        'Method','Bag','Weights','weights',...
        'OptimizeHyperparameters',...
        {'NumLearningCycles','MinLeafSize','SplitCriterion',...
        'NumVariablesToSample'},...
        'HyperparameterOptimizationOptions',hypopts);
end

% Ensemble of classifiers - Random undersampling boosting
function mdl = RandomUndersamplingBoosting(T,hypopts)
    T.weights = [];
    t = templateTree('MaxNumSplits',20,'Reproducible',true);    
    mdl = fitcensemble(T,'cases','Learner',t,'ClassNames',[0,1],...
        'Method','RUSBoost','OptimizeHyperparameters',...
        {'NumLearningCycles','MinLeafSize','LearnRate',...
        'SplitCriterion'},...
        'HyperparameterOptimizationOptions',hypopts);
end

% Predictions on training and test folds
function [metricsR,metricsT] = Performance(mdl,train,test)
    [labels,scores] = resubPredict(mdl);
    metricsR = PerformanceMetrics(train.cases,labels,scores);
    [labels,scores] = predict(mdl,test);
    metricsT = PerformanceMetrics(test.cases,labels,scores);
end

% Confusion matrix and metrics
function metrics = PerformanceMetrics(yTest,labels,scores)
    c = confusionmat(yTest,labels);
    tn = c(1);
    fn = c(2);
    fp = c(3);
    tp = c(4);
    TPR = tp/(tp+fn);   
    TNR = tn/(tn+fp);
    accBal = (TPR+TNR)/2;
    [~,~,~,AUROC] = perfcurve(yTest,scores(:,2),1);
    metrics = [accBal AUROC];
end

% Shapley values
function shap = ShapleyValues(T,mdl)
    oob = ~mdl.UseObsForLearner(:,:);
    numSamples = height(T);
    numPredictors = width(T)-2;
    shap = NaN(numSamples,numPredictors);
    for ii = 1:numSamples
        oobTrees = oob(ii,:);
        f = @(T) predict(mdl,T,'Learners',oobTrees);
        explainer = shapley(f,T(:,1:end-2));
        explainer = fit(explainer,T(ii,:),...
            'UseParallel',true,'Method','interventional');
        shap(ii,:) = table2array(explainer.ShapleyValues(:,2))';
    end
end