# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary

The dataset used in this project contains data collected during direct marketing campaigns (phone calls) of a Portuguese banking institution. 

It consists of 32950 entries, 20 features containing information about the client, information relative to the `marketing campaign`, and `social and economic` metrics. 

This is a classification problem wich goal is to predict if the client will subscribe (yes/no) to a bank term deposit (variable `y`).

**original source of the data**: 
[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, 
June 2014 (https://repositorio.iscte-iul.pt/bitstream/10071/9499/5/dss_v3.pdf )

## Proposed Solutions:
This project tries to find the optimal solution for the classification task with two routes:
1. Apply hyperparameter tunning to a sklearn logistic regression model fitted by a training script given by udacity. The idea here is to use Hyperdrive module, to find the best model.

2. Use azure automl via the python SDK to build the best model

Finally we can compare both approaches and choose the best overall model.

## Scikit-learn Pipeline
<img align="center" width="700" height="300" src="https://github.com/hualcosa/Optimizing_a_Pipeline_in_Azure/creating-and-optimizing-an-ml-pipeline.png">
 The pipeline starts with the execution of train.py file. This script will download the dataset, preprocess it,
 and fit a sklearn logistic regression classifier with the hyperparameters passed during the script invocation.
 This script is going to be used by Hyperdrive, which are azureml module for performing hyperparameter tunning.
 In order to use it, we will have to specify a sampling strategy. I have used RandomParameterSampling, which will randomly
 select a set of parameter values and fit the model based on it. It usually yields as good results as scanning the complete
 hyperparameter space, with the advantage of consuming less compute-resources. Another component defined was an early termination
 policy, specifically a BanditPolicy. This allow the experiment to avoid spending useless time fitting models
 in a region where parameter combinations are not yielding top results.
 <br><br>
 ## AutoML
 The second branch of the pipeline is going to read the same dataset, apply the preprocessing functions,
 and use azure's AutoML functionality, to create a job that will automatically search for the best performing model, i.e, the
 model with the best accuracy score. This strategy can be beneficial when we want quickly find a good model, without spending
 considerable time and money in R&D.


## Pipeline comparison
The best model the hyperdrive job was capable of finding was a logistic-regression with the following hyperparameters:<br><br>
<img align="center" width="700" height="300" src="https://github.com/hualcosa/Optimizing_a_Pipeline_in_Azure/logistic_regression_hyperdrive.png">
<br><br>
In contrast, the AutoML best performing model was a votingEnsemble that achieved the following mark:<br><br>
<img align="center" width="700" height="300" src="https://github.com/hualcosa/Optimizing_a_Pipeline_in_Azure/voting_ensemble.png">
<br><br>
If we solely consider accuracy as the decisive factor, then the model found the AutoML model is the best pick, but if we also consider
training time and model complexity, the scenario is a little bit different. It took 11 minutes to run the hyperdrive job in the compute cluster
and to fit 20 different model configurations. It took 31 minutes to run the Automl experiment. AutoML is a more expensive
alternative. The the difference between the accuracy of the best model from AutoML and Hyperdrive is only 0.00115%.

The second point is that the best model from AutoML is an ensemble of 6 individual models. If we are concerned about
deploying the model, and low latency becomes a requirement, deploying models like this might become an issue. The fine
tuned logistic regression on the other hand is much more light weight.

## Conclusion
If having the highest accuracy possible is the primary goal, then I would go if the AutoML VotingEnsemble model. Nonetheless,
taking into consideration, training cost, model complexity, minimizing latency during deploying and the shy improvement in the
accuracy when comparing the two strategies, I would choose the fine-tuned logistic regression.

## Future work
A possible path to be explored is to include more hyperparameters in the hyperdrive experiment. Maybe this will allow the
fitting of an even best performing model. <br>
On the AutoML side, we could try to run the experiment longer to let the process try to find better alternatives. But there is
a trade off between the additional cost it will take, and the benefit that improving model accuracy will bring.

**What are some areas of improvement for future experiments? Why might these improvements help the model?**

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
<img align="center" width="700" height="300" src="https://github.com/hualcosa/Optimizing_a_Pipeline_in_Azure/deleting_cpu_cluster.png">

