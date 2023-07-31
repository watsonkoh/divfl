# MSSD 2023 Thesis on Label-Flipping Attack and Defense on Federated Learning :

<b>Title</b>: <i>Label-Flipping Attack and Defense on Federated Learning</i>\
<b>Authors</b>: Watson Koh\
<b>Institutes</b>: ISTD, Singapore University of Technology and Design (SUTD)

### Abstract
With the new branch of AI, Federated Learning (FL) offers autonomy and privacy framework to all active nodes allowing the nodes to withhold their personal data within their smart devices, whereas communally construct a central machine learning (ML) model. Federated learning algorithms encompass calculating the average of model parameters or gradient updates to approximate a universal model at the central node. Nevertheless, decentralized form of machine learning is met with new security threats by hypothetically malicious participants. Malicious users can contaminate the model by piloting poisoning attacks that can be targeted and/or untargeted. One form of targeted poisoning attack is called Label-Flipping (LF) attack when the aggressors performed the attack by altering the labels of the training samples from one class (the original class) to another (the incorrect class). 

## Preparation

### Based upon
- <b>DivFL - Diverse Client Selection for Federated Learning via Submodular Maximization</b>
- <b>FedProx - Federated Optimization in Heterogeneous Networks</b>

### Environment
Macbook M1 processor\
https://developer.apple.com/metal/tensorflow-plugin/


### Dataset generation

Five datasets that are used in the paper are under corresponding data folders. 

<center>
  
| Dataset       | Devices         | Samples|Samples/device <br> mean (stdev) |
| ------------- |-------------| -----| ---|
| MNIST      | 1,000 | 69,035 | 69 (106)| 
| FEMNIST (nist)    | 200      |   18,345 | 92 (159)|
| Synthetic_IID | 30    | | |
| Synthetic_0_0 | 30    | | |
| Synthetic_1_1 | 30    | | |

</center>

## References
See our () paper for more details as well as all references.

## Acknowledgements
Our implementation is based on [DivFL](https://github.com/melodi-lab/divfl)
