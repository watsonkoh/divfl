# Label-flipping Attack in Federated Learning
# Diverse Client Selection for Federated Learning via Submodular Maximization

## MSSD 2023 Thesis on Label-Flipping Attack with Diverse Client Selection :

<b>Title</b>: <i>Label-Flipping Attack with Diverse Client Selection for Federated Learning</i>\
<b>Authors</b>: Watson Koh, Dr. Brian (NTU)\
<b>Institutes</b>: ISTD, Singapore University of Technology and Design (SUTD)

The implementation is based on\
<b>Environment</b>\
Macbook M1 processor\
https://developer.apple.com/metal/tensorflow-plugin/

<b>Abstract</b>\
With the new branch of AI, Federated Learning (FL) offers autonomy and privacy framework to all active nodes allowing the nodes to withhold their personal data within their smart devices, whereas communally construct a central machine learning (ML) model. Federated learning algorithms encompass calculating the average of model parameters or gradient updates to approximate a universal model at the central node. Nevertheless, decentralized form of machine learning is met with new security threats by hypothetically malicious participants. Malicious users can contaminate the model by piloting poisoning attacks that can be targeted and/or untargeted. One form of targeted poisoning attack is called Label-Flipping (LF) attack when the aggressors performed the attack by altering the labels of the training samples from one class (the original class) to another (the incorrect class). 

## Preparation

### Dataset generation

We **already provide four synthetic datasets** that are used in the paper under corresponding folders. For all datasets, see the `README` files in separate `data/$dataset` folders for instructions on preprocessing and/or sampling data.

The statistics of real federated datasets are summarized as follows.

<center>

| Dataset       | Devices         | Samples|Samples/device <br> mean (stdev) |
| ------------- |-------------| -----| ---|
| MNIST      | 1,000 | 69,035 | 69 (106)| 
| FEMNIST     | 200      |   18,345 | 92 (159)|
| Shakespeare | 143    |    517,106 | 3,616 (6,808)|
| Sent140| 772      |    40,783 | 53 (32)|

</center>

## References
See our () paper for more details as well as all references.

## Acknowledgements
Our implementation is based on [DivFL](https://github.com/melodi-lab/divfl)https://github.com/melodi-lab/divfl).
