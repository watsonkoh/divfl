# Label-flipping Attack in Federated Learning
# Diverse Client Selection for Federated Learning via Submodular Maximization

## MSSD 2023 Thesis on Label-Flipping Attack with Diverse Client Selection :

<b>Title</b>: <i>DLabel-Flipping Attack with Diverse Client Selection for Federated Learning</i> <a href="https://openreview.net/pdf?id=nwKXyFvaUm">[pdf]</a> <a href="https://iclr.cc/virtual/2022/poster/7047">[presentation]</a>\
<b>Authors</b>: Watson Koh, Dr Brian (NTU)\
<b>Institutes</b>: Intel Labs, Carnegie Mellon University, University of Washington


Our implementation is based on 
<b>Abstract</b>\
In every communication round of federated learning, a random subset of clients communicate their model updates back to the server which then aggregates them all. The optimal size of this subset is not known and several studies have shown that typically random selection does not perform very well in terms of convergence, learning efficiency and fairness. We, in this paper, propose to select a small diverse subset of clients, namely those carrying representative gradient information, and we transmit only these updates to the server. Our aim is for updating via only a subset to approximate updating via aggregating all client information. We achieve this by choosing a subset that maximizes a submodular facility location function defined over gradient space. We introduce “federated averaging with diverse client selection (DivFL)”. We provide a thorough analysis of its convergence in the heterogeneous setting and apply it both to synthetic and to real datasets. Empirical results show several benefits to our approach including improved learning efficiency, faster convergence and also more uniform (i.e., fair) performance across clients. We further show a communication-efficient version of DivFL that can still outperform baselines on the above metrics.

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

### Downloading dependencies

```
pip3 install -r requirements.txt  
```

## References
See our () paper for more details as well as all references.

## Acknowledgements
Our implementation is based on [DivFL](https://github.com/melodi-lab/divfl)https://github.com/melodi-lab/divfl).
