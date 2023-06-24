# Label-flipping Attack in Federated Learning
# Diverse Client Selection for Federated Learning via Submodular Maximization

## MSSD 2023 Thesis on Label-Flipping Attack with Diverse Client Selection :

<b>Title</b>: <i>DLabel-Flipping Attack with Diverse Client Selection for Federated Learning</i> <a href="https://openreview.net/pdf?id=nwKXyFvaUm">[pdf]</a> <a href="https://iclr.cc/virtual/2022/poster/7047">[presentation]</a>\
<b>Authors</b>: Watson Koh, Dr. Brian (NTU)\
<b>Institutes</b>: ISTD, Singapore University of Technology and Design (SUTD)


Our implementation is based on 
<b>Abstract</b>\
In every communication round of federated learning, a random subset of clients communicate their model updates back to the server which then aggregates them all.

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
