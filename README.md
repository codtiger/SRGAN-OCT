# SRGAN-OCT
An SRGAN-based OCT imaging SuperResolution


### Classification Results
Here are the classification comparisons with the original dataset at 128x128 resolution and the GAN-generated dataset:

#### Original Dataset
| Class   | Precision | Recall | F1-score | Support |
|--------:|----------:|-------:|---------:|--------:|
| NORMAL  | 1.00      | 0.96   | 0.98     | 437     |
| DME     | 0.94      | 1.00   | 0.97     | 305     |
| DRUSEN  | 1.00      | 0.98   | 0.99     | 221     |
| **accuracy**   |          |        | **0.98** | 963     |
| **macro avg**  | 0.98      | 0.98   | 0.98     | 963     |
| **weighted avg** | 0.98      | 0.98   | 0.98     | 963     |

F1 Score: 0.9790

#### SRResNet
| Class   | Precision | Recall | F1-score | Support |
|--------:|----------:|-------:|---------:|--------:|
| NORMAL  | 1.00      | 0.77   | 0.87     | 437     |
| DME     | 0.66      | 0.77   | 0.71     | 305     |
| DRUSEN  | 0.66      | 0.81   | 0.73     | 221     |
| **accuracy**   |          |        | **0.78** | 963     |
| **macro avg**  | 0.77      | 0.78   | 0.77     | 963     |
| **weighted avg** | 0.81      | 0.78   | 0.79     | 963     |

F1 Score: 0.7697