# vgg16 on imagenet2012
## 原始模型
|model_name|comp_ratio|top1|top3|top5|top1-e|top3-e|top5-e|top1-e-inc|top3-e-inc|top5-e-inc|explain|
| ---- | ---- |---| ---- |--- | ------ | --- | --- | --- |---|---|---|
|original|-|35814|43300|45184|0.28372|0.134|0.09632|-|-|-|-|
|ours(without fine-tune)|x5|26244|35495|38759|0.47512|0.2901|0.22482|0.1914|0.1561|0.1285|conv1_x~conv4_x保留0.2比率的通道数，并且向上取整;conv5_x不变|