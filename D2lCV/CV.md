## CV

先修CNN/Resnet/backward/batch normalization……



### 锚框

![../_images/output_anchor_f592d1_66_0.svg](https://zh-v2.d2l.ai/_images/output_anchor_f592d1_66_0.svg)
$$
s\in (0,1)
\\
r>0\\
height=\frac{hs}{\sqrt{r}}
\\
width=hs\sqrt{r}
$$

#### anchor–GT 匹配

image
  ↓
feature map
  ↓
generate anchors
  ↓
IoU(anchor, GT)
  ↓
anchor matching
  ↓
class labels
bbox offsets
bbox mask
  ↓
training targets