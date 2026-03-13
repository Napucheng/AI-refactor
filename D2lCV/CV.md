## 多目标检测

先修CNN/Resnet/backward/batch normalization……

<u>Wait for Understanding and Updating</u>

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

#### non-maximum suppression

##### bbox decode

```python
def offset_inverse(anchors, offset_preds):
```

输入锚框和偏移量反编码输出最终预测的边界框。

Q：锚框和真实框如何经过神经网络训练？

损失 = 分类损失 + 回归损失

### 伪代码

```python
# -----------------------------
# 多目标检测全过程伪代码
# -----------------------------

# 假设输入：
# image: (3, H, W)    原始图像
# gt_boxes: (N_gt, 4) 真实目标框
# gt_labels: (N_gt,) 真实类别

# 1. Backbone 提取特征图
feature_maps = backbone_cnn(image)
# feature_maps[i]: (C_i, H_i, W_i)

all_cls_preds, all_bbox_preds = [], []

# 2. 对每个尺度特征图进行 anchor 生成和检测头预测
for fmap in feature_maps:
    # fmap: (C, H, W)
    
    # 2a. 生成 anchor
    anchors = generate_anchors(fmap)
    # anchors: (H*W*num_anchors_per_pos, 4)
    
    # 2b. 检测头预测
    cls_pred, bbox_pred = detection_head(fmap)
    # cls_pred: (H*W*num_anchors_per_pos, num_classes)
    # bbox_pred: (H*W*num_anchors_per_pos, 4)
    
    all_cls_preds.append(cls_pred)
    all_bbox_preds.append(bbox_pred)

# 3. Anchor 与 GT 匹配（训练阶段）
matched_labels, matched_offsets = match_anchors_to_gt(anchors, gt_boxes, gt_labels)
# matched_labels: (num_anchors_total,)
# matched_offsets: (num_anchors_total, 4)

# 4. 损失计算
cls_loss = classification_loss(all_cls_preds, matched_labels)  # 分类损失
bbox_loss = regression_loss(all_bbox_preds, matched_offsets)   # 回归损失
total_loss = cls_loss + alpha * bbox_loss

# 5. 反向传播更新参数
total_loss.backward()
optimizer.step()


# -----------------------------
# 推理阶段
# -----------------------------
all_pred_boxes, all_confidences, all_class_ids = [], [], []

for fmap in feature_maps:
    anchors = generate_anchors(fmap)
    cls_pred, bbox_pred = detection_head(fmap)
    
    # 将 anchor + 偏移 → 预测框
    pred_boxes = decode_bbox(anchors, bbox_pred)
    # pred_boxes: (H*W*num_anchors_per_pos, 4)
    
    # 每个 anchor 最大类别和对应置信度
    conf, class_id = max_over_classes(cls_pred)
    # conf: (H*W*num_anchors_per_pos,)
    # class_id: (H*W*num_anchors_per_pos,)
    
    all_pred_boxes.append(pred_boxes)
    all_confidences.append(conf)
    all_class_ids.append(class_id)

# 合并所有尺度预测
pred_boxes = concat(all_pred_boxes)       # (num_total_anchors, 4)
confidences = concat(all_confidences)     # (num_total_anchors,)
class_ids = concat(all_class_ids)         # (num_total_anchors,)

# 6. 非极大值抑制 NMS
final_boxes, final_classes, final_scores = NMS(pred_boxes, class_ids, confidences, iou_threshold)
# final_boxes: (num_final, 4)
# final_classes: (num_final,)
# final_scores: (num_final,)

# 输出最终预测
return final_boxes, final_classes, final_scores
```



### Notes: Detection Head

在SSD/R-CNN/yolo里面，会分类成两类检测头：分类检测头和回归检测头。

### Notes: Feature Map

特征图是 CNN 对原图卷积后的中间输出，包含每个位置的局部语义信息。

每个位置对应原图上的一个**感受野**

特征图上的每个位置生成锚框，用于预测目标类别和位置

### Notes: Receptive Field

感受野是特征图上某个单元在原图上看到的区域。

- 感受野越大 → 能看到更多上下文 → 适合大目标

- 感受野越小 → 只看到局部 → 适合小目标

多尺度目标检测就是利用不同层特征图不同大小的感受野检测不同大小目标

### Key Stack: FPN/SSD/Faster R-CNN/Yolo



