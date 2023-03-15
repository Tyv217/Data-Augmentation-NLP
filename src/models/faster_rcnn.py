from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class FasterRCNN(pl.LightningModule):
  def __init__(self, leraning_rate, max_epochs, use_pretrained_weights = True, trainable_backbone_layers = 3):
    super().__init__()
    if use_pretrained_weights:
      weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    else:
      weights = None
    roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
    self.model = fasterrcnn_resnet50_fpn(pretrained=use_pretrained_weights, progress=True, num_classes=44, pretrained_backbone=True)
    # anchor_generator = AnchorGenerator(
        # sizes=((16,), (32,), (64,), (128,)),
        # aspect_ratios=((0.5, 1.0, 2.0),) * 4)
    # self.model.rpn.anchor_generator = anchor_generator

    # 256 because that's the number of features that FPN returns
    # self.model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    # self.model = fasterrcnn_resnet50_fpn(weights = weights, 
    #                                      trainable_backbone_layers = trainable_backbone_layers, 
    #                                      num_classes = 44,
    #                                      rpn_anchor_generator=anchor_generator, 
    #                                      box_roi_pool=roi_pooler
    #                                      )
    self.learning_rate = learning_rate
    self.weight_decay = 1e-4
    self.max_epochs = max_epochs

  def forward(self, input:torch.Tensor, target:torch.Tensor):
    boxes = target['boxes']
    labels = target['labels']
    target = []
    for box, label in zip(boxes, labels):
      target.append({"boxes": box, "labels": torch.stack([label])})
    output = self.model(input, target)
    return output

  def configure_optimizers(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr = self.learning_rate, weight_decay=self.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
    return [optimizer], [lr_scheduler]

  def training_step(self, batch, batch_idx):
    input, target = batch
    output = self(input,target)
    loss = sum(loss for loss in output.values())
    if(np.isnan(loss.cpu().detach())):
      print(input)
      print(torch.any(input.isnan()))
      print(target)
      print(output)
      raise Exception
    self.log(
      "training_loss",
      loss,
      on_step = True,
      on_epoch = True,
      prog_bar = True,
      logger = True,
      sync_dist = True,
    )
    return loss
  
  def validation_step(self, batch, batch_idx):
    input, target = batch
    outputs = self(input,target)
    metric = MeanAveragePrecision()
    boxes = target['boxes']
    labels = target['labels']
    target = []
    for box, label in zip(boxes, labels):
      target.append({"boxes": box, "labels": torch.stack([label])})
    metric.update(outputs, target)
    maps = metric.compute()
    map = maps['map']
    self.log(
      "validation_map",
      map,
      on_step = False,
      on_epoch = True,
      prog_bar = True,
      logger = True,
      sync_dist = True,
    )
    return map

  def test_step(self, batch, batch_idx):
    
    input, target = batch
    outputs = self(input,target)
    metric = MeanAveragePrecision()
    boxes = target['boxes']
    labels = target['labels']
    target = []
    for box, label in zip(boxes, labels):
      target.append({"boxes": box, "labels": torch.stack([label])})
    metric.update(outputs, target)
    maps = metric.compute()
    map = maps['map']

    self.log(
      "test_map",
      map,
      on_step = False,
      on_epoch = True,
      prog_bar = True,
      logger = True,
      sync_dist = True,
    )
    return map