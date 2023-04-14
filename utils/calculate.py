def iter_model(model , img_metas , images , bboxes , labels):
    images = images.cuda()
    bboxes = [bbox.cuda() for bbox in bboxes]
    labels = [label.cuda() for label in labels]
    outputs = model(images)
    losses = model.bbox_head.loss(
        cls_scores=outputs[0],
        bbox_preds=outputs[1],
        gt_bboxes=bboxes,
        gt_labels=labels,
        img_metas=img_metas
    )
    loss_cls_total = losses['loss_cls'][0] + losses['loss_cls'][1] + losses['loss_cls'][2]
    loss_bbox_total = losses['loss_bbox'][0] +  losses['loss_bbox'][1] + losses['loss_bbox'][2]
    total_loss = loss_cls_total + loss_bbox_total
    return outputs , loss_bbox_total , loss_cls_total, total_loss