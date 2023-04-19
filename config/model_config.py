import config.backbone 
import config.neck
import config.dense_head
import torch.nn as nn

class model(nn.Module):
    def __init__(self, backbone , neck , bbox_head):
        super().__init__()
        self.backbone = backbone
        self.backbone.init_weights()
        self.neck = neck
        self.bbox_head = bbox_head

    def forward(self, x):
        out = self.backbone(x)
        out = self.neck(out)
        out = self.bbox_head(out)
        return out

def get_model(backbone_name , neck_name , bbox_head_name , neck_type):
    b = getattr(config.backbone , backbone_name)
    n = getattr(config.neck , neck_name)
    h = getattr(config.dense_head ,bbox_head_name)
    return model(b() , n(neck_type) , h())


if __name__ == "__main__":
    model = get_model('Swin_L' , 'Swin_L_neck' ,'Retina_head')
