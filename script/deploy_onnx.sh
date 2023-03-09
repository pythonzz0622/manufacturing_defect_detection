python tools/deployment/pytorch2onnx.py \
custom_configs/retinanet_swin-l_ms/run.py \
ckpts/new_retinanet_swin-l/latest.pth \
--output-file ./retina-swin-transformer.onnx \
--input-img ./demo.jpg \
--test-img ./demo.jpg \
--opset-version 11
--shape 512 461 \
--dynamic-export \
--show \ 
