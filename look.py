import torch
ck = torch.load("/data/renhongzhang/DeepLabV3Plus-Pytorch/checkpoints/mobilenetv2_140.pth")
ck_140 =list(ck.keys())
print("len",len(ck_140))

new_ckpt = dict()

ck_us = torch.load("/data/renhongzhang/DeepLabV3Plus-Pytorch/checkpoints/mobilenet_v2.pth")
ck_100 = list(ck_us.keys())
for i in range(len(ck_100)):
    key = ck_100[i]
    key_140 = ck_140[i]
    new_ckpt[key] = ck[key_140]
    # assert ck_us[key].shape==ck[key_140].shape
    print("key",key,ck_us[key].shape)
    print("key_140",key_140,ck[key_140].shape)
print("-------------------------------------------------------------------")
torch.save(new_ckpt,"/data/renhongzhang/DeepLabV3Plus-Pytorch/checkpoints/mymobilenetv2_140.pth")
for key in new_ckpt.keys():
    print("key",key,new_ckpt[key].shape)
