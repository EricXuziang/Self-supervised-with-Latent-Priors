import networks
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input = torch.rand(1, 3, 224, 224).to(device)
input_1 = torch.rand(1, 64, 4, 4).to(device)
input_2 = torch.rand(6, 3, 480, 480).to(device)


# resnet encoder and decoder
# encoder_orgin = networks.ResnetEncoder(18, pretrained = True).to(device)
# decoder_orgin = networks.DepthDecoder(encoder_orgin.num_ch_enc, [0, 1, 2, 3]).to(device)
# features = encoder_orgin(input_2)
# outputs = decoder_orgin(features)

# latent_bank
# latent_bank = networks.StyledGenerator().to(device)
# encoder = networks.Encoder().to(device)
# decoder = networks.Decoder().to(device)
# EDLatentBank  = networks.EDLatentBank(encoder, decoder, latent_bank).to(device)
# features, codes = encoder(input)
# for i in range(len(features)):
#     print(features[i].shape)
# print(decoder_mask(input_1).shape)
# gen = torch.randn(input.shape[0], 512).to(device)
# bank_codes = latent_bank(gen, step=6, bank=True, feats=features, codes=codes)
# outputs = decoder(feats=features[-4:], codes=bank_codes)

# new latent_bank
# unetencoder = networks.UNetEncoder().to(device)
# unetdecoder = networks.UNetDecoder().to(device)
# rrdbencoder = networks.RRDBEncoder().to(device)
# x1, x2, x3, x4, x5, x_compressed = rrdbencoder(input_2)
# output = unetdecoder(x1, x2, x3, x4, x5, x_compressed)

gan_path  = '/well/rittscher/users/ycr745/monodepth2_revised/pretrained_gan/rgenerator_wp_final.pth'
latent_bank = networks.StyledGenerator().to(device)
rrdbencoder = networks.RRDBEncoder().to(device)
unetdecoder = networks.UNetDecoder().to(device)
x1, x2, x3, x4, x5, x_compressed = rrdbencoder(input_2)
print(x1.shape,x2.shape,x3.shape,x4.shape,x5.shape,x_compressed.shape)
latent_bank.load_state_dict(torch.load(gan_path))
for p in latent_bank.parameters():
    p.requires_grad=False
for p in latent_bank.fusions.parameters():
    p.requires_grad=True
for p in latent_bank.process_compress.parameters():
    p.requires_grad=True    
code = latent_bank(x_compressed, x1, x2, x3, x4, x5, mode='latent_bank')
print(code[0].shape,code[1].shape,code[2].shape,code[3].shape,code[4].shape)

# output = unetdecoder(x1, x2, x3, x4, x5, x_compressed, code)
# print(output[0].shape,output[1].shape,output[2].shape,output[3].shape,output[4].shape)


# test transformer 
# transformer_encoder = networks.TransformerEncoder(transformer_model = 'deit-base', pretrained = False, img_size=(224, 224)).to(device)
# num_ch_enc = transformer_encoder.num_ch_enc
# transformer_decoder = networks.DepthDecoderDpt(num_ch_enc = num_ch_enc).to(device)

# feature_out = transformer_encoder(input, mask=None, mode = 'train')
# output = transformer_decoder(feature_out)
# print(output)




            