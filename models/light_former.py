import sys
sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from .encoder import Encoder
from .decoder import Decoder
import pytorch_lightning as pl
from dataset.dataset import LightFormerDataset
from pathlib import Path
from functools import partial


def _resnet_forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x) # C: 256
    x = self.layer4(x) # C: 512
    return x

class LightFormer(nn.Module):


    def __init__(self, config):
        super().__init__()
        self.config = config

        # Remove last two layers: Average Pooling and Fully Connected
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = None
        self.resnet.avgpool = None

        # Adjust resnet forward to reflect removed last two layers
        self.resnet.forward = partial(_resnet_forward, self.resnet)


        self.down_conv = nn.Sequential(
            nn.Conv2d(512, 1024, 3),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 256, 3),
            nn.BatchNorm2d(256)
        )
        self.mlp = nn.Sequential(
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,1024),
        )
        self.head1 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,1024)
        )
        self.head2 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,1024)
        )

        # Query embeddings
        self.embed_dim = self.config["embed_dim"] # 256
        self.num_query = self.config["num_query"]
        self.query_embed = nn.Embedding(self.num_query, self.embed_dim)

        # Encoder
        self.encoder = Encoder(self.config)


    def forward(self, images, features=None):
        """
        images: self.config['image_num'] number of buffered sequential images (default 10)
        """
        image_num = self.config['image_num']
        B,_,c,h,w = images.shape

        # Reshape Image Buffer to accommodate Resnet input shape
        images = images.reshape(B*image_num,c,h,w)

        # Modified Resnet Backbone
        vectors = self.resnet(images)

        # Down convolution encoding
        vectors = self.down_conv(vectors) # 512 -> 256

        # Reshape back to batch, image number structure
        _,c,h,w = vectors.shape
        vectors = vectors.view(B, image_num, c, h, w) # [bs,num_img, 256, h, w]

        # Grab query embedding
        query = self.query_embed.weight

        # Run encoder architecture
        agent_all_feature = self.encoder(query, vectors) # [bs, 1, 256]

        # Run simple multilayer perceptrons
        agent_all_feature = self.mlp(agent_all_feature)

        # two headed outputs for use in straight and left decoders
        head1_out = self.head1(agent_all_feature)
        head2_out = self.head2(agent_all_feature)
        head1_out = head1_out.unsqueeze(3)
        head2_out = head2_out.unsqueeze(3)
        return head1_out, head2_out


class LightFormerPredictor(pl.LightningModule, nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Model
        self.model = LightFormer(self.config)
        self.index = 0
        self.class_decoder_st = Decoder(self.config)
        self.class_decoder_lf = Decoder(self.config)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.config['optim']['init_lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.config['optim']['step_size'],
                                              gamma=self.config['optim']['step_factor'])
        return [optimizer], [scheduler]

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        print("TRAINING STEP")
        loss  = self.cal_loss_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        print("VALIDATION STEP")
        loss = self.cal_loss_step(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        print("VALIDATION STEP")
        _ = self.cal_ebeding_step(batch)
        return

    def cal_loss_step(self, batch):
        images = batch["images"]
        head1_out, head2_out = self.model(images) # (bs, 1, 1024, 1)
        st_lightstatus_class = self.class_decoder_st(head1_out, batch["label"][:,:2])
        lf_lightstatus_class = self.class_decoder_lf(head2_out, batch["label"][:,2:4])
        st_class_loss = self.prob_loss(st_lightstatus_class, batch["label"][:,:2])
        lf_class_loss = self.prob_loss(lf_lightstatus_class, batch["label"][:,2:4])
        class_loss = st_class_loss + lf_class_loss
        self.index = self.index+1
        return class_loss

    def prob_loss(self, lightstatus, gt_label):
        """
        Calculate the
        """
        gt_label_idx = torch.argmax(gt_label,dim=-1)
        pred_cls_score = torch.log(lightstatus)
        loss = F.nll_loss(pred_cls_score.squeeze(-1), gt_label_idx, reduction='mean')
        return loss

    def cal_ebeding_step(self, batch):
        images = batch["images"]
        head1_out, head2_out = self.model(images)
        st_lightstatus = self.class_decoder_st(head1_out,None)
        lf_lightstatus = self.class_decoder_lf(head2_out,None)
        B, K, _ = st_lightstatus.shape
        st_prob = st_lightstatus.view(B, self.config["out_class_num"], 1)
        lf_prob = lf_lightstatus.view(B, self.config["out_class_num"], 1)
        st_predict=st_prob.argmax(dim=1)[0][0]
        lf_predict=lf_prob.argmax(dim=1)[0][0]
        st_target=batch["label"][:,:2].argmax(dim=1)[0]
        lf_target=batch["label"][:,2:4].argmax(dim=1)[0]
        with open("complete_model_Kaggle_daytime_n=1_res1.txt","a+") as f:
            flag='right'
            if(st_predict!=st_target) or (lf_predict!=lf_target):
                flag = 'error'
            ss = "{} {} {} {} {} {}\n".format(batch["name"], st_predict, st_target, lf_predict, lf_target, flag)
            f.write(ss)
            f.flush()
        return 0

    def train_dataloader(self):
        image_norm = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
        transform = transforms.Normalize(mean=image_norm[0], std=image_norm[1])
        train_set = LightFormerDataset(self.config['training']['sample_database_folder'], transform)
        # print("what is this", self.config['training']['sample_database_folder'])
        print(f"...............................Total Samples {len(train_set)} .......................................")
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=self.config['training']['batch_size'],
                                  shuffle=True,
                                  # collate_fn=AgentClosureBatch.from_data_list,
                                  num_workers=self.config['training']['loader_worker_num'],
                                  drop_last=True,
                                  pin_memory=True,
                                  persistent_workers=True)
        return train_loader

    def val_dataloader(self):
        image_norm = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
        val_set = LightFormerDataset(self.config['validation']['sample_database_folder'], image_norm)
        val_loader = DataLoader(val_set,
                                batch_size=self.config['validation']['batch_size'],
                                shuffle=False,
                                # collate_fn=AgentClosureBatch.from_data_list,
                                num_workers=self.config['validation']['loader_worker_num'],
                                drop_last=True,
                                pin_memory=True,
                                persistent_workers=True)
        return val_loader

    def test_dataloader(self):
        image_norm = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
        test_set = LightFormerDataset(self.config['test']['sample_database_folder'], image_norm)
        test_loader = DataLoader(test_set,
                                 batch_size=self.config['test']['batch_size'],
                                 shuffle=False,
                                 # collate_fn=AgentClosureBatch.from_data_list,
                                 num_workers=self.config['test']['loader_worker_num'],
                                 drop_last=False,
                                 pin_memory=True,
                                 persistent_workers=True)
        return test_loader
