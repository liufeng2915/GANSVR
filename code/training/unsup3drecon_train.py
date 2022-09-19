import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
import numpy as np  
import scipy.io
import utils.util as utils
import utils.plots as utils_plt
from torch.nn import functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable

class Unsup3DTrainRunner():
    def __init__(self,**kwargs):

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.num_views = self.conf.get_int('train.num_views')
        self.num_views_per_train = self.conf.get_int('train.num_views_per_train')
        self.category = self.conf.get_string('train.category')
        self.render_img_res = self.conf.get_int('train.render_img_res')
        self.train_data_dir=self.conf.get_string('train.data_dir')
        img_res_default = 256
        self.img_scale = self.render_img_res/img_res_default

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../',kwargs['exps_folder_name'],self.category)):
                timestamps = os.listdir(os.path.join('../',kwargs['exps_folder_name'],self.category))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        # create exp dirs
        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.category)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))
        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.img_encoder_params_subdir = "ImgEncoderParameters"
        self.model_params_subdir = "ModelParameters"
        self.feat_params_subdir = "FeatParameters"
        self.cam_params_subdir = "CamParameters"
        self.log_subdir = "logs"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.img_encoder_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.feat_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.log_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.cam_params_subdir))
        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))
        print('shell command : {0}'.format(' '.join(sys.argv)))

        # # load data
        print('Loading data ...')
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(data_dir=self.train_data_dir, img_res=self.render_img_res)
        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            num_workers=4,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           num_workers=1,
                                                           batch_size=self.batch_size,
                                                           shuffle=True,
                                                           )
        print('Finish loading data ...')

        # # define model
        self.img_encoder = utils.get_class(self.conf.get_string('train.image_encoder'))(latent_dim=self.conf.get_int('model.image_encoder_model.latent_dim'), img_res=self.render_img_res)
        self.recon_model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
        if torch.cuda.is_available():
            self.recon_model.cuda()
            self.img_encoder.cuda()

        # # define loss
        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        # # 
        self.num_instances = len(self.train_dataloader)

        # settings for feat optimization
        self.feat_vecs = torch.nn.Embedding(self.num_instances, self.conf.get_int('model.image_encoder_model.latent_dim'), sparse=True).cuda()
        torch.nn.init.xavier_normal_(self.feat_vecs.weight)

        # # learning rate
        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam(list(self.recon_model.parameters())+list(self.img_encoder.parameters()), lr=self.lr)
        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)

        # settings for camera optimization
        reference_cam,intrinsics = utils.get_pose_init(os.path.join("./data", self.category+'_reference_cam.mat'), self.img_scale)
        self.pose_vecs = torch.nn.Embedding(self.num_instances*self.num_views, 7, sparse=True).cuda()
        self.pose_vecs.weight.data.copy_(reference_cam.repeat(self.num_instances, 1))
        self.optimizer_cam = torch.optim.SparseAdam(list(self.pose_vecs.parameters())+list(self.feat_vecs.parameters()), self.conf.get_float('train.learning_rate_cam'))


        self.intrinsics = intrinsics.unsqueeze(0).repeat(self.num_views, 1, 1).cuda()
        uv = utils.get_uv(self.render_img_res)
        self.uv = uv.unsqueeze(0).repeat(self.num_views, 1, 1).cuda()

        # #
        self.start_epoch = 0
        if is_continue:
            old_checkpoints_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpoints_dir, 'ImgEncoderParameters', str(kwargs['checkpoint']) + ".pth"))
            self.img_encoder.load_state_dict(saved_model_state["imgencoder_state_dict"])

            saved_model_state = torch.load(
                os.path.join(old_checkpoints_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.recon_model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpoints_dir, self.feat_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.feat_vecs.load_state_dict(data["feat_vecs_state_dict"])

            data = torch.load(
                os.path.join(old_checkpoints_dir, self.cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.pose_vecs.load_state_dict(data["pose_vecs_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.render_img_res*self.render_img_res
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.plot_conf = self.conf.get_config('plot')

        # # log
        self.summary = SummaryWriter(os.path.join(self.checkpoints_path, self.log_subdir))

        # # alpha
        self.alpha_milestones = self.conf.get_list('train.alpha_milestones', default=[])
        self.alpha_factor = self.conf.get_float('train.alpha_factor', default=0.0)
        for acc in self.alpha_milestones:
            if self.start_epoch > acc:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

    def save_checkpoints(self, epoch):

        torch.save(
            {"epoch": epoch, "imgencoder_state_dict": self.img_encoder.state_dict()},
            os.path.join(self.checkpoints_path, self.img_encoder_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "imgencoder_state_dict": self.img_encoder.state_dict()},
            os.path.join(self.checkpoints_path, self.img_encoder_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "model_state_dict": self.recon_model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.recon_model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "feat_vecs_state_dict": self.feat_vecs.state_dict()},
            os.path.join(self.checkpoints_path, self.feat_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "pose_vecs_state_dict": self.feat_vecs.state_dict()},
            os.path.join(self.checkpoints_path, self.feat_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
            os.path.join(self.checkpoints_path, self.cam_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
            os.path.join(self.checkpoints_path, self.cam_params_subdir, "latest.pth"))

    def visualization(self, epoch, data_iter):

        self.img_encoder.eval()
        self.recon_model.eval()
        self.pose_vecs.eval()
        indices, model_input, ground_truth = next(iter(self.plot_dataloader))
        pose_indices = indices*self.num_views + torch.arange(self.num_views)
        img_sampling_idx = torch.randperm(self.num_views)[:1]

        input_image = model_input["image"].squeeze(0).cuda()
        input_image = input_image[img_sampling_idx+1]
        model_input["intrinsics"] = self.intrinsics.clone()[img_sampling_idx].cuda()
        model_input["uv"] = self.uv.clone()[img_sampling_idx].cuda()
        model_input["object_mask"] = model_input["object_mask"].squeeze(0)[img_sampling_idx].cuda()
        pose_input = self.pose_vecs(pose_indices.cuda())
        model_input['pose'] = pose_input[img_sampling_idx]
        gt_rgb = ground_truth['rgb'].squeeze(0)[img_sampling_idx].cuda()

        encoder_feature_maps, _ = self.img_encoder( model_input["image"].squeeze(0)[:1].cuda())
        encoder_feature_latent = self.feat_vecs(indices.cuda())

        split = utils.split_input(model_input, self.total_pixels)
        res = []
        for s in split:
            out = self.recon_model.forward(s, encoder_feature_latent)
            res.append({
                'points': out['points'].detach(),
                'rgb_values': out['rgb_values'].detach(),
                'network_object_mask': out['network_object_mask'].detach(),
                'object_mask': out['object_mask'].detach()
            })

        batch_size = ground_truth['rgb'].shape[0]
        model_outputs = utils.merge_output(res, self.total_pixels, batch_size)
        utils_plt.plot(self.recon_model,
                    torch.exp(encoder_feature_maps),
                    encoder_feature_latent,
                    indices,
                    data_iter,
                    model_outputs,
                    model_input['pose'],
                    gt_rgb,
                    self.plots_dir,
                    epoch,
                    self.render_img_res,
                    **self.plot_conf
                    )
        self.img_encoder.train()
        self.recon_model.train()
        self.pose_vecs.train()


    def run(self):
        print("training...")

        for epoch in range(self.start_epoch, self.nepochs + 1):
            
            if epoch in self.alpha_milestones:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

            if epoch % 1 == 0:
                self.save_checkpoints(epoch)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):

                if data_index % 500 == 0:
                    self.save_checkpoints(epoch)
                if data_index % self.plot_freq == 0:
                    self.visualization(epoch, data_index)

                # data
                pixel_sampling_idx = torch.randperm(self.total_pixels)[:self.num_pixels]
                img_sampling_idx = torch.randperm(self.num_views)[:self.num_views_per_train]

                input_data = {}
                gt_data = {}
                input_data["intrinsics"] = self.intrinsics.clone()[img_sampling_idx]
                input_data["uv"] = self.uv.clone()[img_sampling_idx][:,pixel_sampling_idx]

                pose_indices = indices*self.num_views + torch.arange(self.num_views)
                input_data["pose"] = self.pose_vecs(pose_indices.cuda())[img_sampling_idx]
                input_data["object_mask"] = model_input["object_mask"].squeeze(0).cuda()[img_sampling_idx][:,pixel_sampling_idx]

                gt_data["rgb"] = ground_truth["rgb"].squeeze(0).cuda()[img_sampling_idx][:,pixel_sampling_idx,:]
                input_image = model_input["image"].squeeze(0).cuda()[img_sampling_idx+1]
                gt_data["feat_vecs"] = self.feat_vecs(indices.cuda())


                # encoder
                encoder_feature_maps, encoder_feature_latent = self.img_encoder(torch.cat((model_input["image"].squeeze(0).cuda()[0:1],input_image),0))
                model_outputs = self.recon_model.forward(input_data, self.feat_vecs(indices.cuda()))
                model_outputs['encoder_feature_maps'] = encoder_feature_maps[1:,pixel_sampling_idx,:]
                model_outputs['esti_feat'] = encoder_feature_latent
   
                # loss
                loss_output = self.loss.forward(model_outputs, gt_data)
                loss = loss_output['loss']

                self.optimizer.zero_grad()
                self.optimizer_cam.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.optimizer_cam.step()

                # log
                for key, value in loss_output.items():
                    self.summary.add_scalar('train/loss/{}'.format(key), value, epoch*self.num_instances+data_index)
                self.summary.add_histogram('embedding/gt_latent', gt_data["feat_vecs"], epoch*self.num_instances+data_index)
                self.summary.add_histogram('embedding/esti_latent', encoder_feature_latent[0:1], epoch*self.num_instances+data_index)

                print(
                    'Time: {0}, Epoch: [{1}]  Data: ({2}/{3}): loss = {4}, rgb_loss = {5}, eikonal_loss = {6}, mask_loss = {7}, latent_loss = {8}, u_term1 = {9}, u_term2 = {10}, alpha = {11}, lr = {12}'
                    .format(str(datetime.now()), epoch, data_index, self.num_instances-1, loss_output['loss'],
                        loss_output['rgb_loss'],
                        loss_output['eikonal_loss'],
                        loss_output['mask_loss'],
                        loss_output['latent_loss'],
                        loss_output['u_term1'],
                        loss_output['u_term2'],
                        self.loss.alpha,
                        self.scheduler.get_last_lr()[0])
                    )
            self.scheduler.step()
