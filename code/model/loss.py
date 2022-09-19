import torch
from torch import nn
from torch.nn import functional as F

class ReconLoss(nn.Module):
    def __init__(self, eikonal_weight, rgb_weight, mask_weight, latent_weight, uncertainty_weight, alpha):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.rgb_weight = rgb_weight
        self.mask_weight = mask_weight
        self.latent_weight = latent_weight
        self.uncertainty_weight = uncertainty_weight
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l1_loss_mean = nn.L1Loss(reduction='mean')

    def get_rgb_loss(self,rgb_values, rgb_gt, network_object_mask, object_mask, u_term1, u_term2):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float(), torch.tensor(0.0).cuda().float(), torch.tensor(0.0).cuda().float()

        u1 = u_term1.reshape(-1,1)[network_object_mask & object_mask]
        u2 = u_term2.reshape(-1,1)[network_object_mask & object_mask]

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = torch.mean(u1*torch.abs(rgb_values-rgb_gt) + u2) / (float(object_mask.shape[0]) + 1e-6)
        return rgb_loss, torch.mean(u_term1), torch.mean(u_term2)

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / (float(object_mask.shape[0])+1e-6)
        return mask_loss

    def get_latent_loss(self, esti_feat, gt_feat):
        latent_loss = torch.mean(torch.sqrt(torch.sum((esti_feat-gt_feat.repeat(esti_feat.shape[0],1))**2,1)))
        return latent_loss
        
    def uncertainty_terms(self, log_sigma):

        term2 = torch.log(1e-6+torch.exp(log_sigma))
        term1 = torch.exp(-1*term2)
        return term1, term2

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb']
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']
        encoder_feature_maps = model_outputs['encoder_feature_maps']

        u_term1, u_term2 = self.uncertainty_terms(encoder_feature_maps)

        rgb_loss, u_value1, u_value2 = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt, network_object_mask, object_mask, u_term1, u_term2)
        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask)
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        latent_loss = self.get_latent_loss(model_outputs['esti_feat'], ground_truth['feat_vecs'])

        loss = rgb_loss + \
            self.eikonal_weight * eikonal_loss + \
            self.mask_weight * mask_loss + \
            self.latent_weight * latent_loss 

        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'mask_loss': mask_loss,
            'latent_loss': latent_loss,
            'u_term1': u_value1.item(),
            'u_term2': u_value2.item()
        }