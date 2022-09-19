import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import utils.util as utils
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork


class ImplicitFunction(nn.Module):
    def __init__(
            self,impl_layers
    ):
        super().__init__()
        self.impl_layers = impl_layers
        self.define_network(impl_layers)

        multires = 6
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn

    def define_network(self,impl_layers):
        self.impl = torch.nn.ModuleList()
        for li, linear in enumerate(impl_layers):
            if li!=len(impl_layers)-1:
                layer = torch.nn.Sequential(
                    linear,
                    torch.nn.LayerNorm(linear.bias.shape[-1],elementwise_affine=False),
                    torch.nn.Softplus(beta=100)
                )
            else:
                layer = torch.nn.Sequential(
                    linear,
                )  
            self.impl.append(layer)

    def forward(self,input): # [B,...,3]
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        feat = input
        # extract implicit features
        for li,layer in enumerate(self.impl):
            if li>0:
                feat = torch.cat([feat,input],dim=-1)
            feat = layer(feat)
        return feat

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class Generator(torch.nn.Module):
    def __init__(
            self,
            dims,
            multires=0
    ):
        super().__init__()

        self.layers_impl = dims
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.dim_in = input_ch

        self.hyper_impl = self.get_module_params(self.layers_impl,k0=self.dim_in,interm_coord=True)

    def get_module_params(self,layers,k0,interm_coord=False):
        posenc_L = 5
        impl_params = torch.nn.ModuleList()
        L = utils.get_layer_dims(layers)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = k0
            if interm_coord and li>0:
                k_in += k0
            params = self.define_hyperlayer(dim_in=k_in,dim_out=k_out)
            impl_params.append(params)
        return impl_params

    def define_hyperlayer(self,dim_in,dim_out):
        layers_hyper = [None,256,None]
        latent_dim = 256
        L = utils.get_layer_dims(layers_hyper)
        hyperlayer = []
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = latent_dim
            if li==len(L)-1: k_out = (dim_in+2)*dim_out # weight and bias
            hyperlayer.append(torch.nn.Linear(k_in,k_out))
            if li!=len(L)-1:
                #hyperlayer.append(torch.nn.Softplus(beta=100))
                hyperlayer.append(torch.nn.ReLU(inplace=False))
        hyperlayer = torch.nn.Sequential(*hyperlayer)
        return hyperlayer

    def forward(self,latent):
        impl_layers = self.hyperlayer_forward(latent,self.hyper_impl,self.layers_impl,k0=self.dim_in)
        impl_func = ImplicitFunction(impl_layers)
        return impl_func

    def hyperlayer_forward(self,latent,module,layers,k0):
        batch_size = len(latent)
        impl_layers = []
        L = utils.get_layer_dims(layers)
        for li,(k_in,k_out) in enumerate(L):
            if li==0: k_in = k0
            if li>0: k_in += k0
            hyperlayer = module[li]
            out = hyperlayer.forward(latent).view(batch_size,k_in+2,k_out)
            impl_layers.append(utils.BatchLinear(weight_v=out[:,2:],weight_g=out[:,0:1], bias=out[:,1:2]))
        return impl_layers




class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0
    ):
        super().__init__()

        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)


        rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        return x

class Unsup3DNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.implicit_network = Generator(**conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

    def forward(self, input, encoder_latent):

        # Parse model input
        intrinsics = input["intrinsics"]  
        uv = input["uv"]                  
        pose = input["pose"]              
        object_mask = input["object_mask"].reshape(-1)
        
        ray_dirs, cam_loc = utils.get_camera_params(uv, pose, intrinsics) 

        batch_size, num_pixels, _ = ray_dirs.shape

        implicit_network = self.implicit_network(encoder_latent)
        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer.forward(sdf=lambda x: implicit_network(x)[:, 0],cam_loc=cam_loc,object_mask=object_mask,ray_directions=ray_dirs)
        implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        sdf_output = implicit_network(points)[:, 0:1]  
        ray_dirs = ray_dirs.reshape(-1, 3) 

        if self.training:
            surface_mask = network_object_mask & object_mask    
            surface_points = points[surface_mask]               
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere    
            n_eik_points = batch_size * num_pixels // 2        
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            output = implicit_network(surface_points)
            surface_sdf_values = output[:N, 0:1].detach()   

            g = implicit_network.gradient(points_all)
            surface_points_grad = g[:N, 0, :].clone().detach()
            grad_theta = g[N:, 0, :]

            differentiable_surface_points = self.sample_network.forward(surface_output,  # with gradient
                                                                surface_sdf_values,      # no gradient
                                                                surface_points_grad,     # no gradien
                                                                surface_dists,           # no gradient
                                                                surface_cam_loc,         # with gradient
                                                                surface_ray_dirs)        # no gradient

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None

        view = -ray_dirs[surface_mask]

        rgb_values = torch.ones_like(points).float().cuda()
        if differentiable_surface_points.shape[0] > 0:
            rgb_values[surface_mask] = self.get_rbg_value(implicit_network, differentiable_surface_points, view)

        output = {
            'points': points,
            'rgb_values': rgb_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta
        }

        return output

    def get_rbg_value(self, implicit_network, points, view_dirs):
        output = implicit_network(points)
        g = implicit_network.gradient(points)
        normals = g[:, 0, :]

        feature_vectors = output[:, 1:]
        rgb_vals = self.rendering_network.forward(points, normals, view_dirs, feature_vectors)

        return rgb_vals
