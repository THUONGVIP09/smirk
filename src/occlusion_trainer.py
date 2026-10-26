"""
Occlusion-aware SMIRK training.

Same framework as SmirkTrainer (reconstruction path through the neural translator + cycle path); what changes:
  * the encoder sees batch['img_input'] (an occluded face, or the clean one), every target comes from the clean image
    (batch['img']): landmarks, reconstruction / perceptual / emotion targets, the pixels given to the translator;
  * landmark and L1 reconstruction losses are weighted by visibility: occluded landmarks / pixels get
    occlusion.landmark_weight_occluded / occlusion.reconstruction_weight_occluded instead of 1;
  * visible-region consistency (loss_weights.visible_consistency): the expression mesh predicted from the occluded
    image must match the one of the teacher (the frozen base encoder, i.e. the pretrained SMIRK, on the clean image)
    on the face vertices that are NOT occluded (weight 1), and only loosely on the occluded ones
    (occlusion.consistency_weight_occluded); this is what keeps the visible parts unaffected by the occluder;
  * cycle path: the translator output is occluded with the sample's occluder before it is encoded again
    (occlusion.cycle_occlusion), so the encoder learns to recover known parameters despite the occlusion;
  * transfer learning: occlusion.freeze_backbone keeps the expression backbone frozen (only the regression head is
    trained), occlusion.optimize_generator=False keeps the translator frozen; frozen modules stay in eval mode;
  * resumable checkpoints (model + optimizers + schedulers + position).
"""
import os
import pickle
import torch
import torch.nn.functional as F

from src.smirk_trainer import SmirkTrainer
from src.renderer.util import batch_orth_proj
import src.utils.masking as masking_utils
import src.utils.utils as utils


def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum().clamp(min=1e-6)


class OcclusionTrainer(SmirkTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.occ = config.occlusion
        with open('assets/FLAME_masks/FLAME_masks.pkl', 'rb') as f:
            face = pickle.load(f, encoding='latin1')['face']
        self.register_buffer('face_vertices', torch.tensor(sorted(face.tolist()), dtype=torch.long), persistent=False)

    # ------------------------------------------------------------------ freezing / modes
    def apply_freezing(self):
        if self.occ.freeze_backbone:
            utils.freeze_module(self.smirk_encoder.expression_encoder.encoder, 'expression backbone')
        if not self.occ.optimize_generator and self.config.arch.enable_fuse_generator:
            utils.freeze_module(self.smirk_generator, 'fuse generator')

    def train(self, mode=True):
        super().train()
        # the original SMIRK training keeps the pose / shape encoders in train mode even when they are not optimized
        # (their batch-norm statistics still change); occlusion.freeze_bn_not_optimized=True keeps them fixed instead
        if self.occ.freeze_bn_not_optimized:
            if not self.config.train.optimize_pose:
                self.smirk_encoder.pose_encoder.eval()
            if not self.config.train.optimize_shape:
                self.smirk_encoder.shape_encoder.eval()
        # SmirkTrainer.step re-enables gradients of the generator after the cycle path: freeze again at every step
        self.apply_freezing()
        return self

    def unfreeze_encoder(self):
        super().unfreeze_encoder()
        self.apply_freezing()

    def optimizers_step(self, step_encoder=True, step_fuse_generator=True):
        super().optimizers_step(step_encoder, step_fuse_generator and self.occ.optimize_generator)

    def scheduler_step(self):
        self.encoder_scheduler.step()
        if self.config.arch.enable_fuse_generator and self.occ.optimize_generator:
            self.smirk_generator_scheduler.step()

    def trainable_parameters(self):
        """Number of trainable parameters per module (to check the freezing)."""
        count = lambda module: sum(p.numel() for p in module.parameters() if p.requires_grad)
        modules = {'pose encoder': self.smirk_encoder.pose_encoder, 'shape encoder': self.smirk_encoder.shape_encoder,
                   'expression backbone': self.smirk_encoder.expression_encoder.encoder,
                   'expression head': self.smirk_encoder.expression_encoder.expression_layers}
        if self.config.arch.enable_fuse_generator:
            modules['generator'] = self.smirk_generator
        optimized = {'pose encoder': self.config.train.optimize_pose, 'shape encoder': self.config.train.optimize_shape,
                     'expression backbone': self.config.train.optimize_expression and not self.occ.freeze_backbone,
                     'expression head': self.config.train.optimize_expression, 'generator': self.occ.optimize_generator}
        return {name: count(m) if optimized[name] else 0 for name, m in modules.items()}

    def set_freeze_status(self, config, batch_idx, epoch_idx):
        super().set_freeze_status(config, batch_idx, epoch_idx)
        if not self.occ.optimize_generator:  # the generator never trains: every cycle step trains the encoder
            self.config.train.freeze_encoder_in_second_path = False
            self.config.train.freeze_generator_in_second_path = True

    # ------------------------------------------------------------------ helpers
    def expression_vertices(self, shape, params):
        """FLAME vertices with a given shape, the expression / jaw / eyelids of params and a neutral pose."""
        return self.flame.forward({'shape_params': shape, 'expression_params': params['expression_params'],
                                   'jaw_params': params['jaw_params'], 'eyelid_params': params['eyelid_params'],
                                   'pose_params': torch.zeros_like(params['pose_params'])})['vertices']

    def hidden_vertices(self, params, occlusion_mask):
        """Face vertices (posed with params) that fall under the occlusion mask of the crop: B x F bool."""
        H = occlusion_mask.shape[-1]
        projected = batch_orth_proj(self.flame.forward(params)['vertices'][:, self.face_vertices], params['cam'])
        x = ((projected[..., 0] + 1) / 2 * H).round().long()
        y = ((-projected[..., 1] + 1) / 2 * H).round().long()  # same y flip as Renderer.forward
        inside = (x >= 0) & (x < H) & (y >= 0) & (y < H)
        rows = torch.arange(x.shape[0], device=x.device)[:, None].expand_as(x)
        return (occlusion_mask[:, 0][rows, y.clamp(0, H - 1), x.clamp(0, H - 1)] > 0.5) & inside

    # ------------------------------------------------------------------ first path
    def step1(self, batch):
        B, C, H, W = batch['img'].shape
        img, img_input = batch['img'], batch['img_input']
        occlusion_mask = batch['occlusion_mask']
        lw = self.config.train.loss_weights

        encoder_output = self.smirk_encoder(img_input)
        flame_output = self.flame.forward(encoder_output)
        renderer_output = self.renderer.forward(flame_output['vertices'], encoder_output['cam'],
                                                landmarks_fan=flame_output['landmarks_fan'], landmarks_mp=flame_output['landmarks_mp'])
        rendered_img = renderer_output['rendered_img']
        flame_output.update(renderer_output)

        with torch.no_grad():
            teacher = self.base_encoder(img)  # pretrained encoder on the clean image

        losses = {}

        # landmarks of the clean image, occluded ones down-weighted
        valid_fan = batch['flag_landmarks_fan']
        losses['landmark_loss_fan'] = 0 if torch.sum(valid_fan) == 0 else \
            F.mse_loss(flame_output['landmarks_fan'][valid_fan, :17], batch['landmarks_fan'][valid_fan, :17])
        visible = batch['landmarks_mp_visible'].float()
        landmark_weights = visible + (1 - visible) * self.occ.landmark_weight_occluded
        landmark_error = ((flame_output['landmarks_mp'] - batch['landmarks_mp']) ** 2).mean(-1)
        losses['landmark_loss_mp'] = weighted_mean(landmark_error, landmark_weights)

        # regularization (w.r.t. the teacher on the clean image, or zero)
        if self.config.train.use_base_model_for_regularization:
            base_output = teacher
        else:
            base_output = {k: torch.zeros_like(encoder_output[k]) for k in ['expression_params', 'shape_params', 'jaw_params']}
        losses['expression_regularization'] = torch.mean((encoder_output['expression_params'] - base_output['expression_params']) ** 2)
        losses['shape_regularization'] = torch.mean((encoder_output['shape_params'] - base_output['shape_params']) ** 2)
        losses['jaw_regularization'] = torch.mean((encoder_output['jaw_params'] - base_output['jaw_params']) ** 2)

        # visible-region consistency with the teacher (expression mesh, neutral pose, teacher shape, in mm)
        if lw.visible_consistency > 0:
            with torch.no_grad():
                teacher_vertices = self.expression_vertices(teacher['shape_params'], teacher)[:, self.face_vertices]
                hidden = self.hidden_vertices(teacher, occlusion_mask).float()
            vertices = self.expression_vertices(teacher['shape_params'], encoder_output)[:, self.face_vertices]
            distance_mm = (((vertices - teacher_vertices) * 1000) ** 2).sum(-1).add(1e-8).sqrt()
            weights = (1 - hidden) + hidden * self.occ.consistency_weight_occluded
            losses['visible_consistency'] = weighted_mean(distance_mm, weights)
            with torch.no_grad():  # for the logs: error on the visible vertices only
                losses['visible_mm'] = weighted_mean(distance_mm, 1 - hidden)
        else:
            losses['visible_consistency'] = 0

        if self.config.arch.enable_fuse_generator:
            masks = batch['mask']
            rendered_mask = 1 - (rendered_img == 0).all(dim=1, keepdim=True).float()
            npoints, _ = masking_utils.mesh_based_mask_uniform_faces(flame_output['transformed_vertices'],
                                                                     flame_faces=self.flame.faces_tensor,
                                                                     face_probabilities=self.face_probabilities,
                                                                     mask_ratio=self.config.train.mask_ratio)
            # the translator gets sparse pixels of the CLEAN image (never pixels of the occluder)
            extra_points = masking_utils.transfer_pixels(img, npoints, npoints)
            masked_img = masking_utils.masking(img, masks, extra_points, self.config.train.mask_dilation_radius, rendered_mask=rendered_mask)
            reconstructed_img = self.smirk_generator(torch.cat([rendered_img, masked_img], dim=1))

            # the clean image is the target; occluded pixels are down-weighted
            pixel_weights = 1 - (1 - self.occ.reconstruction_weight_occluded) * occlusion_mask
            reconstruction_loss = F.l1_loss(reconstructed_img, img, reduction='none')
            loss_img = reconstruction_loss.mean(dim=1, keepdim=True)
            losses['reconstruction_loss'] = weighted_mean(reconstruction_loss, pixel_weights.expand_as(reconstruction_loss))
            losses['perceptual_vgg_loss'] = self.vgg_loss(reconstructed_img, img)

            if lw['emotion_loss'] > 0:
                for param in self.smirk_generator.parameters():
                    param.requires_grad_(False)
                generator_mode = self.smirk_generator.training
                self.smirk_generator.eval()
                reconstructed_img_p = self.smirk_generator(torch.cat([rendered_img, masked_img], dim=1))
                if self.occ.optimize_generator:
                    for param in self.smirk_generator.parameters():
                        param.requires_grad_(True)
                self.smirk_generator.train(generator_mode)
                losses['emotion_loss'] = self.emotion_loss(reconstructed_img_p, img, metric='l2', use_mean=False).mean()
            else:
                losses['emotion_loss'] = 0
        else:
            losses['reconstruction_loss'] = losses['perceptual_vgg_loss'] = losses['emotion_loss'] = 0

        if lw['mica_loss'] > 0:
            losses['mica_loss'] = self.mica.calculate_mica_shape_loss(encoder_output['shape_params'], batch['img_mica'])
        else:
            losses['mica_loss'] = 0

        shape_losses = losses['shape_regularization'] * lw['shape_regularization'] + losses['mica_loss'] * lw['mica_loss']
        expression_losses = losses['expression_regularization'] * lw['expression_regularization'] + \
                            losses['jaw_regularization'] * lw['jaw_regularization']
        landmark_losses = (losses['landmark_loss_fan'] + losses['landmark_loss_mp']) * lw['landmark_loss']
        generator_losses = losses['perceptual_vgg_loss'] * lw['perceptual_vgg_loss'] + \
                           losses['reconstruction_loss'] * lw['reconstruction_loss'] + losses['emotion_loss'] * lw['emotion_loss']

        loss_first_path = (
            (shape_losses if self.config.train.optimize_shape else 0) +
            (expression_losses if self.config.train.optimize_expression else 0) +
            landmark_losses + losses['visible_consistency'] * lw.visible_consistency +
            (generator_losses if self.config.arch.enable_fuse_generator else 0)
        )

        for key, value in losses.items():
            losses[key] = value.item() if isinstance(value, torch.Tensor) else value

        outputs = {'rendered_img': rendered_img, 'vertices': flame_output['vertices'], 'img': img,
                   'landmarks_fan_gt': batch['landmarks_fan'], 'landmarks_fan': flame_output['landmarks_fan'],
                   'landmarks_mp': flame_output['landmarks_mp'], 'landmarks_mp_gt': batch['landmarks_mp']}
        if self.config.arch.enable_fuse_generator:
            outputs.update(loss_img=loss_img, reconstructed_img=reconstructed_img, masked_1st_path=masked_img)
        for key in outputs:
            outputs[key] = outputs[key].detach().cpu()
        outputs['encoder_output'] = encoder_output
        return outputs, losses, loss_first_path, encoder_output

    # ------------------------------------------------------------------ cycle path
    def step2(self, encoder_output, batch, batch_idx, phase='train'):
        if not self.occ.cycle_occlusion:
            return super().step2(encoder_output, batch, batch_idx, phase)

        # run the original cycle path, but occlude the translator output before it is encoded again: the encoder
        # input of SmirkTrainer.step2 is reconstructed_img_2nd_path, produced by self.smirk_generator; wrap the
        # generator so that its output gets the sample's occluder (the clean samples have an empty mask)
        Ke = self.config.train.Ke
        mask = batch['occlusion_mask'].repeat(Ke, 1, 1, 1)
        occluder = batch['img_input'].repeat(Ke, 1, 1, 1)
        generator = self.smirk_generator

        class OccludedOutput(torch.nn.Module):
            def forward(self, x):
                out = generator(x)
                return out * (1 - mask) + occluder * mask

        self.smirk_generator = OccludedOutput()
        try:
            return super().step2(encoder_output, batch, batch_idx, phase)
        finally:
            self.smirk_generator = generator

    # ------------------------------------------------------------------ visualizations
    def create_visualizations(self, batch, outputs):
        visualizations = super().create_visualizations(batch, outputs)
        visualizations['img'] = batch['img_input'].detach().cpu()  # show what the encoder saw
        return visualizations

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, path, epoch, step):
        state = {'model': {k: v for k, v in self.state_dict().items() if k.startswith(('smirk_encoder.', 'smirk_generator.'))},
                 'encoder_optimizer': self.encoder_optimizer.state_dict(),
                 'encoder_scheduler': self.encoder_scheduler.state_dict(),
                 'epoch': epoch, 'step': step}
        if self.config.arch.enable_fuse_generator:
            state['generator_optimizer'] = self.smirk_generator_optimizer.state_dict()
            state['generator_scheduler'] = self.smirk_generator_scheduler.state_dict()
        tmp = path + '.tmp'
        torch.save(state, tmp)
        os.replace(tmp, path)

    def load_checkpoint_model(self, path, device):
        state = torch.load(path, map_location=device)
        self.load_state_dict(state['model'], strict=False)
        return state

    def load_training_state(self, state):
        """After configure_optimizers: restore optimizers and schedulers."""
        self.encoder_optimizer.load_state_dict(state['encoder_optimizer'])
        self.encoder_scheduler.load_state_dict(state['encoder_scheduler'])
        if self.config.arch.enable_fuse_generator and 'generator_optimizer' in state:
            self.smirk_generator_optimizer.load_state_dict(state['generator_optimizer'])
            self.smirk_generator_scheduler.load_state_dict(state['generator_scheduler'])
