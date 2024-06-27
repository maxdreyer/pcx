import copy

import torch
from torchvision.models.efficientnet import MBConv
from torchvision.models.resnet import Bottleneck as ResNetBottleneck, BasicBlock as ResNetBasicBlock
from torchvision.models.vision_transformer import EncoderBlock, Encoder
from torchvision.ops.misc import SqueezeExcitation
from zennit import canonizers as canonizers
from zennit import layer as zlayer
from zennit.canonizers import CompositeCanonizer, SequentialMergeBatchNorm, AttributeCanonizer
from zennit.layer import Sum
from timm.models.resnet import Bottleneck as ResNetBottleneckTimm
from timm.models._efficientnet_blocks import ConvBnAct as ConvBnActTimm, EdgeResidual as EdgeResidualTimm
from timm.models._efficientnet_blocks import InvertedResidual as InvertedResidualTimm, \
    SqueezeExcite as SqueezeExcitationTimm


class SignalOnlyGate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2):
        return x1 * x2

    @staticmethod
    def backward(ctx, grad_output):
        return torch.zeros_like(grad_output), grad_output


class SECanonizer(canonizers.AttributeCanonizer):
    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, SqueezeExcitation):
            attributes = {
                'forward': cls.forward.__get__(module),
                'fn_gate': SignalOnlyGate()
            }
            return attributes
        if isinstance(module, SqueezeExcitationTimm):
            attributes = {
                'forward': cls.forward_timm.__get__(module),
                'fn_gate': SignalOnlyGate(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, input):
        scale = self._scale(input)
        return self.fn_gate.apply(scale, input)

    @staticmethod
    def forward_timm(self, x: torch.Tensor) -> torch.Tensor:
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        # return x * self.gate(x_se)
        return self.fn_gate.apply(self.gate(x_se), x)


class MBConvCanonizer(canonizers.AttributeCanonizer):
    '''Canonizer specifically for MBConvBlock of Mobile Net v2 type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, MBConv):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': zlayer.Sum()
            }
            return attributes
        return None

    @staticmethod
    def forward(self, input):
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)

            # result += input
            result = torch.stack([input, result], dim=-1)
            result = self.canonizer_sum(result)
        return result


class EfficientNetBNCanonizer(canonizers.CompositeCanonizer):
    def __init__(self):
        super().__init__((
            SECanonizer(),
            MBConvCanonizer(),
            canonizers.SequentialMergeBatchNorm()
        ))


class NewAttention(torch.nn.MultiheadAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inp):
        result, _ = super().forward(inp, inp, inp, need_weights=False)
        return result


class EncoderBlockCanonizer(canonizers.AttributeCanonizer):
    '''Canonizer specifically for MBConvBlock of Mobile Net v2 type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, EncoderBlock):

            new_attention = NewAttention(module.self_attention.embed_dim,
                                         module.self_attention.num_heads,
                                         module.self_attention.dropout,
                                         batch_first=True)
            for name, param in module.self_attention.named_parameters():
                if "." in name:
                    getattr(new_attention, name.split(".")[0]).register_parameter(name.split(".")[1], param)
                else:
                    new_attention.register_parameter(name, param)
            attributes = {
                'forward': cls.forward.__get__(module),
                'new_attention': new_attention,
                'sum': zlayer.Sum(),
            }
            return attributes
        return None

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x = self.new_attention(x)
        x = self.dropout(x)
        x = self.sum(torch.stack([x, input], dim=-1))

        y = self.ln_2(x)
        y = self.mlp(y)
        return self.sum(torch.stack([x, y], dim=-1))


class EncoderCanonizer(canonizers.AttributeCanonizer):
    '''Canonizer specifically for MBConvBlock of Mobile Net v2 type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        if isinstance(module, Encoder):
            attributes = {
                'forward': cls.forward.__get__(module),
                'sum': zlayer.Sum(),
            }
            return attributes
        return None

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = self.sum(torch.stack([input, self.pos_embedding.expand_as(input)], dim=-1))
        return self.ln(self.layers(self.dropout(input)))


class VITCanonizer(canonizers.CompositeCanonizer):
    def __init__(self):
        super().__init__((
            canonizers.SequentialMergeBatchNorm(),
            EncoderCanonizer(),
            EncoderBlockCanonizer(),
        ))


class ResNetBottleneckCanonizer(AttributeCanonizer):
    '''Canonizer specifically for Bottlenecks of torchvision.models.resnet* type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a Bottleneck layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of Bottleneck, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, ResNetBottleneck):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        if isinstance(module, ResNetBottleneckTimm):
            attributes = {
                'forward': cls.forward_timm.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified Bottleneck forward for ResNet.'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        out = self.relu(out)
        return out

    @staticmethod
    def forward_timm(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        # x += shortcut
        x = torch.stack([shortcut, x], dim=-1)
        x = self.canonizer_sum(x)
        x = self.act3(x)

        return x


class ResNetBasicBlockCanonizer(AttributeCanonizer):
    '''Canonizer specifically for BasicBlocks of torchvision.models.resnet* type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a BasicBlock layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of BasicBlock, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, ResNetBasicBlock):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified BasicBlock forward for ResNet.'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        if hasattr(self, 'last_conv'):
            out = self.last_conv(out)
            out = out + 0

        out = self.relu(out)

        return out


class ResNetCanonizer(CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''

    def __init__(self):
        super().__init__((
            SequentialMergeBatchNorm(),
            ResNetBottleneckCanonizer(),
            ResNetBasicBlockCanonizer(),
        ))


class EfficientNetConvBnAct(AttributeCanonizer):
    '''Canonizer specifically for ConvBnAct of timm.models._efficientnet_blocks type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):

        if isinstance(module, ConvBnActTimm):
            bn = torch.nn.BatchNorm2d(module.bn1.num_features, module.bn1.eps, module.bn1.momentum, module.bn1.affine,
                                      module.bn1.track_running_stats)
            setattr(bn, 'running_mean', module.bn1.running_mean)
            setattr(bn, 'running_var', module.bn1.running_var)
            setattr(bn, 'weight', module.bn1.weight)
            setattr(bn, 'bias', module.bn1.bias)
            bn.to(module.bn1.weight.device)
            bn.eval() if not module.training else bn.train()

            attributes = {
                'forward': cls.forward_timm.__get__(module),
                'canonizer_sum': Sum(),
                'act': module.bn1.act,
                'conv_': copy.deepcopy(module.conv),
                'bn1_': bn,
            }
            return attributes
        return None

    @staticmethod
    def forward_timm(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv_(x)
        x = self.bn1_(x)
        x = self.act(x)
        if self.has_skip:
            x = torch.stack([self.drop_path(x), shortcut], dim=-1)
            x = self.canonizer_sum(x)
        return x


class EfficientNetInvertedResidual(AttributeCanonizer):
    '''Canonizer specifically for InvertedResidual of timm.models._efficientnet_blocks type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):

        if isinstance(module, InvertedResidualTimm):
            bns = []
            for bn in ['bn1', 'bn2', 'bn3']:
                bn_module = getattr(module, bn)
                bn_new = torch.nn.BatchNorm2d(bn_module.num_features, bn_module.eps, bn_module.momentum,
                                              bn_module.affine, bn_module.track_running_stats)
                setattr(bn_new, 'running_mean', bn_module.running_mean)
                setattr(bn_new, 'running_var', bn_module.running_var)
                setattr(bn_new, 'weight', bn_module.weight)
                setattr(bn_new, 'bias', bn_module.bias)
                bn_new.to(bn_module.weight.device)
                bn_new.eval() if not module.training else bn_new.train()
                bns.append(bn_new)

            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
                'bn1_act': module.bn1.act,
                'bn2_act': module.bn2.act,
                'bn3_act': module.bn3.act,
                'conv_pw_': copy.deepcopy(module.conv_pw),
                'bn1_': bns[0],
                'conv_dw_': copy.deepcopy(module.conv_dw),
                'bn2_': bns[1],
                'conv_pwl_': copy.deepcopy(module.conv_pwl),
                'bn3_': bns[2],
            }

            return attributes
        return None

    @staticmethod
    def forward(self, x):
        shortcut = x
        x = self.conv_pw_(x)
        x = self.bn1_(x)
        x = self.bn1_act(x)
        x = self.conv_dw_(x)
        x = self.bn2_(x)
        x = self.bn2_act(x)
        x = self.se(x)
        x = self.conv_pwl_(x)
        x = self.bn3_(x)
        x = self.bn3_act(x)
        if self.has_skip:
            # x = self.drop_path(x) + shortcut
            x = torch.stack([self.drop_path(x), shortcut], dim=-1)
            x = self.canonizer_sum(x)
        return x


class EfficientNetEdgeResidual(AttributeCanonizer):
    '''Canonizer specifically for EdgeResidual of timm.models._efficientnet_blocks type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):

        if isinstance(module, EdgeResidualTimm):
            bns = []
            for bn in ['bn1', 'bn2']:
                bn_module = getattr(module, bn)
                bn_new = torch.nn.BatchNorm2d(bn_module.num_features, bn_module.eps, bn_module.momentum,
                                              bn_module.affine, bn_module.track_running_stats)
                setattr(bn_new, 'running_mean', bn_module.running_mean)
                setattr(bn_new, 'running_var', bn_module.running_var)
                setattr(bn_new, 'weight', bn_module.weight)
                setattr(bn_new, 'bias', bn_module.bias)
                bn_new.to(bn_module.weight.device)
                bn_new.eval() if not module.training else bn_new.train()

                bns.append(bn_new)
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
                'act1': module.bn1.act,
                'act2': module.bn2.act,
                'conv_exp_': copy.deepcopy(module.conv_exp),
                'bn1_': bns[0],
                'conv_pwl_': copy.deepcopy(module.conv_pwl),
                'bn2_': bns[1],
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        shortcut = x
        x = self.conv_exp_(x)
        x = self.bn1_(x)
        x = self.act1(x)
        x = self.se(x)
        x = self.conv_pwl_(x)
        x = self.bn2_(x)
        x = self.act2(x)
        if self.has_skip:
            # x = self.drop_path(x) + shortcut
            x = torch.stack([self.drop_path(x), shortcut], dim=-1)
            x = self.canonizer_sum(x)

        return x


class EfficientNetV2Canonizer(CompositeCanonizer):
    '''Canonizer for timm.models.efficientnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the building block modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''

    def __init__(self):
        super().__init__((
            EfficientNetInvertedResidual(),
            EfficientNetEdgeResidual(),
            EfficientNetConvBnAct(),
            SequentialMergeBatchNorm(),
            SECanonizer(),
        ))