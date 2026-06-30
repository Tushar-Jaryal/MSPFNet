from __future__ import annotations

from mspf_net.models.mspf_net import MSPFNet, flatten_mspf_kwargs

from .convnext1d import ConvNeXt1D
from .inception_time import InceptionTime
from .informer import Informer
from .lstnet import LSTNet
from .mixnet import MixNet
from .resnet1d import ResNet1D
from .se_cnn1d import SECNN1D
from .timesnet import TimesNet
from .transformer1d import Transformer1D
from .wdcnn import WDCNN


def create_baseline(model_name: str, in_channels: int, num_classes: int, model_cfg: dict):
    name = model_name.lower()
    cfg = dict(model_cfg)
    cfg.pop("name", None)
    cfg["in_channels"] = in_channels
    cfg["num_classes"] = num_classes

    registry = {
        "timesnet": TimesNet,
        "informer": Informer,
        "mixnet": MixNet,
        "wdcnn": WDCNN,
        "resnet1d": ResNet1D,
        "inception_time": InceptionTime,
        "convnext1d": ConvNeXt1D,
        "lstnet": LSTNet,
        "transformer1d": Transformer1D,
        "se_cnn1d": SECNN1D,
        "mspf_net": MSPFNet,
    }
    if name not in registry:
        raise ValueError(f"Unknown baseline model: {model_name}")
    if name == "mspf_net":
        cfg = flatten_mspf_kwargs(cfg)
    return registry[name](**cfg)
