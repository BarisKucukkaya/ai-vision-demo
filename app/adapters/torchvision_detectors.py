"""faster_rcnn/pytorch and retinanet/pytorch: torchvision detection models behind one base class."""
from __future__ import annotations

from .. import config
from ..schemas import ImageInput, Options, RawOutput
from .base import Adapter, BackendUnavailable

FLOOR = 0.05  # score floor set at construction; the API threshold is applied afterwards


class TorchvisionDetector(Adapter):
    framework = "pytorch"

    def __init__(self) -> None:
        super().__init__()
        self._torch = None
        self._model = None

    def _build_model(self, random_weights: bool):
        raise NotImplementedError

    def _nms_holder(self):
        """The object whose `nms_thresh` attribute is read at call time."""
        raise NotImplementedError

    def _select_device(self, torch) -> str:
        choice = config.device_choice()
        if choice == "cuda" and not torch.cuda.is_available():
            raise BackendUnavailable("DEMO_DEVICE=cuda but CUDA is not available.")
        if choice == "cuda" or (choice == "auto" and torch.cuda.is_available()):
            return "cuda"
        return "cpu"

    def _load(self) -> None:
        try:
            import torch
            import torchvision  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailable(
                "PyTorch/torchvision are not installed. Install with: pip install -r requirements-pytorch.txt"
            ) from exc
        random = config.random_weights()
        model = self._build_model(random)
        self.device = self._select_device(torch)
        model.to(self.device).eval()
        self._torch, self._model = torch, model
        self.parameters = sum(p.numel() for p in model.parameters())
        self.model_size_mb = round(self.parameters * 4 / 1e6, 1)  # float32
        if random:
            self.mock = True
            self.model_name += " - RANDOM weights - test only"
        with torch.inference_mode():  # warm-up
            model([torch.zeros(3, 320, 320, device=self.device)])

    # excerpt: retinanet_pytorch (shared predict path)
    def _predict(self, image_input: ImageInput, options: Options) -> RawOutput:
        torch = self._torch
        from torchvision.transforms.functional import pil_to_tensor

        tensor = pil_to_tensor(image_input.image).float() / 255   # CHW, float 0..1, RGB
        holder = self._nms_holder()
        previous = holder.nms_thresh
        holder.nms_thresh = options.nms_iou                       # applied inside the inference lock
        try:
            with torch.inference_mode():
                out = self._model([tensor.to(self.device)])[0]    # a list of tensors in, original pixels out
        finally:
            holder.nms_thresh = previous
        return RawOutput(
            out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["labels"].cpu().numpy(),
            {"nms_iou": options.nms_iou, "nms_applied": True},
        )
    # endexcerpt


class RetinaNetPyTorch(TorchvisionDetector):
    algorithm = "retinanet"
    model_name = "RetinaNet ResNet50-FPN COCO (torchvision: COCO_V1)"

    def _build_model(self, random_weights: bool):
        from torchvision.models.detection import RetinaNet_ResNet50_FPN_Weights, retinanet_resnet50_fpn

        if random_weights:
            return retinanet_resnet50_fpn(weights=None, weights_backbone=None, score_thresh=FLOOR)
        # excerpt: retinanet_pytorch
        return retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.COCO_V1, score_thresh=FLOOR)
        # endexcerpt

    def _nms_holder(self):
        return self._model


class FasterRCNNPyTorch(TorchvisionDetector):
    algorithm = "faster_rcnn"
    model_name = "Faster R-CNN ResNet50-FPN COCO (torchvision: COCO_V1)"

    def _build_model(self, random_weights: bool):
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

        if random_weights:
            return fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, box_score_thresh=FLOOR)
        # excerpt: fasterrcnn_pytorch
        return fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1, box_score_thresh=FLOOR)
        # endexcerpt

    def _nms_holder(self):
        return self._model.roi_heads
