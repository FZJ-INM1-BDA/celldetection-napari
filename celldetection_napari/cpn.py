import torch
import celldetection as cd
import cv2


class CpnInterface:
    def __init__(self, model, device=None):
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = cd.models.LitCpn(model).to(device)
        self.model.eval()
        self.tile_size = 512
        self.overlap = 128

    def __call__(
            self,
            img,
            div=255,
            return_labels=True,
            return_viewable_contours=True,
    ):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.ndim == 3 and img.shape[-1] > 3:
            img = img[..., :3]
        img = img / div
        x = cd.data.to_tensor(img, transpose=True, dtype=torch.float32)[None]
        with torch.no_grad():
            out = cd.asnumpy(self.model(x, crop_size=self.tile_size,
                                        stride=max(64, self.tile_size - self.overlap)))

        contours, = out['contours']
        boxes, = out['boxes']
        scores, = out['scores']

        labels = viewable_contours = None

        if return_labels or return_viewable_contours:
            labels = cd.data.contours2labels(contours, img.shape[:2])
        if return_viewable_contours:  # TODO: Napari is buggy when it comes to displaying uncompressed contours
            # Note that fragmented objects would be ignored
            viewable_contours = cd.data.labels2contours(labels, method=cv2.CHAIN_APPROX_SIMPLE,
                                                        raise_fragmented=False)

        return dict(
            contours=contours,
            viewable_contours=viewable_contours,
            labels=labels,
            boxes=boxes,
            scores=scores
        )
