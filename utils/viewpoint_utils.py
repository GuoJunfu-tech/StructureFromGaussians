from random import randint


class ViewpointLoader:
    def __init__(self, scene) -> None:
        self._scene = scene
        self._full_viewpoint = self._scene.getTrainCameras().copy()
        self._viewpoint_frames = {}
        self._current_stack = None

    def _load_frame_viewpoint(self):
        for viewpoint_cam in self._full_viewpoint:
            fid = viewpoint_cam.fid.item()
            if fid not in self._viewpoint_frames:
                self._viewpoint_frames[fid] = []
            self._viewpoint_frames[fid].append(viewpoint_cam)

    def get_viewpoint_frame(self, fid: int):
        if not self._viewpoint_frames:
            self._load_frame_viewpoint()

        return self._viewpoint_frames[fid].copy()

    @property
    def current_stack(self):
        return self._current_stack

    def refresh_current_stack(self, fid=None):
        if not fid:
            self._current_stack = self._full_viewpoint
            return

        self._current_stack = self.get_viewpoint_frame(fid)
        self._current_fid = fid

    @property
    def viewpoint_cam(self, load2device: bool = False):
        if not self._current_stack:
            self.refresh_current_stack(self._current_fid)

        cam = self._current_stack.pop(randint(0, len(self._current_stack) - 1))
        return cam if not load2device else cam.load2device()
