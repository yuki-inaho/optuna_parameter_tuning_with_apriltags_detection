class Intrinsic:
    def __init__(self):
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None

    def set(self, parameter_dict):
        self._set_intrinsic_parameter(*[parameter_dict["camera"][elem] for elem in ["fx", "fy", "cx", "cy"]])

    def _set_intrinsic_parameter(self, fx, fy, cx, cy):
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy

    @property
    def parameters(self):
        return self._fx, self._fy, self._cx, self._cy

    @property
    def center(self):
        return self.cx, self.cy

    @property
    def focal(self):
        return self.fx, self.fy