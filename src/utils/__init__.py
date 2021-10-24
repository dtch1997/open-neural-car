import numpy as np


def rotate_by_angle(vec, th):
    """Rotate a 2D vector by angle
    :param vec: np.array of shape (2,)
    :param th: angle in radians
    """
    M = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return M @ vec


class AttrDict(dict):
    """ Dictionary that also lets you get the entries as properties """

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        setattr(self, key, value)

    @staticmethod
    def from_dict(dict):
        attrdict = AttrDict()
        for key, value in dict.items():
            attrdict[key] = value  # Calls __setitem_
        return attrdict
