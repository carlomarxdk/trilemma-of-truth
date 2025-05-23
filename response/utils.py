from matplotlib.colors import LinearSegmentedColormap, rgb2hex
from IPython.display import display, HTML
import numpy as np
import logging
log = logging.getLogger(__name__)


class TextVizualizer:
    def __init__(self, cmap='none', transform='none'):
        assert transform in ['none', 'min_max']
        self.transform = transform
        if cmap == 'none':
            cmap = LinearSegmentedColormap.from_list(
                "RdGn", ["red", 'white', "green"])
        self.cmap = cmap

    def colorize(self, tokens, scores):
        # words is a list of words
        # color_array is an array of numbers between 0 and 1 of length equal to words
        tokens = [token.lstrip('Ġ') for token in tokens]
        if self.transform == 'min_max':
            scores = self._min_max_transform(scores)
        template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
        colored_string = ''
        for word, color in zip(tokens, scores):
            color = rgb2hex(self.cmap(color)[:3])  # type: ignore
            colored_string += template.format(color, '&nbsp' + word + '&nbsp')
        return colored_string

    def display(self, tokens, scores):
        colored_string = self.colorize(tokens, scores)
        display(HTML(colored_string))

    def _min_max_transform(self, x):
        _min = np.min(x)
        _max = np.max(x)
        return (x - _min) / (_max - _min)
