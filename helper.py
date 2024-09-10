import xml.etree.ElementTree as ET
import xml.etree
from PIL import Image
import numpy as np

class Helper:
    def RandomHelper(weights, r):
        sum = 0
        for i in range(len(weights)):
            sum += weights[i]
        threshold = r * sum

        partial_sum = 0
        for i in range(len(weights)):
            partial_sum += weights[i]
            if (partial_sum >= threshold):
                return i
        return 0
    
    def Elements(xelement, names: list[str]):
        return [child for child in xelement if child.tag in names]
    
class BitmapHelper:
    def SaveBitmap(data, width, height, filename):
        # _img: Image = np.empty([width, height])
        _rgb = [ ( (int(argb) & 0xff0000) >> 16 , (int(argb) & 0xff00) >> 8 , int(argb) & 0xff ) for argb in data ]
        _arr = np.array(_rgb).astype(np.uint8).reshape((width, height, -1))
        _img = Image.fromarray(_arr, mode='RGB')
        _img.save(filename)

    def LoadBitmap(filename:str):
        _img = Image.open(filename)
        _data = _img.getdata()
        _img2 = []
        for i in range(len(_data)):
            # print(_data[i])
            r = _data[i][0]
            g = _data[i][1]
            b = _data[i][2]
            _img2.append( int(0xff000000) | r << 16 | g << 8 | b )
        return np.asarray(_img2), _img.width, _img.height 