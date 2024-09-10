import xml.etree.ElementTree as ET
import xml.etree

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
    
    def Elements(xelement, names):
        return [child for child in xelement if child.tag in names]
    
class BitmapHelper:
    def SaveBitmap(data, width, height, filename):
        pass