from typing import Optional
import xml.etree.ElementTree as ET
from model import Model
from helper import BitmapHelper
import numpy as np

class SimpleTiledModel(Model):
    tiles: list[list[int]]
    tile_names: list[str]

    tile_size : int
    black_background: bool

    def __init__(self, name: str, subset_name: str | None, width: int, heigth: int, periodic: bool, black_bg: bool, heuristic):
        super().__init__(width, heigth, 1, periodic, heuristic)
        self.black_background = black_bg
        xroot = ET.parse(f"tilesets/{name}.xml").getroot()
        unique = xroot.get("unique", False)

        self.subset = None

        if subset_name != None:
            xsubset: ET.Element = next([e for e in xroot.find("subsets").findall("subset") if e.get("name") == subset_name], None)
            if xsubset == None: 
                print(f"ERROR: subset {subset_name} is not found")
            else: 
                self.subset = xsubset.findall("tile")

        def tile(f, size):
            result = np.empty(size * size)
            for y in range(size):
                for x in range(size):
                    result[x + y * size] = f(x, y)
            return result
        
        rotate = lambda array, size : tile(lambda x, y : array[size - 1 - y + x * size], size)
        reflect = lambda array, size : tile(lambda x, y : array[size - 1 - x + y * size], size)

        self.tiles: list[list[int]] = [[]]
        self.tile_names: list[str] = []

        self.weight_list : list[float] = []

        self.action: list[list[int]] = [[]]
        self.first_occurance: dict[str, int] = {}

        for xtile in xroot.find("tiles").findall("tile"):
            tile_name:str = xtile.get("name")
            if (self.subset != None and tile_name not in self.subset): continue

            sym = xtile.get("symmetry", "X")
            
            match sym:
                case 'L':
                    cardinality = 4
                    a = lambda i : (i + 1) % 4
                    b = lambda i : i + 1 if i % 2 == 0 else i - 1
                case 'T':
                    cardinality = 4
                    a = lambda i : (i + 1) % 4
                    b = lambda i : i if i % 2 == 0 else 4 - i
                case 'I':
                    cardinality = 2
                    a = lambda i : 1 - i
                    b = lambda i : i
                case '\\':
                    cardinality = 2
                    a = lambda i : 1 - i
                    b = lambda i : 1 - i
                case 'F':
                    cardinality = 8
                    a = lambda i : (i + 1) % 4 if i < 4 else 4 + (i - 1) % 4
                    b = lambda i : i + 4 if i < 4 else i - 4
                case _:
                    cardinality = 1
                    a = lambda i : i
                    b = lambda i : i

            self._T  = len(self.action)
            self.first_occurance[tile_name] = self._T

            _map = np.empty([cardinality,8])
            for t in range(cardinality):
                # _map[t] = np.empty(8)

                _map[t][0] = t
                _map[t][1] = a(t)
                _map[t][2] = a(a(t))
                _map[t][3] = a(a(a(t)))
                _map[t][4] = b(t)
                _map[t][5] = b(a(t))
                _map[t][6] = b(a(a(t)))
                _map[t][7] = b(a(a(a(t))))

                for s in range(8):
                    _map[t][s] += self._T

                self.action.append(_map[t])

            if (unique):
                for t in range(cardinality):
                    # TODO load bitmap, bitmaphelper
                    bitmap, tile_size, tile_size = [1], 8, 12
                    self.tiles.append(bitmap)
                    self.tile_names.append(f"{tile_name} {t}")
            else: 
                # TODO load bitmap, bitmaphelper
                bitmap, tile_size, tile_size = [1], 8, 12
                self.tiles.append(bitmap)
                self.tile_names.append(f"{tile_name} 0")

                for t in range(cardinality):
                    if (t <= 3): self.tiles.append(rotate(self.tiles[self._T + t - 1], tile_size))
                    if (t >= 4): self.tiles.append(reflect(self.tiles[self._T + t - 4], tile_size))
                    self.tile_names.append(f"{tile_name} {t}")

            for t in range(cardinality):
                self.weight_list.append(xtile.get("weight", 1.0))

        self._T = len(self.action)
        self._weights = self.weight_list.copy()

        self._propagator = np.empty([4, self._T])
        dense_propagator = np.array([4, self._T, self._T], dtype=bool)

        for xneighbor in xroot.find("neighbors").findall("neighbor"):
            left = xneighbor.get("left").split()
            right = xneighbor.get("right").split()

            if ((self.subset is not None) and ((left[0] not in self.subset) or (right[0] not in self.subset))):
                continue

            L = int(self.action[self.first_occurance[left[0]]][0 if len(left) == 1 else int(left[1])])
            D = self.action[L][1]
            R = int(self.action[self.first_occurance[right[0]]][0 if len(right) == 1 else int(right[1])])
            U = self.action[R][1]

            dense_propagator[0][R][L] = True
            dense_propagator[0][self.action[R][6]][self.action[L][6]] = True
            dense_propagator[0][self.action[L][4]][self.action[R][4]] = True
            dense_propagator[0][self.action[L][2]][self.action[R][2]] = True

            dense_propagator[1][U][D] = True
            dense_propagator[1][self.action[D][6]][self.action[U][6]] = True
            dense_propagator[1][self.action[U][4]][self.action[D][4]] = True
            dense_propagator[1][self.action[D][2]][self.action[U][2]] = True

        for t2 in range(self._T):
            for t1 in range(self._T):
                dense_propagator[2][t2][t1] = dense_propagator[0][t1][t2]
                dense_propagator[3][t2][t1] = dense_propagator[1][t1][t2]

        sparse_propagator = np.empty([4, self._T])

        for d in range(4):
            for t1 in range(self._T):
                sp = sparse_propagator[d][t1]
                tp = dense_propagator[d][t1]
                
                for t2 in range(self._T):
                    if (tp[t2]):
                        sp.append(t2)

                ST = len(sp)
                if (ST == 0):
                    print(f"ERROR: tile {self.tile_names[t1]} has no neighbors in direction {d}")

                self._propagator[d][t1] = np.empty(ST)

                for st in range(ST):
                    self._propagator[d][t1][st] = sp[st]

        
    def Save(self, filename:str):
        bitmap_data = np.empty(self._MX * self._MY * self.tile_size * self.tile_size)

        if self._observed[0] >= 0:
            for x in range(self._MX):
                for y in range(self._MY):
                    tile = self.tiles[self._observed[x + y * self._MX]]
                    for dy in range(self.tile_size):
                        for dx in range(self.tile_size):
                            bitmap_data[x * self.tile_size + dx + (y * self.tile_size + dy) * self._MX * self.tile_size] = tile[dx + dy * self.tile_size]
        else:
            for i in range(len(self._wave)):
                x = i % self._MX
                y = i / self._MX
                if (self.black_background and self._sumsOfOnes[i] == self._T):
                    for yt in range(self.tile_size):
                        for xt in range(self.tile_size):
                            bitmap_data[x * self.tile_size + xt + (y * self.tile_size + yt) * self._MX * self.tile_size] = 255 << 24
                else:
                    w = self._wave[i]
                    normalization = 1.0 / self._sumsOfWeights[i]
                    for yt in range(self.tile_size):
                        for xt in range(self.tile_size):
                            idi = x * self.tile_size + xt + (y * self.tile_size + yt) * self._MX * self.tile_size
                            r = 0.0
                            g = 0.0
                            b = 0.0
                            for t in range(self._T):
                                argb = self.tiles[t][xt + yt * self.tile_size]
                                r += ((argb & 0xff0000) >> 16) * self._weights[t] * normalization
                                g += ((argb & 0xff00) >> 8) * self._weights[t] * normalization
                                b +=  (argb & 0xff) * self._weights[t] * normalization
                            bitmap_data[idi] = [r, g, b, 255] #int(0xff000000) | int(r) << 16 | int(g) << 8 | int(b)

        BitmapHelper.SaveBitmap(bitmap_data, self._MX * self.tile_size, self._MY * self.tile_size, filename)

    def TextOutput(self):
        result = []
        for y in range(self._MY):
            for x in range(self._MX):
                result.append(f"{self.tile_namesnames[self._observed[x + y * self._MX]]}, ")
        return result