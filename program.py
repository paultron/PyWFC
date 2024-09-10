import random
import numpy as np

from model import Model
from helper import Helper

from simple_tiled_model import SimpleTiledModel

import xml.etree.ElementTree as ET

if __name__ == "__main__":
    folder = "output\\"
    
    _rand = random.Random

    xdoc = ET.parse("samples.xml")

    for xelem in Helper.Elements(xdoc.getroot(), ['simpletiled']):
        model: Model
        name = xelem.get("name")
        print(f"< {name}")

        isOverlapping = xelem.get("name") == "overlapping"
        size = int(xelem.get("size", 48 if isOverlapping else 24))
        width = int(xelem.get("width", size))
        height = int(xelem.get("height", size))
        periodic = bool(xelem.get("periodic", False))
        heuristicString = xelem.get("heuristic")
        heuristic = Model.Heuristic.Scanline if heuristicString == "Scanline" else (Model.Heuristic.MRV if heuristicString == "MRV" else Model.Heuristic.Entropy)

        if (isOverlapping):
            # TODO Overlapping Model
            continue
            N = xelem.Get("N", 3);
            periodicInput = xelem.Get("periodicInput", true);
            symmetry = xelem.Get("symmetry", 8);
            ground = xelem.Get("ground", false);
            
            #model = new OverlappingModel(name, N, width, height, periodicInput, periodic, symmetry, ground, heuristic);
        
        else:
        
            subset = xelem.get("subset")
            blackBackground = xelem.get("blackBackground", False)

            model = SimpleTiledModel(name, subset, width, height, periodic, blackBackground, heuristic)
        

        #for (int i = 0; i < xelem.Get("screenshots", 2); i++):
        for i in range(int(xelem.get("screenshots", 2))):
            #for (int k = 0; k < 10; k++):
            for k in range(10):
                print("> ")
                seed = random.randint(0, 1234123)
                success = model.Run(seed, int(xelem.get("limit", -1)))
                if (success):
                    print("DONE")
                    #Console.WriteLine("DONE")
                    model.Save(f"output/{name} {seed}.png")
                    if (model is SimpleTiledModel and xelem.get("textOutput", False)):
                        with open('output.txt', 'rw') as f:
                            f.writelines(model.TextOutput())
                        # System.IO.File.WriteAllText(f"output/{name} {seed}.txt", stmodel.TextOutput())
                    break
                
                else: 
                    print("Contradiction")
                    #Console.WriteLine("CONTRADICTION");
            
        
    

        #Console.WriteLine(f"time = 123");