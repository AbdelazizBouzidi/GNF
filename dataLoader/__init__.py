from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .your_own_data import YourOwnDataset
from .image import ImageDataset
from .colmap import ColmapDataset
from .sdf import SDFDataset
from .blender_set import BlenderDatasetSet
from .google_objs import GoogleObjsDataset
from .dtu_objs import DTUDataset


dataset_dict = {'blender': BlenderDataset,
                'blender_set': BlenderDatasetSet,
               'llff':LLFFDataset,
               'tankstemple':TanksTempleDataset,
               'nsvf':NSVF,
                'own_data':YourOwnDataset,
                'image':ImageDataset,
                'sdf':SDFDataset,
                'colmap':ColmapDataset,
                'google_objs':GoogleObjsDataset,
                'dtu':DTUDataset}