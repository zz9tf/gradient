import glob
import cv2
import numpy as np
import scipy.io as sio
import xml.etree.ElementTree as ET


class __AbstractDataset(object):
    """Abstract class for interface of subsequent classes.
    Main idea is to encapsulate how each dataset should parse
    their images and annotations.
    
    """

    def load_img(self, path):
        raise NotImplementedError

    def load_ann(self, path, with_type=False):
        raise NotImplementedError


####
class __Kumar(__AbstractDataset):
    """Defines the Kumar dataset as originally introduced in:

    Kumar, Neeraj, Ruchika Verma, Sanuj Sharma, Surabhi Bhargava, Abhishek Vahadane, 
    and Amit Sethi. "A dataset and a technique for generalized nuclear segmentation for 
    computational pathology." IEEE transactions on medical imaging 36, no. 7 (2017): 1550-1560.

    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        assert not with_type, "Not support"
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
class __CPM17(__AbstractDataset):
    """Defines the CPM 2017 dataset as originally introduced in:

    Vu, Quoc Dang, Simon Graham, Tahsin Kurc, Minh Nguyen Nhat To, Muhammad Shaban, 
    Talha Qaiser, Navid Alemi Koohbanani et al. "Methods for segmentation and classification 
    of digital microscopy tissue images." Frontiers in bioengineering and biotechnology 7 (2019).

    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        assert not with_type, "Not support"
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)["inst_map"]
        ann_inst = ann_inst.astype("int32")
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
class __CoNSeP(__AbstractDataset):
    """Defines the CoNSeP dataset as originally introduced in:

    Graham, Simon, Quoc Dang Vu, Shan E. Ahmed Raza, Ayesha Azam, Yee Wah Tsang, Jin Tae Kwak, 
    and Nasir Rajpoot. "Hover-Net: Simultaneous segmentation and classification of nuclei in 
    multi-tissue histology images." Medical Image Analysis 58 (2019): 101563
    
    """

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False):
        # assumes that ann is HxW
        ann_inst = sio.loadmat(path)["inst_map"]
        if with_type:
            ann_type = sio.loadmat(path)["type_map"]

            # merge classes for CoNSeP (in paper we only utilise 3 nuclei classes and background)
            # If own dataset is used, then the below may need to be modified
            ann_type[(ann_type == 3) | (ann_type == 4)] = 3
            ann_type[(ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 4

            ann = np.dstack([ann_inst, ann_type])
            ann = ann.astype("int32")
        else:
            ann = np.expand_dims(ann_inst, -1)
            ann = ann.astype("int32")

        return ann


class __MoNuSAC(__AbstractDataset):
    """
    MoNuSAC: tif image + xml polygon annotations.
    We output:
      - with_type=False: ann is HxWx1 (inst_map)
      - with_type=True : ann is HxWx2 (inst_map, type_map)
    """

    TYPE2ID = {
        "epithelial": 1,
        "lymphocyte": 2,
        "macrophage": 3,
        "neutrophil": 4,
    }

    def load_img(self, path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    def load_ann(self, path, with_type=False, img_hw=None):
        assert img_hw is not None, "MoNuSAC.load_ann requires img_hw=(H,W)"
        H, W = img_hw

        inst_map = np.zeros((H, W), dtype=np.int32)
        type_map = np.zeros((H, W), dtype=np.int32)

        root = ET.parse(path).getroot()

        inst_id = 1

        # type is stored at Annotation level
        for ann in root.findall(".//Annotation"):
            # read type name
            attr = ann.find(".//Attributes/Attribute")
            type_name = ""
            if attr is not None:
                # your print shows it's in Name
                type_name = (attr.attrib.get("Name", "") or "").strip().lower()

            tid = self.TYPE2ID.get(type_name, 0) if with_type else 0

            # iterate regions under this annotation
            for region in ann.findall(".//Regions/Region"):
                pts = []
                for v in region.findall(".//Vertices/Vertex"):
                    x = int(round(float(v.attrib["X"])))
                    y = int(round(float(v.attrib["Y"])))

                    # clip to bounds to be safe
                    if x < 0: x = 0
                    if x >= W: x = W - 1
                    if y < 0: y = 0
                    if y >= H: y = H - 1

                    pts.append([x, y])

                if len(pts) < 3:
                    continue

                poly = np.array([pts], dtype=np.int32)

                cv2.fillPoly(inst_map, poly, inst_id)
                if with_type and tid != 0:
                    cv2.fillPoly(type_map, poly, tid)

                inst_id += 1

        if with_type:
            ann = np.dstack([inst_map, type_map]).astype("int32")  # HxWx2
        else:
            ann = np.expand_dims(inst_map, -1).astype("int32")     # HxWx1
        return ann

####
def get_dataset(name):
    """Return a pre-defined dataset object associated with `name`."""
    name_dict = {
        "kumar": lambda: __Kumar(),
        "cpm17": lambda: __CPM17(),
        "consep": lambda: __CoNSeP(),
        "monusac": lambda: __MoNuSAC(),
    }
    key = name.lower()
    if key in name_dict:
        return name_dict[key]()
    else:
        assert False, "Unknown dataset `%s`" % name
