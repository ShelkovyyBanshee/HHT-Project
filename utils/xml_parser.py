import xml.etree.ElementTree as ET


CLASS_MAP = {
    "oil/water": 0,
    "oil/coast": 1,
    "no_oil/water": 2,
    "no_oil/coast": 3
}


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")

        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        boxes.append([xmin, ymin, xmax, ymax])

    path = root.find("path").text
    folder = "/".join(path.split("/")[:2])

    label = CLASS_MAP.get(folder, -1)

    return boxes, label