import argparse
import xmltodict
import glob, os
from lxml import etree


def get_time_value(time_order, time_slot):
    result = [x['@TIME_VALUE'] for x in time_order['TIME_SLOT'] if x['@TIME_SLOT_ID'] == time_slot]
    if len(result) > 0:
        return result[0]
    return None


def convert_elan_to_anvil(input_path, output_path):
    with open(input_path) as fd:
        doc = xmltodict.parse(fd.read())
    time_order = doc['ANNOTATION_DOCUMENT']['TIME_ORDER']

    head_nodding = [x for x in doc['ANNOTATION_DOCUMENT']['TIER'] if x['@TIER_ID'] == "Head Nodding"][0]
    if not isinstance(head_nodding['ANNOTATION'], list):
        head_nod_start_end = [(head_nodding['ANNOTATION']['ALIGNABLE_ANNOTATION']['@TIME_SLOT_REF1'], head_nodding['ANNOTATION']['ALIGNABLE_ANNOTATION']['@TIME_SLOT_REF2'])]
    else:
        head_nod_start_end = [(x['ALIGNABLE_ANNOTATION']['@TIME_SLOT_REF1'], x['ALIGNABLE_ANNOTATION']['@TIME_SLOT_REF2']) for x in head_nodding['ANNOTATION']]
    head_nod_start_end = [(get_time_value(time_order, x[0]), get_time_value(time_order, x[1])) for x in head_nod_start_end]
    head_nod_start_end = [(float(x[0]) / 1000, float(x[1]) / 1000) for x in head_nod_start_end]

    root_el = etree.Element("annotation")

    head_el = etree.SubElement(root_el, "head")
    coder_info_el = etree.SubElement(head_el, "info")
    coder_info_el.set("key", "coder")
    coder_info_el.set("type", "String")
    coder_info_el.text = "auto"

    body_el = etree.SubElement(root_el, "body")
    track_el = etree.SubElement(body_el, "track")
    track_el.set("name", "nod")
    track_el.set("type", "primary")

    for i, (start, end) in enumerate(head_nod_start_end):
        ann_el = etree.SubElement(track_el, "el")
        ann_el.set("index", str(i))
        ann_el.set("start", str(start))
        ann_el.set("end", str(end))

    with open(output_path, 'w') as outfile:
        xml_string = etree.tostring(root_el, encoding="utf-8", xml_declaration=True, pretty_print=True)
        outfile.write(xml_string.decode('utf-8'))


def convert_all_elan_to_anvil(dir_path):
    for file in os.listdir(dir_path):
        if file.endswith(".eaf"):
            convert_elan_to_anvil(os.path.join(dir_path, file), os.path.join(dir_path, file.replace(".eaf", ".anvil")))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--input", required=False, default="./")
    args = arg_parser.parse_args()

    convert_all_elan_to_anvil(args.input)