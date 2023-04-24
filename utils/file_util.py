import json
from decimal import Decimal
import dataclasses, json
from typing import Tuple
import pickle
import math
import hashlib


class FileUtil(object):
    """
    文件工具类
    """
    @classmethod
    def read_raw_text(cls, raw_text_path):
        """
        读取原始文本数据，每行均为纯文本
        """
        all_raw_text_list = []
        with open(raw_text_path, "r", encoding="utf-8") as raw_text_file:
            for item in raw_text_file:
                item = item.strip()
                all_raw_text_list.append(item)
        return all_raw_text_list

    @classmethod
    def write_raw_text(cls, texts, file_path):
        """
        写入文本数据，每行均为纯文本
        """
        with open(file_path, "w", encoding="utf-8") as f:
            for item in texts:
                f.write(f'{item}\n')

    @classmethod
    def write_tsv(cls, data_list, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            for mention_type_score in data_list:
                f.write("\t".join(mention_type_score) + "\n")

    @classmethod
    def read_tsv(cls, file_path):
        results = []
        with open(file_path, "r", encoding="utf-8") as f:
            for item in f:
                item = item.strip()
                results.append(item.split("\t"))
        return results

    @classmethod
    def write_jsonl(cls, data_list, file_path, ensure_ascii=False):
        with open(file_path, "w", encoding="utf-8") as f:
            for text_obj in data_list:
                f.write(json.dumps(text_obj, ensure_ascii=ensure_ascii) + "\n")

    @classmethod
    def write_jsonl_general_cls(cls, data_list, file_path, ensure_ascii=False):
        """ the data contains class or other object which is not supported by json dump by default. """
        with open(file_path, "w", encoding="utf-8") as text_format_file:
            for text_obj in data_list:
                text_format_file.write(json.dumps(text_obj, ensure_ascii=ensure_ascii, cls=JSONEncoder) + "\n")                

    @classmethod
    def read_jsonl(cls, file_path):
        results = []
        with open(file_path, "r", encoding="utf-8") as f:
            for item in f:
                item = item.strip()
                text_obj = json.loads(item)
                results.append(text_obj)
        return results

    @classmethod
    def read_json(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    
    @classmethod
    def write_json(cls, data, file_path, ensure_ascii=False):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=4)

    @classmethod
    def write_json_general_cls(cls, data, file_path, ensure_ascii=False):
        """ the data contains class or other object which is not supported by json dump by default. """
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=ensure_ascii, indent=4, cls=JSONEncoder)

    @classmethod
    def load_pickle(cls, filename):
        with open(filename, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        return data

    @classmethod
    def save_as_pickle(cls, filename, data):
        with open(filename, 'wb') as output:
            pickle.dump(data, output)

    
def dataclass_from_dict(raw_class_obj, dict_data:dict):
    """ 
    Args: 
        raw_class_obj is the raw class_obj e.g. SentenceNerOutput.
        dict_data is the data loaded by json.
    """
    try:
        fieldtypes = raw_class_obj.__annotations__
        return raw_class_obj(**{f: dataclass_from_dict(fieldtypes[f], dict_data[f]) for f in dict_data})
    except AttributeError:
        # Must to support List[dataclass]
        if isinstance(dict_data, (tuple, list)):
            return [dataclass_from_dict(raw_class_obj.__args__[0], f) for f in dict_data]
        return dict_data


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, (Tuple, set)):
            return list(o)
        if isinstance(o, bytes):
            return o.decode()
        try:
            import numpy
            if isinstance(o, numpy.ndarray):
                return o.tolist()            
        except:
            pass

        return super().default(o)


def get_partial_files(input_files, part_seq=-1, total_parts_num=-1, start_index=-1) ->List:
    """ part_seq starts from 1.
        If set start_index > 0, directly get partial input_files[start_index:]
    """
    if start_index > 0:
        partial_files = input_files[start_index:]
    elif part_seq > 0 and total_parts_num > 0:
        input_files_num = len(input_files)
        num_per_part = math.ceil(input_files_num / total_parts_num)
        start_i = (part_seq - 1) * num_per_part
        end_i = part_seq * num_per_part
        partial_files = input_files[start_i: end_i]
    return partial_files


def calculate_file_md5(filename):
    """ For small file """
    with open(filename,"rb") as f:
        bytes = f.read()
        readable_hash = hashlib.md5(bytes).hexdigest()
        return readable_hash


def calculate_file_md5_large_file(filename):
    """ For large file to read by chunks in iteration. """
    md5_hash = hashlib.md5()
    with open(filename,"rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
        return md5_hash.hexdigest()
        