import pickle
import struct
import os
import argparse
from typing import Any, Union
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def read_pickle_file(pkl_path: str) -> Any:
    """
    读取 pickle 文件

    Args:
        pkl_path: pickle 文件路径

    Returns:
        pickle 文件中的数据对象
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Successfully read pickle file: {pkl_path}")
        return data
    except Exception as e:
        logger.error(f"Error reading pickle file: {e}")
        raise

def convert_to_binary(data: Any) -> bytes:
    """
    将数据转换为二进制格式

    Args:
        data: 要转换的数据对象

    Returns:
        二进制数据
    """
    try:
        if isinstance(data, (int, float)):
            # 数值类型转换
            return struct.pack('d', float(data))
        elif isinstance(data, str):
            # 字符串转换
            return data.encode('utf-8')
        elif isinstance(data, (list, tuple)):
            # 列表或元组转换
            return b''.join(convert_to_binary(item) for item in data)
        elif isinstance(data, dict):
            # 字典转换
            binary_data = b''
            for key, value in data.items():
                binary_key = convert_to_binary(key)
                binary_value = convert_to_binary(value)
                # 存储键值对的长度信息
                binary_data += struct.pack('I', len(binary_key))
                binary_data += binary_key
                binary_data += struct.pack('I', len(binary_value))
                binary_data += binary_value
            return binary_data
        else:
            # 其他类型，尝试直接序列化
            return pickle.dumps(data)
    except Exception as e:
        logger.error(f"Error converting data to binary: {e}")
        raise

def save_binary_file(binary_data: bytes, output_path: str) -> None:
    """
    保存二进制数据到文件

    Args:
        binary_data: 二进制数据
        output_path: 输出文件路径
    """
    try:
        with open(output_path, 'wb') as f:
            f.write(binary_data)
        logger.info(f"Successfully saved binary file: {output_path}")
    except Exception as e:
        logger.error(f"Error saving binary file: {e}")
        raise

def convert_pkl_to_bin(input_path: str, output_path: str) -> None:
    """
    将 PKL 文件转换为 BIN 文件

    Args:
        input_path: 输入的 PKL 文件路径
        output_path: 输出的 BIN 文件路径
    """
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 读取 pickle 文件
        data = read_pickle_file(input_path)

        # 转换为二进制
        binary_data = convert_to_binary(data)

        # 保存二进制文件
        save_binary_file(binary_data, output_path)

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Convert PKL file to BIN format')
    parser.add_argument('input', help='Input PKL file path')
    parser.add_argument('output', help='Output BIN file path')

    args = parser.parse_args()

    try:
        convert_pkl_to_bin(args.input, args.output)
        logger.info("Conversion completed successfully")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return 1

    return 0

if __name__ == '__main__':
    exit(main())