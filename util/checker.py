import subprocess


def check_gpu(device: int) -> bool:
    """
    GPUデバイスが使用可能か確認する
    Args:
        device: デバイス番号
    """
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"])
        lines = output.decode().split('\n')
        for line in lines:
            if line == "":
                continue
            gpu_device_id = line.split(":")[0]
            if gpu_device_id == f"GPU {device}":
                return True
    except Exception as e:
        return False


if __name__ == '__main__':
    res = check_gpu(0)
    print(res)