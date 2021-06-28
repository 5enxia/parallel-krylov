import subprocess


def check_gpu_device(device: int) -> bool:
    """
    GPUデバイスが使用可能か確認する
    Args:
        device: デバイス番号
    """
    output = subprocess.check_output(["nvidia-smi", "-L"])
    lines = output.decode().split('\n')
    print(lines)
    for line in lines:
        if line == "":
            continue
        gpu_device_id = line.split(":")[0]
        if gpu_device_id == f"GPU {device}":
            return True
    return False

res = check_gpu_device(0)
print(res)
