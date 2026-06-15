"""
FRP 下载管理工具 - exo_manager 本地版本
独立于 exo 主项目，可单独部署
"""
import os
import platform
import zipfile
import tarfile
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Optional
import requests

FRP_VERSION = "0.52.3"
FRP_BASE_URL = "https://github.com/fatedier/frp/releases/download"


def get_system_info() -> tuple:
    """获取系统信息，用于确定 frp 下载链接"""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "windows":
        system = "windows"
    elif system == "darwin":
        system = "darwin"
    else:
        system = "linux"

    if machine in ["amd64", "x86_64"]:
        arch = "amd64"
    elif machine in ["arm64", "aarch64"]:
        arch = "arm64"
    elif machine in ["arm", "armv7l"]:
        arch = "arm"
    elif machine == "386":
        arch = "386"
    else:
        arch = "amd64"

    return system, arch


def get_frp_download_url() -> str:
    """获取 frp 下载链接"""
    system, arch = get_system_info()
    filename = f"frp_{FRP_VERSION}_{system}_{arch}"

    if system == "windows":
        filename += ".zip"
    else:
        filename += ".tar.gz"

    return f"{FRP_BASE_URL}/v{FRP_VERSION}/{filename}"


def get_frp_bin_dir() -> Path:
    """获取 frp 二进制文件存放目录"""
    home = Path.home()
    frp_dir = home / ".exo" / "frp"
    frp_dir.mkdir(parents=True, exist_ok=True)
    return frp_dir


def get_frpc_path() -> Path:
    """获取 frpc 可执行文件路径"""
    frp_dir = get_frp_bin_dir()
    system, _ = get_system_info()

    if system == "windows":
        return frp_dir / "frpc.exe"
    else:
        return frp_dir / "frpc"


def get_frps_path() -> Path:
    """获取 frps 可执行文件路径"""
    frp_dir = get_frp_bin_dir()
    system, _ = get_system_info()

    if system == "windows":
        return frp_dir / "frps.exe"
    else:
        return frp_dir / "frps"


def download_file(url: str, dest: Path) -> bool:
    """下载文件"""
    try:
        print(f"正在下载: {url}")
        response = requests.get(url, stream=True, timeout=300, verify=False)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192

        with open(dest, "wb") as f:
            if total_size == 0:
                f.write(response.content)
            else:
                from tqdm import tqdm
                with tqdm(total=total_size, unit="iB", unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            size = f.write(chunk)
                            pbar.update(size)

        print(f"下载完成: {dest}")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        if dest.exists():
            dest.unlink()
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """解压压缩包"""
    try:
        print(f"正在解压: {archive_path}")

        if str(archive_path).endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        elif str(archive_path).endswith(".tar.gz"):
            with tarfile.open(archive_path, "r:gz") as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"不支持的压缩格式: {archive_path}")
            return False

        print(f"解压完成: {extract_to}")
        return True
    except Exception as e:
        print(f"解压失败: {e}")
        return False


def find_and_move_binaries(extract_dir: Path, target_dir: Path) -> bool:
    """查找并移动 frp 二进制文件"""
    try:
        system, _ = get_system_info()

        frp_subdirs = [d for d in extract_dir.iterdir() if d.is_dir() and d.name.startswith("frp_")]
        if not frp_subdirs:
            print("未找到 frp 解压目录")
            return False

        frp_source_dir = frp_subdirs[0]

        frpc_source = frp_source_dir / ("frpc.exe" if system == "windows" else "frpc")
        frps_source = frp_source_dir / ("frps.exe" if system == "windows" else "frps")

        if frpc_source.exists():
            frpc_dest = target_dir / frpc_source.name
            shutil.move(str(frpc_source), str(frpc_dest))
            if system != "windows":
                frpc_dest.chmod(0o755)
            print(f"已安装: {frpc_dest}")

        if frps_source.exists():
            frps_dest = target_dir / frps_source.name
            shutil.move(str(frps_source), str(frps_dest))
            if system != "windows":
                frps_dest.chmod(0o755)
            print(f"已安装: {frps_dest}")

        return True
    except Exception as e:
        print(f"移动二进制文件失败: {e}")
        return False


def download_and_install_frp() -> bool:
    """下载并安装 frp"""
    frp_dir = get_frp_bin_dir()
    frpc_path = get_frpc_path()

    if frpc_path.exists():
        print(f"frp 已安装: {frpc_path}")
        return True

    print("=" * 60)
    print("  正在安装 frp...")
    print("=" * 60)

    temp_dir = Path(tempfile.mkdtemp())

    try:
        download_url = get_frp_download_url()
        archive_path = temp_dir / Path(download_url).name

        if not download_file(download_url, archive_path):
            return False

        if not extract_archive(archive_path, temp_dir):
            return False

        if not find_and_move_binaries(temp_dir, frp_dir):
            return False

        print()
        print("=" * 60)
        print("  frp 安装成功!")
        print("=" * 60)
        print()

        return True
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def check_frpc_available() -> bool:
    """检查 frpc 是否可用"""
    frpc_path = get_frpc_path()
    if not frpc_path.exists():
        return False

    try:
        result = subprocess.run(
            [str(frpc_path), "-v"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except:
        return False


def ensure_frpc_installed() -> bool:
    """确保 frpc 已安装，如未安装则自动下载"""
    if check_frpc_available():
        return True

    print("frpc 未找到，正在自动下载安装...")
    return download_and_install_frp()
