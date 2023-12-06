from os import sysconf
from pathlib import Path
from psutil import Process

def convert_bytes(num: int, precision: int = 3, use_binary: bool = True) -> str:
    """
    Converte um número de bytes para uma string com a unidade de medida mais adequada.

    Parâmetros
    ----------
    num : int
        Número de bytes.
    precision : int, default=3
        Precisão de exibição do valor em ponto flutuante.
    use_binary : bool, default=True
        Se True, a conversão será feita em base 2, resultando em múltiplos de 1024 em vez de 1000.

    Retorna
    -------
    str
        String com o valor convertido e a unidade de medida.
    """
    constant = 1024 if use_binary else 1000
    units = ["bytes", "KiB", "MiB", "GiB", "TiB"] if use_binary else ["bytes", "KB", "MB", "GB", "TB"]
    for unit in units:
        if num < constant:
            return f"{num:.{precision}f} {unit}"
        num /= constant

def get_bytes(path: str | Path) -> int:
    """
    Retorna o tamanho de um arquivo ou diretório em bytes.

    Parâmetros
    ----------
    path : str or Path
        Caminho para o arquivo ou diretório.

    Retorna
    -------
    int
    """
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file())

def get_size(path: str | Path) -> str:
    """
    Retorna o tamanho de um arquivo ou diretório em forma legível.

    Parâmetros
    ----------
    path : str or Path
        Caminho para o arquivo ou diretório.

    Retorna
    -------
    str
    """
    path = Path(path)
    if path.is_file():
        return convert_bytes(path.stat().st_size)
    return convert_bytes(sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()))

def get_total_available_memory() -> str:
    """
    Retorna a memória total disponível no sistema.

    Pode não ser o valor real quando usado máquinas virtuais como o WSL.

    Retorna
    -------
    str
    """
    return convert_bytes(sysconf("SC_PAGE_SIZE") * sysconf("SC_PHYS_PAGES"))

def get_memory_consumption() -> str:
    """
    Retorna a quantidade de memória consumida pelo processo atual em forma legível.

    Retorna
    -------
    str
    """
    process = Process()
    return convert_bytes(process.memory_info().rss)
