import numpy as np

from typing import Optional

class CircularList:
    """
    Classe que representa uma lista circular de tamanho fixo.

    Toda a memória necessária é alocada na inicialização e o parâmetro `shape`
    representa as dimensões de cada elemento do array, pois a primeira dimensão
    é reservada para o tamanho máximo da lista.

    Parâmetros
    ----------
    max_size : int
        Tamanho máximo e constante da lista.
    shape : tuple[int, ...]
        Dimensões de cada elemento do array interno.
    dtype : numpy.dtype, default=numpy.uint8
        Tipo de dado dos elementos do array interno.
    """
    def __init__(self, max_size: int, shape: tuple[int, ...] | int = 1, dtype: np.dtype = np.uint8) -> None:
        if isinstance(shape, int):
            if shape == 1:
                shape = (max_size,)
            else:
                shape = (max_size, shape)
        else:
            shape = (max_size, *shape)
        self.arr = np.empty(shape, dtype)
        self.start = 0
        self.size = 0

    def _get_index(self, index: int) -> int:
        """Função auxiliar para obter o índice real de um item."""
        return (self.start + index) % self.arr.shape[0]

    @property
    def shape(self) -> tuple[int, ...]:
        """Retorna as dimensões de cada elemento do array interno."""
        return (self.size, *self.arr.shape[1:])

    @staticmethod
    def compute_size(max_size: int, shape: tuple[int, ...], dtype: np.dtype = np.uint8) -> int:
        """Obtém o número de bytes que um array com as dimensões especificadas ocuparia."""
        return 112 + len(shape) * 16 + max_size * np.dtype(dtype).itemsize * np.prod(shape)

    def append(self, item: np.ndarray | float | int) -> None:
        """
        Adiciona um item ao final da lista. Se o array estiver cheio, o item mais antigo
        é sobrescrito.

        Parâmetros
        ----------
        item : np.ndarray | float | int
            Item a ser adicionado.
        """
        # Se o array está cheio, sobrescreve o item mais antigo
        self.arr[self._get_index(self.size)] = item
        if self.size < self.arr.shape[0]:
            self.size += 1
        else:
            self.start = self._get_index(1)

    def clear(self) -> None:
        """Define o tamanho como 0, mas não limpa a memória."""
        self.start = 0
        self.size = 0

    def adapt_key(self, key: int | tuple[int, ...]) -> int | tuple:
        """
        Função auxiliar para adaptar uma chave (índice ou tupla de índices)
        de acesso da lista circular ao array interno.
        """
        if isinstance(key, tuple):
            key = (self._get_index(key[0]), *key[1:])
        else:
            key = self._get_index(key)
        return key

    def __getitem__(self, key: int | tuple[int, ...]) -> np.ndarray | int | float:
        return self.arr[self.adapt_key(key)]

    def __setitem__(self, key: int | tuple[int, ...], value: np.ndarray | int | float) -> None:
        self.arr[self.adapt_key(key)] = value

    def __correct_order(self) -> np.ndarray:
        """Corrige a ordem dos elementos do array interno criando uma cópia."""
        end = self.start + self.size
        return np.concatenate((
            self.arr[self.start:min(end, self.arr.shape[0])],
            self.arr[:min(end % self.arr.shape[0], self.start)]
        ))

    def __repr__(self) -> str:
        return repr(self.__correct_order())

    def __str__(self) -> str:
        return str(self.__correct_order())
