from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound=np.dtype)


class CircularList(Generic[T]):
    """
    Classe que representa uma lista circular de tamanho fixo.

    Toda a memória necessária é alocada na inicialização e o parâmetro `shape`
    representa as dimensões de cada elemento do array, pois a primeira dimensão
    é reservada para o tamanho máximo da lista.

    Parâmetros
    ----------
    max_size : int
        Tamanho máximo e constante da lista.
    shape : tuple[int, ...] ou int
        Dimensões de cada elemento do array interno.
    dtype : numpy.dtype, default=numpy.uint8
        Tipo de dado dos elementos do array interno.
    """
    def __init__(self, max_size: int, shape: tuple[int, ...] | int = 1, dtype: T = np.uint8) -> None:
        self.data: np.ndarray[T] = np.empty(self._process_shape(max_size, shape), dtype)
        self.start = 0
        self.size = 0

    @staticmethod
    def _process_shape(max_size: int, shape: tuple[int, ...] | int) -> tuple[int, ...]:
        """Processa os argumentos para permitir maior flexibilidade na definição do objeto."""
        if isinstance(shape, int):
            if shape == 1:
                return (max_size,)
            return (max_size, shape)
        return (max_size, *shape)

    @staticmethod
    def compute_size(max_size: int, shape: tuple[int, ...] | int = 1, dtype: T = np.uint8) -> int:
        """Obtém o número de bytes que um array com as dimensões especificadas ocuparia."""
        shape = CircularList._process_shape(max_size, shape)
        return (
            112 + 48  # Tamanho do objeto do array e tamanho do objeto CircularList
            + (len(shape) - 1) * 16  # 16 bytes para cada dimensão adicional
            # Quantidade de itens multiplicada pelo tamanho de cada item
            + np.dtype(dtype).itemsize * np.prod(shape)
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Retorna as dimensões de cada elemento do array interno."""
        return (self.size, *self.data.shape[1:])

    @property
    def capacity(self) -> int:
        """Retorna a capacidade ou tamanho máximo do array interno."""
        return self.data.shape[0]

    def _get_index(self, index: int) -> int:
        """Função auxiliar para obter o índice real de um item."""
        return (self.start + index) % self.data.shape[0]

    def append(self, item: np.ndarray[T] | T) -> None:
        """
        Adiciona um item ao final da lista. Se o array estiver cheio, o item mais antigo
        é sobrescrito.

        Parâmetros
        ----------
        item : numpy.ndarray[T] | T
            Item a ser adicionado.
        """
        # Se o array está cheio, sobrescreve o item mais antigo
        self.data[self._get_index(self.size)] = item
        if self.size < self.data.shape[0]:
            self.size += 1
        else:
            self.start = self._get_index(1)

    def clear(self) -> None:
        """Define o tamanho como 0, mas não limpa a memória."""
        self.start = 0
        self.size = 0

    def adapt_key(self, key: int | tuple[int, ...] | npt.ArrayLike) -> int | tuple | list[int]:
        """
        Função auxiliar para adaptar uma chave (índice ou tupla de índices)
        de acesso da lista circular ao array interno.
        """
        if isinstance(key, int):
            return self._get_index(key)
        if isinstance(key, tuple):
            return (self._get_index(key[0]), *key[1:])
        return [self._get_index(i) for i in key]

    def __getitem__(self, key: int | tuple[int, ...] | npt.ArrayLike | slice) -> np.ndarray[T] | T:
        if isinstance(key, slice):
            return self._correct_order(key)
        return self.data[self.adapt_key(key)]

    def __setitem__(self, key: int | tuple[int, ...] | npt.ArrayLike, value: np.ndarray[T] | T) -> None:
        self.data[self.adapt_key(key)] = value

    def __iter__(self):
        for i in range(self.size):
            yield self[i]

    def __len__(self) -> int:
        return self.size

    # Por padrão, não cria uma cópia
    def _correct_order(self, key: slice = slice(None)) -> np.ndarray[T]:
        """
        Corrige a ordem dos elementos do array interno.

        Se for necessário criar uma cópia, ela sobrescreve o array antigo.

        Parâmetros
        ----------
        key : slice
            Slice usado para obter os valores do array. Passos diferentes de 1 não são suportados.

        Retorna
        -------
        numpy.ndarray[T]
        """
        idx = key.indices(self.size)
        if idx[2] != 1:
            raise NotImplementedError("Slicing com passo diferente de 1 não suportado.")
        if self.size == 0:
            # Retorna um array vazio
            return self.data[0:0]
        # Não cria uma cópia se o slice não transborda o array contíguo
        start, end = self._get_index(idx[0]), self._get_index(idx[1] - 1) + 1
        if end > start:
            return self.data[start:end]

        self.data = np.concatenate((
            self.data[start:self.data.shape[0]], self.data[0:end]
        ))
        self.start = 0
        return self.data

    def __repr__(self) -> str:
        return repr(self._correct_order())

    def __str__(self) -> str:
        return str(self._correct_order())
