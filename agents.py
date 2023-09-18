from os import makedirs, path
import numpy as np
import torch
from typing import Iterable, Optional

class Agent:
    """
    Classe base que define a interface para um agente.
    Também funciona como um agente aleatório.

    Parâmetros
    ----------
    action_space_size : int
        O tamanho do espaço de observação do ambiente.
    random_seed : int
        A semente usada para gerar números aleatórios.
    """
    def __init__(self, action_space_size: int = 1, random_seed: Optional[int] = None) -> None:
        """"""
        self.high_action = action_space_size
        self.rng = np.random.default_rng(random_seed)

    def save(self) -> None:
        """Salva os pesos do modelo no disco."""
        pass

    def load(self) -> None:
        """Carrega os pesos do modelo do disco."""
        pass

    def free(self) -> None:
        """Libera a memória usada pelo agente."""
        pass

    def choose_action(self, observation: np.ndarray) -> int:
        """Ignora a observação e retorna uma ação aleatória."""
        return self.rng.integers(0, self.high_action)

class HumanAgent(Agent):
    """
    Classe usada para representar um agente controlado por um jogador humano.

    Cada chave do dicionário deve ser uma tecla ou iterável de teclas do pygame,
    sendo que teclas pressionadas simultaneamente devem ser uma chave à parte.

    Teclas não presentes em qualquer combinação serão desconsideradas no processamento.

    Parâmetros
    ----------
    mapping : dict[Iterable[int] | int]
        Um dicionário que mapeia as teclas para as ações do agente.
    """
    def __init__(self, mapping: dict[Iterable[int] | int, int]) -> None:
        self.keys_held: list[int] = []
        self.all_keys: set[int] = set()
        self.mapping: dict[Iterable[int], int] = {}
        for key_list, action in mapping.items():
            if isinstance(key_list, int):
                key_list = (key_list,)
            # Salva a tupla ordenada para que possa ser conferida com hash
            self.mapping[tuple(sorted(key_list))] = action
            # Adiciona cada tecla à lista de teclas que podem interagir com o jogo
            for key in key_list:
                self.all_keys.add(key)

    def reset(self) -> None:
        """Restaura o estado inicial do agente, sem teclas pressionadas."""
        self.keys_held = []

    def process_input(self, keys_down: list[int], keys_up: list[int]) -> int:
        """
        Processa as teclas pressionadas e retorna a ação escolhida de acordo
        com o mapeamento dado no construtor.

        Parâmetros
        ----------
        keys_down : list[int]
            Lista de teclas que foram pressionadas.
        keys_up : list[int]
            Lista de teclas que foram soltas.

        Retorna
        -------
        int
            A ação escolhida. Se nenhuma ação for escolhida, retorna 0.
        """
        for key in filter(lambda key: key in self.all_keys, keys_up):
            try:
                self.keys_held.remove(key)
            except ValueError:
                print(f"Key of value {key} was released without being held.")

        for key in filter(lambda key: key in self.all_keys, keys_down):
            self.keys_held.append(key)

        self.keys_held.sort()
        return self.mapping.get(tuple(self.keys_held), 0)

class RLAgent(Agent):
    def __init__(self, output_path: str = "weights") -> None:
        self.output_path = output_path
        self.model = self.build_model()

    def build_model(self) -> torch.nn.Module:
        """Constrói o modelo do agente."""
        pass

    def save(self):
        makedirs(self.output_path, exist_ok=True)
        torch.save(self.model, path.join(self.output_path, "model.pth"))

    def load(self):
        torch.load(path.join(self.output_path, "model.pth"))
