import numpy as np

from datetime import datetime
from pathlib import Path
from tensorflow import image, keras
from typing import Iterable, Optional

from keras.backend import clear_session

from neural_network import DeepQNet

class Agent:
    """
    Classe base que define a interface para um agente.
    Também funciona como um agente aleatório.

    Parâmetros
    ----------
    n_actions : int
        O tamanho do espaço de observação do ambiente.
    n_frames_per_action : int
        O número de frames antes de uma nova ação ser escolhida.
    random_seed : int, optional
        A semente usada para gerar números aleatórios.
    """
    def __init__(self, n_actions: int = 1, n_frames_per_action: int = 4,
                 random_seed: Optional[int] = None) -> None:
        """"""
        self.n_actions = n_actions
        self.n_frames_per_action = n_frames_per_action
        self.rng = np.random.default_rng(random_seed)
        self.max_steps_per_episode = float("inf")

        self.previous_action: int = 0
        self.action_index: int = 0

    def save(self) -> None:
        """Salva os pesos do modelo no disco."""
        pass

    def load(self) -> None:
        """Carrega uma versão do modelo do disco."""
        pass

    def choose_action(self, observation: np.ndarray, training: bool) -> int:
        """Ignora a observação e retorna uma ação aleatória."""
        self.action_index += 1
        if self.action_index == self.n_frames_per_action:
            self.previous_action = self.rng.integers(0, self.n_actions)
            self.action_index = 0
        return self.previous_action

class HumanAgent(Agent):
    """
    Classe usada para representar um agente controlado por um jogador humano.

    Cada chave do dicionário como parâmetro deve ser uma tecla ou iterável de teclas do
    pygame, sendo que teclas pressionadas simultaneamente devem ser uma chave à parte.

    Teclas não presentes em qualquer combinação serão desconsideradas no processamento.

    Parâmetros
    ----------
    mapping : dict[Iterable[int] | int]
        Um dicionário que mapeia as teclas para as ações do agente.
    n_frames_per_action : int
        O número de frames antes de uma nova ação ser escolhida.
    """
    def __init__(self, mapping: dict[Iterable[int] | int, int], n_frames_per_action: int = 1) -> None:
        self.keys_held: list[int] = []
        self.all_keys: set[int] = set()
        self.mapping: dict[Iterable[int], int] = {}
        self.n_frames_per_action = n_frames_per_action
        self.max_steps_per_episode = float("inf")

        self.previous_action: int = 0
        self.action_index: int = 0

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
                print(f"Tecla de valor {key} foi solta sem estar sendo pressionada.")

        for key in filter(lambda key: key in self.all_keys, keys_down):
            self.keys_held.append(key)

        self.action_index += 1
        if self.action_index == self.n_frames_per_action:
            self.action_index = 0
            self.keys_held.sort()
            self.previous_action = self.mapping.get(tuple(self.keys_held), 0)

        return self.previous_action

class RLAgent(Agent):
    """
    Classe usada para representar um agente de Reinforcement Learning.

    Parâmetros
    ----------
    n_actions : int
        O número de ações que o agente pode executar.
    input_size : tuple[int, int]
        As dimensões da imagem de entrada.
    n_frames_per_action : int
        O número de frames que serão agrupados para cada ação.
    output_path : str
        A pasta na qual os pesos do modelo serão salvos.
    gamma : float
        O fator de desconto usado para calcular a recompensa futura.
    epsilon_min : float
        O valor mínimo de epsilon.
    epsilon_max : float
        O valor máximo de epsilon.
    epsilon_decay : float
        O fator de decaimento de epsilon.
    batch_size : int
        O tamanho do batch usado para treinar o modelo.
    max_steps : int
        O número máximo de passos que o agente pode executar em um episódio.
    random_state : int
        A semente usada para gerar números aleatórios.
    """
    def __init__(
        self, n_actions: int, input_size: tuple[int, int], n_frames_per_action: int = 4,
        output_dir: str | Path = "weights", gamma: float = 0.99, epsilon_min: float = 0.1,
        epsilon_max: float = 0.9, epsilon_decay: float = 0.9995, batch_size: int = 32,
        max_memory_length: int = 100_000, checkpoint_interval: int = 10_000,
        n_random_frames: int = 50_000, n_greedy_frames: int = 1_000_000,
        random_state: int | None = None, name: str | None = None
    ) -> None:
        self.n_actions = n_actions
        self.input_size = input_size
        self.output_dir = Path(output_dir)
        self.n_frames_per_action = n_frames_per_action
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.max_memory_length = max_memory_length
        self.checkpoint_interval = checkpoint_interval
        self.n_random_frames = n_random_frames
        self.n_greedy_frames = n_greedy_frames
        self.rng = np.random.default_rng(random_state)
        self.name = name or datetime.now().strftime("%Y%m%d_%H%M%S")

        self.model = self._build_model("QValueNet")
        self.target_model = self._build_model("TargetQNet")

        self.previous_action: int = 0
        self.action_index: int = 0
        self.recent_observations: np.zeros((*self.input_size, self.n_frames_per_action))
        self.checkpoint_number: int = 0

        self.action_history: list[int] = []
        self.state_history = []
        self.observation_history: list[np.ndarray] = []
        self.rewards_history: list[float] = []
        self.episode_reward_history: list[float] = []
        self.running_reward: float = 0.0

        self.update_target_network_every = 10_000

    def _build_model(self, name: str) -> DeepQNet:
        """Constrói o modelo do agente."""
        return DeepQNet(
            [
                keras.layers.Input(shape=(*self.input_size, self.n_frames_per_action)),
                keras.layers.Conv2D(32, 9, 4, activation="relu"),
                keras.layers.Conv2D(64, 5, 2, activation="relu"),
                keras.layers.Conv2D(64, 3, 1, activation="relu"),
                keras.layers.Flatten(),
                keras.layers.Dense(512, activation="relu"),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(self.n_actions, activation="linear")
            ],
            metrics=[keras.metrics.Mean(name="mean_q_values")],
            name=name
        )

    def _choose_action(self, training: bool) -> int:
        """Escolhe uma ação de acordo com o modelo do agente."""
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_actions)
        else:
            return np.argmax(self.model.predict(self.recent_observations[np.newaxis, ...]))

    def choose_action(self, observation: np.ndarray, training: bool) -> int:
        self.recent_observations[self.action_index] = observation
        self.action_index += 1
        if self.action_index == self.n_frames_per_action:
            self.action_index = 0
            self.previous_action = self._choose_action(
                image.resize(self.recent_observations, self.input_size),
                training
            )

    def on_episode_end(self, episode: int) -> None:
        """Executa ao final de um episódio."""
        self.episode_reward_history.append(self.running_reward)
        if len(self.episode_reward_history) > 100:
            del self.episode_reward_history[:1]
        self.running_reward = np.mean(self.episode_reward_history[-100:])

    def save(self) -> None:
        """
        Salva o modelo na pasta raiz da classe e na subpasta com o nome do modelo.
        O número do checkpoint é incrementado a cada chamada.
        """
        path = self.output_dir / self.name
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_weights(path / f"{self.checkpoint_number}_q.h5")
        self.target_model.save_weights(path / f"{self.checkpoint_number}._t.h5")
        self.checkpoint_number += 1

    def load(self, name: str, n: int = -1) -> None:
        """
        Carrega o modelo com nome `name` e número de checkpoint `n` do disco.

        Parâmetros
        ----------
        name : str
            O nome do modelo, que é o nome da pasta em que seus pesos estão salvos.
        n : int, default=-1
            O número do checkpoint a ser carregado. Se for -1, carrega o checkpoint mais recente.
        """
        self.name = name
        path = self.output_dir / self.name
        files = sorted(path.glob("*.h5"))
        index = len(files) - 2 if n == -1 else n * 2
        clear_session()
        self.model.load_weights(path / f"{index}_q.h5")
        self.target_model.load_weights(path / f"{index + 1}_t.h5")
