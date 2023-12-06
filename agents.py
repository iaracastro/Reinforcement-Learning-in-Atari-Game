import json

from abc import ABC, abstractmethod
from datetime import datetime
from io import TextIOWrapper
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import h5py
import keras
import numpy as np
import pygame
import tensorflow as tf

from circular_list import CircularList
from neural_network import DeepQNet
from utils import convert_bytes


class HumanAgent:
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
        # Como poucas teclas são pressionadas, lista pode ser mais eficiente que conjunto
        self.keys_held: list[int] = []
        self.all_keys: set[int] = set()
        self.mapping: dict[Iterable[int], int] = {}
        self.n_frames_per_action = n_frames_per_action

        self.step_index: int = 0
        self.action: int = 0

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
        pygame.event.clear()
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
                pass

        for key in filter(lambda key: key in self.all_keys, keys_down):
            if key not in self.keys_held:
                self.keys_held.append(key)

        self.step_index += 1
        if self.step_index == self.n_frames_per_action:
            self.step_index = 0
            self.keys_held.sort()
            self.action = self.mapping.get(tuple(self.keys_held), 0)

        return self.action


class RLAgent(ABC):
    """Classe base que define a interface para um agente."""
    max_actions_per_episode: int | float = np.inf
    n_frames_per_action: int = 1

    @abstractmethod
    def choose_action(self, observation: np.ndarray, previous_reward: float) -> int:
        """Ignora a observação e retorna uma ação aleatória."""
        ...

    @abstractmethod
    def is_training_done(
        self, episode_reward_history: list[float], episode_length_history: list[int]
    ) -> bool:
        """Retorna se o objetivo já foi atingido e o treinamento pode se encerrado."""
        ...

    @abstractmethod
    def on_episode_end(
        self, episode_reward_history: list[float], episode_length_history: list[int]
    ) -> bool:
        """
        Atualiza o estado do agente ao final de um episódio e retorna
        se o treinamento deve ser encerrado.

        Parâmetros
        ----------
        episode_reward_history : list[float]
            Valores de recompensa ao longo dos episódios.
        episode_length_history : list[int]
            Valores de tamanho de episódios em número de ações.
        """
        ...

    @abstractmethod
    def set_training(self, is_training: bool) -> None:
        """Define o modo de treinamento ou inferência do agente."""
        ...


class RandomAgent(RLAgent):
    """
    Agente que sempre escolhe uma ação aleatória.

    Parâmetros
    ----------
    n_actions : int
        O tamanho do espaço de observação do ambiente.
    n_frames_per_action : int
        O número de frames antes de uma nova ação ser escolhida.
    random_seed : int, optional
        A semente usada para gerar números aleatórios.
    """
    def __init__(self,
        n_actions: int = 1,
        n_frames_per_action: int = 4,
        random_seed: Optional[int] = None
    ) -> None:
        self.n_frames_per_action = n_frames_per_action
        self.rng = np.random.default_rng(random_seed)
        self.n_actions = n_actions

        self.step_index: int = 0
        self.action: int = 0

    def choose_action(self, observation: np.ndarray, previous_reward: float) -> int:
        """Ignora a observação e retorna uma ação aleatória."""
        self.step_index += 1
        if self.step_index == self.n_frames_per_action:
            self.action = self.rng.integers(0, self.n_actions)
            self.step_index = 0
        return self.action

    def is_training_done(
        self, episode_reward_history: list[float], episode_length_history: list[int]
    ) -> bool:
        """Sempre retorna False."""
        return False

    def on_episode_end(
        self, episode_reward_history: list[float], episode_length_history: list[int]
    ) -> bool:
        """Sempre retorna True, indicando que não há mais atualizações a serem feitas."""
        return True

    def set_training(self, is_training: bool) -> None:
        """Define o modo de treinamento ou inferência do agente."""
        pass


class DeepQAgent(RLAgent):
    """
    Classe usada para representar um agente de Reinforcement Learning baseado em
    Double Deep Q-Learning.

    Epsilon representa a probabilidade de o agente escolher uma ação aleatória.
    A abordagem epsilon-greedy é usada para balancear a exploração do ambiente
    e o aproveitamento do conhecimento adquirido.

    Parâmetros
    ----------
    name : str, opcional
        O nome do modelo e da sua pasta. Se não for especificado, é gerado com base em data e hora.
    n_actions : int, opcional
        O número de ações que o agente pode executar. Opcional apenas ao carregar um modelo.
    input_size : tuple[int, int], opcional
        As dimensões da imagem de entrada. Opcional apenas ao carregar um modelo.
    n_frames_per_action : int
        O número de frames que serão agrupados para processar cada ação.
    output_dir : str, default="weights"
        A pasta na qual os pesos do modelo serão salvos.
    alpha : float, default=0.001
        A taxa de aprendizado do otimizador.
    gamma : float, default=0.95
        O fator de desconto que determina a importância das recompensas futuras.
    tau : float, default=0.005
        O fator de interpolação usado para atualizar a rede alvo. Pode ser visto como
        a velocidade com que a rede alvo se aproxima da rede principal.
    max_episodes : int, default=100
        O número máximo de episódios que o agente vai executar.
    epsilon_max : float, default=0.9
        O valor máximo e inicial de epsilon.
    epsilon_min : float, default=0.1
        O valor mínimo a que epsilon pode chegar.
    n_frames_until_decay : int, opcional
        Número de frames até que o valor de epsilon decaia para o mínimo.
    n_episodes_until_decay: int, opcional
        Número de episódios até que o valor de epsilon decaia para o mínimo.
        Se omitido, o máximo de episódios é usado.
    batch_size : int, default=32
        O tamanho do batch de amostras usado para treinar o modelo.
    target_reward : float, default=inf
        A recompensa alvo. Quando a recompensa média atinge esse, o treinamento é encerrado.
    update_network_action_interval : int, default=8
        O número de ações que devem ser executadas antes de atualizar a rede.
    update_target_network_action_interval : int, default=1_000
        O número de ações que devem ser executadas antes de atualizar a rede alvo.
    max_memory_length : int, default=100_000
        O tamanho máximo da memória de replay.
    n_random_actions : int, default=50_000
        O número de ações que o agente deve escolher aleatoriamente antes do epsilon-greedy.
    max_skips_on_reset : int, default=150
        O número máximo de vezes que o agente pode tomar a ação nula ao iniciar um episódio.
        Se for 0, impacta a geração de números aleatórios, mas tem o mesmo resultado que None.
    max_actions_per_episode : int ou float, default=20_000
        O número máximo de ações que o agente pode executar em um episódio.
    n_top_saved : int, default=3
        O número de melhores checkpoints que devem ser mantidos no disco.
    share_memory : bool, default=False
        Se True, compartilha a memória de replay entre modelos de nomes diferentes, propenso
        a sobrescrever. Caso contrário, o modelo cria sua própria pasta de memória.
    random_seed : int, opcional
        A semente usada para gerar números aleatórios.
    model_layers : list[keras.layers.Layer], opcional
        A lista de camadas que compõem o modelo, excluindo entrada e saída.
        Se não for especificada, uma arquitetura convolucional padrão é usada.

    Notas
    -----
    O decaimento do epsilon ocorre pelo método que o tornar menor, seja por episódio ou por frame.

    Todos os métodos relacionados a salvar ou carregar o modelo dependem do nome do modelo
    e dos parâmetros `output_dir` e `share_memory`.
    """
    relevant_parameters = [
        "n_actions", "input_size", "n_frames_per_action", "alpha", "gamma", "tau",
        "epsilon_max", "epsilon_min", "n_frames_until_decay", "batch_size",
        "update_network_action_interval", "update_target_network_action_interval",
        "max_memory_length", "n_random_actions", "max_actions_per_episode",
        "max_skips_on_reset", "share_memory",
    ]
    extra_parameters = [
        "max_episodes", "target_reward", "n_top_saved"
    ]

    @staticmethod
    def now_formatted() -> str:
        """Retorna a data e hora atual formatada."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def __init__(
        self,
        name: str | None = None,
		n_actions: int | None = None,
		input_size: tuple[int, int] | None = None,
		n_frames_per_action: int = 4,
        output_dir: str | Path = "saves",
		alpha: float = 0.001,
		gamma: float = 0.95,
        tau: float = 0.005,
        max_episodes: int = 100,
		epsilon_max: float = 0.9,
		epsilon_min: float = 0.1,
        n_frames_until_decay: int | None = None,
        n_episodes_until_decay: int | None = None,
		batch_size: int = 32,
		target_reward: float = np.inf,
        update_network_action_interval: int = 8,
		update_target_network_action_interval: int = 1_000,
        max_memory_length: int = 100_000,
        n_random_actions: int = 50_000,
        max_skips_on_reset: int | None = 150,
		max_actions_per_episode: int | float = 20_000,
		n_top_saved: int = 3,
		share_memory: bool = False,
		random_seed: int | None = None,
        model_layers: list[keras.layers.Layer] | None = None,
    ) -> None:
        if name is None and (n_actions is None or input_size is None):
            raise ValueError("É necessário especificar o nome ou o número de ações e o tamanho da entrada.")
        self.n_actions = n_actions
        self.input_size = input_size
        self.n_frames_per_action = n_frames_per_action
        self.output_dir = Path(output_dir)
        self.rename(name, share_memory)

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.max_episodes = max_episodes
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.n_frames_until_decay = n_frames_until_decay
        self.n_episodes_until_decay = n_episodes_until_decay or max_episodes
        self.adjust_epsilon(frame_number=0, episode_number=0)
        self.batch_size = batch_size
        self.target_reward = target_reward
        self.update_network_action_interval = update_network_action_interval
        self.update_target_network_action_interval = update_target_network_action_interval
        self.max_memory_length = max_memory_length
        self.n_random_actions = n_random_actions
        self.max_skips_on_reset = max_skips_on_reset
        self.max_actions_per_episode = max_actions_per_episode
        self.n_top_saved = n_top_saved

        self.running_reward = 0.0
        self.episode_reward = 0.0
        self.recent_reward = 0.0
        self.is_training = False
        self.action = 0

        if random_seed is not None:
            keras.utils.set_random_seed(random_seed)
        self.rng = np.random.default_rng(random_seed)
        self._set_skip()
        self._init_memory()

        try:
            self.load(name=self.name)
        except FileNotFoundError:
            self.model = self._build_model("QValueNet", model_layers)
            self.target_model = self._build_model("TargetQNet", model_layers)
        self.cpu = tf.device("CPU")

    @staticmethod
    def get_replay_memory_consumption(
        input_size: tuple[int, ...], n_frames_per_action: int = 4,
        max_memory_length: int = 100_000, max_episodes: int = 10_000
    ) -> str:
        """
        Obtém uma estimativa do consumo de RAM da memória de replay do agente.

        Parâmetros
        ----------
        input_size : tuple[int, ...]
            As dimensões da imagem de entrada.
        n_frames_per_action : int, default=4
            O número de frames que serão agrupados para processar cada ação.
        max_memory_length : int, default=100_000
            Tamanho total da memória em ações.
        max_episodes : int, default=10_000
            Máximo de episódios de treinamento do modelo.

        Retorna
        -------
        str
            Memória total formatada e em uma unidade legível.
        """
        # Observações, ações, terminações e recompensas
        dtypes = [np.uint8, np.uint8, bool, np.float32]
        shapes = [(*input_size, n_frames_per_action), 1, 1, 1]

        total = 0
        for shape, dtype in zip(shapes, dtypes):
            total += CircularList.compute_size(max_memory_length, shape, dtype)
        # Recompensas por episódio
        total += CircularList.compute_size(max_episodes, 1, np.float32)

        return convert_bytes(total)

    @property
    def memory_path(self) -> Path:
        return (self.output_dir if self.share_memory else self.path) / "memory.h5"

    def _init_memory(self) -> None:
        """Inicializa a memória de replay do agente."""
        stack_shape = (*self.input_size, self.n_frames_per_action)
        self.recent_observations = np.ndarray(stack_shape, dtype=np.float32)
        self.observation_history = CircularList(self.max_memory_length, stack_shape)
        self.reward_history = CircularList(self.max_memory_length, dtype=np.float32)
        self.action_history = CircularList(self.max_memory_length)

    def _build_model(self, name: str, layers: list[keras.layers.Layer] | None = None) -> DeepQNet:
        """Constrói o modelo do agente."""
        return DeepQNet(
            [keras.layers.Input(shape=(*self.input_size, self.n_frames_per_action))]
            + (layers or [
                # Menos parâmetros que a arquitetura do DeepMind, além de usar atrous convolution
                keras.layers.Conv2D(32, 5, activation="relu", dilation_rate=(3, 3)),
                keras.layers.Conv2D(32, 3, activation="relu"),
                keras.layers.Flatten(),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(128, activation="relu"),
            ]) + [keras.layers.Dense(self.n_actions, activation="linear", name="output")],
            learning_rate=self.alpha,
            name=name
        )

    def _set_skip(self) -> None:
        """Define o número de frames que serão pulados no início de cada episódio."""
        self.remaining_skips = (
            self.rng.integers(0, self.max_skips_on_reset + 1) if self.max_skips_on_reset
            else 0
        )

    def _sample_tensor(self, indices: np.ndarray) -> tf.Tensor:
        """Amostra um tensor de acordo com os índices dados."""
        # Inicialmente era usado tensorflow.gather, mas ficou extremamente lento
        sample = self.observation_history[indices]
        sample = tf.convert_to_tensor(sample, dtype=tf.float32)
        sample /= 255.0
        return sample

    def _sample_memory(self) -> tuple[tf.Tensor, tf.Tensor, np.ndarray, np.ndarray]:
        """Amostra a memória de replay do agente."""
        indices = self.rng.choice(len(self.observation_history) - 1, self.batch_size)
        next_indices = indices + 1
        # Estima as recompensas futuras de cada ação na amostra
        return (
            self.target_model(self._sample_tensor(next_indices)),
            self._sample_tensor(indices),
            self.reward_history[next_indices],
            self.action_history[indices]
        )

    def set_training(self, is_training: bool) -> None:
        self.is_training = is_training

    def adjust_epsilon(
        self, frame_number: int | None = None, episode_number: int | None = None
    ) -> None:
        """
        Recalcula o valor de epsilon com base no método que resulta no menor valor.

        Se novos valores não forem passados, os atuais da classe serão usados.

        Parâmetros
        ----------
        frame_number : int, opcional
            O número de frames passados.
        episode_number : int, opcional
            O número de episódios passados.
        """
        if frame_number is not None:
            self.frame_number = frame_number
        if episode_number is not None:
            self.episode_number = episode_number

        self.epsilon_decay = 0.0
        self.epsilon = self.epsilon_max - (
            (self.epsilon_max - self.epsilon_min)
            * self.episode_number / self.n_episodes_until_decay
        )
        if self.n_frames_until_decay is not None:
            decay = (
                (self.epsilon_max - self.epsilon_min) / self.max_episodes
            )
            epsilon = self.epsilon_max - self.epsilon_decay * self.episode_reward
            if epsilon <= self.epsilon:
                self.epsilon = epsilon
                self.epsilon_decay = decay

    def choose_action(self, observation: np.ndarray, previous_reward: float) -> int:
        """
        Escolhe uma ação com base na observação atual do ambiente.

        Parâmetros
        ----------
        observation : numpy.ndarray
            A observação atual do ambiente.
        previous_reward : float
            A recompensa recebida na última ação.
        """
        # Usa o valor gerado aleatoriamente ao final do episódio anterior para definir
        # quantos frames são ignorados ao começo desse, então retorna a ação nula para todos
        if self.remaining_skips > 0:
            self.remaining_skips -= 1
            return 0

        frame_index = self.frame_number % self.n_frames_per_action
        self.frame_number += 1
        with self.cpu:
            resized = tf.image.resize(np.expand_dims(observation, axis=2), self.input_size)

        # Armazena em float32 para possibilitar o uso da rede neural
        # Usa um slice para manter a terceira dimensão e possibilitar o broadcast
        self.recent_observations[:, :, frame_index:frame_index + 1] = resized
        # Acumula a recompensa até o momento da ação
        self.recent_reward += previous_reward

        # Enquanto não acumular frames suficientes, repete a última ação
        if frame_index < self.n_frames_per_action - 1:
            return self.action

        self.episode_reward += self.recent_reward
        self.recent_reward = 0.0
        # Cria a dimensão de batch e normaliza os valores
        stacked_frames = self.recent_observations[np.newaxis] / 255.0
        # Explora um número fixo de vezes antes de começar a usar a abordagem epsilon-greedy no treinamento
        if self.is_training and (self.action_history.size < self.n_random_actions or self.epsilon > self.rng.random()):
            self.action = self.rng.integers(0, self.n_actions)
        else:
            self.action = self.model.infer(stacked_frames)

        if not self.is_training:
            return self.action

        self.epsilon -= self.epsilon_decay
        # Armazena em uint8 para economizar memória
        self.observation_history.append(self.recent_observations)
        self.reward_history.append(previous_reward)
        self.action_history.append(self.action)

        if self.action_history.size % self.update_network_action_interval == 0:
            self.model.train(
                self.gamma,
                *self._sample_memory()
            )

        if self.action_history.size % self.update_target_network_action_interval == 0:
            # Na primeira atualização dos pesos, copia os valores diretamente
            if self.action_history.size // self.update_target_network_action_interval == 1:
                tau = 0.5
            else:
                tau = self.tau
            for target, weights in zip(
                self.target_model.trainable_variables, self.model.trainable_variables
            ):
                target.assign(tau * weights + (1 - tau) * target)

        return self.action

    def on_episode_end(
        self, episode_reward_history: list[float], episode_length_history: list[int]
    ) -> bool:
        """
        Atualiza o estado interno do agente e retorna se ele está pronto para encerrar o episódio.

        Isso ocorre quando um frame stack está pronto durante o treino, ou em qualquer momento
        durante a inferência. Também salva a memória se estiver no último episódio.
        """
        frame_index = self.frame_number % self.n_frames_per_action
        if self.is_training:
            # Enquanto o stack não estiver completo, não encerra o episódio
            if frame_index > 0:
                return False

            n = min(len(episode_reward_history), self.n_top_saved)
            best = np.partition(episode_reward_history, -n)[-n:]
            if self.episode_reward >= np.min(best):
                self.save(
                    include_memory=False,
                    episode_reward_history=episode_reward_history,
                    episode_length_history=episode_length_history
                )
        else:
            # Descarta os frames que não completaram o stack
            self.frame_number -= frame_index

        self.running_reward = np.mean(episode_reward_history[-100:])
        self.episode_number += 1
        self.adjust_epsilon()
        # Redefine a informação baseada no episódio
        self.episode_reward = 0.0
        self.recent_reward = 0.0
        self._set_skip()

        return True

    def is_training_done(
        self, episode_reward_history: list[float], episode_length_history: list[int]
    ) -> bool:
        """
        Retorna se a métrica objetivo foi atingida.

        Durante o treinamento, isso é quando a média móvel da recompensa atinge o valor alvo
        ou quando o número máximo de episódios é atingido.
        Durante a avaliação, isso é sempre falso.
        """
        conditions = (
            self.is_training
            and self.running_reward > self.target_reward
            or len(episode_length_history) > self.max_episodes
        )
        # Não faz sentido manter cópias menores ou checkpoints da memória, então ela
        # só é salva por chamadas de função do usuário ou ao final do treinamento
        if conditions:
            self.save(
                episode_reward_history=episode_reward_history,
                episode_length_history=episode_length_history
            )
        return conditions

    def memory_to_dict(self) -> dict[str, CircularList]:
        """Retorna um dicionário com os dados não copiados da memória de replay."""
        return {
            "observations": self.observation_history,
            "rewards": self.reward_history,
            "actions": self.action_history
        }

    def _read_checkpoints(self) -> tuple[TextIOWrapper | None, dict[str, Any], int]:
        """Obtém o arquivo de histórico, os dados e o índice do modelo atual em uma ordenação decrescente."""
        info_path = self.path / "info.json"
        if not info_path.exists():
            return None, {}, 0
        file = open(info_path, "r+")
        info = json.load(file)

        get_reward = lambda key: info[key]["episode_reward"]
        best = sorted(info, key=get_reward, reverse=True)
        index = 0
        while index < len(info) and get_reward(best[index]) >= self.episode_reward:
            index += 1
        return file, info, index

    def read_checkpoints(self) -> dict[str, Any]:
        """
        Retorna um dicionário com informações sobre cada checkpoint do modelo, ordenado por última
        recompensa obtida e podendo estar vazio caso o modelo não tenha sido salvo anteriormente.
        """
        file, info, index = self._read_checkpoints()
        if file is not None:
            file.close()
        return info

    def save(
        self, include_memory: bool = True, include_params: bool = True,
        episode_reward_history: list[float] = [], episode_length_history: list[int] = [],
        compression: Literal["gzip", "szip", "lzf"] | None = "gzip"
    ) -> None:
        """
        Salva o modelo na pasta raiz da classe e na subpasta com o nome do modelo.

        Antes de realizar um salvamento manual, é recomendado renomear o modelo para evitar
        manter informações desatualizadas e/ou sobrescrever o histórico de treinamento.

        Parâmetros
        ----------
        include_memory : bool, default=True
            Se True, salva também a memória de replay do agente.
        include_params : bool, default=True
            Se True, salva também os parâmetros do agente e o histórico de recompensas.
        episode_reward_history : list[float], opcional
            Valores de recompensa ao longo dos episódios, incluídos em
            params.json quando `include_params=True`.
        episode_length_history : list[int], opcional
            Valores de tamanho de episódios em número de ações, incluídos em
            params.json quando `include_params=True`.
        compression : {"gzip", "szip", "lzf", None}, default="gzip"
            O algoritmo de compressão usado para salvar os dados da memória.
            Estão listados em ordem decrescente de tempo e compressão.
        """
        file, info, index = self._read_checkpoints()
        if file is None:
            file = open(self.path / "info.json", "w")

        if index < self.n_top_saved:
            now = self.now_formatted()
            keys = list(info.keys())
            # Insere o modelo na posição correta
            keys.insert(index, now)
            info[now] = {
                "frame_number": self.frame_number,
                "memory_size": self.observation_history.size,
                "episode_reward": self.episode_reward,
                "running_reward": self.running_reward,
                "episode_number": len(episode_reward_history)
            }
            info = {key: info[key] for key in keys[:self.n_top_saved]}
            file.seek(0)
            json.dump(info, file, indent=4, check_circular=False, default=str)
            file.truncate()
        file.close()

        if include_memory:
            with h5py.File(self.memory_path, "w") as file:
                for name, value in self.memory_to_dict().items():
                    # Salva um arquivo por vez, mas cada chamada pode criar uma cópia
                    file.create_dataset(
                        name, data=value._correct_order(), compression=compression
                    )
        if include_params:
            params = {
                param: getattr(self, param) for param in
                chain(self.relevant_parameters, self.extra_parameters)
            }
            params["model_parameter_count"] = self.model.count_params()
            params["episode_rewards"] = episode_reward_history
            params["episode_lengths"] = episode_length_history
            with open(self.path / "params.json", "w") as file:
                json.dump(params, file, indent=4, check_circular=False, default=str)

        if index >= self.n_top_saved:
            return

        save_path = self.path / f"{now}.keras"
        if len(keys) > self.n_top_saved:
            overwritten_path = self.path / f"{keys[self.n_top_saved]}.keras"
            self.model.save(overwritten_path, overwrite=True)
            overwritten_path.rename(save_path)
        else:
            self.model.save(save_path)

    def rename(self, name: str | None, share_memory: bool | None = None) -> None:
        """
        Renomeia o modelo e suas pastas, desvinculando-o de outros modelos.

        Parâmetros
        ----------
        name : str, opcional
            O novo nome do modelo. Se não for especificado, um novo é criado.
        share_memory : bool, opcional
            Se True, compartilha a memória de replay entre modelos em `output_dir`.
            Se False, usa uma pasta de memória separada para o modelo.
            Se None, mantém o valor atual.
        """
        self.name = name or self.now_formatted()
        self.path = self.output_dir / self.name
        self.path.mkdir(parents=True, exist_ok=True)
        if share_memory is not None:
            self.share_memory = share_memory

    def load(
        self, checkpoint: int | str = 0, name: str | None = None, load_memory: bool | None = None
    ) -> None:
        """
        Carrega o modelo com nome `name` e número de checkpoint `n` do disco.

        Parâmetros
        ----------
        n : int ou str, default=0
            O número do modelo a ser carregado em uma lista ordenada por recompensa decrescente
            ou o nome do checkpoint a ser carregado.
        name : str
            O nome do modelo, que é o nome da pasta em que seus pesos estão salvos.
        load_memory : bool, opcional
            Se True, carrega também a memória de replay do agente. Por padrão, a memória
            só é carregada se um nome de modelo é passado.

        Notas
        -----
        O histórico de valores de loss e acurácia do modelo não é preservado.
        """
        if name is not None:
            self.rename(name)
            if load_memory is None:
                load_memory = True
        elif load_memory is None:
            load_memory = False

        info = self.read_checkpoints()
        if not info:
            raise FileNotFoundError("Arquivo de histórico não encontrado.")
        if isinstance(checkpoint, int):
            key = list(info.keys())[checkpoint]
        else:
            key = checkpoint
        model_path = self.path / f"{key}.keras"

        if not model_path.exists():
            raise FileNotFoundError(f"{model_path.parts[-1]} não encontrado.")
        keras.backend.clear_session()

        params_path = self.path / "params.json"
        if not params_path.exists():
            raise FileNotFoundError("Arquivo de parâmetros não encontrado.")
        with open(params_path, "r") as file:
            params: dict[str, Any] = json.load(file)
        params.pop("model_parameter_count")

        loaded_memory_length = params.pop("max_memory_length")
        if loaded_memory_length > self.max_memory_length and load_memory:
            self.max_memory_length = loaded_memory_length
            self._init_memory()

        self.__dict__.update(params)
        self.adjust_epsilon(
            info[key]["frame_number"], info[key]["episode_number"]
        )

        if load_memory:
            with h5py.File(self.memory_path, "r") as file:
                for name, memory in self.memory_to_dict().items():
                    size = file[name].shape[0]
                    if size > memory.capacity:
                        raise RuntimeError(
                            "A memória em disco é maior que a capacidade do agente e o valor registrado."
                        )
                    file[name].read_direct(memory.data[:size])
                    memory.size = size
                    memory.start = 0

        # Carrega o modelo sem inicializar a classe customizada
        self.model: DeepQNet = keras.models.load_model(model_path, compile=False)
        # Atualiza o modelo com os parâmetros da classe
        self.model.finish_setup(self.alpha)
        self.target_model = DeepQNet.from_config(self.model.get_config())
        self.target_model.set_weights(self.model.get_weights())
