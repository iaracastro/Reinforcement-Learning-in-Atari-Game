from abc import ABC, abstractmethod
from time import time
from typing import Any, Iterable, Optional

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame

from IPython.display import clear_output, display
from tensorflow import keras

from agents import HumanAgent, RandomAgent, DeepQAgent


class RewardCallback(ABC):
    @abstractmethod
    def __call__(self, observation: np.ndarray, reward: float) -> float:
        """
        Analisa a observação atual e retorna a nova recompensa.

        Parâmetros
        ----------
        observation : numpy.ndarray
            A observação atual do ambiente.
        reward : float
            A recompensa atual do ambiente.

        Retorna
        -------
        float
            O novo valor da recompensa.
        """
        ...


class TerminationCallback(ABC):
    @abstractmethod
    def __call__(
        self, observation: np.ndarray, reward: float, info: dict[str, Any]
    ) -> tuple[bool, bool, float]:
        """
        Analisa a observação e determina terminação e recompensa.

        Parâmetros
        ----------
        observation : numpy.ndarray
            A observação atual do ambiente.
        reward : float
            A recompensa atual do ambiente.
        info : dict[str, Any]
            Informações adicionais do ambiente.

        Retorna
        -------
        bool
            Indicador de perda de uma vida.
        bool
            Indicador de fim de episódio.
        float
            A nova recompensa.
        """
        ...


class PunishPacifistCallback(RewardCallback):
    """
    Callback que diminui a recompensa se o intervalo entre recompensas for muito grande.
    A redução é linear e começa após um número de ações sem recompensa definido.

    Parâmetros
    ----------
    reduction_factor : float, default=0.1
        Fator de redução da recompensa, isto é, razão da redução por recompensa anterior.
    actions_threshold : int, default=300
        Quantidade de ações sem recompensa para que a redução seja aplicada.
    """
    def __init__(self, reduction_factor: float = 0.1, actions_threshold: int = 300) -> None:
        self.factor = reduction_factor / actions_threshold
        self.actions_threshold = actions_threshold
        self.actions_since_last_reward = 0

    def __call__(self, observation: np.ndarray, reward: float) -> float:
        if reward > 0:
            if self.actions_since_last_reward > self.actions_threshold:
                reward -= self.factor * (
                    self.actions_since_last_reward - self.actions_threshold
                )
            self.actions_since_last_reward = 0
        else:
            self.actions_since_last_reward += 1
        return reward


class AtariGame:
    """
    Jogo representado por um ambiente Atari do gymnasium usado para treinar um agente.
    Note que jogos em "modo retrato", isto é, altura maior que largura, devem ser rotacionados.

    Parâmetros
    ----------
    game : str
        Nome do jogo a ser inicializado.
    agent : RLAgent ou RandomAgent
        Agente de Reinforcement Learning que irá controlar o jogo.
    rotated : bool
        Se a tela do jogo deve ser rotacionada em 90 graus.
    render_scale : int
        Multiplicador da resolução do jogo.
    human_agent : HumanAgent, opcional
        Agente humano que irá controlar o jogo.
    reward_callbacks : list[RewardCallback], opcional
        Lista de callables que, a cada ação do agente, recebem a observação atual e a recompensa e
        retornam a nova recompensa. As chamas são feitas na ordem da lista.
    termination_callback : TerminationCallback, opcional
        Callable que, a cada ação do agente, recebe a observação atual e a informação do ambiente e
        retorna um booleano indicando se uma vida foi perdida e a nova recompensa. Para jogos que não possuem vidas,
        o retorno deve indicar se o episódio foi terminado.
    print_info_seconds_interval : float, default=2.0
        Intervalo de tempo em segundos entre cada exibição de informações do jogo no treinamento.
    plot_interval_seconds : float, default=20.0
        O intervalo em segundos entre cada plot produzido de loss e reward do modelo.
    plot_scale : int, default=5
        Multiplicador do tamanho da figura do matplotlib.
    plot_episode_history_size : int, default=100
        O tamanho do histórico de recompensas por episódio exibido no plot.
    extra_plot_data : dict[str, tuple[int, Iterable[float]]], opcional
        Dicionário com o nome dos dados e uma tupla com o tamanho dos dados mais recentes.
        e os dados em si. Os dados são selecionados para o plot por meio de um slice.
        É esperado que esses dados sejam atualizados sem que a referência seja alterada.
    **kwargs
        Argumentos adicionais para o ambiente do jogo.

    Atributos
    ---------
    episode_durations : list[int]
        Lista com a duração de cada episódio.
    episode_reward_history : list[float]
        Lista com a recompensa total de cada episódio.

    Os atributos são atualizados apenas em chamadas de `train` ou `test`.

    Notas
    -----
    Algumas teclas podem interagir com o jogo na janela do pygame:
    1. O jogo pode ser interrompido a qualquer momento pressionando Esc, o que reinicia o ambiente
    e atualiza o agente.
    2. Ao pressionar R, o ambiente é reiniciado no modo de jogo, mas o modo de treinamento apenas
    registra a tecla e encerra a execução ao fim do próximo episódio. Pressionar novamente cancela.
    3. Pressionar Tab alterna entre renderizar e não renderizar no modo de treinamento e entre o
    agente humano e o de Reinforcement Learning no modo de jogo.

    Para interromper o jogo sem afetar o estado interno, use um KeyboardInterrupt.
    """
    def __init__(
        self,
		game: str,
		agent: DeepQAgent | RandomAgent,
		rotated: bool = False,
        render_scale: int = 4,
		human_agent: Optional[HumanAgent] = None,
        reward_callbacks: list[RewardCallback] = [],
        termination_callback: TerminationCallback | None = None,
        print_info_seconds_interval: float = 2.0,
        plot_interval_seconds: float | None = 20.0,
		plot_scale: int = 5,
        plot_episode_history_size: int = 100,
        extra_plot_data: dict[str, tuple[int, list[float]]] = {},
        random_seed: int | None = None,
        **kwargs
    ) -> None:
        self.env: gym.Env = gym.make(game, obs_type="grayscale", render_mode="rgb_array", **kwargs)
        self.env.metadata["render_fps"] = 60
        self.print_info_seconds_interval = print_info_seconds_interval
        self.termination_callback = termination_callback
        self.reward_callbacks = reward_callbacks
        self.random_seed = random_seed
        self.human_agent = human_agent
        self.last_print_time = time()
        self.rl_agent = agent
        self.game = game

        self.env.reset(seed=self.random_seed)
        values = self.env.step(0)
        self.observation: np.ndarray = values[0]
        self.info: dict[str, int] = values[4]
        self.lives: int = self.info["lives"]
        self.episode_reward = 0.0
        self.terminated = False
        self.life_lost = False
        self.reward = 0.0

        self.fig, self.axes, self.last_plot_time = None, None, 0.0
        self.inline_plot = "inline" in matplotlib.get_backend()
        self.plot_episode_history_size = plot_episode_history_size
        self.plot_interval_seconds = plot_interval_seconds
        self.set_plot_data(extra_plot_data, plot_scale)

        self.episode_reward_history: list[float] = []
        self.episode_length_history: list[int] = []
        self._screen: pygame.Surface = None
        self._clock = pygame.time.Clock()
        self.render_scale = render_scale
        self.exit_after_episode = False
        self.is_human_playing = False
        self.is_rendering = False
        self.should_exit = False
        self.rotated = rotated
        self.logging = True
        self.fps = 60

    def set_plot_data(
        self, data: dict[str, tuple[int, Iterable[float]]], plot_scale: int | None = None
    ) -> None:
        """
        Define os dados a serem plotados. O layout dos subplots é ajustado
        conforme a quantidade deles, com um limite de 3 por linha.

        Parâmetros
        ----------
        data : dict[str, tuple[int, Iterable[float]]]
            Dicionário com o nome dos dados e uma tupla com o tamanho do histórico e os dados.
            É esperado que esses dados sejam atualizados sem que a referência seja alterada.
        plot_scale : int, opcional
            Multiplicador do tamanho da figura do matplotlib.
        """
        if plot_scale is not None:
            self.plot_scale = plot_scale
        self.extra_plot_data = data
        if self.plot_interval_seconds:
            plt.ioff()
            n_plots = 2 + len(data)
            row_size = 2 if n_plots in [2, 4] else 3
            column_size = int(np.ceil(n_plots / row_size))
            self.fig, self.axes = plt.subplots(
                column_size, row_size,
                figsize=(self.plot_scale * row_size, self.plot_scale * column_size)
            )
            self.axes = self.axes.flatten()
            self.last_plot_time = time()

    def plot_results(self) -> None:
        """
        Exibe os plots de recompensa e duração de episódios do agente ao longo do tempo.
        Se especificados, também exibe os plots de dados adicionais.

        Se já existiu um plot, limpa o anterior, que pode estar sendo exibido
        inline em um notebook ou em uma janela separada, e exibe o novo.
        """
        for ax in self.axes:
            ax.cla()

        slice_ = slice(-self.plot_episode_history_size, None)
        x = range(*slice_.indices(len(self.episode_reward_history)))
        self.axes[0].set_title("Episode Rewards")
        self.axes[0].plot(x, self.episode_reward_history[slice_])
        self.axes[1].set_title("Episode Lengths")
        self.axes[1].plot(x, self.episode_length_history[slice_])

        for i, (name, (size, data)) in enumerate(self.extra_plot_data.items(), start=2):
            self.axes[i].set_title(name)
            self.axes[i].plot(data[-size:])

        if self.inline_plot:
            clear_output()
            display(self.fig)
        else:
            plt.show(block=False)
            plt.pause(1)

    def set_logging(self, value: bool | None = None) -> None:
        """Ativa/desativa o print de informações e os plots. Por padrão, alterna o estado atual."""
        if value is None:
            self.logging = not self.logging
        else:
            self.logging = value

    def _detect_life_lost(self) -> None:
        """Verifica se o número de vidas mudou e atualiza a variável correspondente."""
        self.life_lost = self.info["lives"] < self.lives

    def _update_agent(self) -> bool:
        """
        Usa o callback de fim de episódio do agente para atualizar seu estado e retorna
        se o treinamento deve ser encerrado.
        """
        is_done = self.rl_agent.on_episode_end(
            self.episode_reward_history, self.episode_length_history
        )
        while not is_done:
            self._step_env(self.rl_agent.choose_action(self.observation, self.reward))
            is_done = self.rl_agent.on_episode_end(
                self.episode_reward_history, self.episode_length_history
            )
        return self.rl_agent.is_training_done(
            self.episode_reward_history, self.episode_length_history
        )

    def _reset_env(self) -> None:
        """Redefine o ambiente do jogo com o estado aleatório da classe e atualiza o agente."""
        # Garante que o ambiente e o agente não serão atualizados se nenhum frame foi computado
        if self.info["episode_frame_number"] > 1:
            self.episode_length_history.append(self.info["episode_frame_number"])
            self.episode_reward_history.append(self.episode_reward)
            if self.is_human_playing:
                self.human_agent.reset()
            else:
                self.should_exit = self._update_agent() or self.should_exit
            # O frame 0 é ruído aleatório, então é ignorado
            self.env.reset(seed=self.random_seed)
            self.observation, self.reward, self.terminated, _, self.info = self.env.step(0)
            self.lives = self.info["lives"]
            self.episode_reward = 0.0

        self.should_exit = self.exit_after_episode or self.should_exit
        self.exit_after_episode = False

    def _step_env(self, action: int, ignore_callbacks: bool = False) -> None:
        """Realiza uma ação no ambiente e detecta mudanças na quantidade de vidas."""
        self.observation, self.reward, terminated, truncated, self.info = self.env.step(action)
        self.terminated = terminated or truncated

        if not ignore_callbacks:
            for callback in self.reward_callbacks:
                self.reward = callback(self.observation, self.reward)
        if self.termination_callback is None or ignore_callbacks:
            self._detect_life_lost()
        else:
            self.life_lost, self.terminated, self.reward = self.termination_callback(
                self.observation, self.reward, self.info
            )
        self.lives = self.info["lives"]

    def _set_agent(self, human_player: bool) -> None:
        """
        Define o agente atual que interage com o jogo.

        Parâmetros
        ----------
        human_player : bool
            Define se o agente atual é humano.
        """
        if human_player:
            if self.human_agent is None:
                raise RuntimeError("Não é possível jogar sem um agente humano.")
            self.human_agent.reset()
        self.is_human_playing = human_player

    def _create_display(self, fps: int) -> None:
        """
        Cria a janela do pygame e inicializa o jogo.

        Parâmetros
        ----------
        fps : int
            Quantidade de frames por segundo.
        """
        self.fps = fps
        if self._screen is None:
            if pygame.display.get_active():
                pygame.display.quit()
            pygame.init()
            window_size = (
                (160 * self.render_scale, 210 * self.render_scale) if self.rotated
                else (210 * self.render_scale, 160 * self.render_scale)
            )
            self._screen = pygame.display.set_mode(window_size)
            pygame.display.set_caption(self.game)

    def _blackout(self) -> None:
        """Desenha um fundo preto na tela e redefine o ambiente."""
        self._screen.fill((0, 0, 0))
        pygame.display.update()
        self._reset_env()

    def _render(self) -> None:
        """Renderiza o conteúdo da observação e mantém a taxa de quadros."""
        frame = self.env.render()
        frame = frame.repeat(self.render_scale, axis=1).repeat(self.render_scale, axis=0)
        if self.rotated:
            frame = np.flip(np.rot90(frame, k=3), axis=1)
        self._screen.blit(pygame.surfarray.make_surface(frame), (0, 0))
        pygame.display.update()
        self._clock.tick(self.fps)

    def _render_after_game_over(self) -> None:
        """
        Renderiza frames adicionais após perder uma vida ou encerrar um episódio para não haver
        cortes abruptos. Isso é feito entre o callback de terminação e a sinalização do ambiente.
        """
        self._render()
        # Se não há detecção adiantada, não há frames adicionais
        if self.termination_callback is None:
            return
        self.life_lost = False
        self.terminated = False
        i = 0
        # Espera até que o jogo tenha sido realmente terminado
        while not (self.life_lost or self.terminated):
            self._step_env(0, ignore_callbacks=True)
            self._render()
            i += 1
            if i > 180:
                raise RuntimeError(
                    "O jogo não foi terminado após 180 frames. "
                    "É provável que o callback de terminação não esteja correto."
                )
        # Ignora interações do usuário durante a espera
        pygame.event.clear()
        if self.is_human_playing:
            self.human_agent.reset()

    def _wait_for_input(self) -> None:
        """Fica em espera até que o usuário interaja com a janela."""
        while True:
            if pygame.event.get(pygame.QUIT) or pygame.event.get(pygame.KEYDOWN):
                return
            self._clock.tick(10)

    def set_seed(self, seed: int | None = None) -> None:
        """
        Define a semente aleatória do jogo aplicada a cada episódio.

        Parâmetros
        ----------
        seed : int, default=None
            Semente aleatória do jogo.
        """
        self.random_seed = seed

    def _try_run(self, no_human: bool, max_episodes: int, polling_interval: int | None) -> None:
        """
        Tenta executar o jogo, fechando o ambiente em caso de exceções e mantendo o estado
        interno em caso de KeyboardInterrupt.
        """
        if self.info["episode_frame_number"] < 2:
            # Cria novas listas em vez de usar clear para não apagar acessos externos
            self.episode_reward_history = []
            self.episode_length_history = []
        self.should_exit = False
        self.reward = 0.0
        try:
            if no_human:
                self._train_or_test(
                    max_episodes, polling_interval or self.rl_agent.n_frames_per_action * 2
                )
            else:
                self._play()
        except KeyboardInterrupt:
            return
        except Exception as e:
            self.close()
            raise e
        self._blackout()

    def play(self, human_player: bool = True, fps: int = 60) -> None:
        """
        Inicializa o jogo para ser jogado por um humano ou para visualizar o agente.

        Parâmetros
        ----------
        human_player : bool, default=True
            Define se o jogo deve ser jogado por um humano.
        fps : int, default=60
            Quantidade de frames por segundo.
        """
        self.is_rendering = True
        self._create_display(fps)
        self.rl_agent.set_training(False)
        self._set_agent(human_player)
        self._try_run(no_human=False, max_episodes=0, polling_interval=None)

    def train(
        self, max_episodes: int = 0, render: bool = False,
        fps: int = 60, polling_interval: int | None = None
    ) -> None:
        """
        Inicializa o jogo para treinar o agente.

        Parâmetros
        ----------
        max_episodes : int, default=0
            Quantidade máxima de episódios a serem executados. 0 significa infinito.
        render : bool, default=False
            Se o jogo deve ser renderizado.
        fps : int, default=60
            Quantidade de frames por segundo quando o jogo é renderizado.
        polling_interval : int, opcional
            Define a quantidade de frames entre cada verificação de interação do usuário.
            Por padrão, isso é o dobro do número de frames por ação do agente.
        """
        self._create_display(fps)
        self.is_rendering = render
        self.rl_agent.set_training(True)
        self._set_agent(human_player=False)
        self._try_run(no_human=True, max_episodes=max_episodes, polling_interval=polling_interval)

    def test(
        self, max_episodes: int = 1, render: bool = False,
        fps: int = 60, polling_interval: int | None = None
    ) -> None:
        """
        Inicializa o jogo para testar o agente. Equivalente a `train` com o agente
        definido como não treinável.

        Parâmetros
        ----------
        max_episodes : int, default=1
            Quantidade máxima de episódios a serem executados. 0 significa infinito.
        render : bool, default=False
            Se o jogo deve ser renderizado.
        fps : int, default=60
            Quantidade de frames por segundo quando o jogo é renderizado.
        polling_interval : int, opcional
            Define a quantidade de frames entre cada verificação de interação do usuário.
            Por padrão, isso é o dobro do número de frames por ação do agente.
        """
        self._create_display(fps)
        self.is_rendering = render
        self.rl_agent.set_training(False)
        self._set_agent(human_player=False)
        self._try_run(no_human=True, max_episodes=max_episodes, polling_interval=polling_interval)

    def reset_screen(self) -> None:
        """
        Usado para forçar que uma nova janela seja aberta para casos
        em que ela fecha por conta própria misteriosamente.
        """
        self._screen = None

    def _play(self) -> None:
        while not self.should_exit:
            if pygame.event.get(pygame.QUIT):
                self.should_exit = True
                self._reset_env()
                continue
            events = []
            captured = pygame.event.get(pygame.KEYDOWN)
            for event in captured:
                match event.key:
                    case pygame.K_r:
                        self._reset_env()
                    case pygame.K_TAB:
                        self._set_agent(human_player=not self.is_human_playing)
                    case pygame.K_ESCAPE:
                        self.should_exit = True
                        self._reset_env()
                    case _:
                        events.append(event.key)
            # Se alguma ação especial foi pressionada, volta ao início do loop
            if len(captured) != len(events):
                continue

            if self.is_human_playing:
                # Passa também as teclas que não estão mais sendo pressionadas pelo jogador
                action = self.human_agent.process_input(
                    keys_down=events,
                    keys_up=[
                        event.key for event in pygame.event.get(pygame.KEYUP)
                    ]
                )
            else:
                action = self.rl_agent.choose_action(self.observation, self.reward)

            self._step_env(action)
            self.episode_reward += self.reward

            if self.life_lost or self.terminated:
                self._render_after_game_over()
                if self.terminated:
                    self._reset_env()
            else:
                self._render()

    def _train_or_test(self, max_episodes: int, polling_interval: int) -> None:
        max_frames = self.rl_agent.max_actions_per_episode * self.rl_agent.n_frames_per_action

        while not self.should_exit:
            if self.info["frame_number"] % polling_interval == 0:
                if pygame.event.get(pygame.QUIT):
                    self.should_exit = True
                    self._reset_env()
                    continue
                for event in pygame.event.get(pygame.KEYDOWN):
                    # Verifica se alguma ação especial foi escolhida
                    match event.key:
                        case pygame.K_r:
                            self.exit_after_episode = not self.exit_after_episode
                        case pygame.K_TAB:
                            if self.is_rendering:
                                self._screen.fill((0, 0, 0))
                                pygame.display.update()
                            self.is_rendering = not self.is_rendering
                        case pygame.K_ESCAPE:
                            self.should_exit = True
                            self._reset_env()
                            continue

            if self.logging:
                now = time()
                if (self.print_info_seconds_interval
                and now - self.last_print_time >= self.print_info_seconds_interval):
                    self.last_print_time = now
                    print(self.info, "                ", end="\r")
                if (self.plot_interval_seconds
                and now - self.last_plot_time >= self.plot_interval_seconds):
                    self.plot_results()
                    self.last_plot_time = now

            action = self.rl_agent.choose_action(self.observation, self.reward)
            self._step_env(action)
            self.episode_reward += self.reward

            if self.life_lost or self.terminated or self.info["episode_frame_number"] >= max_frames:
                if self.is_rendering:
                    self._render_after_game_over()

                if self.terminated:
                    # Apenas uma das condições verificadas
                    info = self.info
                    self._reset_env()
                    self.should_exit = len(self.episode_length_history) == max_episodes

                    if self.should_exit:
                        self.plot_results()
                        print(info, "                ")
            elif self.is_rendering:
                self._render()

    def close(self) -> None:
        """Fecha o ambiente do jogo, a janela do pygame e limpa a sessão do Keras."""
        keras.backend.clear_session()
        self.reset_screen()
        self.env.close()
        pygame.quit()

    def __del__(self) -> None:
        pygame.quit()
