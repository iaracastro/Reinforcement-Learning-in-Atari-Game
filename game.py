import gymnasium as gym
import numpy as np
import pygame

from tensorflow import keras
from typing import Optional

from agents import Agent, HumanAgent, RLAgent

class Game:
    """
    Jogo representado por um ambiente do gymnasium usado para treinar um agente.
    Note que jogos em "modo retrato", isto é, altura maior que largura, devem ser rotacionados.

    Parâmetros
    ----------
    game : str
        Nome do jogo a ser inicializado.
    agent : RLAgent
        Agente de Reinforcement Learning que irá controlar o jogo.
    rotated : bool
        Se a tela do jogo deve ser rotacionada em 90 graus.
    render_scale : int
        Multiplicador da resolução do jogo.
    human_agent : Optional[HumanAgent]
        Agente humano que irá controlar o jogo.
    **kwargs
        Argumentos adicionais para o ambiente do jogo.
    """
    def __init__(self, game: str, agent: RLAgent, rotated: bool = False,
                 render_scale: int = 4, human_agent: Optional[HumanAgent] = None,
                 random_seed: int | None = None, **kwargs) -> None:
        self.env = gym.make(game, obs_type="grayscale", render_mode="rgb_array", **kwargs)
        self.random_seed = random_seed
        self.human_agent = human_agent
        self.agent: Agent = None
        self.rl_agent = agent

        self.is_rendering: bool = False
        self.is_human_playing: bool = None
        self.screen: pygame.Surface = None
        self.clock = pygame.time.Clock()
        self.render_scale = render_scale
        self.rotated = rotated
        self.fps: int = 60

    def _reset_env(self) -> tuple[np.ndarray, dict[str, int | float]]:
        """Redefine o ambiente do jogo com o estado aleatório passado."""
        return self.env.reset(seed=self.random_seed)

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
            self.agent = self.human_agent
        else:
            self.agent = self.rl_agent
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
        if self.screen is None:
            pygame.init()
            window_size = (
                (160 * self.render_scale, 210 * self.render_scale) if self.rotated
                else (210 * self.render_scale, 160 * self.render_scale)
            )
            self.screen = pygame.display.set_mode(window_size)

    def _blackout(self) -> None:
        """Desenha um fundo preto na tela."""
        self.screen.fill((0, 0, 0))
        pygame.display.update()

    def _render(self, observation: np.ndarray) -> None:
        """Renderiza o conteúdo da observação e mantém a taxa de quadros."""
        frame = observation.repeat(self.render_scale, axis=1).repeat(self.render_scale, axis=0)
        if self.rotated:
            frame = np.flip(np.rot90(frame, k=3), axis=1)
        self.screen.blit(pygame.surfarray.make_surface(frame), (0, 0))
        pygame.display.update()
        self.clock.tick(self.fps)

    def _wait_for_input(self) -> None:
        """Fica em espera até que o usuário interaja com a janela."""
        while True:
            if pygame.event.get(pygame.QUIT) or pygame.event.get(pygame.KEYDOWN):
                return
            self.clock.tick(10)

    def train(self, render: bool = False, polling_interval: int = 20, fps: int = 60) -> None:
        """
        Inicializa o jogo para treinar o agente.

        Parâmetros
        ----------
        render : bool, default=False
            Se o jogo deve ser renderizado.
        polling_interval : int, default=20
            Define a quantidade de frames entre cada verificação de interação do usuário.
        fps : int
            Quantidade de frames por segundo quando o jogo é renderizado.
        """
        self.is_rendering = render
        self._set_agent(human_player=False)
        self._create_display(fps)
        try:
            self._train(polling_interval)
        except Exception as e:
            print("Treinamento encerrado devido a erro:", e, sep="\n\t")
            self.close()
        finally:
            # Garante que os pesos do agente são salvos mesmo se o treinamento for interrompido
            self.agent.save()

    def play(self, human_player: bool = False, fps: int = 60) -> None:
        """
        Inicializa o jogo para ser jogado por um humano ou para testar o agente.

        Parâmetros
        ----------
        fps : int
            Quantidade de frames por segundo.
        """
        self._set_agent(human_player)
        self._create_display(fps)
        self.is_rendering = True
        self._play()

    def _play(self) -> None:
        observation, info = self._reset_env()
        while info["episode_frame_number"] < self.agent.max_steps_per_episode:
            if pygame.event.get(pygame.QUIT):
                self._blackout()
                return
            events = []
            for event in pygame.event.get(pygame.KEYDOWN):
                # Verifica se alguma ação especial foi escolhida
                match event.key:
                    case pygame.K_r:
                        observation, info = self._reset_env()
                    case pygame.K_TAB:
                        self._set_agent(human_player=not self.is_human_playing)
                        observation, info = self._reset_env()
                    case pygame.K_ESCAPE:
                        self._blackout()
                        return
                    case _:
                        events.append(event.key)

            if self.is_human_playing:
                # Passa também as teclas que não estão mais sendo pressionadas pelo jogador
                action: int = self.agent.process_input(
                    keys_down=events,
                    keys_up=list(map(
                        lambda event: event.key, pygame.event.get(pygame.KEYUP)
                    ))
                )
            else:
                action: int = self.agent.choose_action(observation, False)

            observation, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                observation, info = self._reset_env()

            self._render(self.env.render())

    def _train(self, polling_interval: int) -> None:
        observation, info = self._reset_env()
        while True:
            if info["frame"] % polling_interval == 0:
                if pygame.event.get(pygame.QUIT):
                    self._blackout()
                    return
                for event in pygame.event.get(pygame.KEYDOWN):
                    # Verifica se alguma ação especial foi escolhida
                    match event.key:
                        case pygame.K_r:
                            observation, info = self._reset_env()
                        case pygame.K_TAB:
                            self.is_rendering = not self.is_rendering
                            if not self.is_rendering:
                                self.screen.fill((0, 0, 0))
                                pygame.display.update()
                            observation, info = self._reset_env()
                        case pygame.K_ESCAPE:
                            self._blackout()
                            return

            action = self.agent.choose_action(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)

            if terminated or truncated:
                observation, info = self._reset_env()

            if self.rendering:
                self.render(observation)

    def close(self) -> None:
        """Fecha o ambiente do jogo, a janela do pygame e tenta liberar a memória do agente."""
        keras.backend.clear_session()
        self.env.close()
        pygame.quit()
