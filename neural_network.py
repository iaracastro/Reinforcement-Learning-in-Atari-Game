import numpy as np
import tensorflow as tf


@tf.keras.saving.register_keras_serializable(package="DeepQ")
class DeepQNet(tf.keras.Sequential):
    """
    Classe que define uma rede neural para o algoritmo Deep Q-Learning.

    Parâmetros
    ----------
    layers : list[keras.layers.Layer]
        Uma lista de camadas fora a entrada e saída da rede.
    learning_rate : float, default=0.001
        A taxa de aprendizado do otimizador. Não usada se o otimizador for especificado.
    optimizer : keras.optimizers.Optimizer, opcional
        O otimizador usado para treinar a rede. Por padrão, é usado Adam com `clipnorm=1.0`.
    name : str, default = "DeepQNet"
        O nome do modelo, apenas para identificação.

    Atributos
    ---------
    accuracy : list[float]
        Acurácia entre ação tomada e o estado futuro com maior valor Q.
    huber_loss : list[float]
        O valor da função de perda de Huber.
    mean_squared_error : list[float]
        O valor da função de perda de erro médio quadrático.
    """
    def __init__(
        self,
        layers: list[tf.keras.layers.Layer] | None = None,
        learning_rate: float = 0.001,
        optimizer: tf.keras.optimizers.Optimizer | None = None,
        name: str = "DeepQNet"
    ) -> None:
        super().__init__(layers, name)
        if layers is None:
            self.n_classes: int = None
            self.optimizer: tf.keras.optimizers.Optimizer = optimizer
        else:
            self.finish_setup(learning_rate, optimizer)

        self._mse_function = tf.keras.metrics.MeanSquaredError(name="mse")
        self._accuracy_function = tf.keras.metrics.Accuracy(name="acc")
        self._loss_function = tf.keras.losses.Huber()
        self.mean_squared_error: list[float] = []
        self.huber_loss: list[float] = []
        self.accuracy: list[float] = []

    def finish_setup(
        self, learning_rate: float | None = 0.001,
        optimizer: tf.keras.optimizers.Optimizer | None = None
    ) -> None:
        """Função chamada para definir variáveis após carregar o modelo do disco."""
        self.n_classes = self.output_shape[-1]
        self.optimizer = optimizer or tf.keras.optimizers.Adam(
            learning_rate=learning_rate, clipnorm=1.0
        )

    def infer(self, observations: np.ndarray) -> int:
        """
        Realiza uma inferência com a rede.

        Parâmetros
        ----------
        observations : numpy.ndarray
            O estado a ser inferido.

        Retorna
        -------
        int
            A ação de maior valor Q para o estado.
        """
        return self._infer(observations).numpy()

    def train(
        self, gamma: float, future_q_values: tf.Tensor, observations: np.ndarray,
        rewards: np.ndarray, actions: np.ndarray
    ) -> None:
        """
        Realiza um passo de treinamento com a rede a partir de uma amostra aleatória
        da memória de replay.

        Parâmetros
        ----------
        gamma : float
            O fator de desconto.
        future_q_values : tensorflow.Tensor
            As previsões de valores Q para cada próximo estado.
        observations : numpy.ndarray
            Os estados que serão usados para treinar a rede.
        rewards : numpy.ndarray
            As recompensas referentes a cada estado.
        actions : numpy.ndarray
            As ações tomadas no estado atual.
        """
        loss = self._train(
            gamma, future_q_values, observations, rewards, actions
        )
        self.mean_squared_error.append(self._mse_function.result().numpy())
        self.accuracy.append(self._accuracy_function.result().numpy())
        self.huber_loss.append(loss.numpy())

    @tf.function
    def _infer(self, observations: np.ndarray) -> tf.Tensor:
        q_values = self(observations, training=False)
        return tf.argmax(q_values, axis=1)[0]

    @tf.function
    def _train(
        self, gamma: float, future_q_values: tf.Tensor, observations: np.ndarray,
        rewards: np.ndarray, actions: np.ndarray
    ) -> tf.Tensor:
        best_actions = tf.argmax(future_q_values, axis=1)
        updated_q_values = rewards + (gamma * tf.cast(best_actions, tf.float32))

        # Garante que a perda seja calculada apenas para a ação tomada
        masks = tf.one_hot(actions, self.n_classes)

        with tf.GradientTape() as tape:
            # Estima os valores para cada estado na amostra
            q_values = self(observations, training=True)
            # Obtém o valor Q apenas para a ação tomada em cada estado
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calcula a perda entre os valores atualizados e os valores futuros estimados
            loss = self._loss_function(updated_q_values, q_action)

        gradients = tape.gradient(loss, self.trainable_variables)
        # Atualiza os pesos da rede
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self._accuracy_function.update_state(best_actions, tf.argmax(q_values, axis=1))
        self._mse_function.update_state(updated_q_values, q_action)

        return loss