from tensorflow import keras

class DeepQNet(keras.Sequential):
    """
    Classe que define uma rede neural para o algoritmo Deep Q-Learning.

    Parâmetros
    ----------
    layers : list[keras.layers.Layer]
        Uma lista de camadas fora a entrada e saída da rede.
    input_dim : tuple[int, ...]
        A dimensão da entrada da rede.
    num_classes : int
        O número de classes de saída da rede.
    loss : keras.losses.Loss
        A função de perda usada para treinar a rede.
    optimizer : keras.optimizers.Optimizer
        O otimizador usado para treinar a rede.
    metrics : list[keras.metrics.Metric]
        Uma lista de métricas usadas para avaliar a rede.
    name : str, default = "DeepQNet"
        O nome da rede.

    Atributos
    ---------
    metric_values : dict[str, list[float]]
        Um dicionário que mapeia o nome de cada métrica para uma lista
        com os valores para cada época de treinamento.
    """
    def __init__(
        self, layers: list[keras.layers.Layer],
        loss: keras.losses.Loss = keras.losses.Huber(),
        optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(clipnorm=1.0),
        metrics: list[keras.metrics.Metric] = [],
        name: str = "DeepQNet"
    ) -> None:
        super().__init__(layers, name)
        self.compile(optimizer, loss, metrics)
        self.metric_values: dict[str, list[float]] = {}
