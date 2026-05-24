import flax.linen as nn

class ClothingClassifier(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool):
        x = nn.Conv(features=64, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)

        x = nn.Dense(features=self.num_classes)(x)
        return x