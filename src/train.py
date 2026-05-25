import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm
from flax import serialization

from model import ClothingClassifier

def bce_with_logits_loss(logits, labels):
    return optax.sigmoid_binary_cross_entropy(logits, labels).mean()

def create_train_state(rng, learning_rate, num_classes):
    model = ClothingClassifier(num_classes=num_classes)
    dummy_input = jnp.ones([1, 224, 224, 3], dtype=jnp.float32)
    variables = model.init(rng, dummy_input, train=False)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, 
        params=variables['params'], 
        tx=tx
    )

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'], train=True)
        return bce_with_logits_loss(logits, batch['labels'])
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

def train_model(train_dataloader, num_classes=228, num_epochs=10):
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, learning_rate=1e-4, num_classes=num_classes)

    for epoch in range(num_epochs):
        batch_losses = []
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                jax_batch = {
                    'image': jnp.array(batch['image']),
                    'labels': jnp.array(batch['labels'])
                }
                state, loss = train_step(state, jax_batch)
                batch_losses.append(loss)
                pbar.set_postfix({'loss': f"{float(loss):.4f}"})
        
        print(f"Epoch {epoch+1} completed. Avg Loss: {sum(batch_losses)/len(batch_losses):.4f}")
        
    return state

def save_model(params, filepath='model_weights.msgpack'):
    with open(filepath, 'wb') as f:
        bytes_output = serialization.to_bytes(params)
        f.write(bytes_output)
    print(f"Model successfully saved to {filepath}")

def load_model(empty_params, filepath='model_weights.msgpack'):
    with open(filepath, 'rb') as f:
        restored_params = serialization.from_bytes(empty_params, f.read())
    return restored_params